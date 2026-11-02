"""Exploratory real-TIFF annular-mean constrained fits at cached TM centers.

Independent pixel-integrated Gaussian fitter, NOT the native Picasso backend.
No synthetic noise constants, truth, detection, tracking, or TARDIS are used.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'background_9x9_simulation'))
import calibrated_photometry as base
sys.path.insert(0, str(ROOT / 'background_9x9_simulation/local_background'))
from run_local_background import arbitrary_psf
sys.path.insert(0, str(ROOT / 'trackmate_python_bridge'))
from run_preview import roi_lib, read_mat_vertices


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=str) + '\n')


def fingerprint(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as s:
        for chunk in iter(lambda: s.read(1024 * 1024), b''):
            h.update(chunk)
    st = p.stat()
    return dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=h.hexdigest())


def objective(theta, data, box, variance, auxiliary):
    N = theta[0] * 1000
    x, y = theta[1:3]
    sx, sy = np.exp(theta[3:5])
    bg = theta[5] * 100
    P, dx, dy, dsx, dsy = base.psf_and_derivatives(box, x, y, sx, sy)
    loss, g, gb = base.marginal_noise_loss(data, N * P, bg, variance)
    grad = np.array([1000 * g @ P, N * g @ dx, N * g @ dy,
                     N * sx * g @ dsx, N * sy * g @ dsy, 100 * gb])
    xy, observed_sum, sum_variance = auxiliary
    p, gx, gy, gsx, gsy = arbitrary_psf(xy, x, y, sx, sy)
    n = len(xy)
    extra, gg, gbg = base.marginal_noise_loss(np.array([observed_sum]),
                                           np.array([N * p.sum()]), n * bg, sum_variance)
    grad += np.array([1000 * gg[0] * p.sum(), N * gg[0] * gx.sum(),
                      N * gg[0] * gy.sum(), N * sx * gg[0] * gsx.sum(),
                      N * sy * gg[0] * gsy.sum(), 100 * n * gbg])
    return loss + extra, grad


def covariance(theta, data, box, variance, aux):
    steps = np.array([1e-5, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
    H = np.empty((6, 6))
    for j in range(6):
        d = np.zeros(6); d[j] = steps[j]
        H[:, j] = (objective(theta + d, data, box, variance, aux)[1]
                   - objective(theta - d, data, box, variance, aux)[1]) / (2 * steps[j])
    H = (H + H.T) / 2
    if not np.isfinite(H).all() or np.linalg.eigvalsh(H)[0] <= 1e-8:
        return None
    S = np.diag([1000, 1, 1, np.exp(theta[3]), np.exp(theta[4]), 100])
    return S @ np.linalg.inv(H) @ S


def background_geometry(targets, all_frame, raw, roi_mask, c, camera):
    centers = all_frame[['x_px', 'y_px']].to_numpy(float)
    centers = centers[np.isfinite(centers).all(axis=1)]
    tree = cKDTree(centers)
    records, masks = [], []
    height, width = raw.shape
    movie = (raw.astype(float) - camera['baseline_adu']) * camera['sensitivity_e_per_adu'] / camera['em_gain']
    floor = camera['read_noise_e_rms'] ** 2
    for r in targets.itertuples():
        x, y = r.x_px, r.y_px
        outer, inner = c['annulus_outer_radius_px'], c['annulus_inner_radius_px']
        yy, xx = np.mgrid[max(0, int(np.floor(y-outer))):min(height, int(np.ceil(y+outer))+1),
                          max(0, int(np.floor(x-outer))):min(width, int(np.ceil(x+outer))+1)]
        radial = np.hypot(xx-x, yy-y)
        geom = (radial > inner) & (radial <= outer)
        px, py = xx[geom], yy[geom]
        n_geom = len(px)
        distances = tree.query(np.column_stack([px, py]))[0]
        keep = (distances > c['neighbor_core_radius_px']) & roi_mask[py, px] & (raw[py, px] > 0)
        px, py = px[keep], py[keep]
        pixels = np.column_stack([px, py]).astype(np.int16)
        data = movie[py, px]
        n = len(data)
        supported = n >= c['minimum_background_pixels']
        variance = max(float(np.var(data, ddof=1)), floor) if n > 1 else np.nan
        iid_variance = variance / n if n else np.nan
        sectors = np.floor(((np.arctan2(py-y, px-x) + np.pi) / (2*np.pi)) * 8).astype(int) % 8
        occupied = np.unique(sectors)
        jk = np.array([data[sectors != sector].mean() for sector in occupied if (sectors != sector).any()])
        jk_variance = (len(jk)-1)/len(jk) * float(np.sum((jk-jk.mean())**2)) if len(jk) >= c['minimum_jackknife_sectors'] else np.nan
        mean_variance = max(iid_variance, jk_variance) if np.isfinite(jk_variance) else iid_variance
        masks.append(pixels)
        records.append(dict(candidate_index=int(r.candidate_index), spot_id=int(r.spot_id), frame=int(r.frame),
            background_supported=bool(supported), annulus_geometric_pixels=n_geom, annulus_retained_pixels=n,
            mean_e=float(data.mean()) if n else np.nan, mean_adu=float(raw[py, px].mean()) if n else np.nan,
            variance_e2=variance, mean_iid_se_e=float(np.sqrt(iid_variance)),
            angular_jackknife_se_e=float(np.sqrt(jk_variance)) if np.isfinite(jk_variance) else np.nan,
            mean_constraint_se_e=float(np.sqrt(mean_variance)), occupied_angular_sectors=int(len(occupied)),
            sample_variance_to_nominal_Poisson_ratio=variance/(max(float(data.mean()), 0)+floor) if n else np.nan,
            uncertainty_status='empirical_background_proxy;variance_estimation_and_selection_not_fully_propagated'))
    return pd.DataFrame(records), masks, movie


def fit_one(item, raw, movie, bg_record, pixels, c):
    i, spot_id, frame, x, y = item
    b = c['box_px']; h = b//2
    ax, ay = int(np.floor(x+.5)), int(np.floor(y+.5))
    ox, oy = ax-h, ay-h
    out = dict(candidate_index=i, spot_id=spot_id, frame=frame,
        time_s=frame*c['frame_interval_s'], fit_attempted=False, fit_success=False,
        active_bound=False, fit_usable_interior=False, finite_uncertainty=False,
        fit_x_px=np.nan, fit_y_px=np.nan, photons_e=np.nan, bg_e_per_pixel=np.nan,
        sx_px=np.nan, sy_px=np.nan, x_se_nm=np.nan, y_se_nm=np.nan, N_se_e=np.nan,
        bg_se_e_per_pixel=np.nan, uncertainty_status='unavailable',
        fit_origin_x_px=ox, fit_origin_y_px=oy, unavailable_reason='',
        background_supported=bool(bg_record.background_supported), background_pixels=len(pixels),
        background_variance_e2=float(bg_record.variance_e2),
        background_constraint_se_e=float(bg_record.mean_constraint_se_e),
        calibration_status=c['calibration_status'], quantitatively_calibrated=False,
        estimator='independent_pixel_integrated_Poisson_signal_Gaussian_empirical_background',
        suspect_overlap=False, fit_review_flags='')
    if ox < 0 or oy < 0 or ox+b > raw.shape[1] or oy+b > raw.shape[0]:
        out['unavailable_reason'] = 'fitting_box_outside_image'; return out
    source = raw[oy:oy+b, ox:ox+b]
    if np.any(source == 0):
        out['unavailable_reason'] = 'unknown_zero_pixels_in_fitting_box'; return out
    if not bg_record.background_supported:
        out['unavailable_reason'] = 'too_few_retained_annular_pixels'; return out
    data = movie[oy:oy+b, ox:ox+b].ravel()
    if np.any(data < 0):
        out['unavailable_reason'] = 'pixels_below_assumed_camera_offset'; return out
    variance = float(bg_record.variance_e2)
    aux = (pixels.astype(float)-[ox, oy], float(movie[pixels[:, 1], pixels[:, 0]].sum()),
           float(len(pixels)**2 * bg_record.mean_constraint_se_e**2))
    low, high = c['width_bounds_px']
    bounds = [(0, None), (0, b-1), (0, b-1), (np.log(low), np.log(high)), (np.log(low), np.log(high)), (0, None)]
    bg = max(float(bg_record.mean_e), 0)
    options = []
    for sigma in c['initial_sigma_multistart_px']:
        P = base.psf_and_derivatives(b, x-ox, y-oy, sigma, sigma)[0]
        N = max(30, float(np.sum(data-bg))/P.sum())
        seed = np.array([N/1000, x-ox, y-oy, np.log(sigma), np.log(sigma), bg/100])
        options.append(minimize(objective, seed, args=(data, b, variance, aux), jac=True,
            method='L-BFGS-B', bounds=bounds,
            options=dict(maxiter=c['fit_max_iterations'], ftol=1e-10, gtol=1e-5, maxls=40)))
    opt = min(options, key=lambda v: v.fun)
    t = opt.x; N = t[0]*1000; sx, sy = np.exp(t[3:5]); bg = t[5]*100
    active = (N <= 1e-3 or bg <= 1e-3 or min(sx, sy) <= low+1e-5 or max(sx, sy) >= high-1e-5
              or min(t[1:3]) <= 1e-5 or max(t[1:3]) >= b-1-1e-5)
    usable = bool(opt.success and np.isfinite(t).all() and N > 0 and not active)
    P = base.psf_and_derivatives(b, *t[1:3], sx, sy)[0]
    predicted = N*P+bg
    residual = data-predicted
    yy, xx = np.mgrid[:b, :b]
    peaks = (residual.reshape(b, b) == maximum_filter(residual.reshape(b, b), size=3))
    peaks &= np.hypot(xx-t[1], yy-t[2]) > 2
    peaks &= residual.reshape(b, b) > 4*np.sqrt((variance+N*P).reshape(b, b))
    peaks[0, :] = peaks[-1, :] = False; peaks[:, 0] = peaks[:, -1] = False
    overlap = bool(peaks.any())
    out.update(fit_attempted=True, fit_success=bool(opt.success), fit_message=str(opt.message),
        iterations=int(opt.nit), negative_log_likelihood=float(opt.fun), active_bound=bool(active),
        fit_usable_interior=usable, fit_x_px=float(t[1]+ox), fit_y_px=float(t[2]+oy),
        photons_e=float(N), bg_e_per_pixel=float(bg), sx_px=float(sx), sy_px=float(sy),
        captured_psf_fraction=float(P.sum()), signal_inside_box_e=float(N*P.sum()),
        refinement_shift_px=float(np.hypot(t[1]+ox-x, t[2]+oy-y)),
        residual_rms_e=float(np.sqrt(np.mean(residual**2))),
        standardized_residual_rms=float(np.sqrt(np.mean(residual**2/(variance+N*P)))),
        suspect_overlap=overlap, fit_review_flags='suspect_residual_overlap' if overlap else '')
    if usable:
        C = covariance(t, data, b, variance, aux)
        if C is not None and np.isfinite(C).all() and np.all(np.diag(C) > 0):
            out.update(N_se_e=float(np.sqrt(C[0, 0])), bg_se_e_per_pixel=float(np.sqrt(C[5, 5])),
                x_se_nm=float(np.sqrt(C[1, 1])*c['pixel_size_nm']), y_se_nm=float(np.sqrt(C[2, 2])*c['pixel_size_nm']),
                corr_N_bg=float(C[0, 5]/np.sqrt(C[0, 0]*C[5, 5])), finite_uncertainty=True,
                uncertainty_status='full_observed_Hessian_with_nuisances;empirical_variance_fixed;coverage_unvalidated')
        else:
            out['uncertainty_status'] = 'nonpositive_or_singular_observed_Hessian'
    elif active:
        out['uncertainty_status'] = 'active_bound;covariance_unavailable'
    else:
        out['uncertainty_status'] = 'nonconverged;covariance_unavailable'
    return out


def math_checks():
    checks = base.validate_math()
    # Verify convolution at real-model variances and the ring-coupled gradient.
    for variance in (2.56, 30.):
        lam, z = 24.5, 117.
        loss = base.marginal_noise_loss(np.array([z]), np.array([lam]), 100., variance)[0]
        n = np.arange(600)
        ref = -logsumexp(-lam+n*np.log(lam)-gammaln(n+1)-.5*np.log(2*np.pi*variance)-(z-100-n)**2/(2*variance))
        np.testing.assert_allclose(loss, ref, atol=1e-11)
    yy, xx = np.mgrid[-12:13, -12:13]
    ring = (np.hypot(xx, yy) > 6) & (np.hypot(xx, yy) <= 12)
    xy = np.column_stack([xx[ring]+3, yy[ring]+3]).astype(float)
    aux = (xy, float(len(xy)*155), float(len(xy)*30))
    theta = np.array([.8, 3.2, 2.8, np.log(1.45), np.log(1.6), 1.55])
    data = np.random.default_rng(182).normal(167, 7, 49)
    gradient = objective(theta, data, 7, 30., aux)[1]
    finite = []
    for j in range(6):
        d = np.zeros(6); d[j] = 1e-5
        finite.append((objective(theta+d, data, 7, 30., aux)[0]-objective(theta-d, data, 7, 30., aux)[0])/(2e-5))
    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=2e-5)
    roi = np.array([[0,0], [10,0], [10,10], [0,10]])
    np.testing.assert_array_equal(roi_lib.points_in_polygon(np.array([[0,5], [5,5], [11,5]]), roi), [True, True, False])
    checks.update(empirical_variance_convolution_checked=True, annulus_gradient_checked=True,
                  inclusive_ROI_boundary_checked=True)
    return checks


def stats(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                p10=float(np.quantile(a,.1)) if len(a) else None,
                p90=float(np.quantile(a,.9)) if len(a) else None,
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None)


def plots(out, table, summary):
    shown = table[table.histogram_eligible]
    clean = shown[shown.qc_unflagged_model_fit]
    review = shown[~shown.qc_unflagged_model_fit]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained')
    for ax, field, label in zip(axes, ['x_se_nm','y_se_nm'], ['X','Y']):
        v = shown[field].to_numpy()
        if len(v):
            edges = np.linspace(0, max(float(v.max())*1.05, 1), 36)
            ax.hist(clean[field], bins=edges, histtype='step', linewidth=1.8, color='#16818e', label=f'Unflagged: {len(clean)}')
            ax.hist(review[field], bins=edges, histtype='step', linewidth=1.4, color='#b350a7', label=f'Overlap review: {len(review)}')
            ax.axvline(float(clean[field].median()) if len(clean) else float(np.median(v)), c='#16818e', ls='--', label='Unflagged median')
            ax.set(xscale='linear', yscale='linear', xlim=(0, edges[-1]))
        ax.set(xlabel=f'{label} local model standard error (nm)', ylabel='Localizations',
               title=f'{label}: linear scale, full range')
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    first, last = summary['frames_inclusive']
    fig.suptitle(f'T1 C2 | Original frames {first}–{last} | TrackMate 0.28 um → 7x7 annular-mean constrained fit\n'
                 f'{len(shown)} finite interior fits in ROI / {summary["ROI_candidates"]} targets; ±1 SE, NOT validated coverage', fontsize=11)
    fig.savefig(out/'localization_uncertainty_histogram.png', dpi=175); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained')
    v = shown.photons_e.to_numpy()
    if len(v):
        cutoff = float(np.quantile(v, .99))
        edges = np.linspace(0, max(cutoff,1), 41)
        full = np.linspace(0, max(float(v.max())*1.05, 1), 41)
        for ax, bins in zip(axes, [edges, full]):
            ax.hist(clean.photons_e, bins=bins, histtype='step', linewidth=1.8, color='#16818e', label=f'Unflagged: {len(clean)}')
            ax.hist(review.photons_e, bins=bins, histtype='step', linewidth=1.4, color='#b350a7', label=f'Overlap review: {len(review)}')
            ax.legend(fontsize=8); ax.grid(alpha=.15)
        axes[0].set_title(f'Linear detail to pooled p99; {int((v>cutoff).sum())} above shown range')
        axes[1].set(xscale='linear', yscale='linear', xlim=(0, full[-1]),
                    title='Entire range (linear): all eligible values')
        for ax in axes:
            ax.set(xlabel='Fitted whole-PSF photoelectron-equivalent count', ylabel='Localizations')
    fig.suptitle('Same points as uncertainty histogram | Whole Gaussian integral, NOT only the 7x7 captured signal\n'
                 'Nominal conversion: (ADU − 100) × 0.46; local background jointly fitted; true counts unknown', fontsize=11)
    fig.savefig(out/'photoelectron_number_histogram.png', dpi=175); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); args = ap.parse_args()
    c = yaml.safe_load(args.config.read_text()); started = time.perf_counter()
    checks = math_checks()
    src = Path(c['source_preview'])
    preview = yaml.safe_load((src/'resolved_config.yaml').read_text())
    camera_path = (args.config.parent/c['nominal_camera_config']).resolve()
    camera = yaml.safe_load(camera_path.read_text())['camera']
    start, end = c['frames_inclusive']
    assert (start,end) == (1250,1300)
    assert preview['frame_start'] == start and preview['frame_stop_exclusive'] == end+1
    assert preview['radius_um'] == .28 and preview['threshold'] == 6.5
    assert camera['baseline_adu'] == 100 and camera['sensitivity_e_per_adu'] == .46 and camera['em_gain'] == 1
    assert c['pixel_size_nm'] == 117 and c['frame_interval_s'] == .02
    assert not any(c[k] for k in ['redetect','tracking','tardis','mitochondria_analysis'])
    tiff = Path(preview['input_tiff']); roi_file = Path(preview['roi_file']); mat = Path(preview['mat_file'])
    movie_file = next((src/'native_detection').glob('*/input_frames.npy'))
    protected = [tiff, roi_file, mat, src/'all_detections.csv', src/'roi_vertices.csv',
                 src/'frame_manifest.csv', src/'resolved_config.yaml', src/'output_fingerprints.json',
                 movie_file, camera_path, Path(base.__file__), ROOT/'trackmate_python_bridge/picasso_refinement.py']
    out = tiff.parent/c['output_relative_directory']/('T1_C2_r0p28um_box7_frames1250_1300_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT:', out, flush=True)
    shutil.copyfile(args.config, out/'config.yaml'); shutil.copyfile(camera_path, out/'nominal_camera_source.yaml')
    dump(out/'rules_before_run.json', c)
    print('Checking source fingerprints...', flush=True)
    before = [fingerprint(p) for p in protected]; dump(out/'input_fingerprints_before.json', before)
    assert before[0]['sha256'] == preview['expected_tiff_sha256']
    assert before[1]['sha256'] == preview['expected_roi_sha256']
    saved = json.loads((src/'output_fingerprints.json').read_text())
    assert next(p['sha256'] for p in before if p['path'] == str((src/'all_detections.csv').resolve())) == saved['all_detections_sha256']
    vertices = roi_lib.load_imagej_roi(roi_file)
    mat_vertices, record = read_mat_vertices(mat, preview['mat_record'])
    np.testing.assert_array_equal(vertices, mat_vertices)
    np.testing.assert_array_equal(vertices, pd.read_csv(src/'roi_vertices.csv').to_numpy())
    all_table = pd.read_csv(src/'all_detections.csv')
    assert len(all_table) == 11150 and all_table.spot_id.is_unique
    inside = roi_lib.points_in_polygon(all_table[['x_px','y_px']].to_numpy(), vertices)
    np.testing.assert_array_equal(inside, all_table.roi_inside.eq(1))
    all_table = all_table.copy(); all_table.insert(0, 'candidate_index', np.arange(len(all_table)))
    targets = all_table[inside].copy()
    assert len(targets) == c['expected_roi_candidates']
    cached_movie = np.load(movie_file, mmap_mode='r')
    assert cached_movie.shape == (51,684,428) and cached_movie.dtype == np.uint16
    manifest = pd.read_csv(src/'frame_manifest.csv')
    page_hashes = []
    with tifffile.TiffFile(tiff) as tf:
        assert len(tf.pages) == 2500
        for frame in range(start,end+1):
            raw = tf.pages[frame].asarray()
            np.testing.assert_array_equal(raw, cached_movie[frame-start])
            digest = hashlib.sha256(raw.tobytes()).hexdigest()
            expected = manifest.loc[manifest.frame.eq(frame), 'source_pixel_sha256'].iloc[0]
            assert digest == expected
            page_hashes.append(dict(frame=frame, pixel_sha256=digest))
    dump(out/'verified_source_pages.json', page_hashes)
    yy,xx = np.mgrid[:684,:428]
    roi_mask = roi_lib.points_in_polygon(np.column_stack([xx.ravel(),yy.ravel()]), vertices).reshape(684,428)
    rows, bg_rows, all_masks, mask_ids = [], [], [], []
    for frame in range(start,end+1):
        frame_targets = targets[targets.frame.eq(frame)]
        raw = np.asarray(cached_movie[frame-start])
        geometry, masks, movie = background_geometry(frame_targets, all_table[all_table.frame.eq(frame)], raw, roi_mask, c, camera)
        assert len(geometry) == len(frame_targets)
        items = [(int(r.candidate_index),int(r.spot_id),frame,float(r.x_px),float(r.y_px)) for r in frame_targets.itertuples()]
        # Geometry must be disjoint from fit pixels and exclude neighbors.
        for item,pixels in zip(items,masks):
            _,_,_,x,y = item; ox,oy = int(np.floor(x+.5))-3,int(np.floor(y+.5))-3
            assert not np.any((pixels[:,0]>=ox)&(pixels[:,0]<ox+7)&(pixels[:,1]>=oy)&(pixels[:,1]<oy+7))
            assert roi_mask[pixels[:,1],pixels[:,0]].all()
            assert (raw[pixels[:,1],pixels[:,0]] > 0).all()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c['workers']) as pool:
            pending = [pool.submit(fit_one, item, raw, movie, geometry.iloc[j], masks[j], c) for j,item in enumerate(items)]
            results = [p.result() for p in pending]
        rows.extend(results); bg_rows.append(geometry); all_masks.extend(masks); mask_ids.extend(frame_targets.candidate_index)
        if (frame-start)%5 == 0 or frame == end:
            print(f'Frame {frame}: {len(rows)}/{len(targets)} fitted; elapsed {time.perf_counter()-started:.1f}s', flush=True)
        if (frame-start)%10 == 0:
            pd.DataFrame(rows).to_csv(out/'checkpoint_fits.csv', index=False)
    fitted = pd.DataFrame(rows).set_index('candidate_index').reindex(targets.candidate_index).reset_index()
    np.testing.assert_array_equal(fitted.spot_id, targets.spot_id)
    np.testing.assert_array_equal(fitted.frame, targets.frame)
    finite = np.isfinite(fitted[['fit_x_px','fit_y_px']]).all(axis=1)
    fit_inside = np.zeros(len(fitted), bool)
    fit_inside[finite] = roi_lib.points_in_polygon(fitted.loc[finite,['fit_x_px','fit_y_px']].to_numpy(), vertices)
    fitted['fitted_center_inside_ROI'] = fit_inside
    fitted['histogram_eligible'] = fitted.fit_usable_interior & fitted.finite_uncertainty & fit_inside
    fitted['qc_unflagged_model_fit'] = fitted.histogram_eligible & ~fitted.suspect_overlap
    fitted['lpx_model_nm'] = fitted.x_se_nm
    fitted['lpy_model_nm'] = fitted.y_se_nm
    fitted['lpx_model_px'] = fitted.x_se_nm/c['pixel_size_nm']
    fitted['lpy_model_px'] = fitted.y_se_nm/c['pixel_size_nm']
    fitted['audit_status'] = np.select([~fitted.fit_attempted, ~fitted.fit_success, fitted.active_bound,
        ~fitted.finite_uncertainty, ~fitted.fitted_center_inside_ROI, fitted.suspect_overlap],
        ['unavailable_support','nonconverged','active_parameter_bound','unavailable_covariance','refined_center_outside_ROI','overlap_review'], default='unflagged_model_fit')
    fitted.to_csv(out/'all_ROI_target_fits.csv', index=False)
    bg_table = pd.concat(bg_rows, ignore_index=True)
    bg_table.to_csv(out/'annular_background_all_ROI_targets.csv', index=False)
    np.savez_compressed(out/'retained_annular_pixels.npz',
        xy_pixels=np.concatenate(all_masks), offsets=np.r_[0,np.cumsum([len(p) for p in all_masks])],
        candidate_index=np.array(mask_ids))
    original = all_table.drop(columns='candidate_index')
    joined = original.join(fitted.set_index('candidate_index').drop(columns=['spot_id','frame','time_s']).add_prefix('annular__'))
    # Original columns must survive exactly; no candidate deletion or renumbering.
    pd.testing.assert_frame_equal(joined[original.columns], original)
    joined.to_csv(out/'all_original_detections_with_new_fits.csv', index=False)
    roundtrip = pd.read_csv(out/'all_original_detections_with_new_fits.csv')
    pd.testing.assert_frame_equal(roundtrip[original.columns], original, check_exact=False, rtol=1e-14, atol=1e-14)
    counts = []
    for frame in range(start,end+1):
        q = fitted[fitted.frame.eq(frame)]
        d = dict(frame=frame,time_s=frame*.02,ROI_candidates=len(q),fit_attempted=int(q.fit_attempted.sum()),
                 finite_interior_ROI=int(q.histogram_eligible.sum()),unflagged=int(q.qc_unflagged_model_fit.sum()))
        for reason in sorted(fitted.audit_status.unique()): d[reason] = int(q.audit_status.eq(reason).sum())
        counts.append(d)
    pd.DataFrame(counts).to_csv(out/'frame_counts.csv', index=False)
    summary = dict(frames_inclusive=[start,end], number_frames=51, FOV_candidates=len(original), ROI_candidates=len(fitted),
        counts={str(k):int(v) for k,v in fitted.audit_status.value_counts().items()},
        fit_attempted=int(fitted.fit_attempted.sum()),converged=int(fitted.fit_success.sum()),
        interior_model_fits=int(fitted.fit_usable_interior.sum()), finite_interior_ROI=int(fitted.histogram_eligible.sum()),
        unflagged=int(fitted.qc_unflagged_model_fit.sum()),
        background_mean_e=stats(bg_table.mean_e), background_mean_ADU=stats(bg_table.mean_adu),
        background_variance_e2=stats(bg_table.variance_e2), background_constraint_SE_e=stats(bg_table.mean_constraint_se_e),
        variance_to_nominal_Poisson_ratio=stats(bg_table.sample_variance_to_nominal_Poisson_ratio),
        annular_pixel_counts=stats(bg_table.annulus_retained_pixels),
        unflagged_statistics={key:stats(fitted.loc[fitted.qc_unflagged_model_fit,key]) for key in
            ['x_se_nm','y_se_nm','photons_e','N_se_e','bg_e_per_pixel','sx_px','sy_px','standardized_residual_rms']},
        eligible_statistics={key:stats(fitted.loc[fitted.histogram_eligible,key]) for key in ['x_se_nm','y_se_nm','photons_e']},
        truth_available=False,accuracy_or_correlation_to_truth_not_computable=True,
        model_is_native_Picasso=False, noise_model=c['noise_model'], calibration_status=c['calibration_status'],
        coverage_validated=False,elapsed_s=time.perf_counter()-started, output_directory=str(out))
    dump(out/'summary.json', summary)
    plots(out, fitted, summary)
    print('Verifying read-only sources unchanged...', flush=True)
    after = [fingerprint(p) for p in protected]; dump(out/'input_fingerprints_after.json', after)
    assert before == after
    checks.update(all_original_11150_rows_preserved=True, all_2113_ROI_targets_preserved=True,
        original_TIFF_pages_match_cached_pixels=True, original_frames_preserved=True,
        ROI_and_MAT_vertices_equal=True, inclusive_ROI_boundary_tolerance_px=1e-9,
        no_coordinate_transform=True, annulus_disjoint_from_fit_pixels=True,
        all_retained_background_pixels_inside_cell_ROI=True, source_fingerprints_unchanged=True,
        simulation_variance_500_not_used=True, estimated_empirical_variance_not_camera_calibration=True,
        no_uncertainty_for_bound_or_nonconverged_fits=True, histograms_include_full_range=True,
        original_camera_offset_and_sensitivity_preserved=True)
    assert fitted.loc[fitted.active_bound | ~fitted.fit_success,['x_se_nm','y_se_nm','N_se_e']].isna().all().all()
    np.testing.assert_allclose(fitted.lpx_model_px*117, fitted.x_se_nm, equal_nan=True)
    dump(out/'validation.json', checks)
    dump(out/'field_units.json', dict(frame='original zero-based frame',time_s='seconds',
        fit_x_px='native C2 pixel',fit_y_px='native C2 pixel',photons_e='nominal whole-Gaussian photoelectron-equivalent count',
        signal_inside_box_e='nominal photoelectron-equivalent count inside 7x7 box',bg_e_per_pixel='nominal e-/pixel',
        sx_px='Gaussian standard deviation in pixels',sy_px='Gaussian standard deviation in pixels',
        x_se_nm='local full-Hessian standard error in nm; not validated CRLB',y_se_nm='same, Y',
        N_se_e='local whole-PSF N standard error in nominal e-',bg_se_e_per_pixel='local background standard error in nominal e-/pixel'))
    snap=out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(base.__file__),ROOT/'background_9x9_simulation/local_background/run_local_background.py',
              ROOT/'trackmate_python_bridge/run_preview.py',ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py']:
        shutil.copyfile(p,snap/p.name)
    dump(out/'versions.json', dict(python=sys.version,executable=sys.executable,
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','numba','pandas','matplotlib','tifffile','picassosr','PyYAML']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())])+'\n')
    (out/'REPORT_zh.txt').write_text(
        '真实 T1 C2 原始帧 1250–1300（含两端，51 帧）。复用 TrackMate 0.28 μm、阈值 6.5 检测缓存。\n'
        f'全视野 {len(original)} 行全部保留；细胞 ROI 内 {len(fitted)} 个目标；有限内点拟合 {summary["finite_interior_ROI"]}，无复核标记 {summary["unflagged"]}。\n'
        '采用独立像素积分二维椭圆 Gaussian，7×7 中心框；这不是原生 Picasso 后端。\n'
        '背景外圈 6<r≤12 px，在共同细胞 ROI 内且排除全部同帧 TrackMate 邻居 4 px 核心与零像素；至少 80 像素。\n'
        '使用全部保留像素均值，不选择最暗 25%；背景与信号联合拟合，并计入本点 PSF 外翼。\n'
        '背景均值约束方差取样本方差/n 与 8 扇区删一 jackknife 方差的较大值，避免无限精度固定背景。\n'
        '中心信号暂用 Poisson，背景用外圈实测方差的 Gaussian 近似。外圈均值不替代相机 offset。\n'
        '100 ADU offset、0.46 电子/ADU、1.6 电子 RMS 下限均为未经此次实测的名义值。没有使用模拟方差 500。\n'
        '外圈方差包含局部结构、残留光点与相机/处理噪声；不等于独立相机校准。\n'
        'X/Y 和 N 标准误由包括背景、信号、宽度的完整观测 Hessian 逆计算；外圈方差本身的不确定度尚未完整传播。\n'
        '不是原生 Picasso CRLB，不是已验证的实际定位误差或置信覆盖率。误差条的幅度为 ±1 个局部模型标准误。\n'
        '边界约束、未收敛、协方差不可用的行保留但不进入误差直方图；残差重叠标记的可用拟合单独显示。\n'
        '直方图要求精修中心也在同一 ROI 内；原 TrackMate ROI 选择不被改写。\n'
        'N 是拟合完整 Gaussian 的积分，不仅是 7×7 框内信号，也不是入射光子数。\n'
        '真实图像无真值，不能计算光电子数准确度、真实位置误差或真值相关系数；不预设每点 800。\n'
        '未重新检测，未运行轨迹、TARDIS、mask、去卷积或扩散分析。所有旧文件只读。\n')
    checkpoint=out/'checkpoint_fits.csv'
    if checkpoint.exists(): checkpoint.unlink()
    dump(out/'COMPLETED.json', dict(all_checks_passed=True,output_directory=str(out),elapsed_s=time.perf_counter()-started))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
