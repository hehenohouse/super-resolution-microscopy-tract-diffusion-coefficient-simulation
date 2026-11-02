"""Independent post-run audit of saved real annulus fits; no refitting."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import ndtr
import yaml

from run_real import covariance, dump, fingerprint, roi_lib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    out = args.output
    config = yaml.safe_load((out/'config.yaml').read_text())
    camera = yaml.safe_load((out/'nominal_camera_source.yaml').read_text())['camera']
    source = Path(config['source_preview'])
    original = pd.read_csv(source/'all_detections.csv')
    joined = pd.read_csv(out/'all_original_detections_with_new_fits.csv')
    pd.testing.assert_frame_equal(joined[original.columns], original, check_exact=False, rtol=1e-14, atol=1e-14)
    fits = pd.read_csv(out/'all_ROI_target_fits.csv')
    backgrounds = pd.read_csv(out/'annular_background_all_ROI_targets.csv')
    selected = original.loc[original.roi_inside.eq(1)]
    np.testing.assert_array_equal(fits.candidate_index, selected.index)
    np.testing.assert_array_equal(fits.spot_id, selected.spot_id)
    np.testing.assert_array_equal(fits.frame, selected.frame)
    np.testing.assert_allclose(fits.time_s, fits.frame*.02, rtol=0, atol=1e-14)
    pixels = np.load(out/'retained_annular_pixels.npz')
    np.testing.assert_array_equal(pixels['candidate_index'], fits.candidate_index)
    movie = np.load(next((source/'native_detection').glob('*/input_frames.npy')), mmap_mode='r')
    vertices = pd.read_csv(source/'roi_vertices.csv').to_numpy()
    # Independent direct distances check exclusions for ALL retained pixels.
    for j, row in enumerate(fits.itertuples()):
        xy = pixels['xy_pixels'][pixels['offsets'][j]:pixels['offsets'][j+1]]
        xy0 = original.iloc[row.candidate_index][['x_px','y_px']].to_numpy(float)
        radius = np.sqrt(np.sum((xy-xy0)**2, axis=1))
        assert np.all(radius > 6) and np.all(radius <= 12)
        neighbors = original.loc[original.frame.eq(row.frame), ['x_px','y_px']].to_numpy(float)
        distance2 = np.sum((xy[:,None,:]-neighbors[None,:,:])**2, axis=2)
        assert np.all(distance2.min(axis=1) > 16)
        assert roi_lib.points_in_polygon(xy.astype(float), vertices).all()
        raw = movie[row.frame-1250]
        values = (raw[xy[:,1],xy[:,0]].astype(float)-100)*.46
        bg = backgrounds.iloc[j]
        np.testing.assert_allclose(values.mean(), bg.mean_e, rtol=1e-13)
        np.testing.assert_allclose(max(values.var(ddof=1),2.56), bg.variance_e2, rtol=1e-13)
        assert bg.mean_constraint_se_e**2 + 1e-12 >= bg.variance_e2/len(xy)
    good = fits[fits.histogram_eligible]
    assert len(good) == 1951
    assert np.isfinite(good[['photons_e','x_se_nm','y_se_nm','N_se_e']]).all().all()
    assert (good[['photons_e','x_se_nm','y_se_nm','N_se_e']] > 0).all().all()
    assert roi_lib.points_in_polygon(good[['fit_x_px','fit_y_px']].to_numpy(), vertices).all()
    assert fits.loc[fits.active_bound,['x_se_nm','y_se_nm','N_se_e']].isna().all().all()
    np.testing.assert_allclose(fits.x_se_nm, fits.lpx_model_px*117, equal_nan=True)
    np.testing.assert_allclose(fits.y_se_nm, fits.lpy_model_px*117, equal_nan=True)
    # Reconstruct selected PSFs independently using CDF pixel integrals and
    # reproduce saved nuisance covariance without a new optimization.
    sample = good.iloc[np.linspace(0,len(good)-1,7).astype(int)]
    for row in sample.itertuples():
        j = int(np.flatnonzero(fits.candidate_index.eq(row.candidate_index))[0])
        xy = pixels['xy_pixels'][pixels['offsets'][j]:pixels['offsets'][j+1]].astype(float)
        raw = movie[row.frame-1250]
        converted = (raw.astype(float)-100)*.46
        data = converted[row.fit_origin_y_px:row.fit_origin_y_px+7,row.fit_origin_x_px:row.fit_origin_x_px+7]
        x, y = row.fit_x_px-row.fit_origin_x_px, row.fit_y_px-row.fit_origin_y_px
        grid=np.arange(7)
        px=ndtr((grid+.5-x)/row.sx_px)-ndtr((grid-.5-x)/row.sx_px)
        py=ndtr((grid+.5-y)/row.sy_px)-ndtr((grid-.5-y)/row.sy_px)
        signal=row.photons_e*np.outer(py,px)
        residual=data-(signal+row.bg_e_per_pixel)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)),row.residual_rms_e,rtol=1e-10)
        np.testing.assert_allclose(signal.sum(),row.signal_inside_box_e,rtol=1e-12)
        bg=backgrounds.iloc[j]
        aux=(xy-[row.fit_origin_x_px,row.fit_origin_y_px],float(converted[xy[:,1].astype(int),xy[:,0].astype(int)].sum()),
             len(xy)**2*bg.mean_constraint_se_e**2)
        theta=np.array([row.photons_e/1000,x,y,np.log(row.sx_px),np.log(row.sy_px),row.bg_e_per_pixel/100])
        C=covariance(theta,data.ravel(),7,row.background_variance_e2,aux)
        np.testing.assert_allclose([np.sqrt(C[1,1])*117,np.sqrt(C[2,2])*117,np.sqrt(C[0,0])],
                                   [row.x_se_nm,row.y_se_nm,row.N_se_e],rtol=1e-6)
    summary=json.loads((out/'summary.json').read_text())
    counts=pd.read_csv(out/'frame_counts.csv')
    np.testing.assert_array_equal(counts.frame,np.arange(1250,1301))
    assert counts.ROI_candidates.sum()==2113
    assert counts.finite_interior_ROI.sum()==len(good)
    assert counts.unflagged.sum()==summary['unflagged']==1950
    hashes=json.loads((out/'input_fingerprints_before.json').read_text())
    for item in hashes:
        assert item==fingerprint(item['path'])
    for name in ['localization_uncertainty_histogram.png','photoelectron_number_histogram.png']:
        with Image.open(out/name) as im:
            im.verify()
    checks=dict(all_original_11150_rows_and_fields_retained=True,all_2113_targets_and_frames_retained=True,
        independently_checked_all_retained_pixel_distances_and_neighbors=True,
        all_background_mean_and_variance_estimates_reproduced=True,
        seven_independent_pixel_PSF_residual_and_covariance_checks_passed=True,
        no_double_nm_conversion=True,bound_fits_have_no_uncertainty=True,
        per_frame_histogram_counts_match=True,source_hashes_unchanged=True,
        cached_fits_only_no_new_optimization=True,PNG_files_decode=True)
    dump(out/'independent_validation.json',checks)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
