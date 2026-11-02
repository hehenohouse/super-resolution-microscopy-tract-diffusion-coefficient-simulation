"""Group-specific raw pair statistics, using the existing frozen ROI decomposition.

No localization, segmentation, registration or trajectory computation is invoked.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'picasso_roi_sensitivity'))
from roi_core import (decompose_roi, bootstrap_roi, scalar_row, distribution_table,
                      finite_ci, paired_difference, scopes)
from full_core import (frame_histograms as legacy_frame_histograms,
                       range_histograms, block_histograms, shared_resampling_plan)

GROUPS = ['On', 'Not-on']
LABELS = {'mito-compatible': 'On', 'non-mito-compatible': 'Not-on', 'unknown': 'unknown'}


def histogram(a, b, edges, same_frame=False):
    """Native coordinates converted to um before entry; no correspondence linking."""
    if not len(a) or not len(b):
        return np.zeros(len(edges)-1, np.int64), 0
    d = cdist(a, b).ravel()
    removed = int((d == 0).sum()) if same_frame else 0
    if same_frame:
        d = d[d > 0]
    return np.histogram(d, bins=edges)[0].astype(np.int64), removed


def arrays_by_group(table, c):
    answer = {}
    for name in GROUPS:
        g = table.loc[table.tardis_group == name]
        f = {int(frame): part[['fit_x_px', 'fit_y_px']].to_numpy(float)*c['pixel_size_nm']/1000
             for frame, part in g.groupby('frame')}
        answer[name] = [f.get(frame, np.empty((0, 2))) for frame in range(c['frames'])]
    return answer


def calculate_histograms(table, c, callback=None):
    """Return within/cross-group histograms with every original frame retained."""
    n, k = c['frames'], c['distance_bins']
    edges = np.linspace(0, c['distance_max_um'], k+1)
    a = arrays_by_group(table, c)
    total = np.zeros((2, n, k), np.int64)
    background = np.zeros_like(total)
    cross = np.zeros_like(total)  # 0 On->Not-on; 1 Not-on->On; NOT trajectories
    same_cross = np.zeros((n, k), np.int64)
    all_total = np.zeros((n, k), np.int64)
    all_background = np.zeros_like(all_total)
    zero = np.zeros((2, n), np.int64)
    for f in range(n):
        for g, name in enumerate(GROUPS):
            background[g, f], zero[g, f] = histogram(a[name][f], a[name][f], edges, True)
            if f+1 < n:
                total[g, f], _ = histogram(a[name][f], a[name][f+1], edges)
        # Include both ordered directions, excluding exact coincident coordinates.
        same_cross[f] = 2*histogram(a['On'][f], a['Not-on'][f], edges, True)[0]
        all_now = np.concatenate([a[name][f] for name in GROUPS])
        all_background[f] = histogram(all_now, all_now, edges, True)[0]
        if f+1 < n:
            cross[0, f] = histogram(a['On'][f], a['Not-on'][f+1], edges)[0]
            cross[1, f] = histogram(a['Not-on'][f], a['On'][f+1], edges)[0]
            all_next = np.concatenate([a[name][f+1] for name in GROUPS])
            all_total[f] = histogram(all_now, all_next, edges)[0]
        np.testing.assert_array_equal(total[:, f].sum(0)+cross[:, f].sum(0), all_total[f])
        np.testing.assert_array_equal(background[:, f].sum(0)+same_cross[f], all_background[f])
        if callback and ((f+1) % c['checkpoint_frames'] == 0 or f+1 == n):
            callback(f+1, total, background, cross)
    return dict(total=total, background=background, cross=cross,
                same_cross_background=same_cross, all_total=all_total,
                all_background=all_background, background_zero_removed=zero,
                edges_um=edges)


def independent_histogram_check(table, data, c, frames):
    """Explicit point enumeration, independent of cdist production calculation."""
    coordinates = arrays_by_group(table, c)
    edges = data['edges_um']
    checked = 0
    for g, name in enumerate(GROUPS):
        for f in frames:
            a = coordinates[name][f]
            b = coordinates[name][f+1] if f+1 < c['frames'] else np.empty((0, 2))
            ds = [float(np.sqrt(np.sum((x-y)**2))) for x in a for y in b]
            db = [float(np.sqrt(np.sum((x-y)**2))) for x in a for y in a if np.any(x != y)]
            np.testing.assert_array_equal(np.histogram(ds, edges)[0], data['total'][g, f])
            np.testing.assert_array_equal(np.histogram(db, edges)[0], data['background'][g, f])
            checked += 1
    return checked


def tail_diagnostics(tc, bc, th, bh, start, stop, c):
    width = c['distance_max_um']/c['distance_bins']
    mid = (np.arange(c['distance_bins'])+.5)*width
    tail = (mid >= c['background_fit_um'][0]) & (mid < c['background_fit_um'][1])
    t, b = tc[tail].astype(float), bc[tail].astype(float)
    if t.sum() > 0 and b.sum() > 0:
        tn, bn = t/t.sum(), b/b.sum()
        l1 = float(abs(tn-bn).sum())
        cosine = float(tn@bn/(np.linalg.norm(tn)*np.linalg.norm(bn)))
    else:
        l1 = cosine = np.nan
    blocks = list(range(start, stop, c['bootstrap']['primary_block_frames']))
    length = c['bootstrap']['primary_block_frames']
    return dict(
        tail_normalized_shape_L1=l1, tail_normalized_shape_cosine=cosine,
        B_tail_support_frames=int((bh[start:stop, tail].sum(1) > 0).sum()),
        Total_tail_support_source_frames=int((th[start:stop-1, tail].sum(1) > 0).sum()),
        B_tail_supported_L25_blocks=sum(bh[s:min(s+length, stop), tail].sum() > 0 for s in blocks),
        Total_tail_supported_L25_blocks=sum(th[s:min(s+length-1, stop-1), tail].sum() > 0 for s in blocks),
        L25_blocks=len(blocks))


def paired_curve_summary(a, b, point_a, point_b, minimum):
    """Original replicate IDs retained; one intersection for the whole curve."""
    common = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    enough = common.mean() >= minimum and np.isfinite(point_a).all() and np.isfinite(point_b).all()
    difference = np.full_like(a, np.nan)
    difference[common] = a[common]-b[common]
    low = high = a_low = a_high = b_low = b_high = np.full(a.shape[1], np.nan)
    if enough:
        low, high = np.quantile(difference[common], [.025, .975], axis=0)
        a_low, a_high = np.quantile(a[common], [.025, .975], axis=0)
        b_low, b_high = np.quantile(b[common], [.025, .975], axis=0)
    return dict(common=common, common_count=int(common.sum()), common_fraction=float(common.mean()),
                available=bool(enough), difference=difference, low=low, high=high,
                On_low=a_low, On_high=a_high, Not_on_low=b_low, Not_on_high=b_high)
