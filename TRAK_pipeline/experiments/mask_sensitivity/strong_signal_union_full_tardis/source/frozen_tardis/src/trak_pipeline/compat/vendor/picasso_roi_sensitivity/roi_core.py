"""ROI selection and supported signed distributions; no localization calls."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'picasso_full_sensitivity'))
from full_core import (EXCLUSIONS, key_name, frame_histograms, range_histograms,
                       block_histograms, decompose_counts, point_summary as _point_summary, frame_qc)
from pilot_core import dump_json, sha256_file
from tardis_tau1.locbio import load_imagej_roi, points_in_polygon


def scopes(c):
    return [('full', 0, c['frames'])] + [
        (f'f{s:04d}_{s+c["segment_frames"]-1:04d}', s, s+c['segment_frames'])
        for s in range(0, c['frames'], c['segment_frames'])]


def point_summary(table):
    if len(table):
        return _point_summary(table)
    return {'points':0, **{key+'_'+q:np.nan
        for key in ['lpx_nm','lpy_nm','photons','bg','sx','sy','deviance_per_dof']
        for q in ['q10','q25','median','q75','q90']}}


def read_mat_roi(path, name):
    with h5py.File(path, 'r') as h:
        matches = []
        for ref in h['data'][()].ravel(order='F'):
            if not ref:
                continue
            record = h[ref]
            if not isinstance(record, h5py.Group) or 'name' not in record:
                continue
            value = ''.join(chr(int(x)) for x in record['name'][()].ravel(order='F') if int(x))
            if value == name:
                matches.append(record)
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one MAT record {name}; found {len(matches)}')
        vertices = matches[0]['roiData/roiCoordinates'][()].T.astype(float)
        return vertices, matches[0].name


def roi_flags(table, vertices):
    xy = table[['x', 'y']].to_numpy(float)
    finite = np.isfinite(xy).all(axis=1)
    inside = np.full(len(table), -1, np.int8)
    inside[finite] = points_in_polygon(xy[finite], vertices).astype(np.int8)
    status = np.where(~finite, 'invalid_center', np.where(inside == 1, 'inside', 'outside'))
    return inside, status, table.qc_unflagged_model_fit.to_numpy(bool) & (inside == 1)


def decompose_roi(tc, bc, c):
    """Mask beta-dependent quantities when support is insufficient, not Total/B."""
    tc = np.atleast_2d(tc)
    bc = np.atleast_2d(bc)
    d = decompose_counts(tc, bc, c)
    mids = (np.arange(c['distance_bins']) + .5) * c['distance_max_um'] / c['distance_bins']
    tail = (mids >= c['background_fit_um'][0]) & (mids < c['background_fit_um'][1])
    nt = tc[:, tail].sum(1)
    nb = bc[:, tail].sum(1)
    supported = (nt >= c['minimum_total_tail_pairs']) & (nb >= c['minimum_background_tail_ordered_pairs'])
    d['total_tail_count'] = nt
    d['background_tail_count'] = nb
    d['total_tail_nonempty_bins'] = (tc[:, tail] > 0).sum(1)
    d['background_tail_nonempty_bins'] = (bc[:, tail] > 0).sum(1)
    d['tail_supported'] = supported
    d['valid'] &= supported
    d['shape_valid'] &= supported
    d['ratio_valid'] &= supported
    for k in ['beta', 'beta_raw', 'signed_area', 'main_mass', 'tail_mass', 'tail_main_ratio', 'negative_mass']:
        d[k] = np.where(d['valid'], d[k], np.nan)
    for k in ['inter', 'intra']:
        d[k][~d['valid']] = np.nan
    d['shape'][~d['shape_valid']] = np.nan
    d['shape_status'] = np.where(~supported, 'unavailable_sparse_tail',
        np.where(~d['valid'], 'unavailable_invalid_distribution',
        np.where(~d['shape_valid'], 'unavailable_near_zero_signed_area', 'available_nominal_signed')))
    return d


def bootstrap_roi(th, bh, c, length, weights):
    starts = np.arange(0, c['frames'], length)
    t, b = block_histograms(th, bh, starts, length)
    ns = c['frames'] // c['segment_frames']
    nk = c['segment_frames'] // length
    t = t.reshape(ns, nk, -1)
    b = b.reshape(ns, nk, -1)
    rt = np.einsum('srb,sbk->srk', weights, t, optimize=True)
    rb = np.einsum('srb,sbk->srk', weights, b, optimize=True)
    draws = [decompose_roi(rt.sum(0), rb.sum(0), c)]
    draws.extend(decompose_roi(rt[s], rb[s], c) for s in range(ns))
    controls = [decompose_roi(t.sum((0, 1)), b.sum((0, 1)), c)]
    controls.extend(decompose_roi(t[s].sum(0), b[s].sum(0), c) for s in range(ns))
    return draws, controls


def scalar_row(d):
    keys = ['beta', 'beta_raw', 'signed_area', 'main_mass', 'tail_mass', 'tail_main_ratio',
            'negative_mass', 'valid', 'shape_valid', 'ratio_valid', 'tail_supported',
            'total_tail_count', 'background_tail_count', 'total_tail_nonempty_bins', 'background_tail_nonempty_bins']
    row = {k: d[k][0].item() for k in keys}
    row['shape_status'] = str(d['shape_status'][0])
    return row


def finite_ci(values, minimum=.95, eligible=True):
    values = np.asarray(values)
    good = np.isfinite(values)
    count = int(good.sum())
    if not eligible or count / len(values) < minimum:
        return np.nan, np.nan, count, count / len(values)
    lo, hi = np.quantile(values[good], [.025, .975])
    return float(lo), float(hi), count, count / len(values)


def paired_difference(a, b, minimum=.95, eligible=True):
    """Keep original replicate order; never separately drop invalid values."""
    a, b = np.asarray(a), np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    difference = np.full_like(a, np.nan, dtype=float)
    difference[good] = a[good] - b[good]
    return difference, good, finite_ci(difference, minimum, eligible)


def distribution_table(tc, bc, d, c):
    width = c['distance_max_um'] / c['distance_bins']
    left = np.arange(c['distance_bins']) * width
    values = {'distance_left_um': left, 'distance_right_um': left+width, 'distance_mid_um': left+width/2,
              'total_count': tc, 'background_ordered_nonzero_count': bc}
    for k in ['total', 'background', 'inter', 'intra', 'shape']:
        values[k+'_density_um_inv'] = d[k][0] / width
    return pd.DataFrame(values)
