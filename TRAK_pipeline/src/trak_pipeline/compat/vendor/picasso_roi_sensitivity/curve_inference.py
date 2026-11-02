"""Centered joint-curve maxT inference on fixed signed density bins."""
from itertools import combinations
import numpy as np
from scipy.stats import beta as beta_distribution


def mc_interval(exceedances, trials):
    lo = 0.0 if exceedances == 0 else beta_distribution.ppf(.025, exceedances, trials-exceedances+1)
    hi = 1.0 if exceedances == trials else beta_distribution.ppf(.975, exceedances+1, trials-exceedances)
    return float(lo), float(hi)


def centered_max_test(observed, draws, valid, alpha=.05, minimum=.95):
    """observed[G,K], draws[G,R,K], valid[G,R]; all tests use one joint bank."""
    observed, draws, valid = np.asarray(observed), np.asarray(draws), np.asarray(valid, bool)
    assert draws.shape[0] == len(observed) == valid.shape[0]
    assert draws.shape[1] == valid.shape[1] and draws.shape[2] == observed.shape[1]
    pair_order = list(combinations(range(len(observed)), 2))
    pair_valid = [(valid[i] & valid[j] & np.isfinite(draws[i]).all(1) & np.isfinite(draws[j]).all(1)) for i,j in pair_order]
    joint = valid.all(0) & np.isfinite(draws).all((0,2))
    ids = np.flatnonzero(joint)
    result = {'joint_valid_count': len(ids), 'joint_valid_fraction': len(ids)/draws.shape[1],
              'joint_replicate_ids': ids, 'pair_order': pair_order,
              'pair_valid_counts': np.array([v.sum() for v in pair_valid]),
              'pair_valid_fractions': np.array([v.mean() for v in pair_valid]),
              'available': bool(np.isfinite(observed).all() and joint.mean() >= minimum)}
    if not result['available']:
        result.update({'p_global': np.nan, 'critical': np.nan, 'status': 'unavailable_observed_or_joint_support'})
        return result
    used = draws[:, joint, :]
    mean = used.mean(1)
    error = used - mean[:, None, :]
    # Range across groups equals max absolute pair contrast, and preserves all covariance.
    null_stats = np.ptp(error, axis=0).max(1)
    differences = np.stack([observed[i]-observed[j] for i,j in pair_order])
    pair_stats = np.abs(differences).max(1)
    stat = float(pair_stats.max())
    rank = int(np.ceil((1-alpha)*(len(ids)+1)))
    critical = float(np.sort(null_stats)[rank-1]) if rank <= len(ids) else np.inf
    counts = np.array([(null_stats >= s).sum() for s in pair_stats])
    exceedances = int((null_stats >= stat).sum())
    adjusted = (counts+1)/(len(ids)+1)
    result.update({'p_global': (exceedances+1)/(len(ids)+1), 'statistic': stat,
        'exceedances': exceedances, 'mc_exceedance_ci': mc_interval(exceedances, len(ids)),
        'p_resolution': 1/(len(ids)+1), 'critical': critical, 'critical_rank_1based': rank,
        'null_statistics': null_stats, 'bootstrap_mean': mean, 'pair_statistics': pair_stats,
        'adjusted_p': adjusted, 'pair_exceedances': counts, 'differences': differences,
        'lower': differences-critical, 'upper': differences+critical, 'status': 'available_centered_maxT'})
    return result


def continuous_intervals(lower, upper, edges):
    """Maximal contiguous fixed bins whose family band excludes zero, split by sign."""
    direction = np.where(lower>0, 1, np.where(upper<0, -1, 0))
    intervals = []
    start = None
    for k in range(len(direction)+1):
        current = direction[k] if k<len(direction) else 0
        if start is not None and (current == 0 or current != direction[start]):
            intervals.append((float(edges[start]), float(edges[k]), int(direction[start])))
            start = None
        if start is None and current != 0:
            start = k
    return intervals


def effect_sizes(difference, mids, width):
    if not np.isfinite(difference).all():
        return {k:np.nan for k in ['sup_abs_density_difference','sup_distance_um','integrated_abs_difference',
                                   'normalized_main_mass_difference','normalized_tail_mass_difference']}
    peak = int(np.abs(difference).argmax())
    return {'sup_abs_density_difference': float(np.abs(difference[peak])), 'sup_distance_um': float(mids[peak]),
        'integrated_abs_difference': float(np.abs(difference).sum()*width),
        'normalized_main_mass_difference': float(difference[mids<.10].sum()*width),
        'normalized_tail_mass_difference': float(difference[(mids>=.10)&(mids<.25)].sum()*width)}
