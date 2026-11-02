"""Descriptive annotations and native mask areas for an existing TARDIS run.

This module only reads the run's saved distributions, paired percentile bands,
and verified per-frame table. It does not calculate or modify TARDIS curves.
"""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .summarize_intra import summarize
except ImportError:
    from summarize_intra import summarize


COLORS = {'On': '#159C76', 'Not-on': '#9854CA'}
STAT_FIELDS = [
    'signed_first_moment_nm', 'signed_half_area_location_nm',
    'half_area_crossing_count', 'half_area_plateau', 'mode_bin_center_nm',
    'mode_bin_left_nm', 'mode_bin_right_nm', 'tied_peak_bin_count',
    'negative_bins', 'negative_area_abs', 'signed_area',
    'ordinary_probability_moments_applicable',
]


def _statistics(group, data, c):
    left = data.distance_left_um.to_numpy(float)
    right = data.distance_right_um.to_numpy(float)
    shape = data.shape_density_um_inv.to_numpy(float)
    intra = data.intra_density_um_inv.to_numpy(float)
    area = float(np.sum(intra * (right-left)))
    result = dict(group=group, range_left_um=float(left[0]),
                  range_right_um=float(right[-1]), original_Intra_signed_area=area)
    usable = (np.isfinite(shape).all() and np.isfinite(intra).all()
              and np.isfinite(area) and abs(area) > c['signed_area_min_abs'])
    if not usable:
        result.update({key: None for key in STAT_FIELDS})
        result.update(statistics_status='unavailable_invalid_shape_or_signed_area',
                      half_area_status='unavailable_invalid_shape',
                      peak_status='unavailable_invalid_shape')
        return result
    shape_area = float(np.sum(shape * (right-left)))
    if not np.isfinite(shape_area) or shape_area <= 0:
        result.update({key: None for key in STAT_FIELDS})
        result.update(statistics_status='unavailable_nonpositive_shape_area',
                      half_area_status='unavailable_invalid_shape',
                      peak_status='unavailable_invalid_shape')
        return result
    result.update(summarize(left, right, shape))
    result.update(statistics_status='available_descriptive_signed_curve',
        half_area_status=('unique_signed_half_area_crossing'
            if result['signed_half_area_location_nm'] is not None
            else 'unavailable_ambiguous_or_plateau_half_area_crossing'),
        peak_status=('unique_highest_native_bin'
            if result['mode_bin_center_nm'] is not None
            else 'unavailable_tied_highest_bins'))
    return result


def _number(value, digits=1):
    return f'{value:.{digits}f}' if value is not None and np.isfinite(value) else 'N/A'


def _annotated_intra(out, c, data, statistics):
    bands = pd.read_csv(out/'bootstrap/paired_bands_full_L25.csv', float_precision='round_trip')
    fig = plt.figure(figsize=(14, 7.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], left=.07,
        right=.985, top=.84, bottom=.23, wspace=.18, hspace=.37)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for group, color in COLORS.items():
        d = data[group]
        np.testing.assert_allclose(bands.distance_mid_um, d.distance_mid_um, rtol=0, atol=1e-12)
        for ax, field, title in zip(axes, ['intra', 'shape'],
                ['Signed Intra = Total - beta B', 'Intra / signed area on 0–3 µm']):
            prefix = group.replace('-', '_')+'_'+field
            ax.plot(d.distance_mid_um, d[field+'_density_um_inv'], color=color,
                    lw=1.6, label=group)
            ax.fill_between(bands.distance_mid_um, bands[prefix+'_low'],
                            bands[prefix+'_high'], color=color, alpha=.16)
            ax.set(xlim=(0, .5), xlabel='Distance (µm)', ylabel='Signed density (1/µm)', title=title)
            ax.axhline(0, color='gray', lw=.6)
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis('off')
    cells = []
    for row in statistics:
        peak = _number(row['mode_bin_center_nm'], 0)
        if row['mode_bin_center_nm'] is not None:
            peak += f"  ({row['mode_bin_left_nm']:.0f}–{row['mode_bin_right_nm']:.0f})"
        cells.append([row['group'], _number(row['signed_first_moment_nm']),
                      _number(row['signed_half_area_location_nm']), peak])
    table = table_ax.table(cellText=cells,
        colLabels=['Group', 'Signed mean, nm', '50% signed crossing, nm', 'Peak-bin center (interval), nm'],
        cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1], colWidths=[.12, .23, .29, .36])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor('#D8DCE0')
        if r == 0:
            cell.set_facecolor('#F0F2F4')
            cell.set_text_props(weight='bold')
        else:
            cell.set_text_props(color=COLORS[statistics[r-1]['group']])
    title = f"{c.get('sample_name','TDKd3 T1')} | {c['mask_label']} | Full {c['frames']} frames, τ = {c['tau_frames']*c['frame_interval_s']*1000:g} ms"
    fig.suptitle(textwrap.fill(title, 115), fontsize=13, y=.97)
    notes = [
        'Statistics use all original bins over 0–3 µm; the plots display 0–0.5 µm. Negative bins are retained.',
        'Signed mean is a signed first moment; the 50% crossing is not an ordinary probability median.',
        'N/A means an invalid shape, ambiguous/plateau 50% crossing, or tied peak; reasons are saved in the CSV.',
        'Shading: exploratory paired L25 pointwise 2.5–97.5 percentiles; coverage is unvalidated. No formal p value.',
    ]
    for y, note in zip([.165, .125, .085, .045], notes):
        fig.text(.5, y, note, ha='center', fontsize=9)
    fig.savefig(out/'figures/08_intra_with_statistics.png', dpi=180)
    plt.close(fig)


def _mask_area(out, c, frames):
    fields = ['frame', 'time_s', 'mask_block_index', 'mask_ROI_area_pixel2', 'mask_full_area_pixel2']
    area = frames[fields].copy()
    np.testing.assert_array_equal(area.frame, np.arange(c['frames']))
    np.testing.assert_array_equal(area.mask_block_index, area.frame.to_numpy()//50)
    np.testing.assert_allclose(area.time_s, area.frame*c['frame_interval_s'], rtol=0, atol=1e-12)
    assert np.isfinite(area[fields].to_numpy(float)).all()
    assert (area.mask_ROI_area_pixel2 >= 0).all()
    assert (area.mask_full_area_pixel2 >= area.mask_ROI_area_pixel2).all()
    for field in fields[3:]:
        assert (area.groupby('mask_block_index')[field].nunique() == 1).all()
    pixel_area = (c['pixel_size_nm']/1000)**2
    summaries = []
    for scope, pixels in [('ROI', 'mask_ROI_area_pixel2'), ('full_FOV', 'mask_full_area_pixel2')]:
        field = 'mask_'+scope+'_area_um2'
        area[field] = area[pixels]*pixel_area
        values = area[field].to_numpy(float)
        summaries.append(dict(scope=scope, frames=len(area), blocks=int(area.mask_block_index.nunique()),
            mean_um2=float(values.mean()), median_um2=float(np.median(values)),
            temporal_std_um2=float(values.std(ddof=0)), min_um2=float(values.min()),
            max_um2=float(values.max()), empty_frames=int((values == 0).sum())))
    area.to_csv(out/'mask_area_by_frame.csv', index=False)
    block = area.groupby('mask_block_index', sort=True).agg(
        original_frame_start=('frame', 'min'), original_frame_end_inclusive=('frame', 'max'),
        mask_ROI_area_pixel2=('mask_ROI_area_pixel2', 'first'),
        mask_full_area_pixel2=('mask_full_area_pixel2', 'first'),
        mask_ROI_area_um2=('mask_ROI_area_um2', 'first'),
        mask_full_FOV_area_um2=('mask_full_FOV_area_um2', 'first')).reset_index()
    block.to_csv(out/'mask_area_by_block.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'mask_area_summary.csv', index=False)
    result = dict(pixel_area_um2=pixel_area, area_definition='native binary foreground, no localization-error expansion',
        time_rule='Original frame f uses block floor(f/50); each block mask persists for50 source frames',
        extra_smoothing=False, fallback_frames=[], summaries=summaries)
    (out/'mask_area_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True)
    for ax, scope, color in zip(axes, ['ROI', 'full_FOV'], ['#246B9E', '#697F50']):
        field = 'mask_'+scope+'_area_um2'
        summary = next(row for row in summaries if row['scope'] == scope)
        x = np.r_[area.frame.to_numpy(), c['frames']]
        y = np.r_[area[field].to_numpy(), area[field].iloc[-1]]
        ax.step(x, y, where='post', color=color, lw=1.1, label='50-frame support mask')
        ax.axhline(summary['mean_um2'], color=color, ls='--', lw=.8, label='Full-movie mean')
        description = 'Inside cell ROI' if scope == 'ROI' else 'Full image before ROI clipping'
        ax.set(title=f"{description} | mean {summary['mean_um2']:.2f} µm²; median {summary['median_um2']:.2f} µm²",
               ylabel='Mask area (µm²)', xlim=(0, c['frames']))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    time_axis = axes[0].secondary_xaxis('top', functions=(
        lambda x: x*c['frame_interval_s'], lambda x: x/c['frame_interval_s']))
    time_axis.set_xlabel('Acquisition time (s)')
    axes[1].set_xlabel('Original frame (0-based; steps end at exclusive frame 2500)')
    axes[1].set_xticks([0, 500, 1000, 1500, 2000, c['frames']-1])
    fig.suptitle(textwrap.fill(f"{c.get('sample_name','TDKd3 T1')} | {c['mask_label']} | Native mask area", 115), fontsize=13, y=.98)
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.16, hspace=.43)
    fig.text(.5, .075, 'Area = foreground pixels × (0.117 µm)². Masks update every50 frames (1 s); no extra smoothing or endpoint fallback.',
             ha='center', fontsize=9)
    fig.text(.5, .035, 'Area is measured from the binary masks, without localization-error expansion. Increased area does not establish segmentation accuracy.',
             ha='center', fontsize=9)
    fig.savefig(out/'figures/09_mask_area.png', dpi=180)
    plt.close(fig)
    return result


def render_extras(out, c, frames):
    """Add descriptive signed-curve statistics and verified stepped mask areas."""
    out = Path(out)
    (out/'figures').mkdir(exist_ok=True)
    assert c['distance_max_um'] == 3 and c['bootstrap']['primary_block_frames'] == 25
    data = {group: pd.read_csv(out/'distribution'/group/'full.csv', float_precision='round_trip')
            for group in COLORS}
    statistics = [_statistics(group, data[group], c) for group in COLORS]
    pd.DataFrame(statistics).to_csv(out/'intra_descriptive_statistics.csv', index=False)
    _annotated_intra(out, c, data, statistics)
    area_summary = _mask_area(out, c, frames)
    return dict(statistics=statistics, mask_area=area_summary,
        figures=['figures/08_intra_with_statistics.png', 'figures/09_mask_area.png'])
