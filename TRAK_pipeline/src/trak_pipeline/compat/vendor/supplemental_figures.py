"""Static cached-data figures with exact numerical plot sources; no new fits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import tifffile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '_runtime_mplconfig'))
sys.path.insert(0, str(PROJECT / 'mito_tardis'))
from point_stack_core import gray_crop, display_centers, marker_alpha, composite, roi_alpha, mask_contour_alpha

COLORS = {'On': '#27DDA5', 'Not-on': '#C08FF7', 'unknown': '#A0A0A0', 'QC_excluded': '#FFAB40'}
LABELS = {'mito-compatible': 'On', 'non-mito-compatible': 'Not-on', 'unknown': 'unknown'}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order='C')).hexdigest()


def truth(value):
    return str(value).strip().lower() in ('true', '1')


def native_number(row, key):
    try:
        return float(row[key])
    except (ValueError, TypeError, KeyError):
        return np.nan


def write_rows(path, rows):
    if not rows:
        raise ValueError('Cannot omit the schema for an empty plot-source table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def subset_rows(path, frames):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [row for row in reader if int(row['frame']) in frames]
    return columns, rows


def draw_text(draw, position, text, font_path, maximum_width, size=20):
    for point_size in range(size, 9, -1):
        font = ImageFont.truetype(font_path, point_size)
        if draw.textbbox((0, 0), text, font=font)[2] <= maximum_width:
            draw.text(position, text, font=font, fill='white')
            return
    raise ValueError('Figure annotation does not fit')


def render_frame(raw1, raw2, full_mask, vertices, fit_rows, detection_rows, spec, frame):
    crop, scale = spec['crop_xyxy_px'], spec['display_scale']
    gray1 = np.repeat(gray_crop(raw1, crop, scale, spec['display_C1_ADU'])[..., None], 3, axis=2)
    gray2 = np.repeat(gray_crop(raw2, crop, scale, spec['display_C2_ADU'])[..., None], 3, axis=2)
    height, width = gray1.shape[:2]
    outline = roi_alpha(vertices, crop, scale)
    c1 = composite(composite(gray1, mask_contour_alpha(full_mask, crop, scale), [255, 75, 75]), outline, [140, 140, 140])
    c2 = composite(gray2, outline, [140, 140, 140])
    circles = composite(c2, marker_alpha((height, width),
        display_centers([[float(row['x_px']), float(row['y_px'])] for row in detection_rows], crop, scale),
        spec['TrackMate_circle_radius_native_px'] * scale, stroke=1.0), [75, 210, 255])
    refined = Image.fromarray(c2.copy())
    draw = ImageDraw.Draw(refined)
    detections = {row['candidate_index']: row for row in detection_rows}
    source_rows = []
    counts = {group: 0 for group in COLORS}
    for row in fit_rows:
        candidate = row['candidate_index']
        detection = detections[candidate]
        selected = truth(row['qc_unflagged_model_fit']) and truth(row['fitted_center_inside_ROI'])
        group = LABELS[row['postmin32_pointwise_label']] if selected else 'QC_excluded'
        counts[group] += 1
        x, y = native_number(row, 'fit_x_px'), native_number(row, 'fit_y_px')
        position_source = 'refined_center'
        if not np.isfinite([x, y]).all():
            if selected:
                raise ValueError('A selected localization has no finite refined center')
            x, y = float(detection['x_px']), float(detection['y_px'])
            position_source = 'original_TrackMate_center_fallback'
        cx, cy = display_centers([[x, y]], crop, scale)[0]
        dx = native_number(row, 'x_se_nm') / spec['pixel_size_nm'] * scale
        dy = native_number(row, 'y_se_nm') / spec['pixel_size_nm'] * scale
        color = COLORS[group]
        if selected:
            if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
                raise ValueError('Selected SE must be finite and strictly positive; no capping/substitution')
            draw.line([(cx - dx, cy), (cx + dx, cy)], fill=color, width=1)
            draw.line([(cx, cy - dy), (cx, cy + dy)], fill=color, width=1)
            draw.ellipse((cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5), fill=color)
        else:
            draw.line([(cx - 3, cy - 3), (cx + 3, cy + 3)], fill=color, width=2)
            draw.line([(cx - 3, cy + 3), (cx + 3, cy - 3)], fill=color, width=2)
        source_rows.append(dict(candidate_index=candidate, frame=row['frame'], source_detection_uid=row.get('source_detection_uid', ''),
            plot_group=group, original_fit_x_px=row.get('fit_x_px', ''), original_fit_y_px=row.get('fit_y_px', ''),
            original_x_se_nm=row.get('x_se_nm', ''), original_y_se_nm=row.get('y_se_nm', ''),
            original_TrackMate_x_px=detection['x_px'], original_TrackMate_y_px=detection['y_px'],
            plotted_position_source=position_source, display_x_px=float(cx), display_y_px=float(cy),
            error_extent_x_display_px=float(dx) if selected else '', error_extent_y_display_px=float(dy) if selected else ''))
    gap, margin, header, panel_label, footer = 20, 16, 82, 38, 110
    canvas = Image.new('RGB', (4 * width + 3 * gap + 2 * margin, height + panel_label + header + footer), '#101317')
    draw = ImageDraw.Draw(canvas)
    font = spec['font_path']
    draw_text(draw, (margin, 10), f"{spec['sample_name']} | original frame {frame} | {frame * spec['frame_interval_s']:.2f} s | support block {frame // 50}", font, canvas.width - 2 * margin, 24)
    draw_text(draw, (margin, 44), 'Single original C1/C2 pages; native crop and coordinates. C1/C2 identity registration UNVERIFIED.', font, canvas.width - 2 * margin, 20)
    labels = ['C1 raw + full-mask contour (before ROI clipping)', 'C2 raw + original cell ROI',
              f"C2 + TrackMate ROI candidates | radius 3 native pixels | n={len(detection_rows)}",
              f"C2 + refined +/-1 per-axis model SE | QC excluded X | n={len(fit_rows)}"]
    origins = []
    for index, (array, label) in enumerate(zip([c1, c2, circles, np.asarray(refined)], labels, strict=True)):
        x = margin + index * (width + gap)
        y = header
        origins.append([x, y + panel_label])
        draw_text(draw, (x, y + 5), label, font, width, 19)
        canvas.paste(Image.fromarray(array), (x, y + panel_label))
    y = canvas.height - footer + 12
    for offset, line in enumerate([
        f"C1 fixed {spec['display_C1_ADU']} ADU; C2 fixed {spec['display_C2_ADU']} ADU; same original ROI in all panels.",
        'On green | Not-on purple | unknown gray | QC-excluded orange X. Error bars are uncapped +/-1 nominal model SE.',
        'SE is not PSF width or validated accuracy. No new detection, fitting, classification, segmentation or registration.']):
        draw_text(draw, (margin, y + offset * 28), line, font, canvas.width - 2 * margin, 18)
    info = dict(frame=frame, mask_block=frame // 50, ROI_candidate_counts=counts, panel_origins_xy=origins,
                panel_size_wh=[width, height], raw_C1_page_SHA256=digest(raw1), raw_C2_page_SHA256=digest(raw2),
                full_mask_C_order_SHA256=digest(full_mask), numerical_plot_sources_preserve_original_field_strings=True)
    return canvas, source_rows, info


def render_representatives(localization, classification, mask_dir, output, spec):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'numerical_sources'; source.mkdir()
    frames = set(spec['representative_frames'])
    _, fits = subset_rows(classification / 'classification/all_ROI_candidates_postmin32.csv', frames)
    _, detections = subset_rows(localization / 'all_detections.csv', frames)
    ids = {row['candidate_index'] for row in fits}
    detections = [row for row in detections if row['candidate_index'] in ids]
    if len({row['candidate_index'] for row in detections}) != len(ids):
        raise ValueError('TrackMate and classified ROI candidate IDs must match exactly')
    original = {row['candidate_index']: row for row in detections}
    for row in fits:
        detection = original[row['candidate_index']]
        if row['frame'] != detection['frame'] or row['source_detection_uid'] != detection['source_detection_uid']:
            raise ValueError('Candidate identity/frame/UID mismatch')
    write_rows(source / 'representative_original_classified_rows.csv', fits)
    write_rows(source / 'representative_original_TrackMate_rows.csv', detections)
    mask = np.load(mask_dir / 'clean_full_mask.npy', mmap_mode='r')
    vertices = np.load(mask_dir / 'roi_vertices_xy.npy')
    roi = np.load(mask_dir / 'roi_mask.npy')
    records = []
    with tifffile.TiffFile(spec['C1_file']) as c1, tifffile.TiffFile(spec['C2_file']) as c2:
        for frame in spec['representative_frames']:
            raw1, raw2 = c1.pages[frame].asarray(), c2.pages[frame].asarray()
            full = mask[frame // 50]
            picture, plot_rows, info = render_frame(raw1, raw2, full, vertices,
                [row for row in fits if int(row['frame']) == frame],
                [row for row in detections if int(row['frame']) == frame], spec, frame)
            metadata = PngImagePlugin.PngInfo(); metadata.add_text('source_metadata', json.dumps(info))
            path = output / f'frame_{frame:04d}_four_panels.png'
            picture.save(path, pnginfo=metadata)
            with Image.open(path) as readback:
                np.testing.assert_array_equal(np.asarray(readback), np.asarray(picture))
            if plot_rows:
                write_rows(source / f'frame_{frame:04d}_display_coordinates.csv', plot_rows)
            np.savez_compressed(source / f'frame_{frame:04d}_raw_arrays.npz', C1_uint16=raw1, C2_uint16=raw2,
                                full_mask_bool=full, roi_mask_bool=roi, roi_vertices_xy=vertices,
                                crop_xyxy_px=np.array(spec['crop_xyxy_px']), display_scale=np.array(spec['display_scale']))
            info.update(PNG=path.name, PNG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
            records.append(info)
    (output / 'frame_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


def histogram_counts(values, edges):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    inside = finite & (values >= edges[0]) & (values <= edges[-1])
    return np.histogram(values[inside], edges)[0].astype(np.int64), dict(total=int(len(values)), finite=int(finite.sum()),
        nonfinite=int((~finite).sum()), below_range=int((finite & (values < edges[0])).sum()),
        above_range=int((finite & (values > edges[-1])).sum()), displayed=int(inside.sum()))


def histogram2d_counts(x, y, edges):
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    inside = finite & (x >= edges[0]) & (x <= edges[-1]) & (y >= edges[0]) & (y <= edges[-1])
    counts = np.histogram2d(x[inside], y[inside], bins=(edges, edges))[0].astype(np.int64)
    return counts, dict(total=len(x), nonfinite=int((~finite).sum()), outside_range=int((finite & ~inside).sum()), displayed=int(inside.sum()))


def render_histograms(classification, output, spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    output.mkdir(parents=True, exist_ok=False)
    path = classification / 'classification/selected_localizations_postmin32.csv'
    columns = ['candidate_index', 'frame', 'fit_x_px', 'fit_y_px', 'x_se_nm', 'y_se_nm', 'photons_e', 'postmin32_pointwise_label']
    original = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    original.to_csv(output / 'original_histogram_source_fields.csv', index=False)
    labels = original['postmin32_pointwise_label'].to_numpy()
    selection = {'All': np.ones(len(original), dtype=bool), 'On': labels == 'mito-compatible', 'Not-on': labels == 'non-mito-compatible'}
    numeric = {name: np.array([float(value) if value else np.nan for value in original[name]], dtype=float)
               for name in ('x_se_nm', 'y_se_nm', 'photons_e')}
    edges = np.asarray(spec['SE_histogram_edges_nm'], dtype=float)
    nonnegative_photons = numeric['photons_e'][np.isfinite(numeric['photons_e']) & (numeric['photons_e'] >= 0)]
    photon_edges = np.linspace(0, max(1.0, float(nonnegative_photons.max())) if len(nonnegative_photons) else 1.0, 101)
    groups = ('All', 'On', 'Not-on'); colors = {'All': '#555555', **COLORS}
    rows, outside, arrays, maps = [], {}, dict(SE_edges_nm=edges, photons_edges_e=photon_edges), {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis_index, field in enumerate(('x_se_nm', 'y_se_nm', 'photons_e')):
        axis = axes[axis_index]; bins = photon_edges if field == 'photons_e' else edges
        for group in groups:
            counts, info = histogram_counts(numeric[field][selection[group]], bins)
            outside[field + '__' + group] = info; arrays[field + '__' + group] = counts
            axis.stairs(counts, bins, color=colors[group], label=group)
            rows.extend(dict(field=field, group=group, bin_index=index, left=float(bins[index]), right=float(bins[index + 1]), count=int(count))
                        for index, count in enumerate(counts))
        axis.set(xlabel='Nominal photoelectron-equivalent signal (e)' if field == 'photons_e' else field + ' (nm)', ylabel='Localizations',
                 xlim=(bins[0], bins[-1]), xscale='linear', yscale='linear')
        axis.grid(alpha=.2); axis.legend()
    fig.suptitle(spec['sample_name'] + ' | original QC-selected localizations | linear histograms')
    fig.tight_layout(); fig.savefig(output / 'SE_and_signal_linear_histograms.png', dpi=180)
    fig.savefig(output / 'SE_and_signal_linear_histograms.pdf'); plt.close(fig)
    for group in groups:
        maps[group], outside['SE_2D__' + group] = histogram2d_counts(numeric['x_se_nm'][selection[group]], numeric['y_se_nm'][selection[group]], edges)
        arrays['SE_2D__' + group] = maps[group]
    maximum = max(1, max(int(value.max()) for value in maps.values()))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), constrained_layout=True)
    for axis, group in zip(axes, groups, strict=True):
        image = axis.pcolormesh(edges, edges, maps[group].T, shading='flat', cmap='viridis', vmin=0, vmax=maximum)
        info = outside['SE_2D__' + group]
        axis.set(xlabel='x SE (nm)', ylabel='y SE (nm)', xlim=(0, 200), ylim=(0, 200), aspect='equal',
                 title=f"{group} | shown {info['displayed']}; outside {info['outside_range']}; NA {info['nonfinite']}")
    fig.colorbar(image, ax=axes, label='Localizations per 5 nm x 5 nm bin (shared linear scale)')
    fig.suptitle(spec['sample_name'] + ' | per-axis nominal model SE; no precision threshold')
    fig.savefig(output / 'SE_2D_All_On_NotOn_shared_linear.png', dpi=180)
    fig.savefig(output / 'SE_2D_All_On_NotOn_shared_linear.pdf'); plt.close(fig)
    np.savez_compressed(output / 'histogram_counts_and_edges.npz', **arrays)
    write_rows(output / 'histogram_1D_bin_counts.csv', rows)
    rows2d = [dict(group=group, x_bin=i, y_bin=j, x_left_nm=float(edges[i]), x_right_nm=float(edges[i+1]),
        y_left_nm=float(edges[j]), y_right_nm=float(edges[j+1]), count=int(maps[group][i, j]))
        for group in groups for i in range(len(edges)-1) for j in range(len(edges)-1)]
    write_rows(output / 'histogram_2D_bin_counts.csv', rows2d)
    metadata = dict(outside_counts=outside, SE_bins_nm=edges.tolist(), photons_bins_e=photon_edges.tolist(),
        shared_2D_vmin=0, shared_2D_vmax=maximum, color_normalization='linear shared across All/On/Not-on',
        selection='All original QC-selected rows; no SE-size cutoff; repeated observations retained',
        All_includes_unknown=True, manufacturer_nominal_calibration=True, validated_accuracy=False)
    (output / 'histogram_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
