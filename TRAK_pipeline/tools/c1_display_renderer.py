#!/usr/bin/env python3
"""Repaint C1 displays from saved products; never perform scientific analysis.

This lean renderer uses stdlib CSV, NumPy, Pillow and Matplotlib.  It does not
import pandas or invoke detection, fitting, segmentation, assignment or TARDIS.
Only the ten four-panel PNGs and the saved-mean Not-on overlay are written.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import uuid

REPRESENTATIVE_FRAMES = (0, 49, 200, 625, 1200, 1250, 1875, 2000, 2450, 2499)
UID_FIELDS = ('source_detection_uid', 'detection_uid', 'uid')


def _rows(path):
    path = Path(path)
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _number(value):
    return float(value) if value not in (None, '') else float('nan')


def _frame(row):
    value = _number(row.get('frame'))
    if not math.isfinite(value) or int(value) != value:
        raise ValueError('Saved representative row lacks an integer original frame')
    return int(value)


def _truth(value):
    if value is True or str(value).lower() in ('true', '1'):
        return True
    if value is False or str(value).lower() in ('false', '0'):
        return False
    raise ValueError('Saved mask validity is not explicit: ' + str(value))


def _first_field(row, names):
    for name in names:
        if name in row:
            return _number(row[name])
    raise ValueError('Saved native TrackMate coordinate missing: ' + ','.join(names))


def _range(style, key, default):
    values = list(map(float, style.get(key, default)))
    if len(values) != 2 or not all(map(math.isfinite, values)) or values[0] >= values[1]:
        raise ValueError('Invalid display range: ' + key)
    return values


def _finish(fig, path, footer, *, black=False):
    import matplotlib.pyplot as plt
    fig.text(.5, .015, footer, ha='center', fontsize=8,
             color='white' if black else 'black')
    fig.tight_layout(rect=(0, .055, 1, .95))
    path = Path(path)
    temporary = path.with_name('.' + path.stem + '.' + uuid.uuid4().hex + '.tmp.png')
    try:
        fig.savefig(temporary, dpi=180)
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def render_cell(config):
    """Render saved classifications/positions using only new display limits.

    Required keys: sample_id, C1, C2, stats_dir, c1_mean_path, full_mask_path,
    output_dir. Optional: style, pixel_size_nm, frame_interval_s, frames.
    All scientific inputs are read only. Output must stay below the raw cell.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    from trak_pipeline.display import resolve_c1_display

    c = dict(config)
    sample = str(c['sample_id'])
    c1_path, c2_path = Path(c['C1']).resolve(), Path(c['C2']).resolve()
    output = Path(c['output_dir']).resolve()
    if c1_path.parent != c2_path.parent or not output.is_relative_to(c1_path.parent):
        raise ValueError('Display output must remain within its original cell directory')
    output.mkdir(parents=True, exist_ok=True)
    stats = Path(c['stats_dir'])
    style = dict(c.get('style', {}))
    range2 = _range(style, 'C2_display_ADU', [406, 554])
    pixel = float(c.get('pixel_size_nm', 117))
    interval = float(c.get('frame_interval_s', .02))
    frames = int(c.get('frames', 2500))
    if pixel <= 0 or interval <= 0:
        raise ValueError('Original pixel size and frame interval must be positive')
    colors = {'All': '#555555', 'On': style.get('on_color', '#00B894'),
              'Not-on': style.get('not_on_color', '#A45BDB'),
              'unknown': style.get('unknown_color', '#AAAAAA')}
    plt.rcParams.update({'font.size': style.get('font_size', 10)})
    fields, fits = _rows(stats / 'representative_fits.csv.gz')
    selected_fields, selected = _rows(stats / 'representative_selected.csv.gz')
    uid = next((name for name in UID_FIELDS if name in fields and name in selected_fields), None)
    if uid is None:
        raise ValueError('Saved representative tables lack stable detection UIDs')
    fit_uids = [row[uid] for row in fits]
    selected_uids = [row[uid] for row in selected]
    if len(fit_uids) != len(set(fit_uids)) or len(selected_uids) != len(set(selected_uids)):
        raise ValueError('Saved representative UID is duplicated')
    if not set(selected_uids).issubset(fit_uids):
        raise ValueError('Saved selected UIDs do not correspond to the fit table')
    by_frame, selected_by_frame = {}, {}
    for row in fits:
        by_frame.setdefault(_frame(row), []).append(row)
    for row in selected:
        selected_by_frame.setdefault(_frame(row), []).append(row)
    _, area = _rows(stats / 'mask_area_by_frame.csv')
    mapping = {_frame(row): row for row in area}
    if len(mapping) != len(area) or set(range(frames)) - set(mapping):
        raise ValueError('Saved original-frame-to-mask mapping is incomplete or duplicated')
    with np.load(stats / 'pooled_coordinates.npz', allow_pickle=False) as saved:
        pooled = {name: saved[name].copy() for name in saved.files}
    crop = pooled['crop_xyxy_px']
    x0, y0, x1, y1 = map(int, crop)
    vertices = pooled['vertices']
    full = np.load(c['full_mask_path'], mmap_mode='r', allow_pickle=False)
    chosen_frames = list(c.get('sample_frames', [f for f in REPRESENTATIVE_FRAMES if f < frames]))
    if (not chosen_frames or len(set(chosen_frames)) != len(chosen_frames)
            or any(isinstance(f, bool) or int(f) != f or not 0 <= f < frames for f in chosen_frames)):
        raise ValueError('Representative original frames must be unique integers within the movie')
    chosen_frames = [int(f) for f in chosen_frames]
    rendered, frame_counts = [], []
    try:
        with Image.open(c1_path) as source1, Image.open(c2_path) as source2:
            # Adaptive limits are resolved once per cell and reused for every
            # representative panel and the existing arithmetic-mean overlay.
            # Keep the ten original C1 pages for rendering; no second read is needed.
            raw1_cache = {}
            def frame_reader(frame):
                if frame not in raw1_cache:
                    source1.seek(frame)
                    raw = np.array(source1)
                    if raw.dtype.kind != 'u' or raw.dtype.itemsize != 2:
                        raise ValueError('Original C1 TIFF page is not uint16')
                    if raw.shape != tuple(full.shape[1:]):
                        raise ValueError('Original C1 TIFF page and saved mask shapes differ')
                    raw1_cache[frame] = raw
                return raw1_cache[frame]
            display_policy = resolve_c1_display(style, frame_ids=style.get('C1_display_sample_frames',chosen_frames),
                                                frame_reader=frame_reader,
                                                roi_mask=pooled['roi_mask'])
            range1 = display_policy['limits_ADU']
            footer = (f'C1 display {range1[0]:g}-{range1[1]:g} ADU; '
                      f'C2 display {range2[0]:g}-{range2[1]:g} ADU. '
                      'Display only; raw pixels, masks, positions and labels unchanged.')
            for frame in chosen_frames:
                raw1 = frame_reader(frame)
                source2.seek(frame)
                raw2 = np.array(source2)
                if any(a.dtype.kind != 'u' or a.dtype.itemsize != 2 for a in (raw1, raw2)):
                    raise ValueError('Original C1/C2 TIFF page is not uint16')
                if raw1.shape != tuple(full.shape[1:]) or raw2.shape != raw1.shape:
                    raise ValueError('Original TIFF page and saved mask shapes differ')
                part, sel = by_frame.get(frame, []), selected_by_frame.get(frame, [])
                selected_ids = {row[uid] for row in sel}
                if not selected_ids.issubset({row[uid] for row in part}):
                    raise ValueError('Saved selected UID has a conflicting original frame')
                block_value = _number(mapping[frame].get('block_index'))
                if (not math.isfinite(block_value) or int(block_value) != block_value
                        or not 0 <= block_value < len(full)):
                    raise ValueError('Representative frame has no valid saved mask mapping')
                block = int(block_value)
                mask_valid = _truth(mapping[frame].get('mask_valid', 'True'))
                fig, axes = plt.subplots(1, 4, figsize=(18, 7))
                for i, ax in enumerate(axes):
                    limits = range1 if i == 0 else range2
                    ax.imshow(raw1 if i == 0 else raw2, cmap='gray', vmin=limits[0],
                              vmax=limits[1], origin='upper', interpolation='nearest')
                    ax.plot(vertices[:, 0], vertices[:, 1], color='#888888', lw=.4)
                    ax.set(xlim=(x0-.5, x1-.5), ylim=(y1-.5, y0-.5))
                    ax.axis('off')
                if mask_valid:
                    axes[0].contour(full[block], levels=[.5], colors=['#FF5555'], linewidths=.5)
                else:
                    axes[0].text(.5, .5, 'Mask unavailable / unknown', transform=axes[0].transAxes,
                                 ha='center', color='red')
                for row in part:
                    dx = _first_field(row, ('detection_x_px', 'detector_x_px', 'x_px', 'candidate_x_px'))
                    dy = _first_field(row, ('detection_y_px', 'detector_y_px', 'y_px', 'candidate_y_px'))
                    if not math.isfinite(dx) or not math.isfinite(dy):
                        raise ValueError('Saved native TrackMate center is nonfinite')
                    axes[2].add_patch(Circle((dx, dy), 3, fill=False, color='#FFB347', lw=.5))
                for row in sel:
                    axes[3].errorbar(_number(row['fit_x_px']), _number(row['fit_y_px']),
                                     xerr=_number(row['x_se_nm'])/pixel,
                                     yerr=_number(row['y_se_nm'])/pixel, fmt='.', ms=1,
                                     color=colors[row['tardis_group']], elinewidth=.5)
                excluded = [row for row in part if row[uid] not in selected_ids]
                for row in excluded:
                    x, y = _number(row.get('fit_x_px')), _number(row.get('fit_y_px'))
                    if not math.isfinite(x) or not math.isfinite(y):
                        x, y = _number(row.get('x_px')), _number(row.get('y_px'))
                    if math.isfinite(x) and math.isfinite(y):
                        axes[3].plot(x, y, 'x', color='#F39416', ms=2, mew=.5)
                if excluded:
                    axes[3].text(.02, .02, 'Orange ×: QC excluded', transform=axes[3].transAxes,
                                 color='#F39416', fontsize=7)
                for ax, title in zip(axes, ('C1 + mask', 'C2 raw', 'C2 + TrackMate centers',
                                           'C2 + refined centers / X,Y SE')):
                    ax.set_title(title, fontsize=10)
                fig.suptitle(f'{sample} | original frame {frame} | {frame*interval:.2f} s | '
                             'nominal SE; identity registration unverified')
                name = f'frame_{frame:04d}_four_panels.png'
                _finish(fig, output/name, footer)
                rendered.append(name)
                frame_counts.append(dict(frame=frame, block=block, ROI_candidates=len(part),
                                         selected=len(sel), QC_excluded=len(excluded)))
        # Read the existing scientific arithmetic-mean image; never average raw frames.
        mean = np.load(c['c1_mean_path'], allow_pickle=False).copy()
        if mean.shape != pooled['roi_mask'].shape:
            raise ValueError('Saved C1 mean and ROI mask shapes differ')
        mean[~pooled['roi_mask']] = 0
        fig, ax = plt.subplots(figsize=(7, 9), facecolor='black')
        ax.set_facecolor('black')
        ax.imshow(mean, cmap='gray', vmin=range1[0], vmax=range1[1], origin='upper',
                  extent=(-.5, mean.shape[1]-.5, mean.shape[0]-.5, -.5), interpolation='nearest')
        ax.plot(vertices[:, 0], vertices[:, 1], color='#777777', lw=.4)
        points = pooled['Not-on']
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=7.5*style.get('dot_radius_px', .2)**2,
                       color=colors['Not-on'], linewidths=0, rasterized=True)
        ax.set(xlim=(x0-.5, x1-.5), ylim=(y1-.5, y0-.5), aspect='equal',
               title=f'{sample} | Not-on {len(points):,} | mean C1, all raw frames')
        ax.axis('off')
        bar = 2000/pixel
        ax.plot([x0+12, x0+12+bar], [y1-6, y1-6], color='white', lw=2)
        ax.text(x0+12, y1-9, '2 µm', color='white', fontsize=8)
        ax.title.set_color('white')
        name = 'NotOn_on_C1_mean2500.png'
        _finish(fig, output/name, footer.replace(' ADU. ', ' ADU.\n', 1), black=True)
        rendered.append(name)
    finally:
        mapping_file = getattr(full, '_mmap', None)
        if mapping_file is not None:
            mapping_file.close()
    return dict(sample_id=sample, rendered=rendered, frame_counts=frame_counts,
                C1_display_ADU=range1, C2_display_ADU=range2,
                C1_display_policy=display_policy,
                saved_Not_on_points=len(pooled['Not-on']), scientific_stage_calls=0,
                new_cells_started=0, saved_scientific_products_modified=False,
                raw_pixels_modified=False, inputs_used='Saved masks, UID-linked labels, fits, '
                'pooled coordinates and C1 arithmetic mean; original TIFF representative pages only')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    result = render_cell(json.loads(args.config.read_text(encoding='utf-8')))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
