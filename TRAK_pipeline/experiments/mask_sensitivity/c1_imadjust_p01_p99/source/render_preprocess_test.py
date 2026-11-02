#!/usr/bin/env python3
"""Publish saved C1 preprocessing-mask comparisons without scientific recomputation."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config['run_directory']).resolve()
    if not (root / 'NUMERICAL_COMPLETE.json').is_file():
        raise SystemExit('Saved mask computation is not complete; no rendering was started.')
    for key, rel in [('TMPDIR', 'runtime/tmp'), ('MPLCONFIGDIR', 'runtime/matplotlib'),
                     ('NUMBA_CACHE_DIR', 'runtime/numba'), ('PYTHONPYCACHEPREFIX', 'runtime/pycache')]:
        target = root / rel
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYDANTIC_DISABLE_PLUGINS'] = '__all__'
    sys.dont_write_bytecode = True

    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from point_stack_core import gray_crop, mask_contour_alpha, roi_alpha, composite

    started = time.perf_counter()
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    # The mask arrays are saved scientific products; this program only draws them.
    old = np.load(root / 'masks/baseline_clean_full_mask.npy', mmap_mode='r')
    new = np.load(root / 'masks/clean_full_mask.npy', mmap_mode='r')
    vertices = np.load(root / 'masks/roi_vertices_xy.npy')
    if old.shape != new.shape or old.shape != (3, *config['shape_TYX'][1:]):
        raise ValueError('Saved old/new full-mask shapes are incompatible.')
    with (root / 'data/frame_mapping.csv').open(newline='', encoding='utf-8') as stream:
        mapping = list(csv.DictReader(stream))
    starts = config['block_starts']
    if len(mapping) != 150 or starts != [0, 1200, 2450]:
        raise ValueError('This comparison requires the three explicit disjoint 50-frame blocks.')
    frames = [int(row['original_frame']) for row in mapping]
    expected = [f for start in starts for f in range(start, start + 50)]
    if frames != expected or [int(row['page_index']) for row in mapping] != list(range(150)):
        raise ValueError('Saved page/frame mapping is not the expected zero-based mapping.')

    crop = config['crop_xyxy_px']
    scale = int(config['display_scale'])
    limits = config['display_original_C1_ADU']
    panel_w = (crop[2] - crop[0]) * scale
    panel_h = (crop[3] - crop[1]) * scale
    margin, gap, header, footer = 20, 20, 112, 105
    width = 3 * panel_w + 2 * margin + 2 * gap
    height = header + panel_h + footer
    origins = [(margin + j * (panel_w + gap), header) for j in range(3)]
    font_candidates = [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                       Path('C:/Windows/Fonts/arial.ttf')]
    font_path = next((p for p in font_candidates if p.is_file()), None)
    if font_path is None:
        fonts = {n: ImageFont.load_default() for n in [16, 18, 20, 23, 27]}
    else:
        fonts = {n: ImageFont.truetype(str(font_path), n) for n in [16, 18, 20, 23, 27]}
    roi = roi_alpha(vertices, crop, scale)
    gray_roi = np.array([145, 145, 145], float)
    red = np.array([255, 75, 75], float)
    added = np.array([39, 221, 165], np.uint8)
    removed = np.array([229, 82, 192], np.uint8)
    contours = [(mask_contour_alpha(old[j], crop, scale),
                 mask_contour_alpha(new[j], crop, scale)) for j in range(3)]
    differences = []
    x0, y0, x1, y1 = crop
    for j in range(3):
        a = np.asarray(old[j, y0:y1, x0:x1], bool)
        b = np.asarray(new[j, y0:y1, x0:x1], bool)
        rgb = np.zeros((*a.shape, 3), np.uint8)
        rgb[a & b] = [92, 92, 92]
        rgb[b & ~a] = added
        rgb[a & ~b] = removed
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        differences.append(composite(rgb, roi, gray_roi))

    def bar(draw, ox, oy):
        length = round(2.0 / (config['pixel_size_nm'] / 1000) * scale)
        yy = oy + panel_h - 22
        draw.line((ox + 16, yy, ox + 16 + length, yy), fill='white', width=3)
        draw.text((ox + 16, yy - 23), '2 um', fill='white', font=fonts[16])

    def save_png(array, destination):
        temp = destination.with_name(destination.name + '.tmp')
        Image.fromarray(array).save(temp, format='PNG')
        temp.replace(destination)

    def render(frame, raw):
        local_block = starts.index((frame // 50) * 50)
        g = gray_crop(raw, crop, scale, limits)
        rgb = np.repeat(g[..., None], 3, axis=2)
        a, b = contours[local_block]
        panels = [composite(composite(rgb, a, red), roi, gray_roi),
                  composite(composite(rgb, b, red), roi, gray_roi), differences[local_block]]
        canvas = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 12),
                  f"{config['sample_name']} | original frame {frame} | {frame * config['frame_interval_s']:.2f} s | support block {frame // 50}",
                  fill='white', font=fonts[27])
        draw.text((margin, 49), 'C1 input preprocessing test: per-frame MATLAB-style 1st/99th-percentile contrast adjustment',
                  fill='#D9D9D9', font=fonts[20])
        titles = ['Original C1 + baseline mask', 'SAME original C1 + adjusted-input mask', 'Mask difference (before ROI clipping)']
        for panel, title, origin in zip(panels, titles, origins):
            canvas.paste(Image.fromarray(panel), origin)
            draw.text((origin[0], 82), title, fill='white', font=fonts[20])
            bar(draw, *origin)
        fy = header + panel_h + 12
        draw.text((margin, fy),
                  f'Both C1 backgrounds: fixed {limits[0]}-{limits[1]} ADU. Red: full-mask boundary. Gray outline: original cell ROI.',
                  fill='#D9D9D9', font=fonts[18])
        draw.text((margin, fy + 27), 'Difference: common foreground = gray | added = green | removed = magenta. No localization or TARDIS recomputation.',
                  fill='#D9D9D9', font=fonts[18])
        draw.text((margin, fy + 54), 'DISJOINT SEGMENTS: frames 0-49, 1200-1249, 2450-2499. Adjacent TIFF pages at the joins are not adjacent acquisition frames.',
                  fill='#FFD37E', font=fonts[18])
        return np.asarray(canvas)

    files = []
    samples = set(int(f) for f in config['sample_frames'])
    sample_hashes = {}
    preview_raw = []
    preview_adjusted = []
    original_streams = []
    adjusted_streams = []
    try:
        for start in starts:
            original_streams.append(tifffile.TiffFile(root / f'inputs/C1_original_frames{start:04d}_{start + 49:04d}.ome.tif'))
            adjusted_streams.append(tifffile.TiffFile(root / f'inputs/C1_adjusted_frames{start:04d}_{start + 49:04d}.ome.tif'))
        for j in range(3):
            if len(original_streams[j].pages) != 50 or len(adjusted_streams[j].pages) != 50:
                raise ValueError('Each saved input block must contain exactly 50 original-frame pages.')
            preview_raw.append(original_streams[j].pages[0].asarray())
            preview_adjusted.append(adjusted_streams[j].pages[0].asarray())

        def pages():
            for page, frame in enumerate(frames):
                j, local = divmod(page, 50)
                raw = original_streams[j].pages[local].asarray()
                image = render(frame, raw)
                if frame in samples:
                    sample_hashes[page] = hashlib.sha256(image.tobytes()).hexdigest()
                if local == 0:
                    name = f'C1_old_new_mask_frame{frame:04d}.png'
                    save_png(image, figures / name)
                    files.append(name)
                if (page + 1) % 50 == 0:
                    print(f'Rendered {page + 1}/150 pages; original frame {frame}', flush=True)
                yield image

        tiff_name = 'C1_old_new_mask_comparison_150pages.tif'
        tiff_temp = figures / (tiff_name + '.tmp')
        tifffile.imwrite(tiff_temp, pages(), shape=(150, height, width, 3), dtype=np.uint8,
                         imagej=True, photometric='rgb', compression='deflate',
                         metadata={'axes': 'TYXS', 'mode': 'color', 'loop': False,
                                   'Info': '150 RGB pages from three DISJOINT original-frame segments: 0-49, 1200-1249, 2450-2499. Original frame, acquisition time and mask block are printed in every page header. Do not treat page joins as consecutive acquisition frames. Within-segment frame interval: 0.02 s. Both original-C1 backgrounds: 407-651 ADU. Display crop and scale are unchanged between panels.'})
        tiff_temp.replace(figures / tiff_name)
        files.append(tiff_name)
    finally:
        for stream in original_streams + adjusted_streams:
            stream.close()

    # The input preview has deliberately different, explicitly labelled display
    # windows: original ADU and the full adjusted uint16 output range.
    preview_header, row_header, row_gap = 72, 35, 24
    preview_w = 2 * panel_w + 2 * margin + gap
    preview_h = preview_header + 3 * (row_header + panel_h + row_gap) + 46
    canvas = Image.new('RGB', (preview_w, preview_h), 'black')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{config['sample_name']} | C1 input previews (not mask overlays)", font=fonts[23], fill='white')
    draw.text((margin, 45), 'Original: fixed 407-651 ADU | Adjusted input: fixed 0-65535 (uint16)', font=fonts[18], fill='#DDDDDD')
    for j, frame in enumerate(starts):
        top = preview_header + j * (row_header + panel_h + row_gap)
        draw.text((margin, top), f'Original frame {frame} | {frame * config["frame_interval_s"]:.2f} s', font=fonts[20], fill='white')
        for col, raw, display in [(0, preview_raw[j], limits), (1, preview_adjusted[j], [0, 65535])]:
            image = np.repeat(gray_crop(raw, crop, scale, display)[..., None], 3, axis=2)
            image = composite(image, roi, gray_roi)
            ox, oy = margin + col * (panel_w + gap), top + row_header
            canvas.paste(Image.fromarray(image), (ox, oy))
            bar(draw, ox, oy)
    draw.text((margin, preview_h - 32), 'Input preprocessing changes pixel intensities. A brighter image or a larger mask does not establish greater accuracy.',
              font=fonts[18], fill='#FFD37E')
    preview_name = 'C1_original_adjusted_input_previews.png'
    save_png(np.asarray(canvas), figures / preview_name)
    files.append(preview_name)

    with (root / 'data/mask_comparison.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    scopes = list(dict.fromkeys(row['scope'] for row in rows))
    if len(scopes) != 2:
        raise ValueError(f'Expected ROI and full-image area records, received scopes {scopes!r}')
    scopes.sort(key=lambda s: 0 if 'roi' in s.lower() else 1)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    for ax, scope in zip(axes, scopes):
        selection = sorted((r for r in rows if r['scope'] == scope), key=lambda r: int(r['original_frame_start']))
        if len(selection) != 3:
            raise ValueError('Area comparison must contain exactly three records for each scope.')
        x = np.arange(3)
        vals_old = [float(row['area_old_um2']) for row in selection]
        vals_new = [float(row['area_new_um2']) for row in selection]
        ax.bar(x - .18, vals_old, .36, label='Baseline', color='#687C91')
        ax.bar(x + .18, vals_new, .36, label='Adjusted C1 input', color='#32AE9C')
        ax.set_xticks(x, [f'{int(row["original_frame_start"])}-{int(row["original_frame_stop_exclusive"]) - 1}' for row in selection])
        ax.set_xlabel('Original acquisition-frame block')
        ax.set_ylabel('Binary mask area (um^2)')
        ax.set_title('Inside original cell ROI' if 'roi' in scope.lower() else 'Full image before ROI clipping')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        top = max(vals_old + vals_new) if max(vals_old + vals_new) > 0 else 1
        ax.set_ylim(0, top * 1.20)
        for xi, row, yo, yn in zip(x, selection, vals_old, vals_new):
            ax.text(xi, max(yo, yn) + top * .035, f'Delta {float(row["delta_area_um2"]):+.2f}', ha='center', fontsize=9)
    fig.suptitle(f"{config['sample_name']} | C1 preprocessing mask-area sensitivity", fontsize=15)
    area_name = 'C1_mask_area_comparison.png'
    temp = figures / (area_name + '.tmp')
    fig.savefig(temp, format='png', dpi=180)
    plt.close(fig)
    temp.replace(figures / area_name)
    files.append(area_name)

    # Read back only the ten agreed sample pages; this is not an all-page scan.
    checks = []
    with tifffile.TiffFile(figures / tiff_name) as saved:
        if len(saved.pages) != 150 or not saved.is_imagej:
            raise AssertionError('Published TIFF does not have 150 ImageJ-compatible pages.')
        if saved.series[0].shape != (150, height, width, 3):
            raise AssertionError(f'Unexpected published TIFF series shape: {saved.series[0].shape}')
        for page, expected_hash in sorted(sample_hashes.items()):
            restored = saved.pages[page].asarray()
            observed_hash = hashlib.sha256(restored.tobytes()).hexdigest()
            if observed_hash != expected_hash:
                raise AssertionError(f'Lossless TIFF sample readback differs at page {page}')
            checks.append({'page_index': page, 'original_frame': frames[page], 'sha256': observed_hash, 'match': True})
        imagej_metadata = saved.imagej_metadata

    links = [('C1_original_adjusted_input_previews.png', 'Original / adjusted C1 input previews'),
             ('C1_mask_area_comparison.png', 'ROI and full-image mask area'),
             ('C1_old_new_mask_comparison_150pages.tif', 'Fiji RGB TIFF: 150 pages from three disjoint segments')]
    links += [(f'C1_old_new_mask_frame{f:04d}.png', f'Mask comparison: original frame {f}') for f in starts]
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>C1 preprocessing mask test</title>',
            '<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b36}img{max-width:100%;height:auto}article{margin:32px 0}a{color:#086b88}.notice{background:#fff0cc;padding:16px}</style>',
            f'<h1>{html.escape(config["sample_name"])}: C1 preprocessing mask test</h1>',
            '<p>Per-frame mat2gray / stretchlim(0.01) / imadjust(gamma=1) preprocessing, followed by the unchanged broad → support ≥10/50 → post-min32 → holes &lt;64 recipe.</p>',
            '<p class="notice"><strong>Three disjoint time segments:</strong> original frames 0–49, 1200–1249 and 2450–2499. The TIFF page joins are time jumps. Original frame, actual acquisition time and support-block identity are printed on every page.</p>',
            '<p>Baseline and new mask overlays use the same original C1 pixels and the same fixed 407–651 ADU display. Contours show full masks before ROI clipping; the gray outline is the original cell ROI. Differences: common gray, added green, removed magenta.</p>']
    for filename, title in links:
        body.append(f'<article><h2><a href="{html.escape(filename)}">{html.escape(title)}</a></h2>')
        if filename.endswith('.png'):
            body.append(f'<a href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(title)}"></a>')
        else:
            body.append('<p>Open in Fiji as a TIFF or a virtual stack. Page 1 = original frame 0, page 51 = frame 1200, page 101 = frame 2450.</p>')
        body.append('</article>')
    body += ['<p><a href="../data/mask_comparison.csv">Saved mask-area and overlap measurements</a> · <a href="../data/frame_mapping.csv">Original-frame mapping</a> · <a href="../README_RENDERING.txt">Rendering notes</a></p>',
             '<p>This is a contrast-sensitivity test. Area gain is not evidence of segmentation accuracy. No localization, assignment or TARDIS calculation is performed by this publisher.</p></html>']
    (figures / 'INDEX.html').write_text('\n'.join(body), encoding='utf-8')
    notes = '''C1 PREPROCESSING TEST: RENDERING NOTES

The publisher reads existing transformed input frames, existing baseline/new masks,
and saved area/overlap statistics. It does not invoke Nellie, localization fitting,
spatial classification, TARDIS, or statistical inference.

The 150-page RGB TIFF contains disjoint original-frame segments 0-49, 1200-1249,
and 2450-2499. Page order is a viewing order, not a continuous acquisition time
axis. Original-frame identity, real acquisition time, and support block are printed
on every page. Fiji page numbers are 1-based; saved page_index is 0-based.

Both mask-overlay panels use identical ORIGINAL C1 intensity values, clipped only
for display to 407-651 ADU. The adjusted input is shown separately in previews at
0-65535. Neither display window changes saved masks. Mask contours are calculated
on the full binary mask before display cropping; original cell ROI is gray.
Difference colors: common foreground gray, added green, removed magenta.
The fixed native-coordinate crop is displayed with nearest-neighbor 2x replication.
Contours are antialiased for display only. Scale bars are 2 um.

Only the agreed ten sample TIFF pages are losslessly read back and compared with
the renderer's output pixels. Metadata/page count are checked. This is not an
independent verification of all 150 pages or all 2500 original frames.

Raw-C1 zero-region validity comes from the saved scientific products; newly clipped
zero-valued adjusted pixels are not interpreted as unknown by this publisher.
Identity C1/C2 registration remains independently unverified.
'''
    (root / 'README_RENDERING.txt').write_text(notes, encoding='utf-8')
    manifest = {'sample_id': config['sample_id'], 'scientific_steps_invoked': 0,
                'tiff': {'relative_path': 'figures/' + tiff_name, 'pages': 150,
                         'shape_TYXS': [150, height, width, 3], 'dtype': 'uint8',
                         'compression': 'deflate', 'imagej': True,
                         'disjoint_original_segments': [[s, s + 50] for s in starts],
                         'metadata': imagej_metadata},
                'sample_page_readback': checks, 'sample_pages_checked': len(checks),
                'files': [{'relative_path': 'figures/' + name,
                           'bytes': (figures / name).stat().st_size} for name in files],
                'elapsed_seconds': time.perf_counter() - started}
    (root / 'render_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'render_complete': True, 'sample_pages_checked': len(checks),
                      'scientific_steps_invoked': 0, 'elapsed_seconds': manifest['elapsed_seconds'],
                      'index': str(figures / 'INDEX.html')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
