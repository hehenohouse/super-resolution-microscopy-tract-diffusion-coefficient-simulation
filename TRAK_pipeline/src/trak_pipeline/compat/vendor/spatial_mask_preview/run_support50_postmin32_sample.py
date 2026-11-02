#!/usr/bin/env python3
"""Sample-configured raw-broad support50 mask, frozen classification and views.

Use the existing science environment; geometry runs in the existing cleanup
environment. Scientific inputs and previous results are never modified.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/noton-mplcache')
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tifffile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'mito_tardis'))
LABELS = ['mito-compatible', 'non-mito-compatible', 'unknown']
REGISTRATION = 'exploratory_identity_registration_unverified'
EXTRA = ['postmin32_block_index', 'postmin32_original_frame_start',
         'postmin32_original_frame_end_inclusive', 'postmin32_pointwise_label',
         'postmin32_unknown_reason', 'postmin32_axis_x_px', 'postmin32_axis_y_px',
         'postmin32_center_inside_mask', 'postmin32_mask_sha256',
         'postmin32_registration_status', 'postmin32_old_to_new',
         'postmin32_touches_ROI_boundary', 'postmin32_crosses_ROI_boundary']
REPS = {0, 49, 50, 200, 1199, 1200, 1999, 2000, 2499}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def known_records(folder):
    result = {}
    for name in ['output_fingerprints.json', 'input_fingerprints_before.json', 'input_fingerprints.json']:
        p = Path(folder)/name
        if not p.exists():
            continue
        records = json.loads(p.read_text())
        if isinstance(records, list):
            result.update({r['path']: r['sha256'] for r in records})
        else:
            result.update({str(Path(folder)/k) if not k.startswith('/') else k: v for k, v in records.items()})
    return result


def check_files(paths, known):
    from render_point_stack import fingerprint
    records = [fingerprint(p) for p in paths]
    for r in records:
        if r['path'] in known:
            assert r['sha256'] == known[r['path']], r['path']
    return records


def archive(out):
    for name in [Path(__file__).name, 'support50_postmin32_core.py', 'geometry.py',
                 'run_full.py', 'run_mean50_outputs.py', 'run_support50_postmin32_preview.py',
                 'run_preview.py', 'contracts.py', 'mean50_contract.py',
                 'test_support50_postmin32_core.py', 'test_geometry.py']:
        shutil.copyfile(HERE/name, out/'source_snapshot'/name)
    for name in ['render_point_stack.py', 'point_stack_core.py', 'render_time_projection.py']:
        shutil.copyfile(PROJECT/'mito_tardis'/name, out/'source_snapshot'/name)
    dump(out/'software_versions.json', dict(python=sys.version, executable=sys.executable,
         packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'Pillow', 'tifffile', 'h5py']}))
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (out/'requirements.lock.txt').write_text(freeze.stdout)


def mask_build(config_path):
    from render_point_stack import fingerprint, roi_reader, mat_vertices, Tee
    from support50_postmin32_core import process_counts
    from run_support50_postmin32_preview import write_scientific_tiff
    c = json.loads(config_path.read_text())
    assert c['source_frame_count'] == 2500 and c['support_total_frames'] == 50
    assert c['support_minimum_count'] == 10 and c['minimum_foreground_area_pixel2'] == 32
    assert c['hole_area_exclusive_max_pixel2'] == 64 and c['per_frame_min32'] is False
    assert c['pixel_size_nm'] == 117 and c['frame_interval_s'] == .02
    parent = Path(c['localization_view_source'])
    assert json.loads((parent/'COMPLETED.json').read_text())['delivery_checks_passed']
    original = json.loads((parent/'resolved_config.json').read_text())
    original.update(c)
    c = original
    c['cleanup_python'] = str(PROJECT/c['cleanup_python']) if not Path(c['cleanup_python']).is_absolute() else c['cleanup_python']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = HERE/'results'/f"{c['sample_id']}_support50_postmin32_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for n in ['checkpoints', 'source_snapshot', 'qc']:
        (out/n).mkdir()
    c['external_destination'] = str(Path(c['C1_file']).parent/'mitochondria_masks'/('Nellie_broad_support50_ge10_postmin32_holes_lt64_'+stamp))
    c['previous_mask_directory'] = c['mask_directory']
    for obsolete in ['new_mask_sha256', 'expected_added_ROI_pixels']:
        c.pop(obsolete, None)
    broad_dir = Path(c['broad_source_directory'])
    paths = [config_path, Path(c['C1_file']), Path(c['ROI_file']), Path(c['MAT_file']),
             broad_dir/'full_component_labels.npy', broad_dir/'original_frame_numbers.npy', broad_dir/'frame_completed.npy',
             broad_dir/'parameters.json', broad_dir/'run_full.py', broad_dir/'nellie_worker.py', broad_dir/'COMPLETE.json',
             parent/'resolved_config.json', parent/'classification/all_ROI_candidates_mean50.csv',
             *[Path(c['previous_mask_directory'])/n for n in ['roi_mask.npy', 'roi_vertices_xy.npy', 'clean_full_mask.npy']]]
    known = known_records(parent) | known_records(broad_dir) | known_records(c['previous_mask_directory'])
    (out/'command.txt').write_text(shlex.join([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--config', str(config_path)])+'\n')
    with (out/'run.log').open('w') as log:
        oldout, olderr = sys.stdout, sys.stderr
        sys.stdout = Tee(oldout, log); sys.stderr = Tee(olderr, log)
        try:
            print('MASK_OUTPUT', out, flush=True)
            before = check_files(paths, known); dump(out/'input_fingerprints_before.json', before)
            done = json.loads((broad_dir/'COMPLETE.json').read_text())
            assert done['completed_frames'] == 2500 and done['all_checks_passed']
            np.testing.assert_array_equal(np.load(broad_dir/'original_frame_numbers.npy'), np.arange(2500))
            assert np.load(broad_dir/'frame_completed.npy').shape == (2500,) and np.load(broad_dir/'frame_completed.npy').all()
            labels = np.load(broad_dir/'full_component_labels.npy', mmap_mode='r')
            assert labels.shape == (2500, 684, 428) and labels.dtype == np.uint32
            roi = np.load(Path(c['previous_mask_directory'])/'roi_mask.npy')
            vertices = np.load(Path(c['previous_mask_directory'])/'roi_vertices_xy.npy')
            lib, roi_script = roi_reader()
            np.testing.assert_array_equal(vertices, lib.load_imagej_roi(c['ROI_file']))
            np.testing.assert_array_equal(vertices, mat_vertices(c['MAT_file'], c['MAT_record']))
            assert len(vertices) == c['expected_roi_vertices'] and roi.sum() == c['expected_roi_pixels']
            yy, xx = np.mgrid[:684, :428]
            np.testing.assert_array_equal(roi, lib.points_in_polygon(np.c_[xx.ravel(), yy.ravel()], vertices).reshape(684, 428))
            with tifffile.TiffFile(c['C1_file']) as tf:
                assert len(tf.pages) == 2500 and tf.series[0].shape == labels.shape and tf.series[0].axes == 'TYX'
            # Verify the recorded native-label stage is saved before min16 cleanup.
            source_code = (broad_dir/'run_full.py').read_text()
            assert 'labs[f]=labels' in source_code and 'clean=keep[labels]' in source_code
            # NumPy indexing creates `clean`; it does not modify `labels`.
            # Saved labels can be written later while still representing the
            # pre-cleanup components. Independent all-frame audit verifies it.
            arrays = {n: np.zeros((50, 684, 428), dtype=np.uint8 if n == 'foreground_count' else bool)
                      for n in ['foreground_count', 'input_ge10_full_mask', 'min32_full_mask', 'clean_full_mask', 'filled_holes_full_mask', 'analysis_ROI_mask']}
            blocks, components, holes = [], [], []
            for b in range(50):
                count = np.count_nonzero(labels[b*50:b*50+50] > 0, axis=0).astype(np.uint8)
                r = process_counts(count, 10, 32, 64, 50)
                for name, key in [('foreground_count', 'foreground_count'), ('input_ge10_full_mask', 'support_mask'),
                                  ('min32_full_mask', 'post_min32_mask'), ('clean_full_mask', 'final_mask'), ('filled_holes_full_mask', 'filled_holes')]:
                    arrays[name][b] = r[key]
                arrays['analysis_ROI_mask'][b] = r['final_mask'] & roi
                assert arrays['analysis_ROI_mask'][b].any()
                blocks.append(dict(block_index=b, original_frame_start=b*50, original_frame_end_inclusive=b*50+49,
                    source_frame_count=50, time_start_s=b, time_end_exclusive_s=b+1,
                    full_pixel2=int(r['final_mask'].sum()), ROI_pixel2=int(arrays['analysis_ROI_mask'][b].sum()),
                    ROI_um2=float(arrays['analysis_ROI_mask'][b].sum()*.117**2),
                    filled_holes=int(r['background_component_fill'].sum()), added_hole_pixels=int(r['filled_holes'].sum())))
                for i in range(1, len(r['support_component_areas'])):
                    components.append(dict(block_index=b, component_label=i, area_pixel2=int(r['support_component_areas'][i]), kept=bool(r['support_component_keep'][i])))
                for i in range(1, len(r['background_component_areas'])):
                    holes.append(dict(block_index=b, background_label=i, area_pixel2=int(r['background_component_areas'][i]), touches_image_edge=bool(r['background_component_edge'][i]), filled=bool(r['background_component_fill'][i])))
                if (b+1) % 5 == 0:
                    dump(out/'checkpoints'/f'masks_{(b+1)*50:04d}.json', dict(completed_source_frames=(b+1)*50, completed_blocks=b+1))
                    print('MASK_CHECKPOINT', (b+1)*50, '/2500', flush=True)
            for name, a in arrays.items():
                np.save(out/(name+'.npy'), a)
            np.save(out/'roi_mask.npy', roi); np.save(out/'roi_vertices_xy.npy', vertices)
            np.save(out/'original_frame_to_mask_block.npy', np.arange(2500)//50)
            np.save(out/'original_frame_numbers_by_block.npy', np.arange(2500).reshape(50, 50))
            np.save(out/'block_completed.npy', np.ones(50, bool))
            pd.DataFrame(blocks).to_csv(out/'block_frame_mapping.csv', index=False)
            pd.DataFrame(blocks).to_csv(out/'per_block_statistics.csv', index=False)
            pd.DataFrame(components).to_csv(out/'component_decisions.csv', index=False)
            pd.DataFrame(holes).to_csv(out/'hole_decisions.csv', index=False)
            for n in ['clean_full_mask', 'analysis_ROI_mask']:
                write_scientific_tiff(out/(n+'.ome.tif'), arrays[n], c['sample_name']+' '+n)
            params = dict(c, shape=[50, 684, 428], axes='TYX', source_frame_count=2500, mask_block_interval_s=1.,
                source_frame_interval_s=.02, pixel_size_um=.117, per_frame_min32=False, support_minimum_count=10,
                support_total_frames=50, minimum_foreground_area_pixel2=32, foreground_connectivity=8,
                hole_area_exclusive_max_pixel2=64, background_connectivity=4, foreground_boundary_exemptions=False,
                no_Nellie_rerun=True, no_image_averaging=True, no_alignment=True, no_W5_voting=True,
                original_frame_to_block='floor(frame/50)', selected_as_final=False,
                processing_order=['recorded native individual-frame Nellie broad labels > 0 BEFORE extra min16/min32 cleanup',
                                  'nonoverlapping 50-frame pixel support>=10', 'full-image FG8 area>=32', 'enclosed BG4 holes<64', 'original ROI intersection'],
                registration_status=REGISTRATION, no_TARDIS=True)
            dump(out/'parameters.json', params); dump(out/'resolved_config.json', c)
            shutil.copyfile(config_path, out/'config.json'); shutil.copyfile(roi_script, out/'source_snapshot/roi_reader.py')
            dump(out/'mask_validation.json', dict(all50_blocks_complete=True, ROI_MAT_vertices_and_pixels_identical=True,
                no_empty_ROI_blocks=True, scientific_TIFF_all50_pages_equal=True,
                C_order_array_bytes_sha256={n: array_hash(a) for n, a in arrays.items()},
                independent_reference_pending=True))
            after = check_files(paths, known); assert before == after
            dump(out/'input_fingerprints_after.json', after)
            archive(out)
            dump(out/'COMPLETE.json', dict(all_passed=False, numerical_checks_passed=True, independent_reference_pending=True, output=str(out)))
            (HERE/'latest_postmin32_WT_mask_local.txt').write_text(str(out)+'\n')
            print('MASK_READY', out, flush=True)
        except BaseException:
            dump(out/'FAILED.json', dict(traceback=traceback.format_exc())); raise
        finally:
            sys.stdout, sys.stderr = oldout, olderr


def classification_worker(out):
    from run_full import FrameRows, selected, brute_intersection
    from geometry import classify, point_inside_mask, roi_boundary_flags
    import shapely
    c = json.loads((out/'resolved_config.json').read_text())
    mask_dir = Path(c['mask_directory']); full = np.load(mask_dir/'clean_full_mask.npy', mmap_mode='r')
    polygon = shapely.Polygon(np.load(mask_dir/'roi_vertices_xy.npy'))
    source = Path(c['audit_source']); audit = out/'classification/all_ROI_candidates_postmin32.csv'
    chosen = out/c['selected_classification_csv']; total = Counter(); transitions = Counter(); frames = []; checked = []; ids = set()
    with source.open(newline='') as s, audit.open('w', newline='') as a, chosen.open('w', newline='') as k, tifffile.TiffFile(c['C1_file']) as movie:
        reader = FrameRows(s); fields = reader.fields; assert not set(fields) & set(EXTRA)
        wa = csv.DictWriter(a, fieldnames=fields+EXTRA); wk = csv.DictWriter(k, fieldnames=fields+EXTRA)
        wa.writeheader(); wk.writeheader()
        for f in range(2500):
            rows = reader.for_frame(f); b = f//50; mask = full[b]; zeros = movie.pages[f].asarray() == 0; counts = Counter(); excluded = 0
            for row in rows:
                index = int(row['candidate_index']); assert index not in ids; ids.add(index)
                extra = dict(zip(EXTRA, [b, b*50, b*50+49, 'excluded_qc', '', '', '', '', c['new_mask_sha256'], REGISTRATION, 'excluded_qc']))
                if selected(row):
                    x, y = float(row['fit_x_px']), float(row['fit_y_px'])
                    ax, ay = float(row['x_se_nm'])/117, float(row['y_se_nm'])/117
                    assert np.isfinite([x, y, ax, ay]).all() and min(ax, ay) > 0
                    result = classify(mask, zeros, x, y, ax, ay, 1e-12)
                    boundary = roi_boundary_flags(polygon, x, y, ax, ay, 1e-12)
                    old = row[c['old_label_field']] if c.get('old_label_field') else 'not_previously_classified'
                    new = result['label']
                    assert (old in LABELS or old == 'not_previously_classified') and new in LABELS
                    # Unknown rule is independent of the segmentation method.
                    if old in LABELS:assert (old == 'unknown') == (new == 'unknown')
                    extra.update(postmin32_pointwise_label=new, postmin32_unknown_reason=result['reason'],
                        postmin32_axis_x_px=ax, postmin32_axis_y_px=ay, postmin32_center_inside_mask=point_inside_mask(mask, x, y),
                        postmin32_old_to_new=old+' -> '+new,
                        postmin32_touches_ROI_boundary=boundary['touches_ROI_boundary'],
                        postmin32_crosses_ROI_boundary=boundary['crosses_ROI_boundary'])
                    counts[new] += 1; transitions[(old, new)] += 1
                    if f in REPS:
                        hit = brute_intersection(mask, x, y, ax, ay, 1e-12); zero = brute_intersection(zeros, x, y, ax, ay, 1e-12)
                        outside = x-ax < -.5-1e-12 or x+ax > 427.5+1e-12 or y-ay < -.5-1e-12 or y+ay > 683.5+1e-12
                        expected = 'unknown' if zero or outside else ('mito-compatible' if hit else 'non-mito-compatible')
                        assert new == expected; checked.append(index)
                else:
                    excluded += 1
                updated = dict(row, **extra); wa.writerow(updated)
                if selected(row):
                    wk.writerow(updated)
            record = dict(frame=f, time_s=f*.02, block_index=b, ROI_candidates=len(rows), selected=sum(counts.values()), QC_excluded=excluded,
                          On=counts[LABELS[0]], Not_on=counts[LABELS[1]], unknown=counts[LABELS[2]])
            assert record['selected']+excluded == len(rows); frames.append(record)
            total.update({key: record[key] for key in ['ROI_candidates', 'selected', 'QC_excluded', 'On', 'Not_on', 'unknown']})
            if (f+1) % 250 == 0:
                a.flush(); k.flush()
                dump(out/'checkpoints'/f'classified_{f+1:04d}.json', dict(completed_frames=f+1, counts=dict(total)))
                print('CLASSIFICATION_CHECKPOINT', f+1, json.dumps(dict(total)), flush=True)
        assert reader.head is None
    for key, expected in [('ROI_candidates', c['expected_ROI_candidates']), ('selected', c['expected_selected']), ('QC_excluded', c['expected_QC_excluded'])]:
        assert total[key] == expected
    n = kept = 0
    with source.open(newline='') as s, audit.open(newline='') as a, chosen.open(newline='') as k:
        before = csv.DictReader(s); after = csv.DictReader(a); chosen_reader = csv.DictReader(k)
        for old in before:
            new = next(after); assert {key: new[key] for key in before.fieldnames} == old; n += 1
            if selected(old):
                assert next(chosen_reader) == new; kept += 1
        assert next(after, None) is None and next(chosen_reader, None) is None
    assert n == c['expected_ROI_candidates'] and kept == c['expected_selected']
    frame = pd.DataFrame(frames); frame.to_csv(out/'classification/per_frame_counts.csv', index=False)
    frame.groupby('block_index')[['ROI_candidates', 'selected', 'QC_excluded', 'On', 'Not_on', 'unknown']].sum().reset_index().to_csv(out/'classification/per_block_counts.csv', index=False)
    old_labels = LABELS if c.get('old_label_field') else ['not_previously_classified']
    pd.DataFrame([dict(old_label=old, new_label=new, count=transitions[(old, new)], fraction_of_selected=transitions[(old, new)]/kept)
                  for old in old_labels for new in LABELS]).to_csv(out/'classification/old_new_transition_table.csv', index=False)
    dump(out/'classification/summary.json', dict(total))
    dump(out/'classification/validation.json', dict(all_original_rows_and_cells_preserved_verbatim=True,
        original_QC_coordinates_SE_calibration_and_old_labels_unchanged=True, all2500_frames_preserved=True,
        independent_full_pixel_enumeration_passed=True, independently_checked_candidates=checked,
        original_C1_zero_and_FOV_unknown_policy_preserved=True,
        baseline='previous classification retained; no nesting assumed' if c.get('old_label_field') else 'first classification of frozen annular fits'))
    print('CLASSIFICATION_COMPLETE', json.dumps(dict(total)), flush=True)


def mask_previews(out):
    from render_point_stack import pixel_hash
    from point_stack_core import gray_crop, mask_contour_alpha, composite, ROI_COLOR, MASK_COLOR, roi_alpha
    c = json.loads((out/'resolved_config.json').read_text()); crop = c['crop_xyxy_px']; scale = 2
    (out/'qc').mkdir(exist_ok=True)
    masks = np.load(out/'clean_full_mask.npy'); roi = np.load(out/'roi_mask.npy'); vertices = np.load(out/'roi_vertices_xy.npy')
    x0, y0, x1, y1 = crop
    inside = np.repeat(np.repeat(roi[y0:y1, x0:x1], scale, 0), scale, 1); h, w = inside.shape
    font = lambda n: ImageFont.truetype(c.get('font_path', '/System/Library/Fonts/Supplemental/Arial.ttf'), n)
    records = []
    with tifffile.TiffFile(c['C1_file']) as movie:
        for f in [200, 1200, 2000]:
            raw = movie.pages[f].asarray()
            base = np.repeat(gray_crop(raw, crop, scale, [407, 651])[..., None], 3, axis=2); base[~inside] = 0
            base = composite(base, roi_alpha(vertices, crop, scale), ROI_COLOR)
            alpha = mask_contour_alpha(masks[f//50], crop, scale); alpha[~inside] = 0
            panel = composite(base, alpha, MASK_COLOR)
            image = Image.new('RGB', (w*2+24, h+90)); image.paste(Image.fromarray(base), (0, 90)); image.paste(Image.fromarray(panel), (w+24, 90))
            draw = ImageDraw.Draw(image)
            draw.text((10, 8), f"{c['sample_name']} | original C1 frame {f} | {f*.02:.2f} s | block {f//50}", font=font(22), fill='white')
            draw.text((10, 42), 'C1 raw | ROI only', font=font(22), fill='white')
            draw.text((w+34, 42), 'C1 + support50>=10 / post-min32 / holes<64', font=font(22), fill='white')
            path = out/'qc'/f'frame_{f:04d}_C1_new_mask.png'; image.save(path)
            records.append(dict(frame=f, block=f//50, source_page_sha256=pixel_hash(raw), PNG=path.name, display_ADU=[407, 651]))
    dump(out/'qc/representative_frame_manifest.json', records)


def views(mask_dir, cache_config=None):
    from render_point_stack import fingerprint, Tee, roi_reader, mat_vertices
    from run_mean50_outputs import project_points
    if cache_config:
        c = json.loads(cache_config.read_text()); parent = Path(c['localization_cache'])
        assert Path(c['mask_directory']).resolve() == mask_dir.resolve(), 'configured mask differs from --views'
        fit_config = json.loads((parent/'config.json').read_text())
        for key in ['sample_name', 'sample_id', 'C1_file', 'C2_file', 'crop_xyxy_px', 'expected_roi_pixels', 'expected_roi_vertices']:
            assert c[key] == fit_config[key], key
        for upper, lower in [('ROI_file', 'roi_file'), ('MAT_file', 'mat_file'), ('MAT_record', 'mat_record')]:
            assert c[upper] == fit_config[lower], upper
        summary = json.loads((parent/'summary.json').read_text())
        assert json.loads((parent/'COMPLETED.json').read_text())['delivery_checks_passed']
        assert summary['frames'] == 2500
        c.update(expected_ROI_candidates=int(summary['ROI_candidates']), expected_selected=int(summary['selected']),
                 expected_QC_excluded=int(summary['QC_excluded']))
        audit_source = parent/'all_ROI_target_fits.csv'; old_label_field = None; previous_projection = None
    else:
        c = json.loads((mask_dir/'resolved_config.json').read_text()); parent = Path(c['localization_view_source'])
        audit_source = parent/'classification/all_ROI_candidates_mean50.csv'; old_label_field = 'mean50_pointwise_label'
        previous_projection = str(parent/'projections')
    assert json.loads((mask_dir/'COMPLETE.json').read_text())['all_passed']
    c['cleanup_python'] = str((PROJECT/c['cleanup_python']).absolute()) if not Path(c['cleanup_python']).is_absolute() else c['cleanup_python']
    c.update(mask_directory=str(mask_dir), audit_source=str(audit_source),
        old_label_field=old_label_field, selected_classification_csv='classification/selected_localizations_postmin32.csv',
        classification_label_field='postmin32_pointwise_label', mask_recipe='support50_ge10_postmin32_holes_lt64',
        mask_recipe_display='support50>=10/post-min32/holes<64',
        projection_footer='50-frame raw-broad support >=10; min32 after support only; repeated observations retained.',
        previous_projection=previous_projection, new_mask_sha256=fingerprint(mask_dir/'clean_full_mask.npy')['sha256'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = HERE/'results'/f"{c['sample_id']}_postmin32_projection_{stamp}"; out.mkdir(parents=True, exist_ok=False)
    for n in ['classification', 'projections', 'checkpoints', 'source_snapshot', 'qc']:
        (out/n).mkdir()
    c['external_destination'] = str(Path(c['C1_file']).parent/'analysis_results/support50_postmin32_localization_views'/stamp)
    paths = [Path(c['C1_file']), Path(c['ROI_file']), Path(c['MAT_file']), Path(c['font_path']), Path(c['audit_source']),
             *[mask_dir/n for n in ['clean_full_mask.npy', 'analysis_ROI_mask.npy', 'roi_mask.npy', 'roi_vertices_xy.npy',
                 'original_frame_to_mask_block.npy', 'original_frame_numbers_by_block.npy', 'block_completed.npy', 'parameters.json', 'COMPLETE.json']]]
    if cache_config:
        paths += [cache_config, parent/'config.json', parent/'summary.json', parent/'frame_counts.csv']
    else:
        paths += [parent/'resolved_config.json', parent/'classification/per_frame_counts.csv', parent/'projections/localization_count_maps.npz']
    known = known_records(mask_dir) | known_records(parent)
    before = check_files(paths, known); dump(out/'input_fingerprints_before.json', before)
    parameters = json.loads((mask_dir/'parameters.json').read_text())
    assert parameters['per_frame_min32'] is False and parameters['support_minimum_count'] == 10
    assert parameters['support_total_frames'] == 50 and parameters['minimum_foreground_area_pixel2'] == 32
    assert parameters['hole_area_exclusive_max_pixel2'] == 64 and parameters['foreground_connectivity'] == 8
    assert parameters['background_connectivity'] == 4 and parameters['pixel_size_um'] == .117
    assert parameters['source_frame_interval_s'] == .02 and parameters['mask_block_interval_s'] == 1.
    assert parameters['C1_file'] == c['C1_file']
    full = np.load(mask_dir/'clean_full_mask.npy', mmap_mode='r'); roi = np.load(mask_dir/'roi_mask.npy')
    vertices = np.load(mask_dir/'roi_vertices_xy.npy'); library, _ = roi_reader()
    assert full.shape == (50, 684, 428) and full.dtype == bool
    np.testing.assert_array_equal(np.load(mask_dir/'analysis_ROI_mask.npy'), full & roi)
    np.testing.assert_array_equal(vertices, library.load_imagej_roi(c['ROI_file']))
    np.testing.assert_array_equal(vertices, mat_vertices(c['MAT_file'], c['MAT_record']))
    assert len(vertices) == c['expected_roi_vertices'] and int(roi.sum()) == c['expected_roi_pixels']
    yy, xx = np.mgrid[:684, :428]
    np.testing.assert_array_equal(roi, library.points_in_polygon(np.c_[xx.ravel(), yy.ravel()], vertices).reshape(684, 428))
    np.testing.assert_array_equal(np.load(mask_dir/'original_frame_to_mask_block.npy'), np.arange(2500)//50)
    np.testing.assert_array_equal(np.load(mask_dir/'original_frame_numbers_by_block.npy'), np.arange(2500).reshape(50, 50))
    assert np.load(mask_dir/'block_completed.npy').shape == (50,) and np.load(mask_dir/'block_completed.npy').all()
    dump(out/'source_validation.json', dict(all_passed=True, ROI_MAT_mask_vertices_identical=True,
        ROI_pixels=int(roi.sum()), ROI_vertices=len(vertices), all2500_frames_to50_blocks_exact=True,
        mask_recipe_verified=True, registration_status=REGISTRATION))
    dump(out/'resolved_config.json', c)
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--views', str(mask_dir)]
    if cache_config:
        command += ['--cache-config', str(cache_config)]
        shutil.copyfile(cache_config, out/'config.json')
    (out/'command.txt').write_text(shlex.join(command)+'\n')
    with (out/'run.log').open('w') as log:
        oldout, olderr = sys.stdout, sys.stderr; sys.stdout = Tee(oldout, log); sys.stderr = Tee(olderr, log)
        try:
            cmd = [c['cleanup_python'], '-B', '-u', str(Path(__file__).resolve()), '--worker', str(out)]
            (out/'classification_command.txt').write_text(shlex.join(cmd)+'\n')
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
            assert process.wait() == 0
            frame = pd.read_csv(out/'classification/per_frame_counts.csv')
            old = pd.read_csv(parent/'frame_counts.csv' if cache_config else parent/'classification/per_frame_counts.csv')
            for field in ['frame', 'ROI_candidates', 'selected', 'QC_excluded']:
                np.testing.assert_array_equal(frame[field], old[field])
            roi = np.load(mask_dir/'roi_mask.npy'); vertices = np.load(mask_dir/'roi_vertices_xy.npy')
            project_points(c, out, roi, vertices)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16, 6))
            for key, color, title in [('On', '#00ac83', 'On (mito-compatible)'), ('Not_on', '#9b50df', 'Not-on (non-mito-compatible)'), ('QC_excluded', '#ef8700', 'QC excluded')]:
                ax.plot(frame.frame, frame[key], lw=.8, color=color, label=title)
            ax.set(xlim=(0, 2499), ylim=(0, None), xlabel='Original frame (0-based)', ylabel='Localizations per frame',
                   title=c['sample_name']+' | support50>=10 / post-min32 / holes<64\nUnsmoothed counts; C1/C2 identity registration unverified')
            ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'qc/per_frame_counts.png', dpi=150); plt.close(fig)
            archive(out)
            after = check_files(paths, known); assert before == after; dump(out/'input_fingerprints_after.json', after)
            dump(out/'validation.json', dict(all_numerical_checks_passed=True, source_fingerprints_unchanged=True,
                all_fits_QC_and_old_labels_unchanged=True, all_valid_count_grid_identical=True, visual_review_pending=True, no_TARDIS=True))
            dump(out/'COMPLETED.json', dict(all_implementation_checks_passed=True, visual_review_pending=True, counts=json.loads((out/'classification/summary.json').read_text()), output=str(out)))
            (HERE/('latest_postmin32_'+c['sample_id']+'_projection_local.txt')).write_text(str(out)+'\n')
            if c['sample_id']=='WT_T1':(HERE/'latest_postmin32_WT_projection_local.txt').write_text(str(out)+'\n')
            print('VIEWS_READY', out, flush=True)
        except BaseException:
            dump(out/'FAILED.json', dict(traceback=traceback.format_exc())); raise
        finally:
            sys.stdout, sys.stderr = oldout, olderr


def publish(src, reference=None):
    from render_point_stack import fingerprint
    mask = (src/'clean_full_mask.npy').exists()
    if mask:
        assert reference is not None
        r = json.loads(reference.read_text()); validation = json.loads((src/'mask_validation.json').read_text())
        protected = {record['path']: record['sha256'] for record in json.loads((src/'input_fingerprints_before.json').read_text())}
        assert protected[r['source_broad_labels']['path']] == r['source_broad_labels']['sha256']
        # Reference is an independently implemented full2500-label reconstruction.
        expected = {name: values['C_order_array_bytes_sha256'] for name, values in r['expected_arrays'].items()}
        for n in ['foreground_count', 'input_ge10_full_mask', 'min32_full_mask', 'clean_full_mask', 'analysis_ROI_mask']:
            assert validation['C_order_array_bytes_sha256'][n] == expected[n], n
        # Remove inherited mean50-only metadata, including a stale parent SHA.
        for filename in ['resolved_config.json', 'parameters.json']:
            parameters = json.loads((src/filename).read_text())
            for obsolete in ['new_mask_sha256', 'expected_added_ROI_pixels']:
                parameters.pop(obsolete, None)
            parameters['mask_directory'] = parameters['external_destination']
            parameters['output_relative_directory'] = 'mitochondria_masks'
            dump(src/filename, parameters)
        shutil.copyfile(reference, src/'independent_reconstruction_reference.json')
        validation.update(independent_reference_pending=False, independent_all2500_label_reconstruction_match=True)
        dump(src/'mask_validation.json', validation)
        mask_previews(src)
        for name in ['run_support50_postmin32_preview.py', 'run_preview.py', 'contracts.py', 'mean50_contract.py']:
            shutil.copyfile(HERE/name, src/'source_snapshot'/name)
        dump(src/'COMPLETE.json', dict(all_passed=True, source_frames=2500, output_blocks=50, selected_as_final=False))
        dump(src/'COMPLETED.json', dict(delivery_checks_passed=True, no_TARDIS=True))
    else:
        done = json.loads((src/'COMPLETED.json').read_text()); assert done['all_implementation_checks_passed']
        for name in ['all_timepoints_triptych.png', 'on_all_timepoints.png', 'not_on_all_timepoints.png', 'all_valid_all_timepoints.png']:
            with Image.open(src/'projections'/name) as im:
                im.load()
        done.update(visual_review_pending=False, delivery_checks_passed=True)
        dump(src/'COMPLETED.json', done)
        v = json.loads((src/'validation.json').read_text()); v.update(visual_review_pending=False, visual_review_passed=True); dump(src/'validation.json', v)
    assert json.loads((src/'input_fingerprints_before.json').read_text()) == json.loads((src/'input_fingerprints_after.json').read_text())
    c = json.loads((src/'resolved_config.json').read_text()); dst = Path(c['external_destination'])
    assert not dst.exists()
    shutil.copyfile(Path(__file__), src/'source_snapshot'/Path(__file__).name)
    (src/'publish_command.txt').write_text(shlex.join([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
         '--publish', str(src)]+(['--reference', str(reference)] if reference else []))+'\n')
    (src/'README.txt').write_text(c['sample_name']+' | raw-broad support50 / post-min32 / holes<64\n'
         'Original frames 0-2499; 50 nonoverlapping blocks, frame f uses block f//50.\n'
         'Use the recorded native Nellie broad labels before extra per-frame area cleanup.\n'
         'Pixel support >=10 of 50 -> full-image 8-connected foreground area >=32 -> enclosed 4-connected holes <64 -> ROI.\n'
         'No additional per-frame min32, image averaging, alignment, W5 voting or Nellie rerun.\n'
         'Masks represent blockwise temporal support, not instantaneous boundaries.\n'
         'Frozen TrackMate0.28um +7x7 annular-background fits and original QC/nominal SEs retained.\n'
         'Spatial classification: own k1 diagonal ellipse, SE_nm/117, closed pixel squares, tangency included, q<=1+1e-12.\n'
         'Unknown priority uses ORIGINAL same-frame C1 zero pixels and closed field of view.\n'+
         ('Previous labels retained verbatim; no nesting of mask methods is assumed.\n'
          if c.get('old_label_field') else 'First spatial classification; all original fit/QC fields retained verbatim.\n')+
         'On/Not-on are spatial compatibility labels; C1/C2 identity registration remains unverified.\n'
         'No TARDIS, trajectories, state fitting or biological inference. Old results and default/batch state unchanged.\n')
    dump(src/'output_fingerprints.json', [fingerprint(p) for p in sorted(src.rglob('*')) if p.is_file() and p.name != 'output_fingerprints.json'])
    # External-drive metadata/xattr copying is unsupported; copy bytes only.
    dst.mkdir(parents=True, exist_ok=False)
    for p in sorted(src.rglob('*')):
        q = dst/p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            assert p.is_file() and not p.is_symlink()
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
    files = []
    for p in sorted(src.rglob('*')):
        if p.is_file():
            q = dst/p.relative_to(src); a, b = fingerprint(p), fingerprint(q)
            assert a['sha256'] == b['sha256'] and a['bytes'] == b['bytes']; files.append(b)
    dump(dst/'publication.json', dict(copied_files_SHA256_equal=True, files_verified=len(files), local_source=str(src)))
    done = json.loads((dst/'COMPLETED.json').read_text())
    done.update(output=str(dst), local_stage=str(src), external_copy_verified=True)
    dump(dst/'COMPLETED.json', done)
    dump(dst/'output_fingerprints.json', [fingerprint(p) for p in sorted(dst.rglob('*'))
         if p.is_file() and p.name != 'output_fingerprints.json'])
    suffix = 'mask_output.txt' if mask else 'projection_output.txt'
    (HERE/('latest_postmin32_'+c['sample_id']+'_'+suffix)).write_text(str(dst)+'\n')
    if c['sample_id']=='WT_T1':(HERE/('latest_postmin32_WT_'+suffix)).write_text(str(dst)+'\n')
    print('PUBLISHED', dst, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path); p.add_argument('--worker', type=Path)
    p.add_argument('--views', type=Path); p.add_argument('--cache-config', type=Path)
    p.add_argument('--mask-qc', type=Path, help='Render only C1/mask representative QC from a completed mask stage')
    p.add_argument('--publish', type=Path); p.add_argument('--reference', type=Path)
    a = p.parse_args()
    if a.mask_qc:
        assert json.loads((a.mask_qc/'COMPLETE.json').read_text())['all_passed']
        mask_previews(a.mask_qc)
    elif a.worker:
        classification_worker(a.worker)
    elif a.views:
        views(a.views, a.cache_config)
    elif a.publish:
        publish(a.publish, a.reference)
    else:
        mask_build(a.config.resolve())
