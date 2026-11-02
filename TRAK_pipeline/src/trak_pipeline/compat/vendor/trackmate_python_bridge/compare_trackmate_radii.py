"""Exact-frame raw and TrackMate radius comparison; no additional filters."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import yaml

from run_preview import annotate_roi, draw_lib, dump, fingerprint, video_and_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--middle',type=Path,help='Optional 0.28 um run, between the 0.25 and 0.30 um columns')
    parser.add_argument('--reference-fingerprint-file', type=Path,
                        help='Prior recorded fingerprints; required for exact native-cache recovery if a reference CSV is unreadable')
    args = parser.parse_args()
    paths = [args.reference.resolve()]+([args.middle.resolve()] if args.middle else [])+[args.comparison.resolve()]
    assert all((p/'COMPLETED.json').exists() for p in paths)
    configs = [yaml.safe_load((p/'resolved_config.yaml').read_text()) for p in paths]
    radii=[c['radius_um'] for c in configs]
    assert radii == ([.25,.28,.30] if args.middle else [.25,.30])
    tags=['0p'+f'{int(round(r*100)):02d}' for r in radii]
    start=int(configs[0]['frame_start']);stop=int(configs[0]['frame_stop_exclusive']);count=stop-start
    common = ['input_tiff','roi_file','mat_file','mat_record','expected_tiff_sha256',
              'expected_roi_sha256','trackmate_version','frame_start','frame_stop_exclusive',
              'pixel_size_nm','frame_interval_s','threshold','median_filter',
              'subpixel_localization','tracking','picasso_fitting','uncertainty_estimation',
              'tardis','mitochondria_analysis','resolved_display_range_adu',
              'circle_radius_source_px','circle_width_source_px','circle_color','roi_color']
    assert all(configs[0][key] == candidate[key] for candidate in configs[1:] for key in common)
    assert configs[0]['trackmate_version']=='7.14.0' and configs[0]['threshold']==6.5
    assert not any(configs[0][k] for k in ['tracking','picasso_fitting','uncertainty_estimation','tardis','mitochondria_analysis'])
    movies = [next((p/'native_detection').glob('*/input_frames.npy')) for p in paths]
    native_reports = [next((p/'native_detection').glob('*/report.json')) for p in paths]
    native_tables = [next((p/'native_detection').glob('*/spots.csv')) for p in paths]
    protected = movies+native_reports+native_tables
    if args.reference_fingerprint_file:
        protected.append(args.reference_fingerprint_file.resolve())
    for p in paths:
        protected.extend([p/'all_detections.csv',p/'roi_detections.csv',p/'frame_counts.csv',
                          p/'frame_manifest.csv',p/'roi_vertices.csv',p/'resolved_config.yaml',
                          p/f'frames_{start:04d}_{stop-1:04d}_10fps.mp4'])
    before = [fingerprint(p) for p in protected]
    out = (args.middle.resolve() if args.middle else paths[-1])/('comparison_radius_0p25_0p28_0p30_four_panels' if args.middle else 'comparison_radius_0p25_vs_0p30')
    out.mkdir(exist_ok=False)
    dump(out/'source_fingerprints_before.json',before)
    dump(out/'config.json',{'source_runs':[str(p) for p in paths],'radii_um':radii,
          'common_parameters':{k:configs[0][k] for k in common},'frames':[start,stop-1],
          'crop_native_xyxy':[82,232,347,430],'display_scale':3,
          'crop_is_display_only':True,'selection':'Original floating center inside or on identical cell ROI; tolerance 1e-9',
          'additional_point_filters':False,'video_fps':10,
          'spot_ids_are_run_local':'No matching by spot ID across the two detector runs'})
    for file in [Path(__file__),Path(__file__).parent/'run_preview.py',Path(__file__).parent.parent/'localization_overlays/render_frames.py']:
        shutil.copyfile(file,out/file.name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    versions = {key:importlib.metadata.version(key) for key in ['numpy','pandas','Pillow','tifffile','h5py','PyYAML']}
    versions['native_detector'] = [json.loads(p.read_text())['versions'] for p in native_reports]
    dump(out/'software_versions.json',versions)
    vertices = [pd.read_csv(p/'roi_vertices.csv')[['x_px','y_px']].to_numpy() for p in paths]
    for other in vertices[1:]:np.testing.assert_array_equal(vertices[0],other)
    tables = []
    recovery = []
    for index,p in enumerate(paths):
        try:
            table = pd.read_csv(p/'all_detections.csv',float_precision='round_trip')
        except UnicodeDecodeError as exc:
            # Never repair or overwrite the old result. A recovery is permitted
            # only when the reconstructed bytes match its independently saved,
            # pre-existing checksum exactly.
            assert index == 0 and args.reference_fingerprint_file, 'Unreadable CSV needs prior independent checksum evidence'
            old_record = json.loads(args.reference_fingerprint_file.read_text())
            matching = [r for r in old_record if r['path'] == str(p/'all_detections.csv')]
            assert len(matching) == 1
            recovered = pd.read_csv(native_tables[index],float_precision='round_trip')
            recovered = annotate_roi(recovered,vertices[0])
            recovered['radius_setting_px'] = configs[index]['radius_px']
            content = recovered.to_csv(index=False,float_format='%.17g').encode('utf-8')
            recovered_hash = hashlib.sha256(content).hexdigest()
            assert recovered_hash == matching[0]['sha256'], 'Native-cache recovery differs from original recorded checksum'
            assert len(content) == matching[0]['bytes']
            recovered_path = out/'reference_0p25_all_detections_recovered.csv'
            recovered_path.write_bytes(content)
            table = pd.read_csv(io.BytesIO(content),float_precision='round_trip')
            intact_roi = pd.read_csv(p/'roi_detections.csv',float_precision='round_trip')
            pd.testing.assert_frame_equal(table[table.roi_inside==1].reset_index(drop=True),intact_roi,check_exact=True)
            recovery.append({'unreadable_reference':str(p/'all_detections.csv'),
                             'read_error':str(exc),'native_source':str(native_tables[index]),
                             'prior_checksum_evidence':str(args.reference_fingerprint_file.resolve()),
                             'reconstructed_sha256':recovered_hash,'original_sha256':matching[0]['sha256'],
                             'byte_exact_recovery_verified':True,'intact_ROI_table_exactly_verified':True,
                             'new_copy':str(recovered_path),'old_file_overwritten':False})
        tables.append(table)
    dump(out/'reference_cache_integrity.json',{'recovery':recovery})
    for table,c in zip(tables,configs):
        assert table.spot_id.is_unique and table.frame.between(start,stop-1).all()
        assert not any('picasso' in name or 'net_gradient' in name for name in table.columns)
        np.testing.assert_array_equal(table.roi_inside,annotate_roi(table,vertices[0]).roi_inside)
        np.testing.assert_allclose(table.time_s,table.frame*.02,atol=1e-12,rtol=0)
        np.testing.assert_allclose(table.radius_setting_um,c['radius_um'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(table.x_um,table.x_px*.117,atol=1e-12,rtol=0)
        np.testing.assert_allclose(table.y_um,table.y_px*.117,atol=1e-12,rtol=0)
    selected = [table[table.roi_inside==1] for table in tables]
    manifests = [pd.read_csv(p/'frame_manifest.csv') for p in paths]
    source_counts = [pd.read_csv(p/'frame_counts.csv').set_index('frame') for p in paths]
    arrays = [np.load(p,mmap_mode='r') for p in movies]
    assert all(a.shape==(count,684,428) and a.dtype==np.uint16 for a in arrays)
    c = dict(configs[0],display_scale=3)
    lo,hi = c['resolved_display_range_adu']
    crop = tuple(3*v for v in [82,232,347,430])
    w,h = crop[2]-crop[0],crop[3]-crop[1]
    panels=1+len(paths)
    width = panels*w+(panels+1)*14; width += width%2
    height = h+186
    big = ImageFont.truetype(c['font_path'],28)
    small = ImageFont.truetype(c['font_path'],23)
    detail = ImageFont.truetype(c['font_path'],20)
    frames = out/'frames';frames.mkdir()
    rows = []
    for frame in range(start,stop):
        raw = arrays[0][frame-start]
        for other in arrays[1:]:np.testing.assert_array_equal(raw,other[frame-start])
        raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
        for manifest in manifests:assert manifest.loc[manifest.frame==frame,'source_pixel_sha256'].iloc[0]==raw_hash
        gray = Image.fromarray(draw_lib.display_gray(raw,lo,hi,3)).convert('RGBA')
        groups = [table[table.frame==frame] for table in selected]
        for group,source_count in zip(groups,source_counts):assert len(group)==int(source_count.loc[frame,'roi_detections'])
        image = Image.new('RGB',(width,height),'#101317');draw=ImageDraw.Draw(image)
        draw.text((14,12),f'T1 C2 | Original frame {frame:04d} | Acquisition {frame*.02:.2f} s | TrackMate radius comparison | Same cell crop',font=big,fill='white')
        labels = ['Raw C2 single frame']+[f'Radius {r:.2f} um / {r/.117:.3f} px | {len(g)} ROI points' for r,g in zip(radii,groups)]
        for index,label in enumerate(labels):
            x = 14+index*(w+14)
            draw.text((x,60),label,font=small,fill='white')
            if index==0:
                panel=gray.convert('RGB')
            else:
                xy=groups[index-1][['x_px','y_px']].to_numpy()
                panel=Image.alpha_composite(gray,draw_lib.overlay_layer(raw.shape,xy,vertices[0],c)).convert('RGB')
            image.paste(panel.crop(crop),(x,104))
        draw.text((14,h+116),f'TrackMate 7.14.0 LoG | Threshold 6.5 | Median + subpixel ON | Fixed {lo:g}-{hi:g} ADU | 117 nm/pixel | No additional point filters',font=detail,fill='#DCE3EB')
        draw.text((14,h+144),'Display-only crop. Cyan circles have fixed radius 3 native pixels and mark position, not detector radius or uncertainty. Video: 10 fps.',font=detail,fill='#DCE3EB')
        meta=PngImagePlugin.PngInfo()
        for key,val in {'original_frame':frame,**{f'roi_count_radius_{tag}':len(g) for tag,g in zip(tags,groups)}}.items():meta.add_text(key,str(val))
        path=frames/f'frame_{frame:04d}.png';image.save(path,pnginfo=meta,compress_level=5)
        with Image.open(path) as readback:
            assert int(readback.info['original_frame'])==frame
            for tag,g in zip(tags,groups):assert int(readback.info[f'roi_count_radius_{tag}'])==len(g)
            np.testing.assert_array_equal(np.asarray(readback.crop((14,104,14+w,104+h))),np.asarray(gray.convert('RGB').crop(crop)))
        rows.append({'frame':frame,'time_s':frame*.02,
                     **{f'roi_radius_{tag}_count':len(g) for tag,g in zip(tags,groups)},
                     **{f'FOV_radius_{tag}_count':int(source_count.loc[frame,'all_detections']) for tag,source_count in zip(tags,source_counts)},
                     'source_pixel_sha256':raw_hash,'raw_display_verified':True})
    counts=pd.DataFrame(rows);counts.to_csv(out/'frame_counts_comparison.csv',index=False)
    for tag,table in zip(tags,selected):assert counts[f'roi_radius_{tag}_count'].sum()==len(table)
    video,checks=video_and_checks(out,c)
    after=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_after.json',after)
    assert before==after
    summary={'output':str(out),'radii_um':radii,'panel_order':['raw C2']+[f'radius {r:.2f} um' for r in radii],
             'roi_counts_by_radius':{str(r):len(t) for r,t in zip(radii,selected)},
             'FOV_counts_by_radius':{str(r):len(t) for r,t in zip(radii,tables)},
             'roi_count_change_percent_vs_0p25':{str(r):100*(len(t)/len(selected[0])-1) for r,t in zip(radii,selected)},
             'frames':count,'original_frames':[start,stop-1],'video_fps':10,'duration_s':count/10,'video':str(video),'video_checks':checks,
             'only_detector_radius_changed':True,'same_selected_source_pages_verified':True,
             'all_raw_display_panels_verified':True,'same_ROI_and_all_circle_counts_verified':True,
             'all_source_fingerprints_unchanged':True,'additional_point_filters':False,
             'reference_cache_recovery':recovery,
             'tracking_fitting_uncertainty_TARDIS_mito_analysis':False}
    dump(out/'COMPLETED.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
