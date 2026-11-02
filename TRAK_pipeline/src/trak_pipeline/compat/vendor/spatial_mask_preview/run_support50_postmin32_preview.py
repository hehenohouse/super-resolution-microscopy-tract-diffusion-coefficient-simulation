#!/usr/bin/env python3
"""Isolated post-support-only min32 mask trial and raw-C1 Fiji comparison."""
import argparse
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
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/noton-mplcache')
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tifffile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'mito_tardis'))
from render_point_stack import dump, fingerprint, pixel_hash, roi_reader, mat_vertices, Tee
from point_stack_core import gray_crop, mask_contour_alpha, roi_alpha, composite, ROI_COLOR, MASK_COLOR
from support50_postmin32_core import process_counts

EXAMPLES = [200, 1200, 2000]
STACK_NAME = 'Fiji_C1_old_vs_postmin32_ROI_frames0000_2499.ome.tif'


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def resolve(config_path):
    c = json.loads(config_path.read_text())
    parent = json.loads((Path(c['source_view']) / 'resolved_config.json').read_text())
    old = Path(parent['mask_directory'])
    p = json.loads((old / 'parameters.json').read_text())
    c.update(config_path=str(config_path), old_mask_directory=str(old),
             source_broad_labels=p['source_broad_labels'], raw_count_cache=str(Path(c['verified_stage_audit']) / 'raw_broad_foreground_count.npy'))
    for key in ['C1_file', 'ROI_file', 'MAT_file', 'MAT_record', 'font_path']:
        c[key] = parent[key]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    c['timestamp_UTC'] = stamp
    c['external_destination'] = str(Path(c['C1_file']).parent / 'mitochondria_masks' /
        ('Nellie_broad_support50_ge10_postmin32_holes_lt64_' + stamp))
    return c


def source_paths(c):
    old = Path(c['old_mask_directory'])
    return [Path(c['config_path']), Path(c['C1_file']), Path(c['ROI_file']), Path(c['MAT_file']),
        Path(c['source_broad_labels']), Path(c['raw_count_cache']),
        *[old / name for name in ['foreground_count.npy', 'input_ge10_full_mask.npy', 'min32_full_mask.npy',
          'clean_full_mask.npy', 'analysis_ROI_mask.npy', 'roi_mask.npy', 'roi_vertices_xy.npy',
          'original_frame_to_mask_block.npy', 'parameters.json', 'COMPLETE.json']]]


def verify_sources(c, out):
    old = Path(c['old_mask_directory']); audit = Path(c['verified_stage_audit']); view = Path(c['source_view'])
    assert json.loads((audit / 'COMPLETED.json').read_text())['delivery_checks_passed']
    assert json.loads((old / 'COMPLETE.json').read_text())['all_passed']
    known = {}
    for folder in [audit, view]:
        for filename in ['output_fingerprints.json', 'input_fingerprints_before.json']:
            records = json.loads((folder / filename).read_text())
            assert isinstance(records, list)
            known.update({r['path']: r['sha256'] for r in records})
    known.update({str(old / name): digest for name, digest in json.loads((old / 'output_fingerprints.json').read_text()).items()})
    known.update(json.loads((old / 'input_fingerprints_before.json').read_text()))
    before = []
    for path in source_paths(c):
        record = fingerprint(path)
        if path.name not in [Path(c['config_path']).name, 'COMPLETE.json']:
            assert record['sha256'] == known[str(path)], str(path)
        before.append(record)
    dump(out / 'input_fingerprints_before.json', before)
    p = json.loads((old / 'parameters.json').read_text())
    assert p['foreground_connectivity'] == 8 and p['background_connectivity'] == 4
    assert p['support_minimum_count'] == 10 and p['minimum_foreground_area_pixel2'] == 32
    assert p['hole_area_exclusive_max_pixel2'] == 64 and p['pixel_size_um'] == .117
    assert c['support_total_frames'] == 50 and c['source_frame_count'] == 2500 and not c['per_frame_min32']
    assert c['support_minimum_count'] == 10 and c['minimum_foreground_area_pixel2'] == 32
    assert c['hole_area_exclusive_max_pixel2'] == 64 and c['source_frame_interval_s'] == .02
    assert c['display_C1_ADU'] == [407, 651] and c['display_scale'] == 2
    np.testing.assert_array_equal(np.load(old / 'original_frame_to_mask_block.npy'), np.arange(2500) // 50)
    broad = np.load(c['source_broad_labels'], mmap_mode='r')
    assert broad.shape == (2500, 684, 428) and broad.dtype == np.uint32
    counts = np.load(c['raw_count_cache'], mmap_mode='r')
    assert counts.shape == (50, 684, 428) and counts.dtype == np.uint8 and counts.max() <= 50
    roi = np.load(old / 'roi_mask.npy'); vertices = np.load(old / 'roi_vertices_xy.npy')
    assert roi.dtype == bool and roi.shape == (684, 428) and roi.sum() == 32179 and len(vertices) == 454
    library, roi_script = roi_reader()
    np.testing.assert_array_equal(vertices, library.load_imagej_roi(c['ROI_file']))
    np.testing.assert_array_equal(vertices, mat_vertices(c['MAT_file'], c['MAT_record']))
    old_full = np.load(old / 'clean_full_mask.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(old / 'analysis_ROI_mask.npy'), old_full & roi)
    with tifffile.TiffFile(c['C1_file']) as tf:
        assert len(tf.pages) == 2500 and tf.series[0].axes == 'TYX'
        assert tf.series[0].shape == (2500, 684, 428) and tf.pages[0].dtype == np.uint16
    x0, y0, x1, y1 = c['crop_xyxy_px']; yy, xx = np.where(roi)
    assert 0 <= x0 <= xx.min() and xx.max() < x1 <= 428 and 0 <= y0 <= yy.min() and yy.max() < y1 <= 684
    for name, values in [('roi_mask.npy', roi), ('roi_vertices_xy.npy', vertices)]:
        np.save(out / name, values)
    shutil.copyfile(roi_script, out / 'source_snapshot' / 'roi_reader.py')
    dump(out / 'source_validation.json', dict(all_checks_passed=True, ROI_MAT_mask_identical=True,
        source_raw_count_cache_SHA_verified=True, original_broad_source_SHA_verified=True,
        TYX=True, source_frames=2500, ROI_pixels=32179, ROI_vertices=454, pixel_size_um=.117))
    return counts, old_full, roi, vertices


def write_scientific_tiff(path, array, name):
    tifffile.imwrite(path, array.astype(np.uint8), bigtiff=True, ome=True,
        photometric='minisblack', compression='deflate', compressionargs={'level': 6},
        metadata=dict(axes='TYX', Name=name, PhysicalSizeX=.117, PhysicalSizeXUnit='µm',
            PhysicalSizeY=.117, PhysicalSizeYUnit='µm', TimeIncrement=1., TimeIncrementUnit='s',
            Description='50 nonoverlapping original-frame support blocks; block b uses frames50b..50b+49. Not instantaneous20ms masks. Binary values0/1.'))
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome and len(tf.pages) == 50 and tf.series[0].axes == 'TYX'
        p = ET.fromstring(tf.ome_metadata).find('.//{*}Pixels')
        assert p.get('SizeT') == '50' and p.get('SizeZ') == '1' and float(p.get('TimeIncrement')) == 1.
        for block, page in enumerate(tf.pages):
            np.testing.assert_array_equal(page.asarray(), array[block].astype(np.uint8))


def build_masks(c, out, counts, old_full, roi):
    arrays = {name: np.zeros((50, 684, 428), dtype=bool) for name in
        ['input_ge10_full_mask', 'min32_full_mask', 'clean_full_mask', 'filled_holes_full_mask', 'analysis_ROI_mask']}
    old_count = np.load(Path(c['old_mask_directory']) / 'foreground_count.npy', mmap_mode='r')
    old_support = np.load(Path(c['old_mask_directory']) / 'input_ge10_full_mask.npy', mmap_mode='r')
    old_post = np.load(Path(c['old_mask_directory']) / 'min32_full_mask.npy', mmap_mode='r')
    statistics = []; components = []; holes = []; mapping = []
    for b in range(50):
        result = process_counts(counts[b], 10, 32, 64, 50)
        for name, key in [('input_ge10_full_mask', 'support_mask'), ('min32_full_mask', 'post_min32_mask'),
                          ('clean_full_mask', 'final_mask'), ('filled_holes_full_mask', 'filled_holes')]:
            arrays[name][b] = result[key]
        final = arrays['clean_full_mask'][b]; arrays['analysis_ROI_mask'][b] = final & roi
        assert (counts[b] >= old_count[b]).all()
        assert not (old_support[b] & ~result['support_mask']).any()
        assert not (old_post[b] & ~result['post_min32_mask']).any()
        assert not (old_full[b] & ~final).any()
        for label in range(1, len(result['support_component_areas'])):
            components.append(dict(block_index=b, component_label=label,
                area_pixel2=int(result['support_component_areas'][label]), kept=bool(result['support_component_keep'][label])))
        for label in range(1, len(result['background_component_areas'])):
            holes.append(dict(block_index=b, background_label=label,
                area_pixel2=int(result['background_component_areas'][label]),
                touches_image_edge=bool(result['background_component_edge'][label]),
                enclosed=bool(result['background_component_enclosed'][label]), filled=bool(result['background_component_fill'][label])))
        old_roi = old_full[b] & roi; new_roi = final & roi
        statistics.append(dict(block_index=b, original_frame_start=50*b, original_frame_end_inclusive=50*b+49,
            old_full_pixel2=int(old_full[b].sum()), new_full_pixel2=int(final.sum()),
            old_ROI_pixel2=int(old_roi.sum()), new_ROI_pixel2=int(new_roi.sum()),
            added_ROI_pixel2=int((new_roi & ~old_roi).sum()), removed_ROI_pixel2=int((old_roi & ~new_roi).sum()),
            new_ROI_um2=float(new_roi.sum() * .117**2), old_ROI_um2=float(old_roi.sum() * .117**2),
            filled_holes_count=int(result['background_component_fill'].sum()), filled_holes_pixel2=int(result['filled_holes'].sum()),
            empty_ROI=not new_roi.any()))
        mapping.append(dict(block_index=b, original_frame_start=50*b, original_frame_end_inclusive=50*b+49,
            source_frame_count=50, time_start_s=float(b), time_end_exclusive_s=float(b+1)))
        if (b+1) % 5 == 0:
            dump(out / 'checkpoints' / f'masks_{(b+1)*50:04d}.json', dict(completed_blocks=b+1, completed_source_frames=(b+1)*50))
            print('MASK_CHECKPOINT', (b+1)*50, 'original frames;', b+1, 'blocks', flush=True)
    np.save(out / 'foreground_count.npy', np.asarray(counts))
    for name, values in arrays.items(): np.save(out / (name+'.npy'), values)
    np.save(out / 'original_frame_to_mask_block.npy', np.arange(2500)//50)
    np.save(out / 'original_frame_numbers_by_block.npy', np.arange(2500).reshape(50,50))
    np.save(out / 'block_completed.npy', np.ones(50, bool))
    pd.DataFrame(mapping).to_csv(out / 'block_frame_mapping.csv', index=False)
    pd.DataFrame(statistics).to_csv(out / 'per_block_statistics.csv', index=False)
    pd.DataFrame(components).to_csv(out / 'component_decisions.csv', index=False)
    pd.DataFrame(holes).to_csv(out / 'hole_decisions.csv', index=False)
    reference = json.loads((PROJECT / c['independent_reference']).read_text())
    # These reference hashes were independently recomputed from all2500 raw broad labels.
    actual = {'foreground_count': array_hash(counts), 'input_ge10': array_hash(arrays['input_ge10_full_mask']),
        'post_min32': array_hash(arrays['min32_full_mask']), 'final_full': array_hash(arrays['clean_full_mask']),
        'final_ROI': array_hash(arrays['analysis_ROI_mask'])}
    reference_names={'foreground_count':'foreground_count','input_ge10':'input_ge10_full_mask',
        'post_min32':'min32_full_mask','final_full':'clean_full_mask','final_ROI':'analysis_ROI_mask'}
    expected={name:reference['expected_arrays'][other]['C_order_array_bytes_sha256'] for name,other in reference_names.items()}
    assert actual == expected, (actual, expected)
    assert sum(r['old_ROI_pixel2'] for r in statistics) == 358406
    assert sum(r['new_ROI_pixel2'] for r in statistics) == 380394
    shutil.copyfile(PROJECT / c['independent_reference'], out / 'independent_reconstruction_reference.json')
    for name in ['clean_full_mask', 'analysis_ROI_mask']:
        write_scientific_tiff(out / (name+'.ome.tif'), arrays[name], c['sample']+' post-support-only min32 '+name)
    params = dict(c, shape=[50,684,428], axes='TYX', block_count=50, mask_block_interval_s=1.,
        processing_order=['original individual-frame Nellie broad >0', 'nonoverlapping50-frame per-pixel support>=10',
                          'full-image8conn foreground area>=32', 'full-image4conn enclosed holes<64', 'original ROI intersection'],
        foreground_connectivity=8, background_connectivity=4, foreground_boundary_exemptions=False,
        no_Nellie_rerun=True, no_image_averaging=True, no_alignment=True, no_W5_voting=True,
        no_post_holefill_size_filter=True, selected_as_final=False, original_frame_to_block='floor(frame/50)',
        viewing_stack_frames=2500, viewing_stack_interval_s=.02, source_scale_nm_per_pixel=117,
        no_localization_reclassification=True, no_TARDIS=True)
    dump(out / 'parameters.json', params)
    dump(out / 'mask_validation.json', dict(all50_blocks_complete=True,
        independent_all2500_raw_labels_reconstruction_hashes_match=True, array_sha256_C_order_bytes=actual,
        new_final_contains_old_final=True, removed_old_full_pixels=0, ROI_only_after_full_operations=True,
        original_ROI_unchanged=True, new_ROI_pixel2_sum=380394, old_ROI_pixel2_sum=358406,
        added_ROI_pixel2_sum=21988, empty_ROI_blocks=0, scientific_TIFF_all50_pages_equal=True))
    dump(out / 'MASKS_READY.json', dict(mask_blocks=50, source_frames=2500, old_final_preserved=True))
    return arrays['clean_full_mask']


class ComparisonView:
    def __init__(self, c, old_full, new_full, roi, vertices):
        self.c = c; self.crop = c['crop_xyxy_px']; self.scale = 2
        self.w = (self.crop[2]-self.crop[0])*2; self.h = (self.crop[3]-self.crop[1])*2
        self.margin = 12; self.gap = 16; self.header = 104; self.footer = 32
        self.width = 2*self.margin+3*self.w+2*self.gap; self.height = self.header+self.h+self.footer
        self.origins = [self.margin+i*(self.w+self.gap) for i in range(3)]
        x0,y0,x1,y1 = self.crop
        self.roi = np.repeat(np.repeat(roi[y0:y1,x0:x1],2,0),2,1)
        self.outline = roi_alpha(vertices,self.crop,2)
        self.contours = [[mask_contour_alpha(m[b],self.crop,2) for b in range(50)] for m in [old_full,new_full]]
        self.masks = [old_full, new_full]
        self.areas = [[int((m[b]&roi).sum()) for b in range(50)] for m in self.masks]
        self.font = {n: ImageFont.truetype(c['font_path'],n) for n in [13,14,16,18]}

    def render(self, frame, raw):
        assert raw.shape == (684,428) and raw.dtype == np.uint16
        b = frame//50; base = np.repeat(gray_crop(raw,self.crop,2,[407,651])[...,None],3,axis=2)
        layers = [composite(base,self.outline,ROI_COLOR)]
        for i in range(2):
            layer = composite(composite(base,self.contours[i][b],MASK_COLOR),self.outline,ROI_COLOR)
            untouched = self.roi & (self.contours[i][b] == 0) & (self.outline == 0)
            np.testing.assert_array_equal(layer[untouched],base[untouched])
            layers.append(layer)
        canvas = Image.new('RGB',(self.width,self.height),'black'); draw = ImageDraw.Draw(canvas)
        draw.text((12,5),'TDKd3 T1 | C1 and support50 masks | min32 moved after temporal support',font=self.font[18],fill='white')
        draw.text((12,30),f'Original frame {frame:04d} | {frame*.02:.2f} s | mask block {b:02d}: frames {b*50}-{b*50+49}, [{b},{b+1}) s',font=self.font[16],fill='#DDDDDD')
        titles = ['C1 original single frame | cell ROI', 'OLD: per-frame min32 + post-support min32', 'NEW: support >=10, then min32 only']
        subtitles = ['407-651 ADU | display 2x | source117 nm/pixel',
            f'ROI foreground: {self.areas[0][b]:,} pixel2',
            f'ROI foreground: {self.areas[1][b]:,} pixel2 | +{self.areas[1][b]-self.areas[0][b]:,}']
        for i,layer in enumerate(layers):
            layer[~self.roi] = 0
            assert not layer[~self.roi].any()
            canvas.paste(Image.fromarray(layer),(self.origins[i],self.header))
            draw.text((self.origins[i],60),titles[i],font=self.font[14],fill='white')
            draw.text((self.origins[i],81),subtitles[i],font=self.font[13],fill='#DDDDDD')
        draw.text((12,self.header+self.h+8),'Red: full-mask boundary clipped for viewing. Gray: cell ROI. Both: holes<64. Masks update every50frames; raw C1 is never averaged.',font=self.font[14],fill='#CCCCCC')
        image = np.asarray(canvas)
        info = dict(page_index=frame,frame=frame,time_s=frame*.02,block_index=b,
            block_original_frame_start=b*50,block_original_frame_end_inclusive=b*50+49,
            old_ROI_pixel2=self.areas[0][b],new_ROI_pixel2=self.areas[1][b],
            raw_C1_page_sha256=pixel_hash(raw),old_mask_page_sha256=pixel_hash(self.masks[0][b]),
            new_mask_page_sha256=pixel_hash(self.masks[1][b]),rendered_sha256=pixel_hash(image))
        return image, info


def previews(c,out,view):
    records=[]
    with tifffile.TiffFile(c['C1_file']) as source:
        for f in EXAMPLES:
            image, info = view.render(f,source.pages[f].asarray())
            Image.fromarray(image).save(out/'previews'/f'frame_{f:04d}_C1_old_new.png')
            records.append(info)
    dump(out/'previews/image_manifest.json',records)


def export_view(c,out,view):
    target=out/STACK_NAME; assert not target.exists()
    rows=[]; started=time.perf_counter()
    with tifffile.TiffFile(c['C1_file']) as source, tifffile.TiffWriter(target,bigtiff=True,ome=True) as writer, (out/'frame_manifest.csv').open('w',newline='') as stream:
        csvout=None
        def pages():
            nonlocal csvout
            for f in range(2500):
                image, info=view.render(f,source.pages[f].asarray())
                if csvout is None:
                    csvout=csv.DictWriter(stream,fieldnames=list(info));csvout.writeheader()
                csvout.writerow(info);rows.append(info);yield image
                if (f+1)%250==0:
                    stream.flush();writer.filehandle.flush()
                    dump(out/'checkpoints'/f'rendered_{f+1:04d}.json',dict(rendered_frames=f+1,elapsed_s=time.perf_counter()-started))
                    print('RENDER_CHECKPOINT',f+1,'/2500',flush=True)
        writer.write(pages(),shape=(2500,view.height,view.width,3),dtype='uint8',photometric='rgb',
            compression='deflate',compressionargs={'level':6},rowsperstrip=128,maxworkers=2,
            metadata=dict(axes='TYXS',Name='TDKd3 T1 C1 old vs post-support-only min32',TimeIncrement=.02,TimeIncrementUnit='s',
                Description='Viewing composite; not quantitative fluorescence or binary mask. Original20ms C1 frames; masksf//50. '
                'Three panels: rawC1/oldmask/newmask; ROI exterior black; native117nm/px;2x display;crop[82,232,347,430].'))
    dump(out/'checkpoints/rendered_2500.json',dict(rendered_frames=2500,elapsed_s=time.perf_counter()-started,TIFF_finalized=True))
    print('RENDER_CHECKPOINT',2500,'/2500; TIFF finalized',flush=True)
    checked=[]
    with tifffile.TiffFile(target) as tf:
        assert tf.is_bigtiff and tf.is_ome and len(tf.pages)==2500 and tf.series[0].axes=='TYXS'
        assert tf.series[0].shape==(2500,view.height,view.width,3)
        p=ET.fromstring(tf.ome_metadata).find('.//{*}Pixels')
        assert p.get('SizeT')=='2500' and p.get('SizeZ')=='1' and float(p.get('TimeIncrement'))==.02
        for f,page in enumerate(tf.pages):
            image=page.asarray();assert pixel_hash(image)==rows[f]['rendered_sha256']
            assert page.keyframe.photometric.name=='RGB' and page.keyframe.compression.name in ['DEFLATE','ADOBE_DEFLATE']
            for origin in view.origins:
                body=image[view.header:view.header+view.h,origin:origin+view.w]
                assert not body[~view.roi].any()
            if f in EXAMPLES:
                np.testing.assert_array_equal(image,np.asarray(Image.open(out/'previews'/f'frame_{f:04d}_C1_old_new.png')))
            checked.append(dict(frame=f,block_index=f//50,decoded_pixels_equal=True,all3_ROI_exteriors_black=True))
            if (f+1)%250==0:print('READBACK_CHECKPOINT',f+1,'/2500',flush=True)
        (out/'OME_view_metadata.xml').write_text(tf.ome_metadata)
    manifest=pd.DataFrame(rows)
    np.testing.assert_array_equal(manifest.frame,np.arange(2500))
    np.testing.assert_array_equal(manifest.block_index,np.arange(2500)//50)
    for f in [0,49,50,1199,1200,1999,2000,2498,2499]:
        assert manifest.iloc[f].block_original_frame_start==50*(f//50)
    pd.DataFrame(checked).to_csv(out/'view_readback_validation.csv',index=False)
    dump(out/'view_validation.json',dict(all2500_pages_readback_equal=True,RGB8=True,lossless_Deflate=True,
        SizeT=2500,SizeZ=1,TimeIncrement_s=.02,shape=[2500,view.height,view.width,3],ROI_exterior_black=True,
        correct_source_frame_and_block_mapping=True,mask_boundaries_extracted_before_ROI_clipping=True,
        source_grayscale_pixels_untouched_outside_contours=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume-output',type=Path)
    args=parser.parse_args();args.config=args.config.resolve()
    if args.resume_output:
        out=args.resume_output.resolve();c=json.loads((out/'resolved_config.json').read_text())
        assert c['config_path']==str(args.config)
        assert json.loads((out/'run_status.json').read_text())['status']=='previews_ready'
    else:
        c=resolve(args.config);out=HERE/'results'/('support50_postmin32_'+c['timestamp_UTC'])
        out.mkdir(parents=True,exist_ok=False)
        for name in ['previews','source_snapshot','checkpoints']:(out/name).mkdir()
        dump(out/'resolved_config.json',c)
    (HERE/'latest_support50_postmin32_local.txt').write_text(str(out)+'\n')
    (out/('resume_command.txt' if args.resume_output else 'prepare_command.txt')).write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(args.config),
        *(['--prepare-only'] if args.prepare_only else []),*(['--resume-output',str(out)] if args.resume_output else [])])+'\n')
    with (out/'run.log').open('a') as stream:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,stream);sys.stderr=Tee(stderr,stream)
        try:
            print('OUTPUT',out,flush=True)
            if not args.resume_output:
                counts,old_full,roi,vertices=verify_sources(c,out)
                test=subprocess.run([sys.executable,'-B',str(HERE/'test_support50_postmin32_core.py')],capture_output=True,text=True)
                (out/'core_tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
                full=build_masks(c,out,counts,old_full,roi)
            else:
                full=np.load(out/'clean_full_mask.npy',mmap_mode='r');roi=np.load(out/'roi_mask.npy');vertices=np.load(out/'roi_vertices_xy.npy')
                old_full=np.load(Path(c['old_mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
            view=ComparisonView(c,old_full,full,roi,vertices)
            if not args.resume_output:previews(c,out,view)
            if args.prepare_only:
                dump(out/'run_status.json',dict(status='previews_ready'))
                print('PREVIEWS_READY',out,flush=True);return
            export_view(c,out,view)
            before=json.loads((out/'input_fingerprints_before.json').read_text());after=[fingerprint(path) for path in source_paths(c)]
            assert before==after;dump(out/'input_fingerprints_after.json',after)
            for name in ['run_support50_postmin32_preview.py','support50_postmin32_core.py','test_support50_postmin32_core.py']:
                shutil.copyfile(HERE/name,out/'source_snapshot'/name)
            for name in ['render_point_stack.py','point_stack_core.py']:
                shutil.copyfile(PROJECT/'mito_tardis'/name,out/'source_snapshot'/name)
            shutil.copyfile(args.config,out/'config.json')
            dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
                packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','Pillow','tifffile']}))
            freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
            (out/'requirements.lock.txt').write_text(freeze.stdout)
            (out/'README.txt').write_text('TDKd3 T1 exploratory post-support-only min32 trial.\n'
                'Original per-frame Nellie broad >0 ->50-frame per-pixel support>=10 ->full-image8conn min32 ->4conn enclosed holes<64 ->originalROI.\n'
                'Removed only per-frame min32. No new Nellie segmentation,image averaging,sliding voting,alignment or localization classification.\n'
                'Scientific masks50x684x428 TYX,binaryNPY/uint80/1OME-TIFF,117nm/px,1s per block,originalframef->f//50.\n'
                'clean_full_mask is beforeROI clipping;analysis_ROI_mask is finalfull&roi. Threshold32keepsforeground;hole64isexclusive.\n'
                'Fiji comparison:2500 original20ms C1 pages;3columns raw/oldmask/newmask;ROIoutsideblack,2x,crop[82,232,347,430],407–651ADU.\n'
                'Mask contours from fullmask beforeROI clipping;scientificdata unchangedbydisplay. Viewingcomposite is RGB8,not quantitative C1or binarymask.\n'
                'Oldfinal is containedinnewfinal,zero removedpixels. Summed50-blockROIarea358406->380394px2,+21988(+6.13494%).\n'
                'Larger coverage doesnot establish better segmentation or simultaneous structures. No C1/C2registration validation,noTARDIS/trajectories/biologicalinference.\n'
                'Rawcountcachehashverified;independentfull2500rawlabelreconstructionmatchesall50newarraycontenthashes. AllTIFFpagesreadbackverified.\n'
                'OriginalrawTIFF,broadlabels,oldmask,ROI,MAT,localizations,selecteddefaultflagsandpaused47samplebatchunchanged.\n')
            dump(out/'validation.json',dict(all_implementation_checks_passed=True,source_fingerprints_unchanged=True,
                independent_all2500_raw_labels_reconstruction_matches=True,all50_scientific_TIFF_pages_equal=True,
                all2500_view_pages_equal=True,old_final_preserved=True,ROI_MAT_verified=True,visual_review_pending=True))
            dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,output=str(out)))
            dump(out/'run_status.json',dict(status='complete_pending_visual_review'))
            print('IMPLEMENTATION_COMPLETE',out,flush=True)
        except BaseException:
            dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':main()
