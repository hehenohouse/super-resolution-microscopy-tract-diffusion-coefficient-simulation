"""Read-only TrackMate detection of T1 pages 0-49, with exact-frame video QC.

The rendering/ROI process uses existing scientific Python dependencies. A child
process in the independent PyImageJ environment executes native TrackMate only.
No Picasso algorithm, trajectory linking, or TARDIS calculation is invoked.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import tifffile
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Read-only reuse of validated polygon and annotation primitives; no analysis run.
ROI_SOURCE = REPO / 'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py'
DRAW_SOURCE = REPO / 'localization_overlays/render_frames.py'
roi_lib = load_module('preview_roi_primitives', ROI_SOURCE)
draw_lib = load_module('preview_drawing_primitives', DRAW_SOURCE)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'sha256': draw_lib.sha256(path)}


def read_mat_vertices(path, name):
    with h5py.File(path, 'r') as h:
        matches = []
        for ref in h['data'][()].ravel(order='F'):
            if not ref:
                continue
            record = h[ref]
            if not isinstance(record, h5py.Group) or 'name' not in record:
                continue
            text = ''.join(chr(int(x)) for x in record['name'][()].ravel(order='F') if int(x))
            if text == name:
                matches.append(record)
        assert len(matches) == 1, 'MAT ROI record must be unique'
        return matches[0]['roiData/roiCoordinates'][()].T.astype(float), matches[0].name


def annotate_roi(table, vertices):
    table = table.copy()
    xy = table[['x_px', 'y_px']].to_numpy(float)
    finite = np.isfinite(xy).all(axis=1)
    flag = np.full(len(table), -1, dtype=np.int8)
    flag[finite] = roi_lib.points_in_polygon(xy[finite], vertices).astype(np.int8)
    table['roi_inside'] = flag
    table['roi_status'] = np.where(~finite, 'position_undetermined',
                                    np.where(flag == 1, 'inside_or_boundary', 'outside'))
    return table


def render(raw, points, vertices, frame, total, low, high, c):
    plain = Image.fromarray(draw_lib.display_gray(raw, low, high, c['display_scale'])).convert('RGB')
    marked = Image.alpha_composite(plain.convert('RGBA'),
                                  draw_lib.overlay_layer(raw.shape, points, vertices, c)).convert('RGB')
    margin, gap, header, footer = 24, 28, 104, 82
    width, height = plain.size
    canvas = Image.new('RGB', (2 * margin + 2 * width + gap, header + height + footer), '#101317')
    canvas.paste(plain, (margin, header))
    canvas.paste(marked, (margin + width + gap, header))
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype(c['font_path'], 28)
    small = ImageFont.truetype(c['font_path'], 20)
    detail = ImageFont.truetype(c['font_path'], 17)
    draw.text((margin, 12), f'T1 C2 | Frame {frame:04d} | Acquisition {frame*c["frame_interval_s"]:.2f} s | ROI {len(points)} / FOV {total} detections', font=large, fill='white')
    draw.text((margin, 62), 'Original single frame (ADU display)', font=small, fill='#DCE3EB')
    draw.text((margin + width + gap, 62), 'Same frame + TrackMate detections inside cell ROI', font=small, fill=c['circle_color'])
    y = header + height + 8
    draw.text((margin, y), f'TrackMate {c["trackmate_version"]} LoG | Radius {c["radius_px"]:.5g} px / {c["radius_um"]:g} um | Threshold {c["threshold"]:g} | Median + subpixel ON', font=detail, fill='#DCE3EB')
    draw.text((margin, y+24), f'Fixed display {low:g}-{high:g} ADU | 117 nm/pixel | Cyan radius-3-px circles: position markers, NOT uncertainty | Yellow: cell ROI', font=detail, fill='#DCE3EB')
    draw.text((margin, y+48), 'Native C2 coordinates; no averaging or fitting. Video: 10 fps, 5x slower than acquisition; labels use original acquisition time.', font=detail, fill='#DCE3EB')
    return canvas, (margin, header, margin + width, header + height)


def video_and_checks(out, c):
    start=int(c.get('frame_start',0));stop=int(c.get('frame_stop_exclusive',50))
    count=stop-start
    video = out / f'frames_{start:04d}_{stop-1:04d}_10fps.mp4'
    command = [shutil.which('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'warning',
               '-framerate', '10', '-start_number', str(start), '-i', str(out/'frames/frame_%04d.png'),
               '-frames:v', str(count), '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-threads', str(c['video_threads']), '-crf', str(c['video_crf']),
               '-pix_fmt', 'yuv420p', '-r', '10', '-fps_mode', 'cfr', '-movflags', '+faststart', str(video)]
    (out/'video_command.txt').write_text(shlex.join(command)+'\n')
    with (out/'video_encode.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    probe_cmd = [shutil.which('ffprobe'), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                 '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:format=duration',
                 '-of', 'json', str(video)]
    (out/'video_probe_command.txt').write_text(shlex.join(probe_cmd)+'\n')
    probe = json.loads(subprocess.check_output(probe_cmd, text=True))
    dump(out/'video_probe.json', probe)
    stream = probe['streams'][0]
    assert stream['codec_name']=='h264'
    assert int(stream['nb_frames']) == int(stream['nb_read_frames']) == count
    assert stream['r_frame_rate'] == stream['avg_frame_rate'] == '10/1'
    assert abs(float(probe['format']['duration']) - count/10) < 1e-6
    width, height = stream['width'], stream['height']
    qa = out/'validation_video_frames'
    qa.mkdir()
    checks=[]
    for frame in sorted({start,start+(count-1)//2,stop-1}):
        cmd=[shutil.which('ffmpeg'),'-nostdin','-v','error','-i',str(video),
             '-vf',f'select=eq(n\\,{frame-start})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-']
        decoded=subprocess.check_output(cmd)
        assert len(decoded)==width*height*3
        actual=np.frombuffer(decoded,np.uint8).reshape(height,width,3)
        expected=np.asarray(Image.open(out/'frames'/f'frame_{frame:04d}.png').convert('RGB'))
        error=float(np.abs(actual.astype(float)-expected.astype(float)).mean())
        assert error<8, (frame,error)
        Image.fromarray(actual).save(qa/f'decoded_{frame:04d}.png')
        checks.append({'original_frame':frame,'mean_absolute_rgb_error':error,'passed':True})
    return video,checks


def run(c, cfg, out):
    started=time.perf_counter()
    start=int(c['frame_start']);stop=int(c['frame_stop_exclusive']);count=stop-start
    tiff=Path(c['input_tiff']);roi=Path(c['roi_file']);mat=Path(c['mat_file'])
    cache=(cfg.parent/c['verified_roi_run']).resolve()
    verified=json.loads((cache/'roi_validation.json').read_text())
    protected=[tiff,roi,mat,cache/'roi_vertices.csv',cache/'roi_validation.json',cache/'COMPLETED.json',
               ROOT/'run_detection.py',DRAW_SOURCE,ROI_SOURCE]
    print('Hashing original input and reference files (read-only)...',flush=True)
    before=[fingerprint(p) for p in protected]
    dump(out/'source_fingerprints_before.json',before)
    assert before[0]['sha256']==c['expected_tiff_sha256']==verified['source_tiff_metadata']['source_sha256']
    assert before[1]['sha256']==c['expected_roi_sha256']==verified['roi_sha256']
    vertices=roi_lib.load_imagej_roi(roi)
    mat_vertices,mat_record=read_mat_vertices(mat,c['mat_record'])
    cached=pd.read_csv(cache/'roi_vertices.csv')[['x_px','y_px']].to_numpy(float)
    np.testing.assert_array_equal(vertices,mat_vertices)
    np.testing.assert_array_equal(vertices,cached)
    assert len(vertices)==454
    # Vertex/boundary and independent simple-square cases, including nonfinite centers.
    assert roi_lib.points_in_polygon(vertices,vertices).all()
    square=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    test=pd.DataFrame({'x_px':[5.,0.,-1.,np.nan],'y_px':[5.,5.,5.,5.]})
    assert annotate_roi(test,square).roi_inside.tolist()==[1,1,0,-1]
    synthetic_checks=draw_lib.coordinate_checks(c)
    dump(out/'roi_validation.json',{'vertices':len(vertices),'roi_sha256':before[1]['sha256'],
         'MAT_vertices_exactly_equal':True,'cached_vertices_exactly_equal':True,'mat_record':mat_record,
         'boundary_included':True,'boundary_tolerance_px':1e-9,'coordinate_transform':'identity/native C2; no inferred transform'})
    pd.DataFrame(vertices,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    snapshots=out/'source_snapshot';snapshots.mkdir()
    for p in [Path(__file__),ROOT/'run_detection.py',DRAW_SOURCE,ROI_SOURCE,cfg,ROOT/'requirements.lock.txt']:
        shutil.copyfile(p,snapshots/p.name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(cfg)])+'\n')
    (out/'render_environment.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    # Preserve the virtualenv executable path: resolving its symlink would run
    # the base interpreter without this environment's PyImageJ dependencies.
    detector_python=(cfg.parent/c['detector_python']).absolute()
    command=[str(detector_python),'-B','-u',str(ROOT/'run_detection.py'),'--input',str(tiff),'--start',str(start),'--count',str(count),
             '--pixel-size-um',str(c['pixel_size_nm']/1000),'--frame-interval-s',str(c['frame_interval_s']),
             '--radius-um',str(c['radius_um']),'--threshold',str(c['threshold']),
             '--fiji',c['fiji'],'--java-home',c['java_home'],'--output-root',str(out/'native_detection')]
    (out/'detection_command.txt').write_text(shlex.join(command)+'\n')
    print(f'Running native TrackMate: pages{start}-{stop-1}, radius{c["radius_px"]:.6g}px/{c["radius_um"]:g}um, threshold{c["threshold"]:g}, median/subpixel ON...',flush=True)
    with (out/'detection_console.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=log,check=True)
    reports=list((out/'native_detection').glob('*/report.json'))
    assert len(reports)==1
    detected_dir=reports[0].parent
    detector_report=json.loads(reports[0].read_text())
    assert detector_report['versions']['trackmate']==c['trackmate_version']
    assert detector_report['tracking_executed'] is False
    table=pd.read_csv(detected_dir/'spots.csv',float_precision='round_trip')
    assert len(table)==detector_report['total_spots'] and table.spot_id.is_unique
    assert table.frame.between(start,stop-1).all()
    np.testing.assert_allclose(table.time_s,table.frame*.02,rtol=0,atol=1e-12)
    np.testing.assert_allclose(table.x_px*.117,table.x_um,rtol=0,atol=1e-12)
    np.testing.assert_allclose(table.y_px*.117,table.y_um,rtol=0,atol=1e-12)
    table=annotate_roi(table,vertices)
    table['radius_setting_px']=table.radius_setting_um/(c['pixel_size_nm']/1000)
    np.testing.assert_allclose(table.radius_setting_um,c['radius_um'],rtol=0,atol=1e-12)
    table.to_csv(out/'all_detections.csv',index=False,float_format='%.17g')
    selected=table.loc[table.roi_inside==1].copy()
    selected.to_csv(out/'roi_detections.csv',index=False,float_format='%.17g')
    index=pd.Index(range(start,stop),name='frame')
    counts=table.groupby('frame').size().reindex(index,fill_value=0).to_frame('all_detections')
    for name,flag in [('roi_detections',1),('outside_roi',0),('position_undetermined',-1)]:
        counts[name]=table.loc[table.roi_inside==flag].groupby('frame').size().reindex(index,fill_value=0)
    counts['time_s']=counts.index*.02
    np.testing.assert_array_equal(counts.all_detections,counts[['roi_detections','outside_roi','position_undetermined']].sum(axis=1))
    np.testing.assert_array_equal(counts.all_detections,[x['count'] for x in detector_report['per_frame_counts']])
    counts.to_csv(out/'frame_counts.csv',float_format='%.17g')
    print(f'Detected {len(table):,} full-FOV points; {len(selected):,} inside/boundary of cell ROI.',flush=True)
    movie=np.load(detected_dir/'input_frames.npy',mmap_mode='r')
    assert movie.shape==(count,684,428) and movie.dtype==np.uint16
    if 'fixed_display_range_adu' in c:
        low,high=c['fixed_display_range_adu']
    else:
        pooled=np.asarray(movie).ravel();nonzero=pooled[pooled!=0]
        low,high=np.percentile(nonzero,c['contrast_percentiles'])
        del nonzero
    assert high>low
    c['resolved_display_range_adu']=[float(low),float(high)]
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    print(f'Fixed display range: {low:g}-{high:g} ADU. Rendering{count} paired PNGs...',flush=True)
    groups={int(f):g for f,g in selected.groupby('frame')}
    frames=out/'frames';frames.mkdir()
    rows=[];annotations=[]
    with tifffile.TiffFile(tiff) as tf:
        assert len(tf.pages)==2500 and tf.series[0].axes=='TYX' and tf.series[0].shape==(2500,684,428)
        assert tf.imagej_metadata['frames']==2500 and tf.imagej_metadata['finterval']==.02
        labels=tf.imagej_metadata['Labels']
        for frame in range(start,stop):
            raw=tf.pages[frame].asarray()
            np.testing.assert_array_equal(raw,movie[frame-start])
            assert f' t:{frame+1}/2500 ' in labels[frame]
            group=groups.get(frame,selected.iloc[:0]);xy=group[['x_px','y_px']].to_numpy(float)
            assert np.isfinite(xy).all()
            assert roi_lib.points_in_polygon(xy,vertices).all()
            png=frames/f'frame_{frame:04d}.png'
            canvas,plain_box=render(raw,xy,vertices,frame,int(counts.loc[frame,'all_detections']),low,high,c)
            meta=PngImagePlugin.PngInfo()
            for key,value in {'original_frame':frame,'time_s':frame*.02,'roi_detection_count':len(xy),
                              'source_page_label':labels[frame],'circle_meaning':'TrackMate position marker; not uncertainty'}.items():
                meta.add_text(key,str(value))
            canvas.save(png,pnginfo=meta,compress_level=5)
            with Image.open(png) as readback:
                expected=np.rint(np.clip((raw.astype(float)-low)/(high-low),0,1)*255).astype(np.uint8).repeat(2,0).repeat(2,1)
                np.testing.assert_array_equal(np.asarray(readback.crop(plain_box).convert('L')),expected)
                assert int(readback.info['original_frame'])==frame and int(readback.info['roi_detection_count'])==len(xy)
            rows.append({'frame':frame,'time_s':frame*.02,'source_tiff_page':frame,
                         'all_detections':int(counts.loc[frame,'all_detections']),'roi_circle_count':len(xy),
                         'png':str(png.relative_to(out)),'source_pixel_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                         'png_sha256':draw_lib.sha256(png),'original_panel_verified':True,'source_label':labels[frame]})
            for p in group.itertuples():
                dx,dy=draw_lib.display_center([p.x_px,p.y_px],2)
                annotations.append({'frame':frame,'spot_id':int(p.spot_id),'source_x_px':p.x_px,'source_y_px':p.y_px,
                                    'right_panel_relative_x_px':dx,'right_panel_relative_y_px':dy,'circle_radius_source_px':3.0})
            if (frame-start+1)%10==0:print(f'PNG frames: {frame-start+1}/{count}',flush=True)
    pd.DataFrame(rows).to_csv(out/'frame_manifest.csv',index=False,float_format='%.17g')
    pd.DataFrame(annotations,columns=['frame','spot_id','source_x_px','source_y_px','right_panel_relative_x_px',
                                     'right_panel_relative_y_px','circle_radius_source_px']).to_csv(out/'annotation_positions.csv',index=False,float_format='%.17g')
    assert sum(r['roi_circle_count'] for r in rows)==len(selected)==len(annotations)
    assert len(list(frames.glob('frame_*.png')))==count
    print(f'Encoding and verifying H.264 video:{count} frames / 10fps / {count/10:g}seconds...',flush=True)
    video,checks=video_and_checks(out,c)
    print('Checking protected input/reference fingerprints again...',flush=True)
    after=[fingerprint(p) for p in protected]
    dump(out/'source_fingerprints_after.json',after)
    assert before==after,'Protected source/reference changed'
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','Pillow','h5py','tifffile','PyYAML']}
    versions.update(python=sys.version,python_executable=sys.executable,platform=platform.platform(),
                    trackmate_process=detector_report['versions'],
                    ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0])
    dump(out/'software_versions.json',versions)
    dump(out/'field_units.json',{'frame':'original zero-based TIFF time page','time_s':'acquisition seconds',
         'x_px/y_px':'native C2 camera pixels; pixel center0, X=column, Y=row increasing downward',
         'x_um/y_um':'um relative to native image origin; x/y_px multiplied by0.117',
         'quality_log_response':'TrackMate LoG filtered peak response; not photons or localization precision',
         'radius_setting_um':f'fixed detector radius{c["radius_um"]:g}um; not fitted PSF width or localization uncertainty',
         'radius_setting_px':f'fixed detector radius{c["radius_px"]:.8g}pixels','roi_inside':'1=inside/boundary,0=outside,-1=position undetermined',
         'roi_status':'center-in-polygon status; no extra localization QC'})
    validation={'all_checks_passed':True,'source_pixel_roundtrip_exact_all_selected':True,'all_original_panels_verified':True,
                'all_circle_counts_verified':True,'all_candidates_preserved':True,'empty_frames_retained':True,
                'roi_mat_cache_vertices_equal':True,'coordinate_checks':synthetic_checks,'video_checks':checks,
                'source_and_reference_fingerprints_unchanged':True,'tracking_picasso_TARDIS_mito_analysis_executed':False}
    dump(out/'validation.json',validation)
    summary={'all_detections':len(table),'roi_detections':len(selected),'outside_roi':int((table.roi_inside==0).sum()),
             'position_undetermined':int((table.roi_inside==-1).sum()),'roi_per_frame_min':int(counts.roi_detections.min()),
             'roi_per_frame_median':float(counts.roi_detections.median()),'roi_per_frame_max':int(counts.roi_detections.max()),
             'display_range_adu':[float(low),float(high)],'elapsed_s':time.perf_counter()-started}
    dump(out/'summary.json',summary)
    (out/'README.txt').write_text(
        f'T1 C2 TrackMate-only detection preview, source frames{start}-{stop-1}.\n'
        f'TrackMate7.14.0 native LoG; radius{c["radius_px"]:.8g}px={c["radius_um"]:g}um; threshold6.5; median3x3 and subpixel ON.\n'
        'No additional detector/spot filtering, Gaussian fitting, uncertainty, tracking, TARDIS or mitochondria analysis.\n'
        'All detector rows retained in all_detections.csv; roi_detections.csv is a center-in-cell-ROI subset.\n'
        'ROI source and454vertices match original MAT and previously verified cache; no coordinate transform.\n'
        'Cyan hollow circles have fixed radius3 native pixels and only indicate TrackMate positions.\n'
        'Yellow line is the common cell ROI. This preview does not certify molecular detections.\n'
        'Left=original single frame; right=same original plus ROI-contained detections.\n'
        f'Fixed display range{low:g}-{high:g}ADU; source: '+c.get('display_range_source','selected frames pooled nonzero pixels p1/p99.9')+'.\n'
        'Raw pixels unmodified; nearest-neighbor2x display; annotation-only antialiasing.\n'
        f'MP4:{count}frames,10fps,{count/10:g}seconds; fivefold slow playback relative to20ms acquisition.\n'
        f'No temporal average, interpolation or repetition; acquisition times{start*.02:g}-{(stop-1)*.02:g}s.\n'
        'PNG is the lossless display artifact; lossy MP4 is for visual inspection, not pixel measurement.\n'
        'All original/reference fingerprints checked before/after. Source snapshots, versions, commands and logs saved.\n')
    dump(out/'output_fingerprints.json',{'video_sha256':draw_lib.sha256(video),
         'all_detections_sha256':draw_lib.sha256(out/'all_detections.csv'),
         'roi_detections_sha256':draw_lib.sha256(out/'roi_detections.csv'),'manifest_sha256':draw_lib.sha256(out/'frame_manifest.csv')})
    dump(out/'COMPLETED.json',{'output_directory':str(out),'frames':count,'source_frame_start':start,'source_frame_stop_exclusive':stop,'video_fps':10,'video_duration_s':count/10,
         'all_checks_passed':True,**summary,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('COMPLETE:',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'preview_config.yaml'))
    args=parser.parse_args();cfg=Path(args.config).resolve();c=yaml.safe_load(cfg.read_text())
    assert 0<=c['frame_start']<c['frame_stop_exclusive']<=2500 and c['video_fps']==10
    assert c['frame_stop_exclusive']-c['frame_start']<=200, 'Preview is restricted to at most 200 frames'
    assert c['trackmate_version']=='7.14.0' and c['threshold']==6.5
    assert c['median_filter'] and c['subpixel_localization']
    assert c['radius_um'] in [.351,.25,.28,.30], 'This preview accepts only the requested detector-radius profiles'
    assert c['radius_px']>0
    assert abs(c['radius_um']*1000-c['radius_px']*c['pixel_size_nm'])<1e-10
    assert not any(c[k] for k in ['tracking','picasso_fitting','uncertainty_estimation','tardis','mitochondria_analysis'])
    assert shutil.which('ffmpeg') and shutil.which('ffprobe')
    data=Path(c['input_tiff']).parent;assert data.is_dir()
    profile={.25:'MATsettings_r0p25um',.28:'r0p28um',.30:'r0p30um',.351:'r3px'}[c['radius_um']]
    out=data/c['output_relative_directory']/(f'T1_C2_{profile}_frames{c["frame_start"]:04d}_{c["frame_stop_exclusive"]-1:04d}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(cfg,out/'config.yaml')
    print('OUTPUT_DIRECTORY:',out,flush=True)
    with (out/'run.log').open('w') as log:
        stdout,stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=draw_lib.Tee(stdout,log),draw_lib.Tee(stderr,log)
        try:run(c,cfg,out)
        except Exception as exc:
            dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc)})
            raise
        finally:sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    main()
