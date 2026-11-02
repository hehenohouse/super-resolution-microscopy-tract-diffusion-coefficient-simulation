#!/usr/bin/env python3
"""Isolated sample runner using unchanged native detector and annular fitter.

Source detection UID is stable across pilot/full and resume. Pilot candidate
indices are provisional; full indices are assigned in original-frame order.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import tifffile

import run_real as core
from run_full import decorate
from parallel_fit import fit_batch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'mito_tardis'))
from render_point_stack import roi_reader,mat_vertices,pixel_hash,Tee


def stage_frames(c,stage):
    return [f for a,b in c['pilot_segments'] for f in range(a,b+1)] if stage!='full' else list(range(2500))


def prepare(c,out):
    m=Path(c['mask_directory']);done=json.loads((m/'COMPLETE.json').read_text())
    assert done['all_passed'] and done['blocks']==50 and done['source_frames']==2500
    paths=[Path(c[k]) for k in ['C1_file','C2_file','roi_file','mat_file']]
    paths += [m/n for n in ['parameters.json','COMPLETE.json','roi_mask.npy','roi_vertices_xy.npy','clean_full_mask.npy',
        'analysis_ROI_mask.npy','block_frame_mapping.csv','original_frame_numbers_by_block.npy','block_completed.npy']]
    paths += [Path(c['fiji'])/'jars/TrackMate-7.14.0.jar',HERE/'run_real.py',HERE/'parallel_fit.py',
        PROJECT/'trackmate_python_bridge/run_detection.py',HERE/'detect_batches.py',Path(c['config_path'])]
    before=[]
    known={str(m/n):h for n,h in json.loads((m/'output_fingerprints.json').read_text()).items()}
    for path in paths:
        print('SOURCE',path.name,flush=True);row=core.fingerprint(path)
        if str(path) in known:assert row['sha256']==known[str(path)]
        before.append(row)
    if (out/'input_fingerprints_before.json').exists():assert json.loads((out/'input_fingerprints_before.json').read_text())==before
    core.dump(out/'input_fingerprints_before.json',before)
    lib,_=roi_reader();vertices=lib.load_imagej_roi(c['roi_file'])
    np.testing.assert_array_equal(vertices,mat_vertices(c['mat_file'],c['mat_record']))
    np.testing.assert_array_equal(vertices,np.load(m/'roi_vertices_xy.npy'))
    assert len(vertices)==c['expected_roi_vertices']
    yy,xx=np.mgrid[:684,:428]
    roi=lib.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428)
    np.testing.assert_array_equal(roi,np.load(m/'roi_mask.npy'));assert roi.sum()==c['expected_roi_pixels']
    for channel in ['C1_file','C2_file']:
        with tifffile.TiffFile(c[channel]) as tf:
            assert len(tf.pages)==2500 and tf.series[0].axes=='TYX' and tf.series[0].shape==(2500,684,428)
            assert tf.pages[0].dtype==np.dtype('uint16')
    pd.DataFrame(vertices,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    np.save(out/'roi_mask.npy',roi)
    core.dump(out/'source_validation.json',dict(ROI_MAT_mask_exact=True,vertices=len(vertices),ROI_pixels=int(roi.sum()),
        source_shape=[2500,684,428],source_axes='TYX',pixel_size_nm=117,frame_interval_s=.02,no_coordinate_transform=True))
    for name in ['run_sample.py','run_real.py','run_full.py','parallel_fit.py','detect_batches.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    shutil.copyfile(PROJECT/'trackmate_python_bridge/run_detection.py',out/'source_snapshot/run_detection.py')
    core.dump(out/'rules_before_run.json',c)
    core.dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','tifffile','Pillow','h5py']}))
    if not (out/'math_validation.json').exists():core.dump(out/'math_validation.json',core.math_checks())
    return roi,vertices,before


def detect_stage(c,out,frames):
    chunks=[[a,a+25] for a in range(0,2500,25) if a in frames]
    assert set(f for a,b in chunks for f in range(a,b))==set(frames)
    job=dict(output=str(out/'native_detection'),input_tiff=c['C2_file'],java_home=c['java_home'],fiji=c['fiji'],chunks=chunks)
    path=out/('pilot_detection_job.json' if len(frames)==150 else 'full_detection_job.json');core.dump(path,job)
    command=[c['detector_python'],'-B','-u',str(HERE/'detect_batches.py'),'--job',str(path),'--max-new-chunks','10']
    (out/(path.stem+'_command.txt')).write_text(shlex.join(command)+'\n')
    print('NATIVE_DETECTION',len(frames),'frames',flush=True)
    with (out/'native_detection.log').open('a') as log:
        # Restart the JVM after at most 250 frames. Completed chunks are verified
        # and reused; this bounds ImageJ/Java retained memory without changing fits.
        while True:
            remaining=[(a,b) for a,b in chunks if not (out/'native_detection'/f'frames{a:04d}_{b-1:04d}.json').exists()]
            if not remaining:break
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():raise RuntimeError('Native detection failed; incomplete chunks are not accepted')
            after=[(a,b) for a,b in remaining if not (out/'native_detection'/f'frames{a:04d}_{b-1:04d}.json').exists()]
            assert len(after)<len(remaining),'Detection made no progress'
            print('DETECTION_CHECKPOINT',json.dumps(dict(completed_frames=(len(chunks)-len(after))*25,
                total_frames=len(frames),bounded_JVM_batch_frames=250)),flush=True)
    groups={};hashes={}
    for a,b in chunks:
        tag=f'frames{a:04d}_{b-1:04d}';p=out/'native_detection'/(tag+'.csv')
        info=json.loads((out/'native_detection'/(tag+'.json')).read_text())
        assert core.fingerprint(p)['sha256']==info['csv_sha256']
        assert info['trackmate_version']=='7.14.0' and info['radius_um']==.28 and info['threshold']==6.5
        table=pd.read_csv(p,float_precision='round_trip')
        table['source_detection_uid']=[c['sample_id']+':'+tag+':'+str(int(i)) for i in table.spot_id]
        table['source_detection_cache']=str(p)
        for f in range(a,b):groups[f]=table[table.frame==f].copy()
        hashes.update({int(r['frame']):r['pixel_sha256'] for r in info['page_hashes']})
    return groups,hashes


def fit_stage(c,out,frames,groups,hashes,roi,vertices,stage):
    if stage=='full':
        assert json.loads((out/'pilot/visual_review.json').read_text())['passed']
    root=out/('pilot' if stage=='pilot' else 'frame_fits');total=0;counts=[];start=time.perf_counter();reused=0
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[key]='1'
    with tifffile.TiffFile(c['C2_file']) as tf,ProcessPoolExecutor(max_workers=c['workers'],
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in frames:
            table=groups[f].copy().reset_index(drop=True)
            table.insert(0,'candidate_index',np.arange(total,total+len(table),dtype=np.int64));total+=len(table)
            inside=core.roi_lib.points_in_polygon(table[['x_px','y_px']].to_numpy(),vertices)
            table['roi_inside']=inside.astype(int);targets=table[inside].copy()
            raw=tf.pages[f].asarray();assert pixel_hash(raw)==hashes[f]
            fit_path=root/f'frame_{f:04d}.csv';bg_path=root/f'background_{f:04d}.csv'
            pixels_path=root/f'annular_pixels_{f:04d}.npz';marker=root/f'frame_{f:04d}.json'
            cache_hit=marker.exists()
            if cache_hit:
                info=json.loads(marker.read_text());assert info['source_pixel_sha256']==hashes[f]
                assert info['UIDs']==targets.source_detection_uid.tolist()
                for path,key in [(fit_path,'fit_sha256'),(bg_path,'background_sha256'),(pixels_path,'pixels_sha256')]:
                    assert core.fingerprint(path)['sha256']==info[key]
                fits=pd.read_csv(fit_path,float_precision='round_trip');bg=pd.read_csv(bg_path,float_precision='round_trip')
                np.testing.assert_array_equal(fits.candidate_index,targets.candidate_index)
            elif stage=='full' and (out/'pilot'/f'frame_{f:04d}.json').exists():
                pilot=out/'pilot';info=json.loads((pilot/f'frame_{f:04d}.json').read_text())
                assert info['UIDs']==targets.source_detection_uid.tolist() and info['source_pixel_sha256']==hashes[f]
                for path,key in [(pilot/f'frame_{f:04d}.csv','fit_sha256'),(pilot/f'background_{f:04d}.csv','background_sha256'),
                                 (pilot/f'annular_pixels_{f:04d}.npz','pixels_sha256')]:
                    assert core.fingerprint(path)['sha256']==info[key]
                fits=pd.read_csv(pilot/f'frame_{f:04d}.csv',float_precision='round_trip')
                bg=pd.read_csv(pilot/f'background_{f:04d}.csv',float_precision='round_trip')
                fits['pilot_candidate_index']=fits.candidate_index;fits['candidate_index']=targets.candidate_index.to_numpy()
                bg['candidate_index']=targets.candidate_index.to_numpy()
                fits['reused_from_pilot']=True;reused+=len(fits)
                with np.load(pilot/f'annular_pixels_{f:04d}.npz') as pp:
                    np.savez_compressed(pixels_path,xy_pixels=pp['xy_pixels'],offsets=pp['offsets'],
                        candidate_index=targets.candidate_index.to_numpy(),source_detection_uid=targets.source_detection_uid.to_numpy(str))
                fits.to_csv(fit_path,index=False);bg.to_csv(bg_path,index=False)
            else:
                if len(targets):
                    bg,masks,movie=core.background_geometry(targets,table,raw,roi,c,c['camera'])
                    items=[(int(r.candidate_index),int(r.spot_id),f,float(r.x_px),float(r.y_px)) for r in targets.itertuples()]
                    for item,pixels in zip(items,masks):
                        _,_,_,x,y=item;ox,oy=int(np.floor(x+.5))-3,int(np.floor(y+.5))-3
                        assert not np.any((pixels[:,0]>=ox)&(pixels[:,0]<ox+7)&(pixels[:,1]>=oy)&(pixels[:,1]<oy+7))
                        assert roi[pixels[:,1],pixels[:,0]].all() and (raw[pixels[:,1],pixels[:,0]]>0).all()
                    step=max(1,int(np.ceil(len(items)/c['workers'])))
                    futures=[pool.submit(fit_batch,(items[a:a+step],raw,movie,bg.iloc[a:a+step].reset_index(drop=True),
                        masks[a:a+step],c)) for a in range(0,len(items),step)]
                    results=[r for future in futures for r in future.result()]
                    fits=decorate(pd.DataFrame(results),vertices,c)
                    bg['source_detection_uid']=targets.source_detection_uid.to_numpy()
                    fits['source_detection_uid']=targets.source_detection_uid.to_numpy()
                    fits['source_detection_cache']=targets.source_detection_cache.to_numpy()
                    fits['reused_from_pilot']=False
                    np.savez_compressed(pixels_path,xy_pixels=np.concatenate(masks).astype(np.int16),
                        offsets=np.r_[0,np.cumsum([len(p) for p in masks])],candidate_index=targets.candidate_index.to_numpy(),
                        source_detection_uid=targets.source_detection_uid.to_numpy(str))
                else:
                    fields=['candidate_index','spot_id','frame','time_s','fit_attempted','fit_success','active_bound',
                        'fit_usable_interior','finite_uncertainty','fit_x_px','fit_y_px','photons_e','bg_e_per_pixel',
                        'sx_px','sy_px','x_se_nm','y_se_nm','suspect_overlap','source_detection_uid']
                    fits=pd.DataFrame(columns=fields)
                    for b in ['fit_attempted','fit_success','active_bound','fit_usable_interior','finite_uncertainty','suspect_overlap']:
                        fits[b]=fits[b].astype(bool)
                    fits=decorate(fits,vertices,c)
                    bg=pd.DataFrame(columns=['candidate_index','frame','source_detection_uid'])
                    np.savez_compressed(pixels_path,xy_pixels=np.empty((0,2),np.int16),offsets=np.array([0]),
                        candidate_index=np.array([],np.int64),source_detection_uid=np.array([],str))
                fits.to_csv(fit_path,index=False);bg.to_csv(bg_path,index=False)
            assert len(fits)==len(targets)
            np.testing.assert_array_equal(fits.source_detection_uid,targets.source_detection_uid)
            if not cache_hit:
                core.dump(marker,dict(frame=f,source_pixel_sha256=hashes[f],UIDs=targets.source_detection_uid.tolist(),
                    fit_sha256=core.fingerprint(fit_path)['sha256'],background_sha256=core.fingerprint(bg_path)['sha256'],
                    pixels_sha256=core.fingerprint(pixels_path)['sha256'],rows=len(fits)))
            status=Counter(fits.audit_status)
            record=dict(frame=f,time_s=f*.02,FOV_candidates=len(table),ROI_candidates=len(fits),
                selected=int(fits.qc_unflagged_model_fit.sum()),QC_excluded=int((~fits.qc_unflagged_model_fit).sum()),
                **{'QC_reason__'+k:int(v) for k,v in status.items() if k!='unflagged_model_fit'})
            counts.append(record)
            if f%10==9 or f in [0,1225,2450]:print('FIT_FRAME',f,'ROI',len(fits),'selected',record['selected'],
                'elapsed_s',round(time.perf_counter()-start,1),flush=True)
            if stage=='full' and (f+1)%250==0:
                ff=pd.DataFrame(counts).fillna(0);ff.to_csv(out/'frame_counts.csv',index=False)
                cp=dict(completed_frames=f+1,ROI_candidates=int(ff.ROI_candidates.sum()),selected=int(ff.selected.sum()),
                    QC_excluded=int(ff.QC_excluded.sum()),elapsed_s=time.perf_counter()-start)
                core.dump(out/'checkpoints'/f'fits_{f+1:04d}.json',cp);print('FIT_CHECKPOINT',json.dumps(cp),flush=True)
    ff=pd.DataFrame(counts).fillna(0);ff.to_csv(root/'frame_counts.csv',index=False)
    all_fits=pd.concat([pd.read_csv(root/f'frame_{f:04d}.csv',float_precision='round_trip') for f in frames],ignore_index=True)
    all_bg=pd.concat([pd.read_csv(root/f'background_{f:04d}.csv',float_precision='round_trip') for f in frames],ignore_index=True)
    assert all_fits.candidate_index.is_unique and all_fits.source_detection_uid.is_unique
    assert int(ff.ROI_candidates.sum())==len(all_fits) and int(ff.selected.sum())==int(all_fits.qc_unflagged_model_fit.sum())
    all_fits.to_csv(root/'all_ROI_target_fits.csv',index=False);all_bg.to_csv(root/'annular_background_all_ROI_targets.csv',index=False)
    selected=all_fits[all_fits.qc_unflagged_model_fit]
    summary=dict(sample_name=c['sample_name'],frames=len(frames),ROI_candidates=len(all_fits),selected=len(selected),
        QC_excluded=len(all_fits)-len(selected),FOV_candidates=int(ff.FOV_candidates.sum()),
        QC_reasons={str(k):int(v) for k,v in all_fits.audit_status.value_counts().items()},
        reused_pilot_fits=reused,calibration_status='manufacturer_nominal',model_SE_coverage_validated=False,
        statistics={k:core.stats(selected[k]) for k in ['x_se_nm','y_se_nm','photons_e','sx_px','sy_px']})
    core.dump(root/'summary.json',summary)
    if stage=='pilot':
        pilot_QC(c,out,all_fits,groups)
        core.dump(out/'run_status.json',dict(status='pilot_ready',counts=summary))
        print('PILOT_READY',json.dumps(summary),flush=True)
    else:
        for name in ['all_ROI_target_fits.csv','annular_background_all_ROI_targets.csv','summary.json']:
            shutil.copyfile(root/name,out/name)
        all_detect=[];offset=0
        for f in range(2500):
            t=groups[f].copy();t.insert(0,'candidate_index',np.arange(offset,offset+len(t),dtype=np.int64));offset+=len(t)
            t['roi_inside']=core.roi_lib.points_in_polygon(t[['x_px','y_px']].to_numpy(),vertices)
            all_detect.append(t)
        detections=pd.concat(all_detect,ignore_index=True);assert detections.candidate_index.is_unique and detections.source_detection_uid.is_unique
        detections.to_csv(out/'all_detections.csv',index=False)
        np.testing.assert_array_equal(all_fits.source_detection_uid,detections.loc[detections.roi_inside,'source_detection_uid'])
        ff.to_csv(out/'frame_counts.csv',index=False)
        core.dump(out/'source_page_manifest.json',[dict(frame=f,pixel_sha256=hashes[f]) for f in range(2500)])
        current=json.loads((out/'input_fingerprints_before.json').read_text())
        after=[core.fingerprint(r['path']) for r in current];assert current==after
        core.dump(out/'input_fingerprints_after.json',after)
        core.dump(out/'validation.json',dict(all_implementation_checks_passed=True,all2500_original_frames_preserved=True,
            all_C2_page_hashes_match_native_detection=True,all_detected_UIDs_unique=True,ROI_MAT_mask_match=True,
            all_ROI_fits_retained=True,QC_counts_conserved=True,pilot_UID_reuse_verified=True,source_fingerprints_unchanged=True,
            nominal_model_SE_not_calibrated_coverage=True,no_tracking_or_TARDIS=True))
        core.dump(out/'COMPLETED.json',dict(delivery_checks_passed=True,all_implementation_checks_passed=True,
            sample_name=c['sample_name'],frames=2500,counts=summary,output=str(out)))
        core.dump(out/'run_status.json',dict(status='complete'))
        (HERE/'latest_WT_full_output.txt').write_text(str(out)+'\n')
        print('FULL_FITS_COMPLETE',json.dumps(summary),flush=True)


def pilot_QC(c,out,fits,groups):
    from PIL import Image,ImageDraw,ImageFont
    from point_stack_core import gray_crop,display_centers,marker_alpha,composite,roi_alpha,ROI_COLOR
    import matplotlib.pyplot as plt
    outdir=out/'pilot';roi_vertices=pd.read_csv(out/'roi_vertices.csv').to_numpy()
    crop=c['crop_xyxy_px'];scale=2;w=(crop[2]-crop[0])*2;h=(crop[3]-crop[1])*2
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    with tifffile.TiffFile(c['C2_file']) as tf:
        for f in [0,1225,2450]:
            raw=tf.pages[f].asarray();gray=gray_crop(raw,crop,scale,[406,554]);base=np.repeat(gray[...,None],3,2)
            candidates=groups[f];inside=core.roi_lib.points_in_polygon(candidates[['x_px','y_px']].to_numpy(),roi_vertices)
            p=display_centers(candidates.loc[inside,['x_px','y_px']].to_numpy(),crop,2)
            right=composite(composite(base,roi_alpha(roi_vertices,crop,2),ROI_COLOR),marker_alpha((h,w),p,6,1.25),[0,220,235])
            im=Image.new('RGB',(2*w+20,h+90),'black');im.paste(Image.fromarray(base),(0,70));im.paste(Image.fromarray(right),(w+20,70))
            draw=ImageDraw.Draw(im);subset=fits[fits.frame==f]
            draw.text((8,8),f'WT TRAK1 T1 | frame {f} | TM r0.28um + annular7x7 | ROI candidates {len(subset)}',font=font,fill='white')
            draw.text((8,38),f'QC selected {int(subset.qc_unflagged_model_fit.sum())} | nominal model errors, not calibrated coverage',font=font,fill='white')
            im.save(outdir/f'frame_{f:04d}_detection_QC.png',compress_level=6)
        chosen=[]
        for status in ['unflagged_model_fit','active_parameter_bound','overlap_review']:
            chosen.extend(fits[fits.audit_status==status].head(2).to_dict('records'))
        if chosen:
            fig,axes=plt.subplots(len(chosen),3,figsize=(9,2.7*len(chosen)),squeeze=False,layout='constrained')
            for row,axs in zip(chosen,axes):
                f=int(row['frame']);raw=tf.pages[f].asarray();ox,oy=int(row['fit_origin_x_px']),int(row['fit_origin_y_px'])
                if ox<0 or oy<0 or ox+7>428 or oy+7>684:continue
                observed=(raw[oy:oy+7,ox:ox+7].astype(float)-100)*.46
                if np.isfinite([row['fit_x_px'],row['fit_y_px'],row['sx_px'],row['sy_px'],row['photons_e'],row['bg_e_per_pixel']]).all():
                    P=core.base.psf_and_derivatives(7,row['fit_x_px']-ox,row['fit_y_px']-oy,row['sx_px'],row['sy_px'])[0].reshape(7,7)
                    model=row['photons_e']*P+row['bg_e_per_pixel'];residual=observed-model
                else:model=np.zeros((7,7));residual=observed.copy()
                lo,hi=float(observed.min()),float(observed.max())
                for ax,a,title in zip(axs,[observed,model,residual],['Raw nominal e-/px','Fitted model','Residual']):
                    ax.imshow(a,cmap='coolwarm' if title=='Residual' else 'gray',vmin=None if title=='Residual' else lo,vmax=None if title=='Residual' else hi)
                    ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
                axs[0].set_ylabel(f'frame{f}, ID{int(row["candidate_index"])}\n{row["audit_status"]}\nSE {row["x_se_nm"]:.1f}/{row["y_se_nm"]:.1f}nm')
            fig.savefig(outdir/'fit_raw_model_residual_examples.png',dpi=150);plt.close(fig)
    clean=fits[fits.qc_unflagged_model_fit]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,['x_se_nm','y_se_nm','photons_e']):
        v=clean[key].dropna().to_numpy();edges=np.linspace(0,200,41) if key!='photons_e' else np.linspace(0,max(float(np.quantile(v,.99)) if len(v) else 1,1),41)
        ax.hist(v,bins=edges,histtype='step');ax.set(xlabel=key,ylabel='Localizations',title=f'{len(v)} QC selected; {int((v>edges[-1]).sum())} above range')
    fig.suptitle('WT pilot | nominal model standard errors and whole-PSF electron equivalents')
    fig.savefig(outdir/'pilot_uncertainty_signal_histograms.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--stage',choices=['detect-pilot','pilot','full'],default='pilot');args=p.parse_args()
    c=json.loads(args.config.read_text());c['config_path']=str(args.config.resolve())
    c['detector_python']=str((PROJECT/c['detector_python']).absolute())
    if args.output:
        out=args.output.resolve();assert json.loads((out/'config.json').read_text())==c
    else:
        out=Path(c['C2_file']).parent/'analysis_results/annular_background_refinement'/(
            c['sample_id']+'_C2_r0p28um_box7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        out.mkdir(parents=True,exist_ok=False)
        for n in ['source_snapshot','native_detection','pilot','frame_fits','checkpoints']:(out/n).mkdir()
        core.dump(out/'config.json',c)
    (HERE/'latest_WT_attempt.txt').write_text(str(out)+'\n')
    (out/('command_'+args.stage+'.txt')).write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),
        '--config',str(args.config.resolve()),'--output',str(out),'--stage',args.stage])+'\n')
    with (out/'run.log').open('a') as log:
        a,b=sys.stdout,sys.stderr;sys.stdout=Tee(a,log);sys.stderr=Tee(b,log)
        try:
            print('OUTPUT',out,flush=True);roi,vertices,before=prepare(c,out)
            frames=stage_frames(c,args.stage);groups,hashes=detect_stage(c,out,frames)
            if args.stage=='detect-pilot':
                core.dump(out/'run_status.json',dict(status='pilot_detection_ready'));print('PILOT_DETECTION_READY',flush=True);return
            fit_stage(c,out,frames,groups,hashes,roi,vertices,args.stage)
            core.dump(out/'output_fingerprints.json',[core.fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
                and p.name not in ['output_fingerprints.json','COMPLETED.json','run.log']])
        except BaseException:
            core.dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),stage=args.stage));raise
        finally:sys.stdout,sys.stderr=a,b


if __name__=='__main__':main()
