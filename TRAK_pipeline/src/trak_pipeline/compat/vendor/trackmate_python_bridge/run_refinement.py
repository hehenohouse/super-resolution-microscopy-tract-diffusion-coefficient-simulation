"""Refine an approved cached TrackMate sample with Picasso; no new detection."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
from picasso import localize
import tifffile
import yaml

from picasso_refinement import refine_candidates
from run_preview import (annotate_roi,draw_lib,dump,fingerprint,read_mat_vertices,
                         roi_lib,video_and_checks)

ROOT=Path(__file__).resolve().parent


def quantiles(table):
    result={'points':len(table)}
    for name in ['lpx_nm','lpy_nm','sx','sy','photons','bg','refinement_shift_px','deviance_per_dof']:
        a=table[name].to_numpy(float);a=a[np.isfinite(a)]
        result[name]={label:float(np.quantile(a,q)) if len(a) else None for label,q in
                   [('q10',.1),('q25',.25),('median',.5),('q75',.75),('q90',.9),('q99',.99)]}
    return result


def save_hdf5(path,table,metadata):
    columns=[]
    for name in table:
        values=table[name].to_numpy()
        if values.dtype.kind in 'OU':
            encoded=[str(v).encode() for v in values]
            values=np.asarray(encoded,dtype=f'S{max([1]+[len(v) for v in encoded])}')
        columns.append((name,values))
    data=np.empty(len(table),dtype=[(k,v.dtype) for k,v in columns])
    for name,values in columns:data[name]=values
    with h5py.File(path,'w') as h:
        h.create_dataset('locs',data=data,compression='gzip')
        h.create_dataset('metadata',data=json.dumps([metadata]))
        h.attrs['calibration_status']='manufacturer_nominal'
        h.attrs['quantitatively_calibrated']=False
    with h5py.File(path,'r') as h:
        for name in ['frame','spot_id','x','y','lpx','lpy','fit_state']:
            np.testing.assert_equal(h['locs'][()][name],table[name].to_numpy())


def render_qc(movie,roi_table,vertices,preview,out,box):
    c=dict(preview,display_scale=3)
    lo,hi=c['resolved_display_range_adu']
    crop=tuple(3*v for v in [82,232,347,430])
    w,h=crop[2]-crop[0],crop[3]-crop[1]
    width=3*w+56;width+=width%2;height=h+210
    big=ImageFont.truetype(c['font_path'],27);small=ImageFont.truetype(c['font_path'],23)
    detail=ImageFont.truetype(c['font_path'],19)
    frames=out/'frames';frames.mkdir()
    manifest=[]
    colors={'unflagged':'#35E6A0','review':'#F28DE9','invalid':'#FFAB40'}
    start=preview.get('frame_start',0);stop=preview.get('frame_stop_exclusive',50)
    for frame in range(start,stop):
        group=roi_table[roi_table.frame==frame]
        raw=movie[frame-start]
        gray=Image.fromarray(draw_lib.display_gray(raw,lo,hi,3)).convert('RGBA')
        log_xy=group[['x_px','y_px']].to_numpy()
        candidates=Image.alpha_composite(gray,draw_lib.overlay_layer(raw.shape,log_xy,vertices,c)).convert('RGB')
        # Draw exactly the native, uncapped per-axis CRLB extent at fitted centers.
        layer=draw_lib.overlay_layer(raw.shape,np.empty((0,2)),vertices,c)
        d=ImageDraw.Draw(layer)
        nvalid=0
        for row in group.itertuples():
            if row.nominal_precision_valid:
                x,y=draw_lib.display_center([row.fit_x_px,row.fit_y_px],3)
                dx,dy=row.lpx*3,row.lpy*3
                color=colors['unflagged' if row.qc_unflagged_model_fit else 'review']
                if np.isfinite([x,y,dx,dy]).all():
                    d.line([(x-dx,y),(x+dx,y)],fill=color,width=1)
                    d.line([(x,y-dy),(x,y+dy)],fill=color,width=1)
                    d.ellipse((x-1.3,y-1.3,x+1.3,y+1.3),fill=color)
                    nvalid+=1
            else:
                x,y=draw_lib.display_center([row.x_px,row.y_px],3)
                d.line([(x-3,y-3),(x+3,y+3)],fill=colors['invalid'],width=1)
                d.line([(x-3,y+3),(x+3,y-3)],fill=colors['invalid'],width=1)
        refined=Image.alpha_composite(gray,layer).convert('RGB')
        image=Image.new('RGB',(width,height),'#101317');d=ImageDraw.Draw(image)
        d.text((14,12),f'T1 C2 | Original frame {frame:04d} | {frame*.02:.2f} s | TrackMate {preview["radius_um"]:g} um -> Picasso {box}x{box} MLE',font=big,fill='white')
        labels=['Raw C2 single frame',f'TrackMate centers | {len(group)} ROI points',f'MLE +/- nominal X/Y CRLB | {nvalid} valid']
        for i,(panel,label) in enumerate(zip([gray.convert('RGB'),candidates,refined],labels)):
            x=14+i*(w+14);d.text((x,60),label,font=small,fill='white');image.paste(panel.crop(crop),(x,104))
        d.text((14,h+116),'Green: unflagged nominal fit | Magenta: nominal fit requiring review | Orange X: unavailable error at original TrackMate center',font=detail,fill='#DCE3EB')
        d.text((14,h+145),f'Crosses are +/- lpx/lpy from model CRLB, not calibrated coverage. TrackMate circles remain radius 3 px. Fixed {lo:g}-{hi:g} ADU.',font=detail,fill='#DCE3EB')
        d.text((14,h+173),f'All {len(roi_table):,} ROI candidates retained; no Picasso gradient threshold, point deletion, trajectories or TARDIS. Cell crop for display only.',font=detail,fill='#DCE3EB')
        meta=PngImagePlugin.PngInfo()
        for k,v in {'original_frame':frame,'TrackMate_ROI_count':len(group),'nominal_error_valid_count':nvalid}.items():meta.add_text(k,str(v))
        path=frames/f'frame_{frame:04d}.png';image.save(path,pnginfo=meta,compress_level=5)
        with Image.open(path) as r:
            assert int(r.info['TrackMate_ROI_count'])==len(group)
            assert int(r.info['nominal_error_valid_count'])==int(group.nominal_precision_valid.sum())
            np.testing.assert_array_equal(np.asarray(r.crop((14,104,14+w,104+h))),np.asarray(gray.convert('RGB').crop(crop)))
        manifest.append({'frame':frame,'time_s':frame*.02,'ROI_candidates':len(group),
                  'nominal_error_valid':nvalid,'source_pixel_sha256':hashlib.sha256(raw.tobytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False)
    video,checks=video_and_checks(out,preview)
    return str(video),checks


def qc_plots(table,theta,patches,used,camera,counts,out,preview):
    box=patches.shape[1]
    valid=table[table.nominal_precision_valid]
    clean=table[table.qc_unflagged_model_fit]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for field,color in [('lpx_nm','#1F77B4'),('lpy_nm','#E07C24')]:
        vals=np.sort(valid[field].to_numpy());ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=field,color=color)
    ax.set(xscale='log',xlabel='Nominal per-axis CRLB SD (nm)',ylabel='Cumulative fraction',title=f'Admissible finite bounds: {len(valid)} / {len(table)}');ax.legend();ax.grid(alpha=.2)
    ax=axes[0,1]
    cap=float(np.quantile(np.r_[clean.lpx_nm,clean.lpy_nm],.99)) if len(clean) else 200
    edges=np.linspace(0,cap,51)
    for field in ['lpx_nm','lpy_nm']:ax.hist(clean[field],bins=edges,histtype='step',linewidth=1.7,label=field)
    ax.set(xlabel='Nominal CRLB SD (nm)',ylabel='Points',title=f'Unflagged fits: {len(clean)}; display to pooled 99th percentile');ax.legend()
    ax=axes[1,0]
    for field in ['sx','sy']:ax.hist(valid[field],bins=np.linspace(0,7,71),histtype='step',label=field)
    ax.set(xlabel='Gaussian PSF standard deviation (pixels)',ylabel='Points',title='Admissible fit widths; values >7 px outside display');ax.legend()
    ax=axes[1,1]
    for field in ['ROI_candidates','fit_converged','nominal_precision_valid','qc_unflagged_model_fit']:
        ax.plot(counts.frame,counts[field],label=field)
    ax.set(xlabel='Original frame',ylabel='Points / frame',title='All candidates retained; counts describe fit status');ax.legend(fontsize=8)
    fig.suptitle(f'TrackMate {preview["radius_um"]:g} um -> Picasso {box}x{box} MLE | Manufacturer-nominal calibration',fontsize=14)
    fig.savefig(out/'precision_and_fit_QC.png',dpi=180);plt.close(fig)
    # Deterministic examples near the median signal of each QC category, not
    # selected to suggest that every accepted spot is a true isolated molecule.
    categories=[('unflagged',table.qc_unflagged_model_fit),
                ('requires review',table.nominal_precision_valid&~table.qc_unflagged_model_fit),
                ('invalid fit',~table.nominal_precision_valid)]
    samples=[]
    for label,mask in categories:
        possible=table.index[mask & table.fit_attempted].to_numpy()
        if len(possible):
            p=table.loc[possible,'photons'].to_numpy(float)
            finite=np.isfinite(p)
            i=possible[finite][np.argmin(np.abs(p[finite]-np.median(p[finite])))] if finite.any() else possible[0]
            samples.append((label,int(i)))
    fig,axes=plt.subplots(len(samples),3,figsize=(10,3.2*len(samples)),squeeze=False,constrained_layout=True)
    examples=[]
    for row,(label,i) in enumerate(samples):
        j=int(np.flatnonzero(used==i)[0]);t=table.iloc[i]
        data=(patches[j]-camera['baseline_adu'])*camera['sensitivity_e_per_adu']/camera['em_gain']
        model=__import__('picasso_refinement').model_images(theta[j:j+1],box)[0]
        lo,hi=float(data.min()),float(data.max())
        for col,array in enumerate([data,model,data-model]):
            ax=axes[row,col]
            if col<2:im=ax.imshow(array,origin='upper',vmin=lo,vmax=hi,cmap='gray')
            else:
                vmax=float(np.nanmax(np.abs(array))) if np.isfinite(array).any() else 1
                im=ax.imshow(array,origin='upper',cmap='coolwarm',vmin=-max(vmax,1),vmax=max(vmax,1))
            ax.set_title(['Raw -> nominal e-/px','Fitted Gaussian (same scale)','Residual, nominal e-/px'][col]);fig.colorbar(im,ax=ax,shrink=.8)
            if col==0:ax.set_ylabel(f'{label}\nframe {int(t.frame)}, spot {int(t.spot_id)}')
            ax.set_xticks(range(box));ax.set_yticks(range(box))
        examples.append({'category':label,'spot_id':int(t.spot_id),'frame':int(t.frame),
            'precision_status':t.precision_status,'fit_parameter_issues':t.fit_parameter_issues,
            'fit_review_flags':t.fit_review_flags})
    fig.savefig(out/'representative_fit_residuals.png',dpi=170);plt.close(fig)
    dump(out/'representative_fit_examples.json',examples)


def run(config_path):
    started=time.perf_counter()
    c=yaml.safe_load(config_path.read_text())
    start,end=map(int,c['frames']);stop=end+1;nframes=stop-start
    box=int(c['box'])
    assert 0<=start<stop<=2500 and 1<=nframes<=200 and box in (4,5,7)
    assert c['picasso_net_gradient_threshold'] is None
    assert not any(c[k] for k in ['tracking','tardis','mitochondria_analysis'])
    assert importlib.metadata.version('picassosr')==c['picasso_version']
    source=Path(c['trackmate_preview']);candidate=Path(c['candidate_csv'])
    preview=yaml.safe_load((source/'resolved_config.yaml').read_text())
    assert preview['radius_um']==c['trackmate_radius_um'] and preview['threshold']==6.5
    assert preview.get('frame_start',0)==start and preview.get('frame_stop_exclusive',50)==stop
    completion=json.loads((source/'COMPLETED.json').read_text())
    assert completion['all_checks_passed'] and completion['frames']==nframes
    expected_all=completion['all_detections'];expected_roi=completion['roi_detections']
    recorded_hashes=json.loads((source/'output_fingerprints.json').read_text())
    assert c['candidate_sha256']==recorded_hashes['all_detections_sha256']
    assert preview['pixel_size_nm']==c['pixel_size_nm']==117.
    assert preview['frame_interval_s']==c['frame_interval_s']==.02
    camera_path=(config_path.parent/c['nominal_camera_config']).resolve()
    camera=yaml.safe_load(camera_path.read_text())['camera']
    assert camera['calibration_status']=='manufacturer_nominal'
    movie_path=next((source/'native_detection').glob('*/input_frames.npy'))
    tiff=Path(preview['input_tiff'])
    protected=[tiff,Path(preview['roi_file']),Path(preview['mat_file']),candidate,
                movie_path,source/'roi_vertices.csv',source/'roi_detections.csv',
                source/'frame_counts.csv',source/'resolved_config.yaml',source/'COMPLETED.json',
                source/'output_fingerprints.json',source/'frame_manifest.csv',camera_path,
                Path(localize.__file__),Path(__import__('picasso_refinement').gaussfit.__file__)]
    radius_tag=f'{preview["radius_um"]:g}'.replace('.','p')
    out=tiff.parent/c['output_relative_directory']/(f'T1_C2_TM_r{radius_tag}um_Picasso_box{box}_frames{start:04d}_{end:04d}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT_DIRECTORY:',out,flush=True)
    shutil.copyfile(config_path,out/'config.yaml')
    with (out/'run.log').open('w') as log:
        old_stdout,old_stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=draw_lib.Tee(old_stdout,log),draw_lib.Tee(old_stderr,log)
        try:
            print('Fingerprinting read-only sources...',flush=True)
            before=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_before.json',before)
            assert next(r['sha256'] for r in before if r['path']==str(candidate))==c['candidate_sha256']
            assert before[0]['sha256']==preview['expected_tiff_sha256']
            assert before[1]['sha256']==preview['expected_roi_sha256']
            assert next(r['sha256'] for r in before if r['path']==str(source/'roi_detections.csv'))==recorded_hashes['roi_detections_sha256']
            assert next(r['sha256'] for r in before if r['path']==str(source/'frame_manifest.csv'))==recorded_hashes['manifest_sha256']
            dump(out/'resolved_parameters.json',{'config':c,'nominal_camera':camera,'TrackMate':preview,
                 'original_TrackMate_fields_preserved':True,'selection_at_detector_center':True,
                 'fitted_center_ROI_marked_separately':True,
                 'QC':'converged, physical amplitude/background/width/box-center, finite positive model bounds; width/positive-residual overlap/fit-box zero flags stored separately',
                 'width_review_rule':{'lower_sigma_px':0.5,'upper_sigma_px':box/2,'upper_is_box_dependent':True},
                 'fit_box_placement':{'anchor':'floor(original TrackMate x/y + 0.5)',
                     'pixel_offsets_relative_to_anchor':list(range(-(box//2),box-(box//2))),
                     'origin':'anchor - floor(box/2)',
                     'geometric_center_relative_to_anchor_px':(box-1)/2-box//2,
                     'global_MLE_coordinate':'local fitted coordinate + recorded box origin; no image resampling, padding, or post-fit half-pixel shift'},
                 'Picasso_detection_support_halo_not_applied':'Candidates come from TrackMate, not Picasso identification',
                 'model':'axis-aligned point-sampled elliptical Gaussian; single emitter + constant background',
                 'precision':'Picasso native Fisher CRLB, identical supplied uniform variance for MLE and precision; SD in pixels multiplied by 117 for nm'})
            command=shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path.resolve())])
            (out/'run_command.txt').write_text(command+'\n')
            snapshots=out/'source_snapshot';snapshots.mkdir()
            from picasso.fitting import precision, seeds
            for p in [Path(__file__),ROOT/'picasso_refinement.py',ROOT/'test_refinement.py',ROOT/'run_preview.py',ROOT.parent/'picasso_pilot/pilot_core.py',Path(localize.__file__),Path(precision.__file__),Path(seeds.__file__),Path(__import__('picasso_refinement').gaussfit.__file__),ROOT.parent/'localization_overlays/render_frames.py']:
                shutil.copyfile(p,snapshots/p.name)
            shutil.copyfile(camera_path,out/'nominal_camera_source.yaml')
            versions={name:importlib.metadata.version(name) for name in ['picassosr','numpy','pandas','numba','scipy','matplotlib','Pillow','tifffile','h5py','PyYAML']}
            versions.update(python=sys.version,python_executable=sys.executable)
            dump(out/'software_versions.json',versions)
            (out/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_refinement.py','-v'],stdout=log,stderr=log,check=True)
            table=pd.read_csv(io.BytesIO(candidate.read_bytes()),float_precision='round_trip')
            assert len(table)==expected_all and table.spot_id.is_unique
            assert table.frame.between(start,end).all()
            original=table.copy(deep=True)
            vertices=pd.read_csv(source/'roi_vertices.csv')[['x_px','y_px']].to_numpy()
            np.testing.assert_array_equal(vertices,roi_lib.load_imagej_roi(preview['roi_file']))
            mat_vertices,_=read_mat_vertices(preview['mat_file'],preview['mat_record'])
            np.testing.assert_array_equal(vertices,mat_vertices)
            np.testing.assert_array_equal(annotate_roi(table,vertices).roi_inside,table.roi_inside)
            roi=table[table.roi_inside==1].reset_index(drop=True)
            assert len(roi)==expected_roi
            pd.testing.assert_frame_equal(roi,pd.read_csv(source/'roi_detections.csv',float_precision='round_trip'),check_exact=True)
            movie=np.load(movie_path,mmap_mode='r')
            assert movie.shape==(nframes,684,428) and movie.dtype==np.uint16
            with tifffile.TiffFile(tiff) as tif:
                assert tif.series[0].axes=='TYX' and tif.series[0].shape==(2500,684,428)
                for frame in range(start,stop):np.testing.assert_array_equal(movie[frame-start],tif.pages[frame].asarray())
            print(f'Fitting all {len(roi):,} ROI candidates on unfiltered original pixels; no net-gradient gate.',flush=True)
            fits,theta,patches,used=refine_candidates(roi,np.arange(start,stop),movie,c,camera)
            fits['fit_center_roi_inside']=-1
            goodcenter=np.isfinite(fits[['fit_x_px','fit_y_px']]).all(axis=1)
            fits.loc[goodcenter,'fit_center_roi_inside']=roi_lib.points_in_polygon(fits.loc[goodcenter,['fit_x_px','fit_y_px']].to_numpy(),vertices).astype(int)
            extra=fits.drop(columns='frame')
            all_rows=table.merge(extra,on='spot_id',how='left',sort=False,validate='one_to_one')
            outside=all_rows.roi_inside!=1
            all_rows.loc[outside,'precision_status']='not_fitted_outside_detector_ROI'
            all_rows.loc[outside,'fit_attempted']=False
            all_rows.loc[outside,'calibration_status']=camera['calibration_status']
            all_rows.loc[outside,'quantitatively_calibrated']=False
            pd.testing.assert_frame_equal(original,all_rows[original.columns],check_exact=True)
            roi_rows=all_rows[all_rows.roi_inside==1].reset_index(drop=True)
            assert len(all_rows)==expected_all and len(roi_rows)==expected_roi
            all_rows.to_csv(out/'all_detections_with_refinement.csv',index=False,float_format='%.17g')
            roi_rows.to_csv(out/'ROI_points_with_refinement.csv',index=False,float_format='%.17g')
            public=roi_rows[roi_rows.nominal_precision_valid==True].copy()
            public.to_csv(out/'ROI_nominal_precision_valid.csv',index=False,float_format='%.17g')
            roi_rows[roi_rows.qc_unflagged_model_fit==True].to_csv(out/'ROI_unflagged_model_fits.csv',index=False,float_format='%.17g')
            hdf=roi_rows.copy();hdf['x']=hdf.fit_x_px;hdf['y']=hdf.fit_y_px
            save_hdf5(out/'roi_picasso_locs.hdf5',hdf,{'Height':684,'Width':428,'Frames':2500,
                    'Original frames processed':[start,end],'Pixelsize':117.,'calibration_status':'manufacturer_nominal',
                    'source':'TrackMate-seeded Picasso MLE, all ROI candidates, including invalid rows',
                    'select before statistics':'nominal_precision_valid / qc_unflagged_model_fit and fit_center_roi_inside as explicitly required'})
            np.savez_compressed(out/'fit_backend_audit.npz',spot_id=roi.spot_id.to_numpy()[used],
                  canonical_theta=theta,original_ADU_patches=patches)
            # CSV roundtrip: all original source columns and per-axis unit conversions.
            reread=pd.read_csv(io.BytesIO((out/'all_detections_with_refinement.csv').read_bytes()),float_precision='round_trip')
            pd.testing.assert_frame_equal(original,reread[original.columns],check_exact=True)
            for key in ['lpx','lpy','sx','sy']:np.testing.assert_allclose(roi_rows[key+'_nm'],roi_rows[key]*117.,rtol=0,atol=0,equal_nan=True)
            counts=pd.DataFrame(index=pd.Index(range(start,stop),name='frame'))
            counts['ROI_candidates']=roi_rows.groupby('frame').size().reindex(counts.index,fill_value=0)
            for key in ['fit_attempted','fit_converged','usable_model_fit','nominal_precision_valid','qc_unflagged_model_fit','raw_negative_width','suspect_psf_width','suspect_overlap']:
                counts[key]=roi_rows.groupby('frame')[key].sum().reindex(counts.index,fill_value=0).astype(int)
            for key in ['lpx_nm','lpy_nm']:
                counts['valid_median_'+key]=public.groupby('frame')[key].median().reindex(counts.index)
            counts['time_s']=counts.index*.02
            counts=counts.reset_index();counts.to_csv(out/'frame_counts_and_precision.csv',index=False)
            reasons={}
            for field in ['fit_parameter_issues','fit_review_flags']:
                reasons[field]=roi_rows[field].fillna('').str.split(';').explode().replace('',np.nan).dropna().value_counts().to_dict()
            dump(out/'fit_quality_reason_counts.json',reasons)
            summary={'output_directory':str(out),'box':box,'frames':nframes,'original_frame_range':[start,end],
                'width_review_range_px':[0.5,box/2],
                'trackmate_radius_um':preview['radius_um'],'all_FOV_rows_preserved':len(all_rows),
                'ROI_candidates':len(roi_rows),'attempted':int(roi_rows.fit_attempted.sum()),
                'backend_converged':int(roi_rows.fit_converged.sum()),
                'nominal_precision_valid':int(roi_rows.nominal_precision_valid.sum()),
                'qc_unflagged_model_fit':int(roi_rows.qc_unflagged_model_fit.sum()),
                'fitted_center_outside_ROI':int((roi_rows.fit_center_roi_inside==0).sum()),
                'fit_state_counts':roi_rows.fit_state_label.value_counts().to_dict(),
                'precision_status_counts':roi_rows.precision_status.value_counts().to_dict(),
                'valid_bounds':quantiles(public),
                'unflagged_bounds':quantiles(roi_rows[roi_rows.qc_unflagged_model_fit==True]),
                'calibration_status':'manufacturer_nominal','actual_localization_error_validated':False}
            dump(out/'summary.json',summary)
            print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)},indent=2),flush=True)
            qc_plots(roi_rows,theta,patches,used,camera,counts,out,preview)
            video,checks=render_qc(movie,roi_rows,vertices,preview,out,box)
            dump(out/'field_units.json',{'frame':'original zero-based TIFF time index','time_s':'frame * 0.02 seconds',
                'x_px/y_px':'unchanged original TrackMate floating centers, pixels',
                'x_um/y_um':'unchanged original TrackMate centers, um',
                'fit_x_px/fit_y_px':'Picasso MLE refined center; native C2 pixels, X=column, Y=row downward',
                'fit_box_origin_x_px/fit_box_origin_y_px':'original C2 coordinate of top-left fit pixel center, no interpolation',
                'fit_box_geometric_center_x_px/fit_box_geometric_center_y_px':'average of first and last fit pixel centers; half-integer for even boxes',
                'fit_x_nm/fit_y_nm':'refined pixels * 117 nm/px, relative to image origin',
                'lpx/lpy':'native Picasso per-axis square-root Fisher CRLB in pixels',
                'lpx_nm/lpy_nm':'lpx/lpy * 117; manufacturer-nominal model SD; not calibrated true error',
                'photons':'Gaussian integrated detected-photoelectron equivalents from nominal conversion, not validated incident photons',
                'bg':'nominal detected-photoelectron equivalents/pixel',
                'sx/sy':'canonical positive Gaussian PSF SD in pixels','sx_nm/sy_nm':'PSF SD * 117 nm',
                'fit_state':'native Picasso/Gpufit codes 0=converged,1=max iterations,2=singular Hessian,3=negative curvature MLE',
                'iterations':'backend iterations','fit_deviance':'shifted-Poisson deviance, dimensionless',
                'deviance_per_dof':f'fit_deviance/{box*box-6}, descriptive not goodness-of-fit p value',
                'precision_status':'fit validity/review status, never calibrated','fit_center_roi_inside':'1=inside or boundary,0=outside,-1=undetermined; selection for fitting was original TrackMate ROI',
                'qc_unflagged_model_fit':'diagnostic subset, not proof of a single molecule; no detections deleted'})
            after=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_after.json',after)
            assert before==after,'Protected input/reference changed'
            dump(out/'validation.json',{'all_checks_passed':True,'synthetic_and_failure_tests_passed':True,
                 'ROI_MAT_454_vertices_exact':True,'source_TIFF_pages_all_exact':nframes,
                 'source_TrackMate_ROI_table_exact':True,'all_original_fields_and_rows_preserved':True,
                 'pixel_to_nm_exact':True,'width_sign_model_invariance_checked':True,
                 'no_Picasso_detection_or_gradient_threshold':True,'sources_unchanged':True,'video_checks':checks})
            report=[f'TrackMate {preview["radius_um"]:g} um -> Picasso {box}x{box}, original C2 frames {start}-{end}; {len(roi_rows)} ROI candidates.',
                f'Backend converged: {summary["backend_converged"]}; physically admissible with finite positive CRLB: {summary["nominal_precision_valid"]}; heuristic unflagged: {summary["qc_unflagged_model_fit"]}.',
                f'No Picasso detector/net-gradient gate. All original {len(all_rows)} full-field TrackMate rows retained; {len(all_rows)-len(roi_rows)} outside-ROI rows explicitly not fitted.',
                'MLE refines X/Y rather than adding sigma/sqrt(N) at frozen positions. Original LoG coordinates are unchanged; refined center and post-fit ROI membership are separate fields.',
                'Only negative Gaussian width signs are canonicalized without changing the model. Nonpositive amplitude, negative background and backend failures remain explicit.',
                f'Width review threshold: sigma <0.5 or sigma >{box/2:g} px on either axis. This diagnostic upper threshold scales with fitting box; compare shared candidates and shared width criteria when judging box effects.',
                f'Actual {box}x{box} original-pixel crop. Anchor=floor(TrackMate x/y+0.5), origin=anchor-floor(box/2). Geometric box center relative to anchor={(box-1)/2-box//2:g} px per axis. Global fitted coordinates restore the recorded origin; no half-pixel coordinate translation, interpolation or padding.',
                'Error summary uses converged admissible fits with finite positive model bounds. An additional unflagged summary excludes suspect width, residual peak or zero-pixel fit boxes. These are diagnostics, not a truth classifier.',
                'Actual fit pixels are unfiltered raw TIFF. Uniform nominal read-noise variance 2.56 e^2 enters both MLE and Fisher CRLB. Negative converted pixels below -2.56 are clipped and counted.',
                'Nominal conversion (ADU-100)*0.46; EM gain 1. Signal/bg are detected electron equivalents; quantum efficiency was not used to estimate incident photon count.',
                'All CRLB SD values remain manufacturer-nominal model estimates. They exclude drift, motion blur and model mismatch. No measured calibration or actual exposure-time confirmation is available. No 50 nm filter or target is used.',
                'Boundary/overlap diagnostics: a single-emitter constant-background fit can be inadequate for faint points, neighboring peaks or structured background; finite CRLB and convergence do not establish a true isolated molecule.',
                'No trajectories, masking, TARDIS, deconvolution or error subtraction.',
                f'Video: raw / original TrackMate detections / MLE centers with uncapped +/- lpx/lpy crosses; all {nframes} frames,10fps,{nframes/10:g} seconds. Green=unflagged,magenta=requires review,orange X=unavailable bound at original LoG position.',
                'Reproduction command is in run_command.txt; config, software versions, source snapshots, input fingerprints and tests are saved.',
                'Official references: https://github.com/jungmannlab/picasso/blob/v0.11.3/picasso/fitting/gaussfit.py ; https://picassosr.readthedocs.io/en/latest/files.html']
            for subset in ['valid_bounds','unflagged_bounds']:
                q=summary[subset];report.append(f'{subset}: n={q["points"]}; lpx_nm median={q["lpx_nm"]["median"]}, IQR=({q["lpx_nm"]["q25"]},{q["lpx_nm"]["q75"]}); lpy_nm median={q["lpy_nm"]["median"]}, IQR=({q["lpy_nm"]["q25"]},{q["lpy_nm"]["q75"]}).')
            (out/'REPORT.txt').write_text('\n'.join(report)+'\n')
            dump(out/'output_fingerprints.json',{str(p.relative_to(out)):fingerprint(p)['sha256'] for p in [out/'all_detections_with_refinement.csv',out/'ROI_points_with_refinement.csv',out/'roi_picasso_locs.hdf5',out/'frame_manifest.csv',Path(video)]})
            dump(out/'COMPLETED.json',dict(summary,video=video,all_checks_passed=True,elapsed_s=time.perf_counter()-started))
            print('COMPLETE:',out,flush=True)
        except Exception as exc:
            dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc)})
            raise
        finally:sys.stdout,sys.stderr=old_stdout,old_stderr


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config.resolve())
