#!/usr/bin/env python3
"""Read-only local spatial/temporal QC of pooled Not-on patterns; no tracking or reclassification."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import tifffile

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'mito_tardis'))
from render_point_stack import fingerprint,dump,pixel_hash,Tee,roi_reader,mat_vertices
from point_stack_core import mask_contour_alpha,display_centers
from cluster_qc_geometry import MaskSquareDistance,longest_occupied_frame_run

ON='#27dda5';OFF='#c08ff7';MASK='#f05c58'
REGIONS=[('R1','Upper edge',137,246),('R2','Upper, near On',202,260),
    ('R3','Right middle',281,322),('R4','Lower central',243,377),
    ('R5','Right lower',315,373),('R6','Left middle',131,331)]
CORE_RADIUS=2.5
COLUMNS=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','sx_px','sy_px',
    'qc_unflagged_model_fit','fitted_center_inside_ROI','support50_pointwise_label','support50_block_index',
    'calibration_status','original_detection_source_uid']


def qstats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,q25=float(np.quantile(a,.25)) if len(a) else None,
        q75=float(np.quantile(a,.75)) if len(a) else None)


def enrich(table,full,support):
    result=table.copy();xy=result[['fit_x_px','fit_y_px']].to_numpy()
    nearest=np.floor(xy+.5).astype(int);blocks=result.frame.to_numpy(int)//50
    result['support_at_nearest_pixel']=support[blocks,nearest[:,1],nearest[:,0]]
    result['final_mask_at_nearest_pixel']=full[blocks,nearest[:,1],nearest[:,0]]
    distances=np.zeros(len(result),float)
    for block in np.unique(blocks):
        indices=np.flatnonzero(blocks==block)
        distances[indices]=MaskSquareDistance(full[block]).distances(xy[indices])*117
    result['distance_to_current_full_mask_nm']=distances
    return result


def plot_context(out,source,regions):
    image=Image.open(source/'projections/not_on_all_timepoints.png').convert('RGB')
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25)
    for r in regions:
        x0,y0,x1,y1=r['bounds'];p=display_centers([[x0,y0],[x1,y1]],[82,232,347,430],4)
        draw.rectangle((p[0,0],p[0,1]+80,p[1,0],p[1,1]+80),outline='#ffcf66',width=3)
        draw.text((p[0,0]+3,p[0,1]+82),r['id'],font=font,fill='#ffdf88',stroke_width=2,stroke_fill='black')
    image.save(out/'overview_regions.png',compress_level=6)


def show_raw(ax,raw,crop,limits):
    x0,y0,x1,y1=crop
    ax.imshow(raw,cmap='gray',vmin=limits[0],vmax=limits[1],origin='upper',interpolation='nearest',
        extent=[x0-.5,x1-.5,y1-.5,y0-.5])
    ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.tick_params(labelsize=7)


def show_mask(ax,alpha,crop):
    x0,y0,x1,y1=crop
    rgba=np.zeros((*alpha.shape,4));rgba[...,:3]=[.94,.36,.35];rgba[...,3]=alpha*.85
    ax.imshow(rgba,extent=[x0-.5,x1-.5,y1-.5,y0-.5],origin='upper',interpolation='nearest')


def show_points(ax,rows,errors=False):
    for label,color in [('mito-compatible',ON),('non-mito-compatible',OFF)]:
        q=rows[rows.support50_pointwise_label==label]
        if errors:
            ax.errorbar(q.fit_x_px,q.fit_y_px,xerr=q.x_se_nm/117,yerr=q.y_se_nm/117,fmt='.',ms=3,
                color=color,ecolor=color,elinewidth=.6,capsize=1,alpha=.9)
        else:ax.scatter(q.fit_x_px,q.fit_y_px,s=12,facecolors='none',edgecolors=color,linewidths=.8)


def contact_sheet(out,r,points,means,representative,raw1,raw2,full,vertices):
    blocks=r['review_blocks'];crop=r['padded_bounds'];x0,y0,x1,y1=crop;rid=r['id']
    fig,axes=plt.subplots(len(blocks),4,figsize=(14,3.45*len(blocks)),squeeze=False,layout='constrained')
    fig.suptitle(f'{rid}: {r["name"]} | native peak ({r["cx"]},{r["cy"]}) | {r["n_off"]:,} pooled Not-on observations\n'
        'Green=On; purple=Not-on; red=final support50 mask. White circle: fixed 2.5 px diagnostic core.',fontsize=12)
    for row,b in enumerate(blocks):
        f=representative[(rid,b)];one=points[points.frame==f];window=points[points.frame.between(b*50,b*50+49)]
        mask_alpha=mask_contour_alpha(full[b],crop,6)
        arrays=[raw1[f][y0:y1,x0:x1],raw1[f][y0:y1,x0:x1],raw2[f][y0:y1,x0:x1],means[rid][b]]
        titles=['C1 raw single frame','C1 raw + mask + same-frame points','C2 raw + points / nominal X,Y SE',
            'C1 50-frame mean + mask + pooled dots\nDISPLAY ONLY; not used to make this mask']
        for col,(ax,raw,title) in enumerate(zip(axes[row],arrays,titles)):
            show_raw(ax,raw,crop,[406,554] if col==2 else [407,651])
            if col in [1,2,3]:show_mask(ax,mask_alpha,crop)
            if col in [1,2]:show_points(ax,one,errors=col==2)
            if col==3:
                for label,color in [('mito-compatible',ON),('non-mito-compatible',OFF)]:
                    q=window[window.support50_pointwise_label==label];ax.scatter(q.fit_x_px,q.fit_y_px,s=3,c=color,alpha=.35,linewidths=0)
            ax.add_patch(Circle((r['cx'],r['cy']),CORE_RADIUS,fill=False,edgecolor='white',ls='--',lw=.65))
            ax.add_patch(Rectangle((r['bounds'][0],r['bounds'][1]),28,28,fill=False,edgecolor='#b0b0b0',lw=.6))
            ax.plot(vertices[:,0],vertices[:,1],c='#969696',lw=.5)
            ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_title(title,fontsize=9)
        off=int((one.support50_pointwise_label=='non-mito-compatible').sum());on=len(one)-off
        axes[row,0].set_ylabel(f'Block {b:02d}: frames {50*b}-{50*b+49}\nOriginal f={f}, t={f*.02:.2f}s\nFrame On={on}, Not-on={off}',fontsize=9)
        axes[row,3].set_xlabel(f'All 50 frames: On={int((window.support50_pointwise_label=="mito-compatible").sum())}, Not-on={int((window.support50_pointwise_label=="non-mito-compatible").sum())}',fontsize=9)
    fig.get_layout_engine().set(rect=(0,.035,1,.94))
    fig.text(.01,.009,'C1/C2 identity registration unverified. SE is nominal-model uncertainty; mask/registration uncertainty is not included. No trajectory linking.\n'
        'Fixed windows: blocks 0,24,49 plus peak core Not-on block (if different). Single frame = earliest core Not-on, else earliest window Not-on, else middle frame.',fontsize=8)
    fig.savefig(out/'regions'/f'{rid}_raw_mask_localizations.png',dpi=150);plt.close(fig)


def temporal_figure(out,regions,points,perblock):
    fig,axes=plt.subplots(len(regions),2,figsize=(13,2.45*len(regions)),layout='constrained')
    for r,axs in zip(regions,axes):
        rid=r['id'];b=perblock[perblock.region_id==rid];p=points[(points.region_id==rid)&points.in_core&points.support50_pointwise_label.eq('non-mito-compatible')]
        axs[0].plot(b.block_index+.5,b.On,c=ON,lw=1,label='On, 28x28 window')
        axs[0].plot(b.block_index+.5,b.Not_on,c=OFF,lw=1,label='Not-on, window')
        axs[0].plot(b.block_index+.5,b.core_Not_on,c='#663399',lw=1,ls='--',label='Not-on, 2.5 px core')
        axs[0].set(title=f'{rid} {r["name"]} | window Not-on {r["n_off"]:,}',xlabel='Original time (s)',ylabel='Observations per 50 frames',xlim=(0,50))
        axs[1].scatter(p.frame*.02,p.fit_y_px-r['cy'],s=2,c=OFF,alpha=.5,rasterized=True)
        axs[1].set(title=f'Fixed core: {len(p):,} Not-on / {p.frame.nunique()} occupied frames',xlabel='Original time (s)',ylabel='Y - core center (pixel)',xlim=(0,50),ylim=(-2.6,2.6))
        for ax in axs:ax.grid(alpha=.15);ax.tick_params(labelsize=8)
        if rid=='R1':axs[0].legend(fontsize=7,loc='upper right')
    fig.suptitle('TDKd3 T1 | local temporal QC: empty frames retained; each dot is one observation, no linked trajectories',fontsize=12)
    fig.savefig(out/'temporal_counts_and_core_positions.png',dpi=155);plt.close(fig)


def core_figure(out,points,perblock):
    r=points[(points.region_id=='R4')&points.in_core];off=r[r.support50_pointwise_label=='non-mito-compatible'];b=perblock[perblock.region_id=='R4']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].plot(b.block_index+.5,b.core_On,c=ON,label='On core');axes[0,0].plot(b.block_index+.5,b.core_Not_on,c=OFF,label='Not-on core')
    axes[0,0].set(xlabel='Original time (s)',ylabel='Observations / 50 frames',title='Fixed radius 2.5 px core at (243,377)');axes[0,0].legend()
    axes[0,1].plot(b.block_index+.5,b.support_at_fixed_peak,c='black',label='Pre-final support count at fixed pixel')
    axes[0,1].axhline(10,c=MASK,ls='--',label='Support threshold 10/50')
    axes[0,1].scatter(b.block_index+.5,np.where(b.final_mask_at_fixed_peak,2,0),c=np.where(b.final_mask_at_fixed_peak,ON,'#999999'),s=16,label='Final mask at fixed pixel: y=2 yes, y=0 no')
    axes[0,1].set(xlabel='Original time (s)',ylabel='Source foreground count',title='Fixed pixel is not each refined localization center');axes[0,1].legend(fontsize=7)
    axes[1,0].plot(b.block_index+.5,b.C1_core3x3_mean_ADU,c='#444444',label='C1 raw, fixed 3x3')
    axes[1,0].plot(b.block_index+.5,b.C2_core3x3_mean_ADU,c='#2e75b6',label='C2 raw, fixed 3x3')
    axes[1,0].set(xlabel='Original time (s)',ylabel='Mean original ADU / 50 frames',title='Brightness diagnostic only; no photon conversion');axes[1,0].legend(fontsize=8)
    bins=np.linspace(0,1000,51);se=off[['x_se_nm','y_se_nm']].max(axis=1)
    axes[1,1].hist(off.distance_to_current_full_mask_nm,bins=bins,histtype='step',color=OFF,label=f'Mask distance median {off.distance_to_current_full_mask_nm.median():.1f} nm')
    axes[1,1].hist(se,bins=bins,histtype='step',color='#247994',label=f'Max(X,Y) nominal SE median {se.median():.1f} nm')
    axes[1,1].set(xlabel='nm',ylabel='Not-on observations',title='Distance to CURRENT mask\nNot a verified mitochondrial boundary');axes[1,1].legend(fontsize=8)
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.suptitle('R4 core: persistent compact Not-on observations; identity / cause remain unresolved',fontsize=12)
    fig.savefig(out/'R4_core_diagnostics.png',dpi=155);plt.close(fig)


def run(args,out):
    source=args.source;assert json.loads((source/'COMPLETED.json').read_text())['delivery_checks_passed']
    c=json.loads((source/'resolved_config.json').read_text());maskdir=Path(c['mask_directory']);C1=Path(c['C1_file']);C2=C1.with_name(C1.name.replace('C1-','C2-',1))
    known={r['path']:r['sha256'] for r in json.loads((source/'output_fingerprints.json').read_text())}
    known.update({r['path']:r['sha256'] for r in json.loads((source/'input_fingerprints_before.json').read_text())})
    known.update({str(maskdir/n):v for n,v in json.loads((maskdir/'output_fingerprints.json').read_text()).items()})
    source_csv=source/'classification/selected_localizations_support50.csv'
    paths=[source_csv,source/'resolved_config.json',source/'projections/not_on_all_timepoints.png',C1,C2,
        Path(c['ROI_file']),Path(c['MAT_file'])]+[maskdir/n for n in ['clean_full_mask.npy','foreground_count.npy','roi_mask.npy','roi_vertices_xy.npy','original_frame_to_mask_block.npy','COMPLETE.json']]
    before=[]
    for path in paths:
        item=fingerprint(path)
        if str(path) in known:assert item['sha256']==known[str(path)],str(path)
        else:assert path==C2 or path.name=='COMPLETE.json',str(path)
        before.append(item)
    dump(out/'input_fingerprints_before.json',before)
    full=np.load(maskdir/'clean_full_mask.npy',mmap_mode='r');support=np.load(maskdir/'foreground_count.npy',mmap_mode='r')
    assert full.shape==support.shape==(50,684,428) and full.dtype==bool
    np.testing.assert_array_equal(np.load(maskdir/'original_frame_to_mask_block.npy'),np.arange(2500)//50)
    lib,_=roi_reader();vertices=lib.load_imagej_roi(c['ROI_file']);np.testing.assert_array_equal(vertices,np.load(maskdir/'roi_vertices_xy.npy'))
    np.testing.assert_array_equal(vertices,mat_vertices(c['MAT_file'],c['MAT_record']))
    table=pd.read_csv(source_csv,usecols=COLUMNS,float_precision='round_trip')
    assert len(table)==115013 and table.candidate_index.is_unique and table.qc_unflagged_model_fit.all() and table.fitted_center_inside_ROI.all()
    assert table.support50_pointwise_label.value_counts().to_dict()=={'mito-compatible':77601,'non-mito-compatible':37412}
    regions=[];local=[];representative={};representative_manifest=[]
    for rid,name,cx,cy in REGIONS:
        bounds=[cx-14,cy-14,cx+14,cy+14];x0,y0,x1,y1=bounds
        p=table[table.fit_x_px.ge(x0)&table.fit_x_px.lt(x1)&table.fit_y_px.ge(y0)&table.fit_y_px.lt(y1)].copy()
        p['region_id']=rid;p['in_core']=np.hypot(p.fit_x_px-cx,p.fit_y_px-cy)<=CORE_RADIUS
        p=enrich(p,full,support);off=p[p.support50_pointwise_label=='non-mito-compatible'];core=off[off.in_core]
        coreblocks=core.groupby(core.frame//50).size().reindex(range(50),fill_value=0)
        peakblock=int(coreblocks.idxmax());blocks=sorted(set([0,24,49,peakblock]))
        r=dict(id=rid,name=name,cx=cx,cy=cy,bounds=bounds,padded_bounds=[x0-1,y0-1,x1+1,y1+1],
            n_on=int(p.support50_pointwise_label.eq('mito-compatible').sum()),n_off=len(off),off_frames=int(off.frame.nunique()),
            off_blocks=int((off.frame//50).nunique()),core_off=len(core),core_off_frames=int(core.frame.nunique()),core_off_blocks=int((core.frame//50).nunique()),
            longest_core_presence_run=longest_occupied_frame_run(core.frame.to_numpy()),review_blocks=blocks,
            off_mask_distance_nm=qstats(off.distance_to_current_full_mask_nm),core_off_mask_distance_nm=qstats(core.distance_to_current_full_mask_nm),
            core_off_x_se_nm=qstats(core.x_se_nm),core_off_y_se_nm=qstats(core.y_se_nm),
            core_off_signal_nominal_e=qstats(core.photons_e),core_off_sx_px=qstats(core.sx_px),core_off_sy_px=qstats(core.sy_px))
        regions.append(r);local.append(p)
        for b in blocks:
            window=off[off.frame.between(b*50,b*50+49)];cp=window[window.in_core]
            if len(cp):f=int(cp.frame.min());rule='earliest_core_Not_on_frame'
            elif len(window):f=int(window.frame.min());rule='earliest_window_Not_on_frame'
            else:f=b*50+24;rule='middle_frame_no_Not_on_observation'
            representative[(rid,b)]=f;representative_manifest.append(dict(region_id=rid,block_index=b,frame=f,time_s=f*.02,selection=rule))
        print('REGION',rid,json.dumps({key:r[key] for key in ['n_on','n_off','off_frames','core_off','core_off_frames']}),flush=True)
    points=pd.concat(local,ignore_index=True);assert points.candidate_index.is_unique
    independent=[]
    for b in range(50):
        q=points[(points.frame//50)==b]
        if len(q):
            sample=q.iloc[np.linspace(0,len(q)-1,min(5,len(q)),dtype=int)]
            yy,xx=np.nonzero(full[b])
            for row in sample.itertuples():
                distance=np.sqrt(np.min(np.maximum(np.abs(xx-row.fit_x_px)-.5,0)**2+np.maximum(np.abs(yy-row.fit_y_px)-.5,0)**2))*117
                np.testing.assert_allclose(distance,row.distance_to_current_full_mask_nm,atol=1e-9,rtol=1e-12);independent.append(int(row.candidate_index))
    r4=points[(points.region_id=='R4')&points.in_core&points.support50_pointwise_label.eq('non-mito-compatible')]
    assert len(r4)==1866 and r4.frame.nunique()==1866 and (r4.frame//50).nunique()==50
    np.testing.assert_allclose(r4.distance_to_current_full_mask_nm.median(),556.1,atol=.1)
    # Stream only the two explicitly specified TIFFs; original frames remain separate.
    means={r['id']:np.zeros((50,30,30),float) for r in regions};core1=np.zeros((6,2500));core2=np.zeros_like(core1)
    raw1={};raw2={};wanted=set(representative.values());page_records=[]
    with tifffile.TiffFile(C1) as f1,tifffile.TiffFile(C2) as f2:
        for tf in [f1,f2]:assert len(tf.pages)==2500 and tf.series[0].shape==(2500,684,428) and tf.series[0].axes=='TYX' and tf.pages[0].dtype==np.uint16
        for f in range(2500):
            a=f1.pages[f].asarray();b=f2.pages[f].asarray()
            for i,r in enumerate(regions):
                x0,y0,x1,y1=r['padded_bounds'];cx,cy=r['cx'],r['cy']
                means[r['id']][f//50]+=a[y0:y1,x0:x1].astype(float)/50
                core1[i,f]=a[cy-1:cy+2,cx-1:cx+2].mean();core2[i,f]=b[cy-1:cy+2,cx-1:cx+2].mean()
            if f in wanted:
                raw1[f]=a;raw2[f]=b;page_records.append(dict(frame=f,C1_sha256=pixel_hash(a),C2_sha256=pixel_hash(b)))
            if (f+1)%250==0:print('RAW_REVIEW_READ',f+1,flush=True)
    perframe=[];perblock=[]
    for i,r in enumerate(regions):
        rid=r['id'];p=points[points.region_id==rid];off=p[p.support50_pointwise_label=='non-mito-compatible'];on=p[p.support50_pointwise_label=='mito-compatible']
        counts={name:q.groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy() for name,q in [('On',on),('Not_on',off),('core_On',on[on.in_core]),('core_Not_on',off[off.in_core])]}
        assert sum(counts['On'])==r['n_on'] and sum(counts['Not_on'])==r['n_off']
        for f in range(2500):
            perframe.append(dict(region_id=rid,frame=f,time_s=f*.02,block_index=f//50,**{key:int(v[f]) for key,v in counts.items()},C1_core3x3_ADU=core1[i,f],C2_core3x3_ADU=core2[i,f]))
        for b in range(50):
            perblock.append(dict(region_id=rid,block_index=b,first_frame=b*50,last_frame=b*50+49,
                **{key:int(v[b*50:b*50+50].sum()) for key,v in counts.items()},
                support_at_fixed_peak=int(support[b,r['cy'],r['cx']]),final_mask_at_fixed_peak=bool(full[b,r['cy'],r['cx']]),
                C1_core3x3_mean_ADU=float(core1[i,b*50:b*50+50].mean()),C2_core3x3_mean_ADU=float(core2[i,b*50:b*50+50].mean())))
    pf=pd.DataFrame(perframe);pb=pd.DataFrame(perblock);pf.to_csv(out/'per_frame_statistics.csv',index=False);pb.to_csv(out/'per_block_statistics.csv',index=False)
    points.to_csv(out/'region_localizations.csv',index=False)
    pd.DataFrame(representative_manifest).to_csv(out/'representative_frames.csv',index=False);dump(out/'source_page_hashes.json',page_records)
    dump(out/'region_summary.json',regions)
    plot_context(out,source,regions)
    for r in regions:
        contact_sheet(out,r,points[points.region_id==r['id']],means,representative,raw1,raw2,full,vertices)
        print('CONTACT_SHEET',r['id'],flush=True)
    temporal_figure(out,regions,points,pb);core_figure(out,points,pb)
    np.savez_compressed(out/'C1_display_only_block_mean_crops.npz',**{rid:v.astype(np.float32) for rid,v in means.items()})
    after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
    for name in ['run_not_on_cluster_qc.py','cluster_qc_geometry.py','test_cluster_qc_geometry.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    for name in ['point_stack_core.py','render_point_stack.py']:
        shutil.copyfile(PROJECT/'mito_tardis'/name,out/'source_snapshot'/name)
    dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scipy','Pillow','tifffile','h5py']}))
    dump(out/'config.json',dict(source=str(source),regions=REGIONS,region_selection='Exploratory fixed windows identified from pooled native-pixel Not-on density smoothed with Gaussian sigma2px, then anatomical coverage; not statistical cluster calling',
        region_membership='x in [cx-14,cx+14), y in [cy-14,cy+14); figures add 1-pixel raw context padding',core_radius_px=2.5,
        contact_sheet_blocks='0,24,49 plus highest core Not-on count block, first tie',frame_rule='earliest core Not-on frame; else earliest window Not-on; else middle frame',
        C1_display_ADU=[407,651],C2_display_ADU=[406,554],classification_and_fits_unchanged=True,
        display_only_C1_mean='Mean of original 50 raw frames, no rescaling, no mask computation',registration='identity_unverified',tracking=False,TARDIS=False))
    dump(out/'validation.json',dict(all_checks_passed=True,sources_unchanged=True,original_QC_and_labels_and_fits_unchanged=True,
        all2500_original_frames_retained_in_per_frame_statistics=True,independent_mask_distance_checks=len(independent),
        distance_checked_candidate_indices=independent,R4_independent_reference_reproduced=True,raw_page_hashes_recorded=True,
        spatial_regions_disjoint=True,visual_review_pending=True))
    (out/'README.txt').write_text('TDKd3 T1 support50 Not-on local QC. Six exploratory fixed windows, no statistical cluster test.\n'
        'overview_regions.png locates windows on unchanged pooled Not-on image. regions/ contains original C1/C2, mask, same-frame points/SE and display-only 50-frame C1 means.\n'
        'C1 mean crops are only viewing aids: support50 masks were NOT generated from these averages.\n'
        'temporal_counts_and_core_positions.png: per-block observation counts plus unlinked core Y observations. Empty frames retained.\n'
        'R4_core_diagnostics.png: local counts, fixed-pixel support, raw original-channel brightness, distances to current mask vs nominal model SE.\n'
        'Continuous-position rectangle membership and fixed radius2.5px define QC windows/cores, not molecular objects or uncertainty bounds.\n'
        'Per-frame counts and consecutive occupied-frame runs describe presence only; no trajectory linking or independent molecule identification.\n'
        'Mask geometry distance uses exact closed foreground squares; it is NOT distance to a validated mitochondrial boundary.\n'
        'Fixed peak-pixel mask status is separate from classification at each refined point center.\n'
        'C1/C2 identity registration unverified; nominal SE excludes mask/registration/model uncertainties. No biological mechanism inferred.\n'
        'Original localizations, labels, QC, raw data and masks untouched. No refitting, segmentation, TARDIS or significance test.\n')
    dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,output=str(out)))
    print('QC_READY',out,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    args.source=args.source or Path((HERE/'latest_support50_projection_output.txt').read_text().strip())
    out=args.output or HERE/'results'/('not_on_cluster_QC_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(out/'regions').mkdir();(out/'source_snapshot').mkdir()
    (HERE/'latest_not_on_cluster_qc_local.txt').write_text(str(out)+'\n')
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--source',str(args.source),'--output',str(out)])+'\n')
    with (out/'run.log').open('w') as log:
        a,b=sys.stdout,sys.stderr;sys.stdout=Tee(a,log);sys.stderr=Tee(b,log)
        try:run(args,out)
        except BaseException:
            import traceback;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=a,b


if __name__=='__main__':main()
