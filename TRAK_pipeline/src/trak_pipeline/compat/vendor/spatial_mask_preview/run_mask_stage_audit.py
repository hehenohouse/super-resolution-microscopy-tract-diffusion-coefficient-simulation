#!/usr/bin/env python3
"""Trace existing Not-on observations through the unchanged support50 mask recipe."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/noton-mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import tifffile

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'mito_tardis'))
from render_point_stack import fingerprint,dump,Tee,pixel_hash,roi_reader
from point_stack_core import mask_contour_alpha
from run_not_on_cluster_qc import show_raw,show_mask,show_points,ON,OFF
from mask_stage_qc_core import ellipse_support_max,first_failure_reason

REGION_IDS=['R4','R3','R1']
REASONS=['no_raw_broad_foreground_in_ellipse_in_block','raw_broad_present_below_per_pixel_support_threshold',
    'per_frame_min32_lowered_below_threshold','post_support_min32_removed']
COLORS=['#777777','#e69f00','#8064a2','#009e73']
SHORT=['No raw broad support','Raw support below 10 per pixel','Per-frame min32 drops below 10','Post-support min32 removes']


def stage_sheet(out,r,representatives,raw_images,broad_frames,clean_frames,crops,saved,vertices,points):
    rid=r['id'];crop=r['padded_bounds'];x0,y0,x1,y1=crop
    reps=representatives[representatives.region_id==rid].sort_values('block_index')
    fig,axes=plt.subplots(len(reps),8,figsize=(21,3.6*len(reps)),squeeze=False,layout='constrained')
    fig.suptitle(f'{rid}: unchanged mask stages | fixed diagnostic center ({r["cx"]},{r["cy"]})\n'
        'Raw C1: 407-651 ADU. Count maps: 0-50. White circle: fixed radius2.5px, not localization uncertainty.',fontsize=12)
    for i,rep in enumerate(reps.itertuples()):
        b=int(rep.block_index);f=int(rep.frame);raw=raw_images[f][y0:y1,x0:x1]
        stage_masks=[None,broad_frames[f],clean_frames[f],None,None,saved['input'][b],saved['post'][b],saved['final'][b]]
        titles=['C1 original single frame','C1 + raw Nellie broad\nSAME original frame','C1 + FIRST min32\nSAME original frame',
            'RAW broad count / 50\nDiagnostic before first min32','After FIRST min32\nCount / 50 (actual input)',
            'C1 + actual support >=10','C1 + SECOND min32','C1 + FINAL holes<64\nOriginal labels / nominal SE']
        for j,ax in enumerate(axes[i]):
            if j in [3,4]:
                values=crops[rid]['raw_count' if j==3 else 'clean_count'][b]
                show_raw(ax,values,crop,[0,50]);ax.images[0].set_cmap('viridis')
                titles[j]+=f'\nFixed peak: {int(values[r["cy"]-y0,r["cx"]-x0])}/50'
            else:
                show_raw(ax,raw,crop,[407,651])
                if stage_masks[j] is not None:show_mask(ax,mask_contour_alpha(stage_masks[j],crop,5),crop)
                if j==7:show_points(ax,points[(points.region_id==rid)&points.frame.eq(f)],errors=True)
            ax.add_patch(Circle((r['cx'],r['cy']),2.5,fill=False,ec='white',ls='--',lw=.65))
            ax.plot(vertices[:,0],vertices[:,1],c='#999999',lw=.5)
            ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_title(titles[j],fontsize=8)
        axes[i,0].set_ylabel(f'Block {b}: frames {b*50}-{b*50+49}\nOriginal frame {f} ({f*.02:.2f}s)',fontsize=9)
    fig.get_layout_engine().set(rect=(0,.03,1,.95),h_pad=.18,hspace=.08,w_pad=.06)
    fig.text(.01,.008,'Per-frame broad/min32 panels and 50-frame count/support panels describe different temporal stages. Final mask and original localization labels are unchanged.\n'
        'Raw-support>=10 is a diagnostic counterfactual, not a new final mask. C1/C2 identity registration remains unverified.',fontsize=8)
    fig.savefig(out/'regions'/f'{rid}_all_mask_stages.png',dpi=150);plt.close(fig)


def overview(out,fixed,point):
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for rid,axs in zip(REGION_IDS,axes):
        s=fixed[fixed.region_id==rid];q=point[(point.region_id==rid)&point.in_core]
        axs[0].plot(s.block_index+.5,s.raw_broad_count,c='#333333',label='Raw broad count')
        axs[0].plot(s.block_index+.5,s.cleaned_count,c='#2e75b6',label='After first min32')
        axs[0].axhline(10,c='#d9534f',ls='--',lw=1,label='Threshold10/50')
        axs[0].set(title=rid+' fixed peak PIXEL (not each refined center)',xlabel='Original time (s)',ylabel='Foreground appearances / 50',xlim=(0,50),ylim=(-.5,50.5))
        bottom=np.zeros(50)
        for reason,color,label in zip(REASONS,COLORS,SHORT):
            counts=q[q.stage_first_failure==reason].groupby(q[q.stage_first_failure==reason].frame//50).size().reindex(range(50),fill_value=0).to_numpy()
            axs[1].bar(np.arange(50)+.5,counts,bottom=bottom,width=.9,color=color,label=label);bottom+=counts
        axs[1].set(title=f'{rid} core: first failed stage for {len(q):,} Not-on observations',xlabel='Original time (s)',ylabel='Observations / 50 frames',xlim=(0,50))
        for ax in axs:ax.grid(alpha=.15);ax.tick_params(labelsize=8)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=7)
    fig.suptitle('TDKd3 T1 | stage diagnostics with unchanged parameters; descriptive observations, no molecule identity',fontsize=12)
    fig.savefig(out/'stage_support_and_point_reasons.png',dpi=155);plt.close(fig)


def render_saved(out):
    """Regenerate only pending-review figures from saved, verified diagnostics."""
    assert json.loads((out/'COMPLETED.json').read_text())['visual_review_pending']
    config=json.loads((out/'config.json').read_text());qc=Path(config['prior_cluster_QC'])
    source=json.loads((Path(config['source'])/'resolved_config.json').read_text());m=Path(source['mask_directory'])
    spec=importlib.util.spec_from_file_location('recorded_support50_union_core',m/'union_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    regions=[r for r in json.loads((qc/'region_summary.json').read_text()) if r['id'] in REGION_IDS]
    regions=sorted(regions,key=lambda r:REGION_IDS.index(r['id']))
    representatives=pd.read_csv(qc/'representative_frames.csv');representatives=representatives[representatives.region_id.isin(REGION_IDS)]
    labels=np.load(json.loads((m/'parameters.json').read_text())['source_broad_labels'],mmap_mode='r')
    raw_images={};broad_frames={};clean_frames={}
    with tifffile.TiffFile(source['C1_file']) as raw_t:
        for f in sorted(set(representatives.frame.astype(int))):
            raw_images[f]=raw_t.pages[f].asarray();broad_frames[f]=labels[f]>0
            clean_frames[f]=core.clean_frame(broad_frames[f])[0]
    arrays=np.load(out/'stage_count_crops.npz')
    crops={r['id']:{name:arrays[r['id']+'_'+name] for name in ['raw_count','clean_count']} for r in regions}
    saved={key:np.load(m/name,mmap_mode='r') for key,name in [('input','input_ge10_full_mask.npy'),('post','min32_full_mask.npy'),('final','clean_full_mask.npy')]}
    all_points=pd.read_csv(qc/'region_localizations.csv',float_precision='round_trip')
    for r in regions:stage_sheet(out,r,representatives,raw_images,broad_frames,clean_frames,crops,saved,np.load(m/'roi_vertices_xy.npy'),all_points)
    overview(out,pd.read_csv(out/'fixed_peak_stage_diagnostics.csv'),pd.read_csv(out/'point_stage_diagnostics.csv',float_precision='round_trip'))


def run(args,out):
    qc=args.qc;qcconfig=json.loads((qc/'config.json').read_text());source=Path(qcconfig['source'])
    c=json.loads((source/'resolved_config.json').read_text());m=Path(c['mask_directory'])
    params=json.loads((m/'parameters.json').read_text());B=Path(params['source_broad_labels']);C1=Path(c['C1_file'])
    assert json.loads((qc/'COMPLETED.json').read_text())['delivery_checks_passed']
    assert json.loads((m/'COMPLETE.json').read_text())['all_passed']
    known={p['path']:p['sha256'] for p in json.loads((qc/'output_fingerprints.json').read_text())}
    known.update({p['path']:p['sha256'] for p in json.loads((qc/'input_fingerprints_before.json').read_text())})
    known.update({str(m/n):h for n,h in json.loads((m/'output_fingerprints.json').read_text()).items()})
    known.update(json.loads((m/'input_fingerprints_before.json').read_text()))
    paths=[qc/'region_localizations.csv',qc/'region_summary.json',qc/'representative_frames.csv',C1,B,m/'union_core.py',m/'parameters.json',
        m/'foreground_count.npy',m/'input_ge10_full_mask.npy',m/'min32_full_mask.npy',m/'clean_full_mask.npy',m/'roi_vertices_xy.npy',m/'roi_mask.npy',
        m/'per_frame_statistics.csv',m/'original_frame_to_mask_block.npy']
    before=[]
    for path in paths:
        item=fingerprint(path);assert item['sha256']==known[str(path)],str(path);before.append(item)
    dump(out/'input_fingerprints_before.json',before)
    spec=importlib.util.spec_from_file_location('recorded_support50_union_core',m/'union_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    tests=core.synthetic_tests();assert all(tests.values())
    dump(out/'original_mask_synthetic_tests.json',tests)
    regions=[r for r in json.loads((qc/'region_summary.json').read_text()) if r['id'] in REGION_IDS]
    regions=sorted(regions,key=lambda r:REGION_IDS.index(r['id']))
    all_points=pd.read_csv(qc/'region_localizations.csv',float_precision='round_trip')
    all_points=all_points[all_points.region_id.isin(REGION_IDS)].copy()
    target=all_points[all_points.support50_pointwise_label=='non-mito-compatible'].copy()
    assert len(target)==8711 and target.candidate_index.is_unique
    representatives=pd.read_csv(qc/'representative_frames.csv');representatives=representatives[representatives.region_id.isin(REGION_IDS)]
    wanted=set(representatives.frame.astype(int))
    broad_labels=np.load(B,mmap_mode='r');assert broad_labels.shape==(2500,684,428)
    np.testing.assert_array_equal(np.load(m/'original_frame_to_mask_block.npy'),np.arange(2500)//50)
    saved={key:np.load(m/name,mmap_mode='r') for key,name in [('count','foreground_count.npy'),('input','input_ge10_full_mask.npy'),('post','min32_full_mask.npy'),('final','clean_full_mask.npy')]}
    frame_reference=pd.read_csv(m/'per_frame_statistics.csv').set_index('frame')
    raw_counts=np.lib.format.open_memmap(out/'raw_broad_foreground_count.npy',mode='w+',dtype=np.uint8,shape=(50,684,428))
    roi=np.load(m/'roi_mask.npy');vertices=np.load(m/'roi_vertices_xy.npy')
    crops={r['id']:{name:np.zeros((50,30,30),np.uint8) for name in ['raw_count','clean_count']} for r in regions}
    instant={};raw_images={};broad_frames={};clean_frames={};frame_records=[];fixed_records=[];point_records=[];validation_samples=[]
    by_frame={int(f):group for f,group in target.groupby('frame')}
    with tifffile.TiffFile(C1) as raw_t:
        assert len(raw_t.pages)==2500 and raw_t.series[0].axes=='TYX'
        for block in range(50):
            raw_count=np.zeros((684,428),np.uint8);clean_count=np.zeros_like(raw_count)
            for f in range(block*50,block*50+50):
                broad=broad_labels[f]>0;clean,lab,area,keep=core.clean_frame(broad)
                raw_count+=broad;clean_count+=clean
                ref=frame_reference.loc[f]
                assert broad.sum()==ref.broad_full_area_pixel2 and clean.sum()==ref.per_frame_min32_full_area_pixel2
                assert (clean&roi).sum()==ref.per_frame_min32_ROI_area_pixel2
                frame_records.append(dict(frame=f,block_index=block,broad_pixels=int(broad.sum()),first_min32_pixels=int(clean.sum()),stored_area_reproduced=True))
                if f in by_frame:
                    area=area.astype(np.uint32);area[0]=0;area_map=area[lab]
                    for row in by_frame[f].itertuples():
                        x,y=row.fit_x_px,row.fit_y_px;a,b=row.x_se_nm/117,row.y_se_nm/117
                        ma=ellipse_support_max(area_map,x,y,a,b)
                        cleanhit=ellipse_support_max(clean,x,y,a,b)>0
                        assert cleanhit==(ma>=32)
                        instant[row.candidate_index]=dict(raw_broad_original_frame_hit=ma>0,first_min32_original_frame_hit=cleanhit,
                            max_touched_raw_component_area_px2=ma)
                if f in wanted:
                    raw_images[f]=raw_t.pages[f].asarray();broad_frames[f]=broad;clean_frames[f]=clean
            np.testing.assert_array_equal(clean_count,saved['count'][block])
            assert np.all(clean_count<=raw_count)
            np.testing.assert_array_equal(clean_count>=10,saved['input'][block])
            post,postlab,postarea,_=core.clean_frame(clean_count>=10)
            np.testing.assert_array_equal(post,saved['post'][block])
            np.testing.assert_array_equal(core.fill_enclosed(post)[0],saved['final'][block])
            raw_counts[block]=raw_count
            for r in regions:
                x0,y0,x1,y1=r['padded_bounds'];x,y=r['cx'],r['cy']
                crops[r['id']]['raw_count'][block]=raw_count[y0:y1,x0:x1];crops[r['id']]['clean_count'][block]=clean_count[y0:y1,x0:x1]
                fixed_records.append(dict(region_id=r['id'],block_index=block,x=x,y=y,raw_broad_count=int(raw_count[y,x]),cleaned_count=int(clean_count[y,x]),
                    raw_ge10=bool(raw_count[y,x]>=10),actual_ge10=bool(saved['input'][block,y,x]),post_min32=bool(post[y,x]),final=bool(saved['final'][block,y,x]),
                    first_failure=first_failure_reason(bool(raw_count[y,x]>=10),bool(clean_count[y,x]>=10),bool(post[y,x]),bool(saved['final'][block,y,x]),bool(raw_count[y,x]>0))))
            subset=target[(target.frame//50)==block]
            postarea=postarea.astype(np.uint32);postarea[0]=0;postareas=postarea[postlab]
            for row in subset.itertuples():
                x,y=row.fit_x_px,row.fit_y_px;a,b=row.x_se_nm/117,row.y_se_nm/117
                raw_max=ellipse_support_max(raw_count,x,y,a,b);clean_max=ellipse_support_max(clean_count,x,y,a,b)
                posthit=ellipse_support_max(post,x,y,a,b)>0;finalhit=ellipse_support_max(saved['final'][block],x,y,a,b)>0
                assert not finalhit,'An original Not-on label disagrees with unchanged final geometry'
                reason=first_failure_reason(raw_max>=10,clean_max>=10,posthit,finalhit,raw_max>0)
                maxpostarea=ellipse_support_max(postareas,x,y,a,b)
                if reason=='post_support_min32_removed':assert 0<maxpostarea<32
                record=dict(row._asdict(),**instant[row.candidate_index],stage_raw_broad_support_max=raw_max,
                    stage_cleaned_support_max=clean_max,stage_raw_ge10_hit=raw_max>=10,stage_actual_ge10_hit=clean_max>=10,
                    stage_post_min32_hit=posthit,stage_final_hit=finalhit,stage_first_failure=reason,
                    max_touched_post_support_component_area_px2=maxpostarea)
                point_records.append(record)
            if len(subset):
                yy,xx=np.indices((684,428))
                for row in subset.iloc[np.linspace(0,len(subset)-1,min(2,len(subset)),dtype=int)].itertuples():
                    q=(np.maximum(np.abs(xx-row.fit_x_px)-.5,0)/(row.x_se_nm/117))**2+(np.maximum(np.abs(yy-row.fit_y_px)-.5,0)/(row.y_se_nm/117))**2
                    hit=q<=1+1e-12
                    assert ellipse_support_max(raw_count,row.fit_x_px,row.fit_y_px,row.x_se_nm/117,row.y_se_nm/117)==int(raw_count[hit].max())
                    validation_samples.append(int(row.candidate_index))
            if (block+1)%5==0:
                raw_counts.flush();dump(out/'checkpoints'/f'frames_{(block+1)*50:04d}.json',dict(completed_frames=(block+1)*50,completed_blocks=block+1,original_support_and_final_masks_reproduced=True))
                print('STAGE_CHECKPOINT', (block+1)*50,'frames; original count/post/final masks reproduced',flush=True)
    raw_counts.flush();points=pd.DataFrame(point_records).sort_values(['frame','candidate_index']).reset_index(drop=True)
    fixed=pd.DataFrame(fixed_records);pd.DataFrame(frame_records).to_csv(out/'per_frame_reconstruction_checks.csv',index=False)
    points.to_csv(out/'point_stage_diagnostics.csv',index=False);fixed.to_csv(out/'fixed_peak_stage_diagnostics.csv',index=False)
    beforecols=list(target.columns);reference=target.set_index('candidate_index',drop=False).loc[points.candidate_index]
    for key in beforecols:np.testing.assert_array_equal(points[key].to_numpy(),reference[key].to_numpy(),err_msg=key)
    countrows=[]
    for rid in REGION_IDS:
        for scope in ['window','core']:
            q=points[(points.region_id==rid)&(points.in_core if scope=='core' else True)]
            for reason in REASONS:
                n=int(q.stage_first_failure.eq(reason).sum());countrows.append(dict(region_id=rid,scope=scope,reason=reason,count=n,total=len(q),fraction=n/len(q)))
    summary=pd.DataFrame(countrows);summary.to_csv(out/'stage_reason_summary.csv',index=False)
    blockrows=[]
    for (rid,block,reason),q in points.groupby(['region_id','support50_block_index','stage_first_failure']):
        blockrows.append(dict(region_id=rid,block_index=block,reason=reason,window_count=len(q),core_count=int(q.in_core.sum())))
    pd.DataFrame(blockrows).to_csv(out/'per_block_stage_reason_counts.csv',index=False)
    np.savez_compressed(out/'stage_count_crops.npz',**{rid+'_'+name:array for rid,data in crops.items() for name,array in data.items()})
    for r in regions:
        stage_sheet(out,r,representatives,raw_images,broad_frames,clean_frames,crops,saved,vertices,all_points)
        print('STAGE_SHEET',r['id'],flush=True)
    overview(out,fixed,points)
    # Frozen fixed-pixel findings independently reproduced by another read-only implementation.
    expected={'R4':(3,47,0,0),'R3':(0,46,4,0),'R1':(0,50,0,0)}
    fixed_summary=[]
    for rid,values in expected.items():
        q=fixed[fixed.region_id==rid];off=q[~q.final]
        observed=(int(q.final.sum()),int((off.raw_broad_count<10).sum()),int(off.first_failure.eq(REASONS[2]).sum()),int(off.first_failure.eq(REASONS[3]).sum()))
        assert observed==values,(rid,observed)
        fixed_summary.append(dict(region_id=rid,final_on_blocks=observed[0],final_off_blocks=len(off),raw_broad_below10_blocks=observed[1],
            first_min32_threshold_loss_blocks=observed[2],second_min32_loss_blocks=observed[3],raw_zero_in_off_blocks=int(off.raw_broad_count.eq(0).sum())))
    pd.DataFrame(fixed_summary).to_csv(out/'fixed_peak_summary.csv',index=False)
    after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
    for name in ['run_mask_stage_audit.py','mask_stage_qc_core.py','test_mask_stage_qc_core.py','run_not_on_cluster_qc.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    shutil.copyfile(m/'union_core.py',out/'source_snapshot/recorded_union_core.py')
    for name in ['point_stack_core.py','render_point_stack.py']:
        shutil.copyfile(PROJECT/'mito_tardis'/name,out/'source_snapshot'/name)
    tests=subprocess.run([sys.executable,'-B',str(HERE/'test_mask_stage_qc_core.py')],capture_output=True,text=True)
    (out/'stage_geometry_tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0,tests.stderr
    dump(out/'config.json',dict(source=str(source),prior_cluster_QC=str(qc),regions=REGION_IDS,classification_and_mask_parameters_unchanged=True,
        original_recipe='Nellie broad -> per-frame full-image8conn min32 -> per-pixel50frame support>=10 -> full-image8conn min32 ->4conn enclosed holes<64',
        point_diagnostic='First failed stage for original Not-on ellipses; raw>=10 is a counterfactual diagnostic only',
        counts_use_all50_frames=True,original_frames=[0,2499],pixel_size_nm=117,frame_interval_s=.02,k=1,q_tolerance=1e-12,
        image_display_C1_ADU=[407,651],Nellie_rerun=False,refitting=False,reclassification=False,TARDIS=False))
    dump(out/'validation.json',dict(all_implementation_checks_passed=True,all2500_frame_full_image_min32_reconstruction_matches=True,
        all50_foreground_counts_input_support_post_min32_and_final_masks_pixel_identical=True,
        exact_ellipse_geometry_integration_checked_candidates=validation_samples,original_fits_labels_and_QC_unchanged=True,
        point_reasons_conserve_counts=True,fixed_pixel_independent_reference_reproduced=True,sources_unchanged=True,visual_review_pending=True))
    (out/'README.txt').write_text('TDKd3 T1 support50 unchanged-stage audit: R4/R3/R1 only.\n'
        'No Nellie rerun. Original broad labels>0 are relabeled across each FULL image with8-connectivity; min32 is reconstructed unchanged.\n'
        'All2500 frames and all50 count/support/post-min32/final arrays match saved outputs pixel-for-pixel. No ROI/image-edge component exemptions.\n'
        'Original Not-on points retain their center, SE, QC and label. Same k1 axis-aligned ellipse, closed foreground pixel squares, tangency included.\n'
        'Raw-broad count>=10 is a diagnostic counterfactual BEFORE first min32, not an alternative final segmentation.\n'
        'First-failure reasons are relative to that specified pipeline; removing a stage is not proven to yield correct mitochondria.\n'
        'Per-frame broad and first-min32 hits are descriptive; actual classification uses50-frame support masks. Different stages can touch different pixels in the ellipse.\n'
        'Fixed peak pixel profiles are separate from each moving refined localization/ellipse. Raw support absent means absent in saved Nellie masks, not proof of biological absence in C1.\n'
        'Hole filling only adds foreground; no post-min32 foreground can be removed. FinalOn hole-fill rescue is not forced into Not-on first-failure labels.\n'
        'C1/C2 identity registration and nominal SE coverage remain unverified. No causal/binding/cytoplasm inference, tracking, TARDIS or significance tests.\n')
    dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,output=str(out)))
    print('POINT_STAGE_SUMMARY',summary.to_json(orient='records'),flush=True)
    print('STAGE_AUDIT_READY',out,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qc',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
    if args.render_only:
        assert args.output is not None
        render_saved(args.output)
        (args.output/'render_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(args.output),'--render-only'])+'\n')
        print('STAGE_FIGURES_RENDERED',args.output);return
    args.qc=args.qc or Path((HERE/'latest_not_on_cluster_qc_output.txt').read_text().strip())
    out=args.output or HERE/'results'/('mask_stage_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(out/'regions').mkdir();(out/'source_snapshot').mkdir();(out/'checkpoints').mkdir()
    (HERE/'latest_mask_stage_audit_local.txt').write_text(str(out)+'\n')
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--qc',str(args.qc),'--output',str(out)])+'\n')
    with (out/'run.log').open('w') as log:
        a,b=sys.stdout,sys.stderr;sys.stdout=Tee(a,log);sys.stderr=Tee(b,log)
        try:run(args,out)
        except BaseException:
            import traceback;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=a,b


if __name__=='__main__':main()
