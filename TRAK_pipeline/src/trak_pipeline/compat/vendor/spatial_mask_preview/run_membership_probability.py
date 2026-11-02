"""Add conditional Gaussian position masses to the completed new-mask cache.

Original selection, fits and k=1 compatibility labels are retained verbatim.
The output is conditional on nominal model SE, diagonal Gaussian position
errors, a fixed W5/min32 mask, and unverified C1/C2 identity registration.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import tifffile

from gaussian_membership import position_mass
from run_full import FrameRows,selected
from run_preview import ROOT,HERE,STATUS,dump,fingerprint,number,flag,stats

plt.style.use('default')
EXTRA=['p_on_model','p_not_on_model','p_unknown_model','p_C1_zero_region_model','p_outside_FOV_model',
       'probability_partition_residual','numerical_tail_mass_bound','p_current_k1_label_model',
       'membership_score_status','membership_probability_interpretation','membership_XY_covariance_assumption']


def independent_mass(mask,zero,x,y,a,b):
    """Full sensor grid, without the production local search or interval helper."""
    h,w=mask.shape;xx=np.arange(w);yy=np.arange(h)
    wx=norm.cdf(xx+.5,loc=x,scale=a)-norm.cdf(xx-.5,loc=x,scale=a)
    wy=norm.cdf(yy+.5,loc=y,scale=b)-norm.cdf(yy-.5,loc=y,scale=b)
    weights=wy[:,None]*wx[None,:]
    outside=1-float(weights.sum())
    return (float(weights[mask&~zero].sum()),float(weights[~mask&~zero].sum()),float(weights[zero].sum())+outside)


def describe(points,frames,out):
    point=pd.DataFrame(points);frame=pd.DataFrame(frames)
    summaries=[]
    ranges=[('full',0,2500)]+[(f'frames_{a:04d}_{a+249:04d}',a,a+250) for a in range(0,2500,250)]
    for scope,lo,hi in ranges:
        pp=point[point.frame.between(lo,hi-1)]
        for label in ['all_selected','mito-compatible','non-mito-compatible','unknown']:
            g=pp if label=='all_selected' else pp[pp.label==label]
            r=dict(scope=scope,label=label,n=len(g))
            for field in ['p_on_model','p_not_on_model','p_unknown_model','p_current_k1_label_model']:
                r.update({field+'_'+k:v for k,v in stats(g[field]).items()})
            # These are descriptive ranges, not new QC or classification cuts.
            v=g.p_current_k1_label_model
            r.update(current_label_mass_below_0p5=int((v<.5).sum()),
                current_label_mass_0p5_to_0p95=int(((v>=.5)&(v<.95)).sum()),
                current_label_mass_at_least_0p95=int((v>=.95).sum()))
            summaries.append(r)
    pd.DataFrame(summaries).to_csv(out/'full_and_250_frame_probability_summary.csv',index=False)
    frame.to_csv(out/'per_frame_probability_summary.csv',index=False)
    edges=np.linspace(0,1,51);bins=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for label,color,name in [('mito-compatible','#159C76','Existing On'),('non-mito-compatible','#9854CA','Existing Not-on')]:
        g=point[point.label==label]
        for ax,field,title in zip(axes,['p_on_model','p_current_k1_label_model'],['Position mass inside mask','Mass supporting existing k=1 label']):
            counts,_=np.histogram(g[field],edges)
            assert counts.sum()==len(g)
            ax.stairs(counts,edges,color=color,lw=1.5,label=f'{name} (n={len(g):,})')
            for lo,hi,n in zip(edges[:-1],edges[1:],counts):bins.append(dict(label=label,field=field,low=lo,high=hi,count=int(n)))
            ax.set(xlim=(0,1),xlabel='Conditional model mass (0–1)',ylabel='Localizations',title=title)
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('TDKd3 | Fixed W5/min32 mask | Nominal diagonal-Gaussian position model',fontsize=11)
    fig.savefig(out/'membership_probability_histograms.png',dpi=180);plt.close(fig)
    pd.DataFrame(bins).to_csv(out/'probability_histogram_counts.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for label,color,name in [('mito-compatible','#159C76','Existing On'),('non-mito-compatible','#9854CA','Existing Not-on')]:
        g=point[point.label==label]
        for ax,field in zip(axes,['p_on_model','p_current_k1_label_model']):
            v=np.sort(g[field].to_numpy());ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,lw=1.5,label=name)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Conditional model mass',ylabel='Cumulative fraction');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('P(On) model mass');axes[1].set_title('Mass for existing assigned label')
    fig.suptitle('Descriptive CDF; mask/registration/SE calibration uncertainty not included',fontsize=11)
    fig.savefig(out/'membership_probability_CDF.png',dpi=180);plt.close(fig)
    full=[r for r in summaries if r['scope']=='full']
    dump(out/'summary.json',dict(groups=full,scored_points=len(point),
        unknown_mass_above_1e_minus_6=int((point.p_unknown_model>1e-6).sum()),
        maximum_unknown_mass=float(point.p_unknown_model.max()),
        maximum_absolute_partition_residual=float(point.partition_abs.max()),
        interpretation='Conditional fixed-mask model mass, not calibrated membership probability or binding evidence',
        descriptive_ranges_only_no_label_or_QC_changes=True))
    return full


def main(args):
    started=time.perf_counter();c=json.loads(args.config.read_text());source=Path(c['source_comparison_directory'])
    complete=json.loads((source/'COMPLETED.json').read_text());assert complete['all_checks_passed'] and complete['delivery_QC_passed']
    verified=json.loads((source/'source_verification.json').read_text());base=Path(verified['config']['data_directory'])
    assert c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    assert c['registration_status']==STATUS and not any(c[k] for k in ['change_existing_labels_or_QC','tracking','tardis'])
    maskdir=Path(verified['new']['mask_directory']);maskpath=maskdir/'clean_full_mask.npy';C1path=Path(verified['new']['C1_file'])
    tablepath=source/'all_ROI_candidates_mask_comparison.csv'
    manifest=json.loads((source/'output_fingerprints.json').read_text());match=[r for r in manifest if Path(r['path'])==tablepath];assert len(match)==1
    expected={str(tablepath):match[0]['sha256'],str(maskpath):verified['expected_sha256'][str(maskpath)],str(C1path):verified['expected_sha256'][str(C1path)]}
    for name in ['roi_mask.npy','frame_completed.npy','original_frame_numbers.npy','parameters.json','frame_windows_and_status.csv']:
        path=maskdir/name;expected[str(path)]=verified['expected_sha256'][str(path)]
    out=base/c['output_relative_directory']/('T1_C2_W5min32_Gaussian_mass_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(out/'source_snapshot').mkdir();(out/'checkpoints').mkdir()
    (HERE/'latest_membership_attempt.txt').write_text(str(out)+'\n')
    print('OUTPUT',out,flush=True)
    dump(out/'config.json',c);dump(out/'source_reference.json',dict(comparison_directory=str(source),mask_directory=str(maskdir),source_completion=complete))
    dump(out/'run_status.json',dict(status='source_checks',completed_frames=0))
    for name in ['run_membership_probability.py','gaussian_membership.py','test_gaussian_membership.py',
                 'run_full.py','run_preview.py','geometry.py','contracts.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve())])+'\n')
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_gaussian_membership.py','-v'],capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    before=[]
    for path,digest in expected.items():
        print('Checking source SHA256:',Path(path).name,flush=True);record=fingerprint(path);assert record['sha256']==digest;before.append(record)
    dump(out/'input_fingerprints_before.json',before)
    mask=np.load(maskpath,mmap_mode='r');assert mask.shape==(2500,684,428) and mask.dtype==bool
    assert np.load(maskdir/'frame_completed.npy',mmap_mode='r').all()
    np.testing.assert_array_equal(np.load(maskdir/'original_frame_numbers.npy',mmap_mode='r'),np.arange(2500))
    frames=[];points=[];scored=excluded=0;checked=0;source_fields=None
    auditpath=out/'all_ROI_candidates_with_model_probability.csv';selectedpath=out/'selected_points_with_model_probability.csv'
    with tablepath.open(newline='') as src,auditpath.open('w',newline='') as aout,selectedpath.open('w',newline='') as sout,tifffile.TiffFile(C1path) as tf:
        reader=FrameRows(src);source_fields=reader.fields;assert not set(source_fields)&set(EXTRA)
        audit=csv.DictWriter(aout,fieldnames=source_fields+EXTRA);chosen=csv.DictWriter(sout,fieldnames=source_fields+EXTRA)
        audit.writeheader();chosen.writeheader()
        assert tf.series[0].shape==mask.shape and tf.series[0].axes=='TYX' and len(tf.pages)==2500
        for frame in range(2500):
            rows=reader.for_frame(frame);zero=tf.pages[frame].asarray()==0;ff=[];kept=0
            for row in rows:
                extra={k:None for k in EXTRA}
                extra.update(membership_score_status='excluded_original_QC',membership_probability_interpretation=c['probability_interpretation'],
                    membership_XY_covariance_assumption=c['covariance_assumption'])
                assert selected(row)==flag(row,'spatial_selected')
                if selected(row):
                    x,y=number(row,'fit_x_px'),number(row,'fit_y_px');sx,sy=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
                    result=position_mass(mask[frame],zero,x,y,sx,sy,c['gaussian_window_sigma'])
                    label=row['new_pointwise_label'];assert label in ['mito-compatible','non-mito-compatible','unknown']
                    key={'mito-compatible':'p_on_model','non-mito-compatible':'p_not_on_model','unknown':'p_unknown_model'}[label]
                    extra.update(result,p_current_k1_label_model=result[key],membership_score_status='conditional_model_only')
                    if frame in c['representative_frames']:
                        independent=independent_mass(mask[frame],zero,x,y,sx,sy)
                        np.testing.assert_allclose([result['p_on_model'],result['p_not_on_model'],result['p_unknown_model']],independent,rtol=0,atol=1e-11);checked+=1
                    record=dict(frame=frame,label=label,**{k:result[k] for k in ['p_on_model','p_not_on_model','p_unknown_model']},
                        p_current_k1_label_model=result[key],partition_abs=abs(result['probability_partition_residual']))
                    ff.append(record);points.append(record);kept+=1
                saved=dict(row,**{k:('' if v is None else v) for k,v in extra.items()})
                audit.writerow(saved)
                if selected(row):chosen.writerow(saved)
            scored+=kept;excluded+=len(rows)-kept
            frames.append(dict(frame=frame,time_s=frame*.02,ROI_candidates=len(rows),scored=kept,QC_excluded=len(rows)-kept,
                mean_p_on_model=stats([r['p_on_model'] for r in ff])['mean'],
                mean_p_unknown_model=stats([r['p_unknown_model'] for r in ff])['mean'],
                existing_on=sum(r['label']=='mito-compatible' for r in ff),existing_not_on=sum(r['label']=='non-mito-compatible' for r in ff)))
            if (frame+1)%250==0:
                for stream in [aout,sout]:stream.flush()
                cp=dict(completed_frames=frame+1,scored_points=scored,QC_excluded=excluded,
                    block_scored=sum(r['scored'] for r in frames[-250:]),block_QC_excluded=sum(r['QC_excluded'] for r in frames[-250:]))
                dump(out/'checkpoints'/f'checkpoint_{frame+1:04d}.json',cp)
                pd.DataFrame(frames).to_csv(out/'per_frame_probability_summary.csv',index=False)
                dump(out/'run_status.json',dict(status='scoring',**cp));print('CHECKPOINT',json.dumps(cp),flush=True)
        assert reader.head is None
    assert scored==115013 and excluded==12046 and scored+excluded==127059
    full=describe(points,frames,out)
    # Independent reread proves all original cells, labels and selected IDs are unchanged.
    with tablepath.open(newline='') as src,auditpath.open(newline='') as dst,selectedpath.open(newline='') as sel:
        original=csv.DictReader(src);actual=csv.DictReader(dst);subset=csv.DictReader(sel);n=0
        for row in original:
            value=next(actual);assert {k:value[k] for k in source_fields}==row;n+=1
            if selected(row):
                assert next(subset)==value
                assert all(0<=float(value[k])<=1 for k in ['p_on_model','p_not_on_model','p_unknown_model','p_current_k1_label_model'])
            else:assert value['p_on_model']==value['p_not_on_model']==value['p_current_k1_label_model']==''
        assert n==127059 and next(actual,None) is None and next(subset,None) is None
    print('Checking source fingerprints after scoring...',flush=True)
    after=[fingerprint(path) for path in expected];dump(out/'input_fingerprints_after.json',after);assert before==after
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','matplotlib','tifffile','shapely']}))
    dump(out/'validation.json',dict(all_checks_passed=True,analytic_unit_tests_passed=True,independent_full_grid_points=checked,
        original_fields_fits_SE_QC_and_labels_preserved_verbatim=True,selected_IDs_unchanged=True,probability_partition_valid=True,
        input_fingerprints_unchanged=True,scored_points=scored,QC_excluded=excluded,
        registration_verified=False,mask_accuracy_validated=False,probabilities_empirically_calibrated=False,no_TARDIS=True))
    (out/'README.txt').write_text('Conditional Gaussian membership mass for TDKd3 W5/min32.\n'
        'Use refined center and position x_se_nm/y_se_nm divided by117, NOT PSF sx/sy. Assume diagonal normal position cloud.\n'
        'Sum products of normal-CDF differences over foreground pixel squares; original uncut full mask fixed.\n'
        'Original C1 zero pixels and outside-FOV probability stay unknown. Cell ROI only selects centers, not integration support.\n'
        'Local +/-8 sigma search includes full pixel squares; omitted mass <=4*Phi(-8)=2.49e-15 plus floating error.\n'
        'No Monte Carlo sampling, no mask dilation, no 50nm assumption, no label/QC changes.\n'
        'Existing On means k=1 ellipse intersects mask; p_on_model is position mass INSIDE mask. These events differ.\n'
        'p_current_k1_label_model selects p_on, p_not_on or p_unknown according to the retained label, as a diagnostic only.\n'
        'Bins below0.5, 0.5-0.95 and >=0.95 are descriptive conventions, not acceptance rules or new classifications.\n'
        'Nominal observed-Hessian SE; Gaussian approximation, XY covariance, segmentation and registration accuracy unvalidated.\n'
        'Not a calibrated posterior, empirical correct-label rate, binding probability, or cytoplasm proof. No TARDIS/tracking.\n')
    dump(out/'run_status.json',dict(status='complete',completed_frames=2500,scored_points=scored))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    dump(out/'COMPLETED.json',dict(all_checks_passed=True,output=str(out),scored_points=scored,QC_excluded=excluded,
        elapsed_s=time.perf_counter()-started))
    (HERE/'latest_membership_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',json.dumps(dict(output=str(out),groups=full)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args())
