#!/usr/bin/env python3
"""Recompute group-specific TARDIS from immutable W5/min32 spatial caches."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import pandas as pd
import tifffile

from mito_core import (ROOT,PROJECT,GROUPS,LABELS,calculate_histograms,
    independent_histogram_check,legacy_frame_histograms,range_histograms,
    decompose_roi,scalar_row,distribution_table,bootstrap_roi,shared_resampling_plan,
    finite_ci,paired_difference,paired_curve_summary,tail_diagnostics,scopes)
from roi_core import load_imagej_roi,read_mat_roi,points_in_polygon
import plots


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,v):self.stream.write(v);self.log.write(v);self.log.flush();return len(v)
    def flush(self):self.stream.flush();self.log.flush()


def jsonable(v):
    if isinstance(v,np.ndarray):return [jsonable(x) for x in v]
    if isinstance(v,np.generic):return jsonable(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
    return v


def dump(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fingerprint(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    st=p.stat()
    return dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h.hexdigest())


def bool_values(series):
    assert series.isin(['True','False','true','false','1','0']).all()
    return series.isin(['True','true','1'])


def snapshot(out):
    snap=out/'source_snapshot'
    for directory in ['mito_tardis','picasso_roi_sensitivity','picasso_full_sensitivity','picasso_pilot',
                      'TARDIS/path_irrelevant_python/tardis_tau1']:
        target=snap/directory;target.mkdir(parents=True,exist_ok=True)
        for file in (PROJECT/directory).glob('*.py'):shutil.copyfile(file,target/file.name)
    shutil.copyfile(ROOT/'config.json',snap/'mito_tardis/config.json')
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'requirements.lock.txt')


def bootstrap(out,c,hist,observed):
    width=c['distance_max_um']/c['distance_bins'];mid=(np.arange(c['distance_bins'])+.5)*width
    scope_list=scopes(c);scalar_rows=[];paired_rows=[];valid_rows=[];control_rows=[]
    path=(ROOT/c['shared_weights']).resolve()
    shutil.copyfile(path,out/'bootstrap/shared_block_resampling.npz')
    with np.load(path) as w:
        weights={length:w[f'L{length}'].copy() for length in c['bootstrap']['block_lengths_frames']}
    for length,w in weights.items():
        np.testing.assert_array_equal(w,shared_resampling_plan(c,length))
        print(f'BOOTSTRAP L{length}: shared 1000 replicates, group-specific re-normalization and beta',flush=True)
        group_draws=[];controls=[]
        for g,group in enumerate(GROUPS):
            draws,control=bootstrap_roi(hist['total'][g],hist['background'][g],c,length,w)
            group_draws.append(draws);controls.append(control)
            for (scope,s,e),d,ctl in zip(scope_list,draws,control):
                point=observed[group,scope]
                control_rows.append(dict(group=group,scope=scope,block_frames=length,
                    possible_block_internal_frame_pairs=e-s-(e-s)//length,
                    original_possible_frame_pairs=e-s-1,**scalar_row(ctl),
                    beta_change_from_full_pairs=ctl['beta'][0]-point['beta'][0],
                    signed_area_change_from_full_pairs=ctl['signed_area'][0]-point['signed_area'][0],
                    max_normalized_shape_change=float(np.max(np.abs(ctl['shape'][0]-point['shape'][0])))/width))
                for metric in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                    lo,hi,n,f=finite_ci(d[metric],c['minimum_joint_valid_fraction'],np.isfinite(point[metric][0]))
                    scalar_rows.append(dict(group=group,scope=scope,block_frames=length,metric=metric,
                        point_estimate=point[metric][0],exploratory_low=lo,exploratory_high=hi,
                        valid_replicates=n,valid_fraction=f,coverage_validated=False))
        shape=np.stack([np.stack([d['shape']/width for d in ds]) for ds in group_draws])
        shape_valid=np.stack([np.stack([d['shape_valid'] for d in ds]) for ds in group_draws])
        np.savez_compressed(out/'bootstrap'/f'shape_draws_L{length}.npz',density=shape,
            valid=shape_valid,group_order=np.array(GROUPS),scope_order=np.array([x[0] for x in scope_list]),
            replicate_id=np.arange(c['bootstrap']['replicates']),distance_mid_um=mid,
            block_point_shape=np.stack([np.stack([d['shape'][0]/width for d in ds]) for ds in controls]))
        metric_arrays={group+'__'+key:np.stack([d[key] for d in group_draws[g]])
                       for g,group in enumerate(GROUPS)
                       for key in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']}
        np.savez_compressed(out/'bootstrap'/f'metric_draws_L{length}.npz',**metric_arrays,
                            replicate_id=np.arange(c['bootstrap']['replicates']))
        for j,(scope,s,e) in enumerate(scope_list):
            a,b=group_draws[0][j],group_draws[1][j]
            pa,pb=observed['On',scope],observed['Not-on',scope]
            columns=dict(distance_mid_um=mid)
            for field in ['total','background','inter','intra','shape']:
                paired=paired_curve_summary(a[field]/width,b[field]/width,
                    pa[field][0]/width,pb[field][0]/width,c['minimum_joint_valid_fraction'])
                for key,value in [('On_low','On_low'),('On_high','On_high'),
                                  ('Not_on_low','Not_on_low'),('Not_on_high','Not_on_high')]:
                    group,ending=key.rsplit('_',1)
                    columns[group+'_'+field+'_'+ending]=paired[value]
                columns[field+'_point_difference']=(pa[field][0]-pb[field][0])/width
                columns[field+'_difference_low']=paired['low'];columns[field+'_difference_high']=paired['high']
                valid_rows.append(dict(scope=scope,block_frames=length,field=field,
                    On_valid_count=int(np.isfinite(a[field]).all(1).sum()),
                    Not_on_valid_count=int(np.isfinite(b[field]).all(1).sum()),
                    common_valid_count=paired['common_count'],common_valid_fraction=paired['common_fraction'],
                    exploratory_range_available=paired['available']))
                if field=='shape':
                    np.savez_compressed(out/'bootstrap'/f'paired_shape_{scope}_L{length}.npz',
                        difference=paired['difference'],common_valid=paired['common'],
                        replicate_id=np.arange(c['bootstrap']['replicates']),distance_mid_um=mid)
            pd.DataFrame(columns).to_csv(out/'bootstrap'/f'paired_bands_{scope}_L{length}.csv',index=False)
            for metric in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                difference,good,(lo,hi,n,f)=paired_difference(a[metric],b[metric],c['minimum_joint_valid_fraction'],
                    np.isfinite(pa[metric][0]) and np.isfinite(pb[metric][0]))
                paired_rows.append(dict(scope=scope,block_frames=length,metric=metric,
                    point_difference=pa[metric][0]-pb[metric][0],exploratory_low=lo,exploratory_high=hi,
                    common_valid_count=n,common_valid_fraction=f,paired_by_original_replicate_id=True,
                    coverage_validated=False))
        print(f'BOOTSTRAP L{length} complete',flush=True)
    pd.DataFrame(scalar_rows).to_csv(out/'bootstrap_scalar_summary.csv',index=False)
    pd.DataFrame(paired_rows).to_csv(out/'paired_metric_differences.csv',index=False)
    pd.DataFrame(valid_rows).to_csv(out/'bootstrap_curve_validity.csv',index=False)
    pd.DataFrame(control_rows).to_csv(out/'block_boundary_exclusion_controls.csv',index=False)


def write_report(out,c,summary,effects):
    full=summary[summary.scope=='full']
    lines=[f'{c.get("sample_name", "TDKd3")} {c.get("mask_label","W5/min32")}: On / Not-on group-specific TARDIS',
        'Frames 0-2499; tau=20 ms; 117 nm/pixel; 300 bins over 0-3 um.',
        'Existing k=1 ellipse-intersection labels retained, no new probability filtering or weighting.',
        'Both endpoints in same group and ROI; ordered same-frame nonzero background from that group.',
        'Pooled count normalization; beta LS on 2.1-3 um clipped [0,1], raw beta saved.',
        'Tail support >=100 Total and >=100 ordered B pairs is a minimum diagnostic, not independent sample size.',
        'Signed shape normalized on 0-3 um; negative bins retained, not a nonnegative PDF.',
        '',full.to_string(index=False),'','On - Not-on effect sizes:',json.dumps(jsonable(effects),indent=2),
        '', 'Unavailable time-window derived estimates:']
    for row in summary[~summary.shape_valid].itertuples():
        lines.append(f'{row.group} {row.scope}: {row.shape_status}; T/B tail counts={row.total_tail_count}/{row.background_tail_count}; signed area={row.signed_area}')
    lines += ['', 'Exploratory paired time-block resampling:',
        '1000 shared draws, stratified every250 frames, L25(0.5 s) primary and L10/L50 sensitivity; exact saved weights reused.',
        'Only source-block-internal true adjacent pairs counted; no block-join adjacency.',
        'Each draw re-normalizes its own Total/B, fits beta and normalizes Intra.',
        'Two-group comparisons use the same original replicate-ID intersection; >=95% valid and valid point estimates required.',
        'Ranges are descriptive 2.5-97.5 percentiles, pointwise, with unvalidated coverage.',
        'No formal p values, no maxT test and no significance conclusion; earlier null calibration failed.',
        '', 'Interpretation:',
        'On means spatial compatibility, not proven binding; Not-on does not establish cytoplasmic identity.',
        'Fixed C1/C2 identity registration remains unverified. SE uses nominal manufacturer calibration.',
        'Dynamic masks, uncertainty-dependent selection and region boundary loss can change the pair statistics.',
        'Mask IoU and far-tail shape agreement are diagnostics, not validation of background exchangeability.',
        'Statistics condition on both endpoints in the same group; region/ROI-crossing pairs are excluded.',
        'Cross-group all-pair histograms are diagnostic only, not trajectories or transition rates.',
        'No refitting, resegmentation, drift correction, deconvolution, uncertainty subtraction, diffusion or biological-state fitting.']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')


def analyze_groups(c,out,table,frame):
    """Run unchanged group-specific TARDIS numerics on a verified input table."""
    def progress(n,total,background,cross):
        cp=dict(completed_frames=n,On_points=int(frame.On[:n].sum()),Not_on_points=int(frame['Not-on'][:n].sum()),
            QC_excluded=int(frame.QC_excluded[:n].sum()),On_total_pairs=int(total[0,:n].sum()),
            Not_on_total_pairs=int(total[1,:n].sum()))
        dump(out/'checkpoints'/f'frames_{n:04d}.json',cp)
        dump(out/'run_status.json',dict(status='pair_histograms',**cp))
        print('CHECKPOINT',json.dumps(cp),flush=True)
    hist=calculate_histograms(table,c,progress)
    np.savez_compressed(out/'pair_histograms/per_frame_raw_counts.npz',**hist,
                        group_order=np.array(GROUPS),original_frame=np.arange(2500))
    independent=independent_histogram_check(table,hist,c,[0,1,49,200,249,625,1200,1250,1875,2000,2450,2498,2499])
    for g,group in enumerate(GROUPS):
        old=table[table.tardis_group==group].rename(columns={'fit_x_px':'x','fit_y_px':'y'})
        th,bh,*_=legacy_frame_histograms(old,c)
        np.testing.assert_array_equal(th,hist['total'][g]);np.testing.assert_array_equal(bh,hist['background'][g])
    print('Raw histograms match existing TARDIS implementation and independent enumeration',flush=True)
    observed={};summary=[];width=c['distance_max_um']/c['distance_bins'];mid=(np.arange(300)+.5)*width
    for g,group in enumerate(GROUPS):
        th,bh=hist['total'][g],hist['background'][g]
        for scope,s,e in scopes(c):
            tc,bc=range_histograms(th,bh,s,e);d=decompose_roi(tc,bc,c);observed[group,scope]=d
            folder=out/'distribution'/group;folder.mkdir(exist_ok=True)
            distribution_table(tc,bc,d,c).to_csv(folder/(scope+'.csv'),index=False)
            subset=table[(table.tardis_group==group)&table.frame.between(s,e-1)]
            row=dict(group=group,scope=scope,start_frame=s,stop_exclusive=e,points=len(subset),
                total_pair_count=int(tc.sum()),background_ordered_nonzero_count=int(bc.sum()),
                background_exact_zero_removed=int(hist['background_zero_removed'][g,s:e].sum()),
                **scalar_row(d),**tail_diagnostics(tc,bc,th,bh,s,e,c))
            for key in ['x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']:
                for name,q in [('q10',.1),('median',.5),('q90',.9)]:row[key+'_'+name]=subset[key].quantile(q)
            summary.append(row)
        # Segments lose exactly the nine pairs crossing 249->250 etc.; nothing else.
        pieces=sum(range_histograms(th,bh,s,s+250)[0] for s in range(0,2500,250))
        np.testing.assert_array_equal(pieces+th[np.arange(249,2499,250)].sum(0),th.sum(0))
    summary=pd.DataFrame(summary);summary.to_csv(out/'per_scope_summary.csv',index=False)
    effects=[]
    for scope,s,e in scopes(c):
        a,b=observed['On',scope],observed['Not-on',scope]
        diff=(a['shape'][0]-b['shape'][0])/width;available=np.isfinite(diff).all()
        main=mid<.10;tail=(mid>=.10)&(mid<.25)
        effects.append(dict(scope=scope,shape_difference_available=available,
            max_abs_density_difference=float(np.max(abs(diff))) if available else np.nan,
            max_abs_difference_distance_um=float(mid[np.argmax(abs(diff))]) if available else np.nan,
            integrated_abs_signed_density_difference=float(abs(diff).sum()*width) if available else np.nan,
            normalized_main_signed_area_difference=float(diff[main].sum()*width) if available else np.nan,
            normalized_tail_signed_area_difference=float(diff[tail].sum()*width) if available else np.nan))
    pd.DataFrame(effects).to_csv(out/'descriptive_shape_differences.csv',index=False)
    dump(out/'raw_histogram_validation.json',dict(independent_group_frames_checked=independent,
        legacy_all_bins_all_frames_match=True,within_plus_cross_equals_all_selected=True,
        segments_exclude_exactly_the_nine_boundary_frame_pairs=True))
    print('FULL POINT ESTIMATES',summary[summary.scope=='full'][['group','points','beta','signed_area','tail_main_ratio','shape_status']].to_json(orient='records'),flush=True)
    bootstrap(out,c,hist,observed)
    plots.all_figures(out,c,summary,frame,effects)
    write_report(out,c,summary,effects)
    return independent


def run(c,config_path,out,source,comparison,verified):
    started=time.perf_counter();maskdir=Path(verified['new']['mask_directory'])
    base=Path(verified['config']['data_directory'])
    dump(out/'config.json',c);dump(out/'RULES_BEFORE_RUN.json',c)
    dump(out/'run_status.json',dict(status='source_verification',completed_frames=0))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path.resolve())])+'\n')
    snapshot(out)
    dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    dump(out/'source_reference.json',dict(membership_directory=str(source),comparison_directory=str(comparison),
        mask_directory=str(maskdir),classification_label='new_pointwise_label',
        probability_scores_used_for_filtering_or_weighting=False,
        original_all_candidates_kept_at=str(source/'all_ROI_candidates_with_model_probability.csv')))
    membership_manifest=json.loads((source/'output_fingerprints.json').read_text())
    expected={r['path']:r['sha256'] for r in membership_manifest
              if Path(r['path']).name in ['all_ROI_candidates_with_model_probability.csv',
                  'selected_points_with_model_probability.csv','validation.json','config.json']}
    raw=[Path(verified['new']['localization_directory'])/'all_ROI_target_fits.csv',
         Path(verified['new']['C2_file']),base/verified['config']['roi_filename'],Path(verified['config']['mat_file'])]
    raw += [maskdir/name for name in ['clean_full_mask.npy','roi_mask.npy','roi_vertices_xy.npy',
             'frame_completed.npy','original_frame_numbers.npy','parameters.json','frame_windows_and_status.csv','per_frame_statistics.csv']]
    for path in raw:expected[str(path)]=verified['expected_sha256'][str(path)]
    mask_manifest=json.loads((maskdir/'output_fingerprints.json').read_text())
    expected[str(maskdir/'adjacent_pair_metrics.csv')]=mask_manifest['adjacent_pair_metrics.csv']
    weightpath=(ROOT/c['shared_weights']).resolve()
    expected[str(weightpath)]=fingerprint(weightpath)['sha256']
    before=[]
    for path,h in expected.items():
        print('SOURCE',Path(path).name,flush=True);record=fingerprint(path)
        assert record['sha256']==h,path;before.append(record)
    dump(out/'input_fingerprints_before.json',before)
    mask=np.load(maskdir/'clean_full_mask.npy',mmap_mode='r')
    assert mask.shape==(2500,684,428) and mask.dtype==bool
    assert np.load(maskdir/'frame_completed.npy',mmap_mode='r').all()
    np.testing.assert_array_equal(np.load(maskdir/'original_frame_numbers.npy'),np.arange(2500))
    roi=load_imagej_roi(base/verified['config']['roi_filename'])
    mat,record=read_mat_roi(verified['config']['mat_file'],verified['config']['mat_record'])
    np.testing.assert_array_equal(roi,mat)
    np.testing.assert_array_equal(roi,np.load(maskdir/'roi_vertices_xy.npy'))
    assert len(roi)==454
    pd.DataFrame(roi,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    with tifffile.TiffFile(verified['new']['C2_file']) as tf:
        assert tf.series[0].axes=='TYX' and tf.series[0].shape==(2500,684,428)
        assert len(tf.pages)==2500 and tf.pages[0].dtype==np.dtype('uint16')
    dump(out/'coordinate_validation.json',dict(ROI_MAT_mask_vertices_exactly_equal=True,vertices=454,
        C2_shape=[2500,684,428],axes='TYX',registration_verified=False,
        registration_status=c['registration_status']))
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_mito_tardis.py')
    with (out/'tests.log').open('w') as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert tests.wasSuccessful(),'Preflight tests failed'
    print('Preflight tests passed',flush=True)
    fields=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm',
            'photons_e','bg_e_per_pixel','sx_px','sy_px','qc_unflagged_model_fit',
            'fitted_center_inside_ROI','spatial_selected','new_pointwise_label',
            'p_on_model','p_not_on_model','spatial_registration_status']
    audit=pd.read_csv(source/'all_ROI_candidates_with_model_probability.csv',usecols=fields,dtype=str,keep_default_na=False)
    assert len(audit)==c['expected_candidates']
    keep=bool_values(audit.qc_unflagged_model_fit)&bool_values(audit.fitted_center_inside_ROI)
    np.testing.assert_array_equal(keep,bool_values(audit.spatial_selected))
    assert keep.sum()==c['expected_selected'] and (~keep).sum()==c['expected_QC_excluded']
    assert audit.candidate_index.is_unique and audit.spatial_registration_status.eq(c['registration_status']).all()
    table=audit[keep].copy();table['tardis_group']=table.new_pointwise_label.map(LABELS)
    for group,n in c['expected_group_counts'].items():assert table.tardis_group.eq(group).sum()==n
    numeric=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px','p_on_model','p_not_on_model']
    # Python's correctly rounded float parser preserves all source double values.
    # pd.to_numeric may round long decimal strings by one ULP.
    for key in numeric:
        table[key]=table[key].map(int if key in ['candidate_index','frame'] else float)
    assert table.frame.between(0,2499).all() and np.isfinite(table[numeric]).all().all()
    np.testing.assert_allclose(table.time_s,table.frame*.02,rtol=0,atol=1e-12)
    assert points_in_polygon(table[['fit_x_px','fit_y_px']].to_numpy(),roi).all()
    table.to_csv(out/'statistical_input.csv',index=False)
    # Full original selected rows, including fit states/QC reasons, are archived verbatim.
    shutil.copyfile(source/'selected_points_with_model_probability.csv',out/'selected_original_rows.csv')
    assert fingerprint(out/'selected_original_rows.csv')['sha256']==fingerprint(source/'selected_points_with_model_probability.csv')['sha256']
    frame=pd.DataFrame(dict(frame=np.arange(2500),time_s=np.arange(2500)*.02))
    for group in GROUPS+['unknown']:
        frame[group]=table[table.tardis_group==group].groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy()
    frame['QC_excluded']=audit[~keep].assign(frame=lambda d:d.frame.astype(int)).groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy()
    frame['ROI_candidates']=audit.assign(frame=lambda d:d.frame.astype(int)).groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy()
    np.testing.assert_array_equal(frame[GROUPS+['unknown','QC_excluded']].sum(1),frame.ROI_candidates)
    m=pd.read_csv(maskdir/'per_frame_statistics.csv');np.testing.assert_array_equal(m.frame,frame.frame)
    for key in ['ROI_area_pixel2','full_area_pixel2','temporal_applied']:frame['mask_'+key]=m[key].to_numpy()
    frame.to_csv(out/'per_frame_counts.csv',index=False)
    shutil.copyfile(maskdir/'adjacent_pair_metrics.csv',out/'mask_adjacent_pair_metrics.csv')

    independent=analyze_groups(c,out,table,frame)
    print('Checking protected source fingerprints after analysis',flush=True)
    after=[fingerprint(path) for path in expected];assert before==after
    dump(out/'input_fingerprints_after.json',after)
    dump(out/'field_units.json',dict(frame='original zero-based frame',time_s='seconds',
        fit_x_px='native C2 camera pixels',fit_y_px='native C2 camera pixels',
        x_se_nm='nominal local-model position SE, nm',y_se_nm='nominal local-model position SE, nm',
        photons_e='nominal photoelectron-equivalent fitted signal',bg_e_per_pixel='nominal electron-equivalent/pixel',
        sx_px='Gaussian PSF sigma, pixels',sy_px='Gaussian PSF sigma, pixels',
        raw_histograms='all-pair counts, not independent observations',
        distribution_density='1/um',beta='dimensionless background weight, not a biological fraction',
        signed_area='dimensionless; sum of signed Intra mass on0-3um',
        shape_density='signed density 1/um, not a nonnegative PDF',
        exploratory_ranges='paired time-block pointwise percentiles, no validated coverage'))
    dump(out/'validation.json',dict(implementation_checks_passed=True,preflight_tests=tests.testsRun,
        independent_group_frames_checked=independent,original_selected=115013,QC_excluded=12046,
        group_counts=c['expected_group_counts'],original_full_rows_archived_verbatim=True,
        probability_scores_not_used=True,input_fingerprints_unchanged=True,
        ROI_MAT_vertices_equal=True,legacy_histograms_all_frames_equal=True,
        shared_saved_weights_reused=True,paired_original_replicate_ID_intersections=True,
        no_cross_gap_segment_or_resampling_block_join_pairs=True,
        bootstrap_intervals_exploratory=True,formal_p_values_computed=False,
        registration_verified=False,absolute_localization_accuracy_validated=False,
        background_model_validated=False))
    dump(out/'runtime.json',dict(elapsed_s=time.perf_counter()-started,refitting_seconds=0))
    dump(out/'run_status.json',dict(status='complete',completed_frames=2500))
    dump(out/'COMPLETED.json',dict(implementation_checks_passed=True,delivery_visual_QC_pending=True,
        frames=2500,groups=GROUPS,selected_points=115013,formal_inference=False,output=str(out)))
    (ROOT/'latest_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',str(out),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=json.loads(args.config.read_text());source=Path(c['membership_directory'])
    complete=json.loads((source/'COMPLETED.json').read_text())
    assert complete['all_checks_passed'] and complete['delivery_QC_passed']
    reference=json.loads((source/'source_reference.json').read_text());comparison=Path(reference['comparison_directory'])
    verified=json.loads((comparison/'source_verification.json').read_text())
    assert c['mask_version']==verified['config']['new_mask_version']
    assert c['frames']==2500 and c['tau_frames']==1 and c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    assert not any(c[k] for k in ['refitting','resegmenting','tracking','probability_filtering',
                                 'uncertainty_deconvolution','diffusion_correction','biological_state_fitting','formal_inference'])
    base=Path(verified['config']['data_directory'])
    out=base/c['output_relative_directory']/('T1_C2_W5min32_On_NotOn_k1_tau1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    for directory in ['','distribution','pair_histograms','bootstrap','figures','checkpoints']:(out/directory).mkdir(parents=True,exist_ok=False)
    (ROOT/'latest_attempt.txt').write_text(str(out)+'\n')
    with (out/'run.log').open('w') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);run(c,args.config,out,source,comparison,verified)
        except BaseException:
            dump(out/'FAILED.json',dict(failed_at_utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()))
            raise
        finally:sys.stdout,sys.stderr=stdout,stderr
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
                                       and p.name not in ['output_fingerprints.json','COMPLETED.json']])


if __name__=='__main__':main()
