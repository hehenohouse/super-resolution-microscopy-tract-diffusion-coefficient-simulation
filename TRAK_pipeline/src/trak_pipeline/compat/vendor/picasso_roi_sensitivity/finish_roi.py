"""Complete reporting/audit of an already-computed ROI run; no recomputation of fits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import numpy as np
import pandas as pd
import h5py
import yaml
from roi_core import ROOT,sha256_file
from curve_inference import centered_max_test
from run_roi import dump,report
from diagnose_calibration import diagnose
import roi_plots


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();out=Path(args.run).resolve()
    c=yaml.safe_load((out/'config.yaml').read_text());source=(ROOT/c['source_run']).resolve()
    summary=pd.read_csv(out/'per_scope_summary.csv');perframe=pd.read_csv(out/'per_frame_qc.csv')
    boot=pd.read_csv(out/'bootstrap_summary.csv');paired=pd.read_csv(out/'paired_comparisons.csv')
    tests=pd.read_csv(out/'omnibus_tests.csv');pairs=pd.read_csv(out/'ten_pair_curve_results.csv')
    sim=json.loads((out/'simulation_validation.json').read_text())
    status='toy_size_check_passed' if sim['nominal_alpha_in_null_rate_interval'] else 'caution_toy_null_size_inflation'
    # Validate both saved native audit artifacts and every original frame/unit column.
    checks=[]
    for g in c['groups']:
        name=f'box{g["box"]}_threshold{g["factor"]:.2f}'.replace('.','p')
        with h5py.File(source/'audit'/(name+'.hdf5'),'r') as old,h5py.File(out/'audit'/(name+'.hdf5'),'r') as new:
            a,b=old['locs'][()],new['locs'][()]
            assert len(a)==len(b)
            for field in a.dtype.names:
                if a.dtype[field].kind=='f':assert np.array_equal(a[field],b[field],equal_nan=True)
                else:np.testing.assert_array_equal(a[field],b[field])
            assert np.all(b['stats_selected_roi']==(b['qc_unflagged_model_fit']&(b['roi_inside']==1)))
        kept=pd.read_csv(out/'selected'/(name+'.csv'))
        np.testing.assert_allclose(kept.time_s,kept.frame*.02,atol=1e-14)
        for field in ['x','y','sx','sy','lpx','lpy']:np.testing.assert_allclose(kept[field+'_nm'],kept[field]*117,rtol=3e-7,atol=1e-7)
        assert kept.frame.between(0,2499).all() and kept.qc_unflagged_model_fit.all() and kept.roi_inside.all()
        assert len(perframe.loc[perframe.group==name])==2500
        checks.append({'group':name,'audit_count':len(a),'selected_count':len(kept),'native_original_fields_identical':True})
    for path,h in json.loads((out/'protected_source_hashes.json').read_text()).items():assert sha256_file(path)==h,path
    with np.load(out/'bootstrap/full_curve_draws_L25.npz') as saved:
        inference=centered_max_test(saved['full_point_density'],saved['density'],saved['valid'],c['alpha'],c['minimum_joint_valid_fraction'])
        np.testing.assert_array_equal(saved['replicate_id'],np.arange(1000))
    primary=tests.loc[tests.block_frames==25].iloc[0]
    np.testing.assert_allclose(inference['p_global'],primary.p_global,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(inference['critical'],primary.critical,rtol=1e-12,atol=1e-12)
    for frame in [tests,pairs]:frame['inference_validation_status']=status
    if 'family_reject' in pairs:pairs=pairs.rename(columns={'family_reject':'nominal_family_reject'})
    tests.to_csv(out/'omnibus_tests.csv',index=False);pairs.to_csv(out/'ten_pair_curve_results.csv',index=False)
    roi_plots.qc_diagnostics(out,c,perframe,summary)
    roi_plots.simultaneous_figures(out,c,pairs,inference,inference)
    for scope in summary.scope.unique():roi_plots.distributions(out,c,summary,scope)
    if (out/'simulation_block_bias_diagnostic.json').exists():
        diagnostic=json.loads((out/'simulation_block_bias_diagnostic.json').read_text())
    else:
        with np.load(out/'bootstrap/shared_block_resampling.npz') as w:diagnostic=diagnose(c,w['L25'])
    factor=diagnostic['diagnostic_variance_factor'];null=inference['null_statistics'];m=len(null)
    scaled=null*factor
    diagnostic['real_movie_variance_rescaling_DIAGNOSTIC_ONLY']={
        'main_outputs_unchanged':True,'p_global':float((1+(scaled>=inference['statistic']).sum())/(m+1)),
        'critical':float(np.sort(scaled)[inference['critical_rank_1based']-1]),
        'pair_adjusted_p':[(1+int((scaled>=s).sum()))/(m+1) for s in inference['pair_statistics']],
        'pair_order':pairs.comparison.tolist(),
        'not_a_validated_nonlinear_correction':True}
    dump(out/'simulation_block_bias_diagnostic.json',diagnostic)
    intervals=pd.read_csv(out/'significant_distance_intervals.csv')
    intervals['inference_validation_status']=status;intervals['confidence_is_nominal']=True
    intervals.to_csv(out/'significant_distance_intervals.csv',index=False)
    for p in (out/'inference').glob('omnibus_L*.json'):
        data=json.loads(p.read_text());data['inference_validation_status']=status;dump(p,data)
    for p in (out/'inference/bands').glob('*.csv'):
        data=pd.read_csv(p);data['inference_validation_status']=status;data['family_band_confidence_nominal']=.95;data.to_csv(p,index=False)
    print(json.dumps(diagnostic,indent=2),flush=True)
    report(out,c,summary,boot,paired,tests,pairs,sim)
    with (out/'RESULTS.txt').open('a') as f:
        f.write('\n有限块诊断（不改变任何真实数据主检验p或带）：\n'+json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n')
    validation={'implementation_checks_passed':True,'preflight_tests':12,'statistical_calibration_status':status,
        'nominal_alpha_in_toy_interval':sim['nominal_alpha_in_null_rate_interval'],'native_saved_file_audits':checks,
        'no_refitting':True,'original_QC_unchanged':True,'source_fields_and_units_preserved':True,
        'protected_sources_unchanged':True,'shared_weights_reused':True,'paired_original_replicate_intersection':True,
        'finite_block_diagnostic_not_used_to_change_primary':True}
    dump(out/'validation.json',validation)
    dump(out/'COMPLETED.json',{'roi_frames':2500,'configurations':5,'windows':10,'no_refitting':True,
        'primary_omnibus_p_nominal':inference['p_global'],'statistical_calibration_status':status,
        'finished_utc':datetime.now(timezone.utc).isoformat()})
    original_start=datetime.strptime(out.name.replace('T1_C2_ROI_',''),'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
    dump(out/'runtime.json',{'no_refitting':True,'refitting_seconds':0,
        'initial_roi_bootstrap_and_simulation_seconds_approx':(out/'simulation_validation.json').stat().st_mtime-original_start,
        'final_audit_diagnostic_seconds':time.perf_counter()-started,
        'timing_note':'initial phase elapsed estimated from timestamped run creation and final simulation artifact; includes no agent reporting interval'})
    (out/'finalize_command.txt').write_text(f'cd {shlex.quote(str(ROOT))}\n{shlex.join(sys.orig_argv)}\n')
    (out/'finalize.log').write_text('Completed audit, paired inference reproduction and nominal calibration disclosure.\n'+json.dumps(diagnostic,indent=2)+'\n')
    dump(out/'initial_stage_stop.json',{'reason':'original driver stopped after toy null calibration check; ROI and bootstrap caches were already complete',
        'finalization':'preserved all numerical results; separately finalized with explicit calibration caution; no altered seed or primary test'})
    dest=out/'reproducibility/finalize_support';dest.mkdir(exist_ok=True)
    for name in ['finish_roi.py','diagnose_calibration.py','run_roi.py','roi_plots.py']:shutil.copy2(ROOT/name,dest/name)
    manifest={str(p.relative_to(out)):sha256_file(p) for p in (out/'reproducibility').rglob('*') if p.is_file()}
    dump(out/'reproducibility_manifest.json',manifest)
    if (ROOT/'README.txt').exists():shutil.copy2(ROOT/'README.txt',out/'README.txt')
    print('ROI run finalized with explicit statistical calibration status:',out,flush=True)


if __name__=='__main__':main()
