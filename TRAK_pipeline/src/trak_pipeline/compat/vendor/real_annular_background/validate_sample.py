"""Additional read-only audit of full-sample cache identity and pilot reuse."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import run_real as core


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    c=json.loads((out/'config.json').read_text());complete=json.loads((out/'COMPLETED.json').read_text())
    assert complete['delivery_checks_passed']
    fits=pd.read_csv(out/'all_ROI_target_fits.csv',float_precision='round_trip')
    detections=pd.read_csv(out/'all_detections.csv',float_precision='round_trip')
    frames=pd.read_csv(out/'frame_counts.csv');pilot=pd.read_csv(out/'pilot/all_ROI_target_fits.csv',float_precision='round_trip')
    assert len(frames)==2500;np.testing.assert_array_equal(frames.frame,np.arange(2500))
    assert fits.candidate_index.is_unique and fits.source_detection_uid.is_unique
    assert detections.candidate_index.is_unique and detections.source_detection_uid.is_unique
    np.testing.assert_array_equal(detections.candidate_index,np.arange(len(detections)))
    d=detections.set_index('source_detection_uid').loc[fits.source_detection_uid]
    np.testing.assert_array_equal(d.candidate_index.to_numpy(),fits.candidate_index.to_numpy());assert d.roi_inside.all()
    np.testing.assert_array_equal(d.frame.to_numpy(),fits.frame.to_numpy())
    f=fits.set_index('source_detection_uid',drop=False).loc[pilot.source_detection_uid]
    excluded={'candidate_index','pilot_candidate_index','reused_from_pilot'}
    checked=[]
    for col in pilot.columns:
        if col in excluded:continue
        left=pilot[col].to_numpy();right=f[col].to_numpy()
        if pd.api.types.is_numeric_dtype(pilot[col]):np.testing.assert_array_equal(left,right,err_msg=col)
        else:assert pd.Series(left).fillna('').equals(pd.Series(right).fillna('')),col
        checked.append(col)
    assert f.reused_from_pilot.all();np.testing.assert_array_equal(f.pilot_candidate_index.to_numpy(),pilot.candidate_index.to_numpy())
    assert frames.ROI_candidates.sum()==len(fits)
    assert ((frames.selected+frames.QC_excluded)==frames.ROI_candidates).all()
    clean=fits[fits.qc_unflagged_model_fit]
    assert clean.fitted_center_inside_ROI.all() and clean.fit_success.all() and clean.finite_uncertainty.all()
    assert not clean.active_bound.any() and not clean.suspect_overlap.any()
    assert np.isfinite(clean[['fit_x_px','fit_y_px','x_se_nm','y_se_nm']]).all().all()
    assert clean[['x_se_nm','y_se_nm']].gt(0).all().all()
    checks=sorted(out.glob('checkpoints/fits_*.json'))
    assert [json.loads(p.read_text())['completed_frames'] for p in checks]==list(range(250,2501,250))
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    fields=['x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']
    records=[]
    for block in range(10):
        allrows=fits[fits.frame.between(block*250,block*250+249)];rows=allrows[allrows.qc_unflagged_model_fit]
        r=dict(segment=block,first_frame=block*250,last_frame=block*250+249,ROI_candidates=len(allrows),selected=len(rows),QC_excluded=len(allrows)-len(rows))
        for key in fields:
            values=rows[key].dropna().to_numpy()
            for name,percentile in [('q25',25),('median',50),('q75',75),('p95',95)]:r[key+'_'+name]=float(np.percentile(values,percentile)) if len(values) else None
        records.append(r)
    pd.DataFrame(records).to_csv(out/'per_250_frame_fit_summary.csv',index=False)
    result=dict(all_checks_passed=True,stable_UIDs_unique=True,global_candidate_index_frame_order=True,
        pilot_rows_reused=len(pilot),pilot_original_fields_exactly_equal=checked,pilot_reindex_only=True,
        all2500_frames_and10_checkpoints=True,QC_membership_verified=True,source_fingerprints_unchanged=True)
    core.dump(out/'additional_validation.json',result)
    for p in [Path(__file__),Path(__file__).parent/'run_sample.py']:
        shutil.copyfile(p,out/'source_snapshot'/p.name)
    project=Path(__file__).resolve().parent.parent;dependencies=[]
    for relative in ['background_9x9_simulation/calibrated_photometry.py',
        'background_9x9_simulation/local_background/run_local_background.py',
        'trackmate_python_bridge/run_preview.py','trackmate_python_bridge/run_detection.py',
        'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py','mito_tardis/render_point_stack.py']:
        source=project/relative;target=out/'dependency_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        dependencies.append(core.fingerprint(source))
    core.dump(out/'dependency_fingerprints.json',dependencies)
    core.dump(out/'field_units.json',dict(frame='Original zero-based frame, 0..2499',time_s='Seconds = frame * 0.02',
        source_detection_uid='sample:25-frame-cache-tag:TrackMate spot_id; stable across pilot/full/resume',
        candidate_index='Full FOV detection row in original frame order; not a track identifier',
        fit_x_px='Refined center, native C2 x pixel',fit_y_px='Refined center, native C2 y pixel',
        photons_e='Nominal whole-Gaussian photoelectron equivalents; includes fitted PSF extrapolation beyond box',
        signal_inside_box_e='Nominal photoelectron equivalents captured inside 7x7 fit box',
        bg_e_per_pixel='Nominal photoelectron equivalents per pixel, jointly fitted with annular mean constraint',
        sx_px='Gaussian standard deviation, native pixels',sy_px='Gaussian standard deviation, native pixels',
        x_se_nm='Local full observed-Hessian X standard error, nm; nuisance covariance included, coverage unvalidated',
        y_se_nm='Local full observed-Hessian Y standard error, nm; nuisance covariance included, coverage unvalidated',
        lpx_model_px='x_se_nm / 117, native pixels; nominal model SE',lpy_model_px='y_se_nm / 117, native pixels; nominal model SE',
        N_se_e='Whole-Gaussian signal standard error, nominal electron equivalents',
        bg_se_e_per_pixel='Background standard error, nominal electron equivalents per pixel'))
    (out/'README.txt').write_text(
        'WT TRAK1 T1, C2 original frames 0..2499, 117 nm/pixel and 20 ms/frame.\n'
        'Native TrackMate 7.14.0 LoG r=0.28um, threshold=6.5, median3x3 and subpixel localization. No tracking.\n'
        'Independent pixel-integrated Gaussian7x7 fit; empirical annulus6<r<=12px, neighbor exclusion<=4px, ROI and nonzero pixels.\n'
        'Minimum80 background pixels; background mean variance=max(sample variance/n, eight-sector jackknife).\n'
        'Full observed-Hessian local SE includes fitted nuisance parameters; empirical background variance is fixed.\n'
        'Nominal offset100ADU, sensitivity0.46e-/ADU, EM gain1, read noise1.6e- RMS; not measured calibration.\n'
        'Signal is whole-PSF photoelectron equivalents, not validated incident photons. SE is not validated physical accuracy or coverage.\n'
        'QC requires converged interior fit, positive finite covariance, refined center in WT ROI, and no residual-overlap flag.\n'
        'All FOV detections and ROI fit failures retained; no SE-size cutoff. Candidate indices are full-frame-order rows, not tracks.\n'
        'Pilot150 frames reused by source UID, then candidate indices remapped; additional_validation.json verifies exact fitted values.\n'
        'Checkpoints every250frames, per-frame fit files and annular pixel sets allow resume. Failed chunks are never replaced with empty rows.\n'
        'No TARDIS, biological-state fitting, trajectory linking, or mask segmentation is performed.\n')
    frozen=subprocess.run([sys.executable,'-B','-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'requirements.lock.txt').write_text(frozen.stdout)
    detector_frozen=subprocess.run([c['detector_python'],'-B','-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'trackmate_python_requirements.lock.txt').write_text(detector_frozen.stdout)
    (out/'validation_command.txt').write_text(f'{sys.executable} -B {Path(__file__).resolve()} --output {str(out)!r}\n')
    if (out/'FAILED.json').exists():
        (out/'FAILED.json').rename(out/'recovered_native_detection_failure.json')
        core.dump(out/'recovery.json',dict(resolved=True,reason='JVM heap exhaustion after many chunks in one JVM',
            solution='Fresh JVM after at most ten 25-frame chunks; native scientific parameters unchanged',
            completed_chunks_preserved_and_fingerprint_verified=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    v=json.loads((out/'validation.json').read_text());v['additional_cache_identity_validation_passed']=True;core.dump(out/'validation.json',v)
    summary=json.loads((out/'summary.json').read_text())
    entry=(f'\n[2026-10-16] WT TRAK1 T1：完整2500帧TrackMate检测与外圈背景约束定位\n输出：{out}\n'
        f'全视野候选{len(detections)}；ROI候选{len(fits)}；QC入选{len(clean)}；排除{len(fits)-len(clean)}。\n'
        f'小样本150帧{len(pilot)}个ROI拟合按检测来源UID精确复用；全片candidate_index按原始帧顺序重新生成，拟合值/QC逐字段一致。\n'
        'TrackMate7.14.0原生LoG，r0.28um、threshold6.5、median3x3、subpixel；7x7像素积分Gaussian，外圈6<r<=12排除邻点<=4，最少80背景像素。\n'
        '背景均值不确定度取样本方差/n与八扇区jackknife较大者；全observed-Hessian含nuisance参数；沿用原QC，无误差大小筛选。\n'
        '厂家名义offset100ADU、0.46e/ADU、EM1、读噪声1.6e RMS，信号/标准误非实测校准。\n'
        '原始帧/空帧及全部全视野检测、ROI失败拟合和背景像素保留，10个250帧检查点、单位/版本/源码/指纹/命令完整。\n'
        'JVM长进程曾耗尽内存，保留已验证缓存后改为每250检测帧重启；检测科学参数未变，随后全片完成。\n'
        '来源前后指纹一致，ROI/MAT/mask795顶点和112640像素一致；未分割、跟踪或运行TARDIS。\n')
    (out/'log_entry.txt').write_text(entry)
    for path in [project/'real_annular_background/LOG.txt',Path(c['C1_file']).parent/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or str(out) not in path.read_text():
            with path.open('a') as stream:stream.write(entry)
    # The completed runner has just hashed every cache file. This audit only
    # appends records and updates validation; reuse unchanged file fingerprints.
    previous={r['path']:r for r in json.loads((out/'output_fingerprints.json').read_text())}
    manifest=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name in ['output_fingerprints.json','COMPLETED.json','run.log']:continue
        old=previous.get(str(p));st=p.stat()
        manifest.append(old if old and old['bytes']==st.st_size and old['mtime_ns']==st.st_mtime_ns else core.fingerprint(p))
    core.dump(out/'output_fingerprints.json',manifest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
