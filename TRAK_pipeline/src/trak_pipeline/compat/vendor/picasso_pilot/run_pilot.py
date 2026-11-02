#!/usr/bin/env python3
"""Run only the approved 150-frame C2 localization experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from picasso import localize, io
from picasso.fitting import gaussfit, precision, seeds, splinefit
import yaml

from pilot_core import (FIELD_UNITS, FIT_STATES, baseline_threshold, dump_json,
                        fit_candidates, identify_candidates, inspect_and_sample,
                        result_summary, sample_ids, save_locs, sha256_file, verify_sample)
from pilot_qc import key_name, make_qc


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.console.flush();self.log.flush()


def provenance(out,config,command):
    root=Path(__file__).resolve().parent
    for name in ['run_pilot.py','pilot_core.py','pilot_qc.py','test_pilot.py','setup_env.sh','requirements.in','requirements.lock.txt','README.md']:
        shutil.copy2(root/name,out/'reproducibility'/name)
    source_hashes={str(Path(inspect.getfile(m)).resolve()):sha256_file(inspect.getfile(m)) for m in [localize,gaussfit,precision,seeds,splinefit]}
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    info={'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),
          'architecture':platform.machine(),'packages':dict(sorted(versions.items())),
          'picasso_module_sha256':source_hashes,'device':'CPU','command':command,
          'started_utc':datetime.now(timezone.utc).isoformat()}
    dump_json(out/'software_versions.json',info)
    (out/'run_command.txt').write_text(f'cd {shlex.quote(str(root))}\n{command}\n',encoding='utf-8')
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    dump_json(out/'field_units.json',FIELD_UNITS)
    dump_json(out/'fit_state_labels.json',FIT_STATES)
    return info


def validation_tests(out):
    result=subprocess.run([sys.executable,'-u','-m','unittest','-v','test_pilot'],cwd=Path(__file__).parent,text=True,capture_output=True)
    (out/'synthetic_validation_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Synthetic validation failed; see synthetic_validation_tests.log')
    print(result.stdout+result.stderr,flush=True)
    return {'unittest_returncode':result.returncode,'synthetic_tests_passed':True,'tests':8}


def data_quality_diagnostics(ids,movie,results,config,out):
    # An image QC rectangle only: never used to calibrate the camera, detect or fit.
    x0,y0,x1,y1=config['background_diagnostic_roi_xyxy']
    roi=movie[:,y0:y1,x0:x1].astype(float)
    mean=roi.mean(axis=(1,2));variance=roi.var(axis=(1,2),ddof=1)
    cam=config['camera'];sensitivity=cam['sensitivity_e_per_adu']/cam['em_gain']
    expected=(mean-cam['baseline_adu'])*sensitivity+cam['read_noise_e_rms']**2
    observed=variance*sensitivity**2
    table=pd.DataFrame({'frame':ids,'time_s':ids*config['frame_interval_s'],
        'roi_mean_adu':mean,'roi_spatial_variance_adu2':variance,'nominal_roi_mean_e_per_pixel':(mean-cam['baseline_adu'])*sensitivity,
        'nominal_observed_spatial_variance_e2':observed,'nominal_poisson_plus_read_variance_e2':expected,
        'observed_to_nominal_expected_variance_ratio':observed/expected})
    table.to_csv(out/'qc/background_rectangle_diagnostic.csv',index=False)
    reasons=[]
    for (box,factor),locs in results.items():
        counts={'box':box,'threshold_factor':factor,'total_candidates':len(locs),
                'inadmissible_model_parameters':int((~locs.usable_model_fit).sum()),
                'nonfinite_or_nonpositive_precision':int((~locs.precision_model_finite).sum()),
                'zero_support':int(locs.suspect_zero_pixel_support.sum()),
                'width_flag':int(locs.suspect_psf_width.sum()),
                'overlap_residual_flag':int(locs.suspect_overlap.sum()),
                'unflagged_model_fits':int(locs.qc_unflagged_model_fit.sum())}
        for name in ['backend_not_converged','nonfinite_parameters','nonpositive_signal','nonpositive_sx','nonpositive_sy','negative_background','center_outside_fitting_box']:
            counts[name]=int(locs.fit_parameter_issues.str.contains(name,regex=False).sum())
        reasons.append(counts)
    pd.DataFrame(reasons).to_csv(out/'qc/fit_quality_reason_counts.csv',index=False)
    diagnostic={'roi_xyxy_pixels':config['background_diagnostic_roi_xyxy'],
                'median_roi_mean_adu':float(np.median(mean)),'median_roi_spatial_variance_adu2':float(np.median(variance)),
                'median_nominal_observed_spatial_variance_e2':float(np.median(observed)),
                'median_nominal_expected_variance_e2':float(np.median(expected)),
                'median_observed_to_nominal_expected_variance_ratio':float(np.median(observed/expected)),
                'interpretation':'Descriptive quiet-image rectangle; spatial noise includes illumination and structure. Not a dark calibration, camera offset estimate or photon-transfer calibration. A discrepancy with the nominal Poisson-plus-read-noise expectation may reflect wrong camera parameters, export processing or model assumptions; cannot distinguish these causes.'}
    dump_json(out/'qc/data_quality_diagnostic.json',diagnostic)
    return diagnostic


def verify_results(results,ids,config):
    all_frames=set(ids)
    import_counts=[]
    for key,locs in results.items():
        assert set(locs.frame).issubset(all_frames)
        assert not locs.candidate_id.duplicated().any()
        np.testing.assert_allclose(locs.time_s,locs.frame*config['frame_interval_s'],rtol=0,atol=0)
        for field in ['x','y','sx','sy','lpx','lpy']:
            np.testing.assert_allclose(locs[field+'_nm'],locs[field]*config['pixel_size_nm'],rtol=0,atol=0,equal_nan=True)
        assert len(locs)==int(locs.fit_converged.sum())+int((~locs.fit_converged).sum())
        assert not locs.quantitatively_calibrated.any()
        assert (locs.calibration_status=='manufacturer_nominal').all()
        csv=pd.read_csv(config['_output_path']+'/fits/'+key_name(*key)+'.csv')
        assert len(csv)==len(locs)
        np.testing.assert_array_equal(csv.frame,locs.frame)
        np.testing.assert_array_equal(csv.fit_state,locs.fit_state)
        for field in ['x','y','lpx','lpy']:
            np.testing.assert_allclose(csv[field],locs[field],rtol=1e-12,atol=1e-12,equal_nan=True)
        path=Path(config['_output_path'])/'fits'/(key_name(*key)+'.hdf5')
        imported,info=io.load_locs(str(path))
        assert info[0]['calibration_status']=='manufacturer_nominal'
        assert len(imported)<=len(locs)
        assert set(imported.candidate_id).issubset(set(locs.candidate_id))
        selected=locs.set_index('candidate_id').loc[imported.candidate_id]
        for field in ['frame','x','y','lpx','lpy','fit_state']:
            np.testing.assert_equal(imported[field].to_numpy(),selected[field].to_numpy())
        import_counts.append({'box':key[0],'threshold_factor':key[1],'audit_rows':len(locs),'picasso_load_locs_rows_after_native_sanity_filter':len(imported)})
    pd.DataFrame(import_counts).to_csv(Path(config['_output_path'])/'native_picasso_import_validation.csv',index=False)
    return {'all_nine_tables_roundtrip_verified':True,'source_frame_indices_preserved':True,
            'time_from_original_frames_verified':True,'pixel_to_nm_verified':True,
            'failed_candidates_preserved_in_audit_tables':True,'all_rows_marked_uncalibrated':True,
            'picasso_native_metadata_and_import_verified':True,
            'native_loader_applies_its_own_sanity_filter':True}


def report(out,summary,stability,results,config,metadata,diagnostic,elapsed):
    ref=next(s for s in summary if s['box']==9 and s['threshold_factor']==1.)
    locs=results[(9,1.)];good=locs.loc[locs.usable_model_fit]
    issues=[
        'Actual acquisition-specific offset (ADU) and sensitivity (electrons/ADU) have not been measured or supplied.',
        'Uniform 1.6-electron RMS read noise is nominal. Pixelwise offset, gain and variance maps, crop origin and camera serial number are missing.',
        'Actual exposure time, Standard/Slow scan, actual camera output bit depth and binning remain unconfirmed. TIFF encoding is uint16; this does not establish the camera operating mode.',
        'This is an ImageJ 1.54p TIFF. Camera correction, background subtraction, rescaling and flips before export are unknown; camera ADU provenance is assumed.',
        'Only a single elliptical, point-sampled Gaussian emitter is fitted per candidate. Structured background, PSF mismatch and unresolved overlapping emitters can bias the result.',
        'CRLB is the precision lower bound of the assumed fitted noise/PSF model. It excludes drift, motion blur and other model mismatch; these outputs are not calibrated TARDIS/DANAE inputs.',
        'Net-gradient thresholding has no ground-truth recall/false-positive rate on these real frames. Lower-threshold-only spots and quality flags require image inspection.',
        'A detection border margin of half-box + 1 pixel is excluded because fit patches and finite-difference gradients need complete support. Border candidates are counted separately.',
        f"The same {metadata['persistent_zero_pixel_count']} source pixels are exactly zero across all 150 frames. Possible static export/calibration padding needs explanation; all patch zeros are flagged, never masked.",
        f"The quiet-rectangle observed/nominal expected variance ratio has median {diagnostic['median_observed_to_nominal_expected_variance_ratio']:.3f}. This and low residual deviance indicate that nominal brightness/noise assumptions require acquisition-specific verification; the diagnostic is not a camera calibration.",
    ]
    counts=[]
    for a,b in config['sample_segments']:
        segment=locs.loc[locs.frame.between(a,b)]
        counts.append({'segment':[a,b],'candidates':len(segment),'usable_model_fits':int(segment.usable_model_fit.sum()),
                       'failed':int((~segment.fit_converged).sum()),'qc_unflagged':int(segment.qc_unflagged_model_fit.sum())})
    dump_json(out/'open_questions.json',{'calibration_status':'manufacturer_nominal','questions_and_limits':issues,
                                      'reference_segment_counts':counts})
    rows=['| Box | Threshold factor | Net gradient (ADU) | Candidates | Converged | Admissible model fits | Heuristic-unflagged | Median sx/sy (px), admissible | Nominal median lpx/lpy (nm), admissible |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        def val(k):return 'unavailable' if s[k] is None else f'{s[k]:.2f}'
        rows.append(f"| {s['box']} | {s['threshold_factor']:g} | {s['threshold_net_gradient']:.2f} | {s['candidates']} | {s['converged']} | {s['usable_model_fits']} | {s['qc_unflagged_model_fit_count']} | {val('median_sx_px')}/{val('median_sy_px')} | {val('median_lpx_nm_nominal')}/{val('median_lpy_nm_nominal')} |")
    lines=[
        '# T1 C2 Picasso 150-frame pilot results', '',
        '**Calibration: manufacturer_nominal. Signal and precision are uncalibrated model estimates.**','',
        f"Source: `{metadata['source_path']}`",'',
        f"Source SHA256: `{metadata['source_sha256']}`",'',
        f"Verified source: {metadata['page_count']} time frames, YX={config['expected_height']}×{config['expected_width']}, uint16, one grayscale channel. Sample: 0–49, 1225–1274, 2450–2499. Frame interval 20 ms; user-confirmed 117 nm/pixel.", '',
        '## Nine configurations','',*rows,'',
        'Reference = box 9, baseline threshold. This is a comparison reference, not a best-parameter claim. No localization quality cut was used to remove rows; all failed candidates remain in CSV/HDF5. `usable_model_fit` is an inspectable flag, not experimental ground truth.', '',
        'Audit HDF5 preserves every candidate in `/locs`. Use h5py or the CSV to read the full audit. Picasso `io.load_locs`/GUI applies its built-in sanity filter and drops some NaN/negative parameters; its imported row count differs by design, as recorded in `native_picasso_import_validation.csv`. No invalid rows were deleted from the saved HDF5.', '',
        '## Reference observations','',
        f"Reference candidates: {ref['candidates']}; backend converged: {ref['converged']}; finite physically admissible model fits: {ref['usable_model_fits']}. Backend states: `{json.dumps(ref['backend_state_counts'])}`.", '',
        f"Usable fits with heuristic overlap flags: {ref['suspect_overlap_count']}; implausible-width flags: {ref['suspect_width_count']}. These are inspection prompts, not validated classifications.", '',
        f"{ref['candidates']-ref['usable_model_fits']} converged fits have inadmissible model parameters. {ref['source_zero_support_count']} candidate patches touch exact-zero source pixels. {ref['qc_unflagged_model_fit_count']} fits pass all heuristic flags; their nominal median X/Y CRLB SD is {ref['unflagged_median_lpx_nm_nominal']:.2f}/{ref['unflagged_median_lpy_nm_nominal']:.2f} nm, median signal {ref['unflagged_median_photons_nominal']:.1f} nominal detected electrons. This subset is not experimentally validated. Full reason counts: `qc/fit_quality_reason_counts.csv`.", '',
        f"External empirical reference supplied by the user: approximately {config['empirical_localization_error_reference_nm']:g} nm localization error. The nominal per-axis CRLB medians are similar in scale. The empirical metric's per-axis/radial definition, estimator and measurement method are unspecified, so this is an order-of-magnitude comparison, not validation of camera calibration or an equality of error metrics. This reference was not used to tune detection or fit parameters.", '',
        f"Median shifted-Poisson deviance/(81−6): {ref['median_deviance_per_dof']}. A value near 1 can be an approximate residual diagnostic, but it is not a calibrated goodness-of-fit test here.", '',
        f"Reference segment counts: `{json.dumps(counts)}`.", '',
        '## Stability','',
        'See `qc/parameter_stability.csv` and all per-candidate match tables. Matching requires mutual nearest neighbours within 1 pixel in the same source frame. Reports include median and 90th percentile absolute position and relative width changes, plus a separate subset without overlap flags. Box-specific threshold estimates are different detector scores; equal threshold factors do not imply equal detection sensitivity.', '',
    ]
    for r in stability.itertuples():
        lines.append(f'- Box {r.box}, {r.threshold_factor:g}×: {r.matched}/{r.reference_usable} admissible reference fits matched; median position change {r.all_median_abs_distance_px*117:.2f} nm; median absolute width changes X/Y {r.all_median_abs_relative_sx_change*100:.1f}%/{r.all_median_abs_relative_sy_change*100:.1f}%. Heuristic-unflagged matches: {r.unflagged_matches}/{r.reference_unflagged}, median position change {r.unflagged_median_abs_distance_px*117:.2f} nm.')
    lines += ['', '## Units and implementation','',
        'Brightness conversion: `(ADU−100)×0.46/1`. `photons` means integrated detected photoelectron-equivalent signal under this convention; it does not measure total emitted photons. `bg` is nominal electrons/pixel. `sx/sy` are Gaussian sigma in camera pixels, not FWHM. `lpx/lpy` are square roots of Picasso Fisher CRLB variances in camera pixels, calculated with the same uniform 2.56 electron² variance as MLE; multiply by 117 for nm. Full definitions: `field_units.json`.', '',
        'Native low-level `gaussfit.fit_spots(ELLIPTIC, ..., mle=True, variance=...)` is used because the high-level Gaussian wrapper discards fit states. Picasso state codes: 0 converged, 1 maximum iterations, 2 singular Hessian, 3 negative curvature / invalid shifted MLE model. Native iteration counts are saved. MLE statistics: `fit_deviance` is shifted-Poisson deviance and `log_likelihood = −0.5 × fit_deviance`, relative to the saturated model, not an absolute normalized log likelihood.', '',
        'Values below the nominal offset are recorded. Like the Picasso MLE path, fitting data below −read-noise variance are clipped to that floor; per-box counts are saved as `mle_pixels_clipped_to_noise_floor`. Source pixels and stored sample pixels are unchanged. Raw residual figures use unclipped source data.', '',
        '`lpx/lpy` that are nonfinite are unavailable under the assumed fit/model; see `precision_model_finite`, `precision_status`, `fit_parameter_issues` and `fit_state`. Gaussian width sign is degenerate: positive-width reparameterization preserves the model image and coordinates, and the raw signed amplitude/width/integral parameters are retained for audit. Genuine negative amplitude/background remains inadmissible. Even finite precision on a failed, unusually broad or overlapping fit is not a reliable localization. Check fitting-patch and detector-support zero flags together. No photon or CRLB column in this pilot is acquisition-calibrated.', '',
        'Heuristic unflagged subset: admissible fit, finite positive precision, no sigma<0.5 or sigma>half-box flag, no distant >4-model-SD positive residual maximum, and no exactly-zero source pixel in the fitting box. These are transparent inspection criteria, not measured detection accuracy or an optimized quality filter. An absence of residual-overlap flags does not establish absence of overlap, especially under uncertain noise scaling.', '',
        '## Validation and reproducibility','',
        'Eight synthetic/adapter tests passed: asymmetric Gaussian X/Y and widths, intensity conversion and nominal noise, source frame IDs/time/nm units, forced non-convergence with persisted states, blank and boundary frames, the ≤200-frame stop guard, zero-support flags without source/candidate removal, and native metadata plus embedded-only Picasso import. All 150 selected images were re-read and compared pixel-for-pixel with the source; CSV/HDF5 status and numeric field round trips and nm conversion were checked. All nine files were also imported by Picasso with its native sanity filter.', '',
        f'Total pipeline elapsed time: {elapsed:.1f} seconds (including validation, input hashing, candidate detection, nine fits, tables and QC; environment installation excluded). See `runtime.json` and `run.log`.', '',
        'Pinned environment: `software_versions.json`, `reproducibility/requirements.lock.txt`. Exact run command: `run_command.txt`; parameters: `config.yaml`; source adapter snapshots: `reproducibility/`.', '',
        '## Questions and limits','',*['- '+issue for issue in issues],'',
        '**Stopped at the 150-frame pilot. Full 2500-frame processing requires the user’s confirmation.**','',
        'Official references: [Picasso 0.11.3](https://pypi.org/project/picassosr/0.11.3/), [Gaussian backend](https://github.com/jungmannlab/picasso/blob/v0.11.3/picasso/fitting/gaussfit.py), [localization and CRLB assembly](https://github.com/jungmannlab/picasso/blob/v0.11.3/picasso/localize.py), [file fields](https://picassosr.readthedocs.io/en/latest/files.html), [Hamamatsu manual](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/en/manual/C13440-20CU_IM_En.pdf#page=83), [ONI specifications](https://oni.bio/wp-content/uploads/2025/03/Nanoimager-Technical-Specifications-April-2025.pdf).', ''
    ]
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return ref


def run(config,out,command):
    started=time.perf_counter();stages={}
    if importlib.metadata.version('picassosr')!='0.11.3':raise RuntimeError('This pilot requires picassosr 0.11.3')
    assert config['fit']['device']=='cpu' and config['fit']['estimator']=='mle' and config['fit']['model']=='elliptical_gaussian'
    assert not config['fit']['gaussian_detection_filter'] and not config['fit']['temporal_median_filter']
    assert config['camera']['calibration_status']=='manufacturer_nominal'
    assert config['box_sizes']==[7,9,11] and config['threshold_factors']==[.75,1.,1.25]
    assert len(sample_ids(config))==150
    for name in ['fits','candidates','qc','reproducibility']:(out/name).mkdir()
    provenance(out,config,command)
    check=validation_tests(out)
    t=time.perf_counter();ids,movie,metadata=inspect_and_sample(config['input_tiff'],config,out)
    stages['fingerprint_and_sample_seconds']=time.perf_counter()-t
    results={};candidate_tables={};threshold_stats={};summary=[]
    for box in config['box_sizes']:
        t=time.perf_counter();all_ids,border=identify_candidates(ids,movie,box)
        detection_time=time.perf_counter()-t
        candidate_tables[box]=all_ids
        all_ids.to_csv(out/'candidates'/f'box{box}_all_local_maxima.csv',index=False)
        border.to_csv(out/'candidates'/f'box{box}_excluded_border_counts.csv',index=False)
        base,stats=baseline_threshold(all_ids,config);threshold_stats[box]=stats
        print(f'Box {box} baseline net gradient: {base:.3f} (median {stats["median"]:.3f}, robust SD {stats["robust_sd"]:.3f}).',flush=True)
        for factor in config['threshold_factors']:
            threshold=base*factor
            selected=all_ids.loc[all_ids.net_gradient>threshold].reset_index(drop=True)
            print(f'Fitting box={box}, factor={factor:g}, threshold={threshold:.3f}, candidates={len(selected)}...',flush=True)
            locs,theta,timing=fit_candidates(selected,ids,movie,box,config)
            results[(box,factor)]=locs
            params={'Height':config['expected_height'],'Width':config['expected_width'],'Frames':config['expected_frames'],
                    'Frames analyzed':len(ids),'Frame ranges inclusive':config['sample_segments'],
                    'Pixelsize':config['pixel_size_nm'],'Pixel size unit':'nm','Frame interval':config['frame_interval_s'],
                    'Generated by':'Picasso 0.11.3 independent 150-frame pilot','Fit method':'CPU elliptical Gaussian MLE',
                    'Box Size':box,'Minimum Net Gradient':threshold,'Threshold factor':factor,
                    'Detection filters':'none','Camera':config['camera'], 'Fit configuration':config['fit'],
                    'Uniform nominal read-noise variance e2':timing['nominal_variance_e2'],
                    'calibration_status':'manufacturer_nominal','quantitatively_calibrated':False,
                    'Source TIFF SHA256':metadata['source_sha256'], 'Failed candidates retained':True,
                    'Threshold reference statistics':stats,'Field units file':'../field_units.json'}
            path=out/'fits'/(key_name(box,factor)+'.hdf5')
            save_locs(path,locs,params)
            path.with_suffix('.yaml').write_text(yaml.safe_dump_all([params],sort_keys=False,allow_unicode=True),encoding='utf-8')
            dump_json(path.with_suffix('.json'),params)
            np.save(path.with_name(path.stem+'_model_parameters.npy'),theta)
            timing['detector_runtime_seconds_for_shared_box_candidates']=detection_time
            s=result_summary(locs,box,factor,threshold,timing);summary.append(s)
            print(f'  {s["converged"]}/{s["candidates"]} converged; {s["usable_model_fits"]} usable; fit {timing["fit_runtime_seconds"]:.2f}s.',flush=True)
    dump_json(out/'threshold_reference_stats.json',threshold_stats)
    dump_json(out/'nine_configuration_summary.json',summary)
    table=pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} for s in summary])
    table.to_csv(out/'nine_configuration_summary.csv',index=False)
    t=time.perf_counter();print('Generating images and within-frame parameter comparisons...',flush=True)
    stability=make_qc(ids,movie,results,candidate_tables,threshold_stats,config,out/'qc')
    diagnostic=data_quality_diagnostics(ids,movie,results,config,out)
    stages['qc_seconds']=time.perf_counter()-t
    check.update(verify_sample(config['input_tiff'],out,metadata))
    temp_config={**config,'_output_path':str(out)}
    check.update(verify_results(results,ids,temp_config))
    dump_json(out/'validation.json',check)
    elapsed=time.perf_counter()-started
    dump_json(out/'runtime.json',{'total_seconds':elapsed,'stages':stages,'finished_utc':datetime.now(timezone.utc).isoformat(),
                                 'full_movie_processed':False,'selected_frame_count':len(ids)})
    reference=report(out,summary,stability,results,config,metadata,diagnostic,elapsed)
    dump_json(out/'COMPLETED.json',{'selected_frame_count':150,'configurations':9,'validation_passed':True,
                                  'reference':reference,'awaiting_user_confirmation_for_full_movie':True})
    print(f'Completed 150 frames and all 9 configurations in {elapsed:.1f}s. Full movie not processed. Output: {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_T1_C2.yaml')
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    config_path=Path(args.config).resolve();config=yaml.safe_load(config_path.read_text())
    out=Path(args.output).resolve() if args.output else root/'runs'/('T1_C2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if not out.is_relative_to(root/'runs'):raise ValueError('Pilot output must be inside picasso_pilot/runs/')
    out.mkdir(parents=True,exist_ok=False)
    command=shlex.join([sys.executable,*sys.argv])
    stdout,stderr=sys.stdout,sys.stderr
    with (out/'run.log').open('w',encoding='utf-8') as log:
        sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:run(config,out,command)
        except Exception:
            traceback.print_exc()
            dump_json(out/'FAILED.json',{'message':'Pilot failed; inspect run.log. Full movie not processed.'})
            raise
        finally:sys.stdout=stdout;sys.stderr=stderr


if __name__=='__main__':main()
