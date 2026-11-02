#!/usr/bin/env python3
"""User-authorized full TIFF localization, preserving all pilot outputs."""
import argparse
from datetime import datetime,timezone
import hashlib
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

import numpy as np
import pandas as pd
from picasso import localize
from picasso.fitting import gaussfit,precision
import tifffile
import yaml

from full_core import ROOT,annotate,fit_candidates,identify_candidates,key_name,frame_qc
from pilot_core import FIELD_UNITS,FIT_STATES,dump_json,model_images,save_locs,sha256_file


class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,v):self.console.write(v);self.log.write(v);self.log.flush();return len(v)
    def flush(self):self.console.flush();self.log.flush()


def read_audit(path):
    t=pd.read_csv(path)
    for k in ['fit_parameter_issues','qc_parameter_reasons','qc_exclusion_reasons']:t[k]=t[k].fillna('')
    t['frame']=t.frame.astype(np.uint32);t['candidate_id']=t.candidate_id.astype(np.uint32)
    return t


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.yaml');parser.add_argument('--output')
    args=parser.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    pilot=(ROOT/c['pilot_run']).resolve();review=(ROOT/c['review_run']).resolve()
    pc=yaml.safe_load((pilot/'config.yaml').read_text())
    thresholds=json.loads((pilot/'threshold_reference_stats.json').read_text())
    assert c['frames']==2500 and c['segment_frames']==250
    assert c['input_tiff']==pc['input_tiff'] and c['pixel_size_nm']==pc['pixel_size_nm'] and c['frame_interval_s']==pc['frame_interval_s']
    assert importlib.metadata.version('picassosr')=='0.11.3'
    assert [(g['box'],g['factor']) for g in c['groups']]==[(9,.75),(9,1.),(9,1.25),(7,1.),(11,1.)]
    for g in c['groups']:assert g['threshold']==thresholds[str(g['box'])]['baseline_threshold_adu_gradient']*g['factor']
    assert not any(c[k] for k in ['tracking','masking','drift_correction','uncertainty_deconvolution','diffusion_correction','biological_state_fitting'])
    out=Path(args.output).resolve() if args.output else ROOT/'runs'/('T1_C2_full_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    assert out.is_relative_to(ROOT/'runs');out.mkdir(parents=True,exist_ok=False)
    for d in ['unions','local_maxima','audit','selected','qc','reproducibility']:(out/d).mkdir()
    log=(out/'fit.log').open('w');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    started=time.perf_counter();print('Full-film output:',out,flush=True)
    shutil.copy2(args.config,out/'config.yaml');shutil.copy2(pilot/'config.yaml',out/'pilot_fit_config.yaml')
    shutil.copy2(pilot/'threshold_reference_stats.json',out/'pilot_threshold_reference_stats.json')
    shutil.copy2(ROOT.parent/'picasso_pilot/requirements.lock.txt',out/'reproducibility/requirements.lock.txt')
    for p in list(ROOT.glob('*.py'))+[ROOT.parent/'picasso_pilot/pilot_core.py',ROOT.parent/'picasso_pilot/pilot_qc.py']:
        shutil.copy2(p,out/'reproducibility'/p.name)
    command=shlex.join(sys.orig_argv);(out/'fit_command.txt').write_text(f'cd {shlex.quote(str(ROOT))}\n{command}\n')
    FIELD_UNITS.update({'reference_segment':'index of original 250-frame interval, 0–9',
          'analysis_segment':'frame // 250, 0–9','box_size':'camera pixels','threshold_factor':'pilot baseline multiplier',
          'absolute_threshold':'ADU-gradient units, fixed pilot threshold','qc_exclusion_reasons':'overlapping exclusion flags; empty iff retained',
          'qc_parameter_reasons':'parameter reasons after width-sign canonicalization; not a false-positive label'})
    for k in ['raw_negative_sx','raw_negative_sy','raw_negative_width','nonpositive_backend_amplitude','negative_background','center_outside_fitting_box']:
        FIELD_UNITS[k]='boolean parameter diagnostic; negative width alone is not exclusion'
    for k in __import__('full_core').EXCLUSIONS:FIELD_UNITS[k]='boolean QC exclusion; categories overlap'
    dump_json(out/'field_units.json',FIELD_UNITS);dump_json(out/'fit_state_labels.json',FIT_STATES)
    software={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'device':'CPU',
              'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
              'picasso_source_sha256':{inspect.getfile(m):sha256_file(inspect.getfile(m)) for m in [localize,gaussfit,precision]}}
    dump_json(out/'software_versions.json',software)
    prior_paths=[pilot/'RESULTS.md',pilot/'threshold_reference_stats.json',review/'REVIEW.md']+list((review/'canonical_fits').glob('*.csv'))
    locked={str(p):sha256_file(p) for p in prior_paths}
    tardis=ROOT.parent/'TARDIS/path_irrelevant_python/tardis_tau1'
    locked.update({str(p):sha256_file(p) for p in tardis.glob('*.py')})
    dump_json(out/'protected_source_hashes.json',locked)
    # Rules are written before tests, input reads or fitting.
    rules={'configuration':c,'camera':pc['camera'],'fit':pc['fit'],'source_pilot':str(pilot),'source_review':str(review),
           'all_candidates_definition':'all local maxima are stored separately; each audit includes every threshold-passing fit, including failed/QC-excluded fits',
           'same_box_fits':'fit the union once, select nested thresholds afterward; independent per-spot MLE is identical for shared candidates',
           'candidate_id':'frame*(height*width)+candidate_y*width+candidate_x; use box/run with ID',
           'near_zero_guard':'abs(signed Intra area)<=0.01 is unavailable; main-region area<=0.01 makes tail/main unavailable',
           'statistical_intervals':'paired time-block resampling; not pointwise CRLB and not between-cell variability',
           'calibration_status':'manufacturer_nominal','user_authorized_full_movie':True}
    dump_json(out/'RULES_BEFORE_RUN.json',rules)
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_full','test_pilot','test_review'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Preflight tests failed')
    source=Path(c['input_tiff']);stat=source.stat();digest=sha256_file(source)
    assert digest==json.loads((pilot/'input_metadata.json').read_text())['source_sha256']
    meta={'source_path':str(source),'source_sha256':digest,'source_bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns}
    sample_hashes=pd.read_csv(pilot/'sample_frames.csv').set_index('frame').sha256.to_dict()
    pc['sample_segments']=[[s,s+249] for s in range(0,2500,250)]
    all_frames=[];borders=[];chunks={7:[],9:[],11:[]};persistent=np.ones((c['height'],c['width']),bool)
    with tifffile.TiffFile(source) as tif:
        assert len(tif.pages)==c['frames'] and tif.series[0].axes=='TYX'
        assert tif.pages[0].shape==(c['height'],c['width']) and tif.pages[0].bitspersample==16
        ij={k:v for k,v in (tif.imagej_metadata or {}).items() if k!='Labels'}
        assert abs(ij['finterval']-c['frame_interval_s'])<1e-10
        meta.update(page_count=len(tif.pages),shape=list(tif.series[0].shape),axes='TYX',imagej_metadata=ij)
        dump_json(out/'input_metadata.json',meta)
        for start in range(0,c['frames'],c['read_chunk_frames']):
            stop=min(start+c['read_chunk_frames'],c['frames']);ids=np.arange(start,stop,dtype=np.uint32);arrays=[]
            for f in ids:
                a=tif.pages[int(f)].asarray();pixel_hash=hashlib.sha256(a.tobytes()).hexdigest()
                if int(f) in sample_hashes:assert pixel_hash==sample_hashes[int(f)],f'Pilot source pixel hash mismatch {f}'
                roi=a[540:590,40:90].astype(float);mean=float(roi.mean());var=float(roi.var(ddof=1))
                cam=pc['camera'];sens=cam['sensitivity_e_per_adu']/cam['em_gain']
                expected=(mean-cam['baseline_adu'])*sens+cam['read_noise_e_rms']**2
                all_frames.append({'frame':int(f),'time_s':int(f)*c['frame_interval_s'],'pixel_sha256':pixel_hash,
                    'zero_pixels':int((a==0).sum()),'saturated_pixels':int((a==65535).sum()),
                    'quiet_mean_adu':mean,'quiet_spatial_variance_adu2':var,'quiet_observed_nominal_variance_ratio':var*sens**2/expected})
                arrays.append(a.astype(np.uint16,copy=False))
            movie=np.stack(arrays);persistent&=np.all(movie==0,axis=0)
            untouched=hashlib.sha256(movie.tobytes()).hexdigest()
            for box in [9,7,11]:
                group=[g for g in c['groups'] if g['box']==box];minimum=min(g['threshold'] for g in group)
                candidates,border=identify_candidates(ids,movie,box)
                candidates['n_id']=(candidates.frame*c['height']*c['width']+candidates.y*c['width']+candidates.x).astype(np.uint32)
                candidates.to_csv(out/'local_maxima'/f'box{box}_{start:04d}_{stop-1:04d}.csv',index=False,float_format='%.17g')
                borders.append(border)
                fit_ids=candidates.loc[candidates.net_gradient>minimum].reset_index(drop=True)
                locs,theta,timing=fit_candidates(fit_ids,ids,movie,box,pc)
                locs=annotate(locs,box,c)
                raw=theta.copy();raw[:,0]=locs.backend_signed_integral_e;raw[:,3]=locs.backend_sx_signed_px;raw[:,4]=locs.backend_sy_signed_px
                np.testing.assert_allclose(model_images(raw,box),model_images(theta,box),rtol=1e-10,atol=1e-8,equal_nan=True)
                path=out/'unions'/f'box{box}_{start:04d}_{stop-1:04d}.hdf5'
                save_locs(path,locs,{'Height':c['height'],'Width':c['width'],'Frames':2500,'Source SHA256':digest,
                                   'Box Size':box,'Minimum Net Gradient':minimum,'Frame range':[start,stop-1],
                                   'Camera':pc['camera'],'calibration_status':'manufacturer_nominal','Failed candidates retained':True})
                np.save(path.with_name(path.stem+'_model_parameters.npy'),theta);chunks[box].append(path.with_suffix('.csv'))
                print(f'Frames {start}–{stop-1}, box {box}: {len(locs)} fit candidates, {int(locs.qc_unflagged_model_fit.sum())} kept; {timing["fit_runtime_seconds"]:.1f}s',flush=True)
            assert hashlib.sha256(movie.tobytes()).hexdigest()==untouched
    pd.DataFrame(all_frames).to_csv(out/'input_frames.csv',index=False)
    pd.concat(borders).to_csv(out/'boundary_excluded_counts.csv',index=False)
    np.save(out/'persistent_zero_pixels.npy',persistent)
    counts=[];patterns=[];pilot_checks=[]
    for box in [9,7,11]:
        union=pd.concat([read_audit(p) for p in chunks[box]],ignore_index=True)
        for g in [g for g in c['groups'] if g['box']==box]:
            name=key_name(box,g['factor']);t=union.loc[union.net_gradient>g['threshold']].copy()
            t['threshold_factor']=g['factor'];t['absolute_threshold']=g['threshold']
            save_locs(out/'audit'/(name+'.hdf5'),t,{'Height':c['height'],'Width':c['width'],'Frames':2500,
                 'Pixelsize':c['pixel_size_nm'],'Box Size':box,'Minimum Net Gradient':g['threshold'],
                 'Camera':pc['camera'],'calibration_status':'manufacturer_nominal','quantitatively_calibrated':False,
                 'QC rules file':'../RULES_BEFORE_RUN.json','Failed candidates retained':True})
            kept=t.loc[t.qc_unflagged_model_fit];kept.to_csv(out/'selected'/(name+'.csv'),index=False,float_format='%.17g')
            counts.append(frame_qc(t,c,box,g['factor'],g['threshold']))
            pattern=t.groupby(['analysis_segment','qc_exclusion_reasons','qc_parameter_reasons','raw_negative_width'],dropna=False).size().reset_index(name='rows')
            pattern['group']=name;patterns.append(pattern)
            old=pd.read_csv(review/'canonical_fits'/(name+'.csv'));keys=['frame','candidate_x','candidate_y']
            current=t.loc[t.frame.isin(old.frame.unique())].sort_values(keys).reset_index(drop=True)
            previous=old.sort_values(keys).reset_index(drop=True)
            np.testing.assert_array_equal(current[keys],previous[keys])
            np.testing.assert_array_equal(current.qc_unflagged_model_fit,previous.qc_unflagged_model_fit)
            for k in ['x','y','photons','bg','sx','sy','lpx','lpy','fit_state']:
                np.testing.assert_allclose(current[k],previous[k],rtol=2e-6,atol=1e-6,equal_nan=True)
            pilot_checks.append({'group':name,'matched_pilot_candidates':len(old),'same_qc_membership':True})
            print(f'{name}: full audit {len(t)}, retained {len(kept)}; all 150 pilot frames reproduce.',flush=True)
    pd.concat(counts).to_csv(out/'per_frame_qc.csv',index=False)
    pd.concat(patterns).to_csv(out/'qc_exclusion_pattern_counts.csv',index=False)
    assert source.stat().st_size==stat.st_size and source.stat().st_mtime_ns==stat.st_mtime_ns
    assert sha256_file(source)==digest
    for p,h in locked.items():assert sha256_file(p)==h,f'Protected input changed: {p}'
    dump_json(out/'fit_validation.json',{'tests_passed':True,'tests':18,'pilot_reproduction':pilot_checks,
        'all_source_frames_read':2500,'all_150_pilot_pixel_hashes_match':True,'source_pixels_unchanged':True,
        'raw_canonical_model_images_invariant':True,'source_and_prior_outputs_and_tardis_unchanged':True})
    dump_json(out/'fit_runtime.json',{'seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat()})
    dump_json(out/'FIT_COMPLETED.json',{'frames':2500,'configurations':5,'analysis_pending':True})
    print('Localization complete; statistical analysis pending:',out,flush=True)


if __name__=='__main__':main()
