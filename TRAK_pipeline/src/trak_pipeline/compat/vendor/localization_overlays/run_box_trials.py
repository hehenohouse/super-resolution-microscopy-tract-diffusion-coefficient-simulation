"""Fit ONLY source frames 0–249 with native Box5 and experimental even Box6.

Box6 reuses the complete original Box7 detection candidate cache, not the
QC-selected coordinates. Its explicitly defined 6x6 pixel crop is [-3,+3),
so the integer candidate is pixel index3 and the window center is index2.5.
The unchanged Picasso Gaussian CPU backend and CRLB consume that exact crop.
No Picasso installation, existing analysis source, cache or raw pixels are edited.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'picasso_pilot'))
sys.path.insert(0, str(PROJECT / 'picasso_full_sensitivity'))
sys.path.insert(0, str(PROJECT / 'picasso_roi_sensitivity'))
import pilot_core as pc
from full_core import annotate, EXCLUSIONS
from roi_core import read_mat_roi, roi_flags
from tardis_tau1.locbio import load_imagej_roi
from render_frames import fingerprint, dump, Tee

NATIVE_CUT = pc.cut_raw_spots
NATIVE_ZERO = pc.detection_zero_counts


def exact_cut(ids, frames, movie, box):
    """Odd boxes are unchanged; for Box6 extract exactly 36 source pixels."""
    if box % 2:
        return NATIVE_CUT(ids, frames, movie, box)
    assert box == 6
    by_frame = {int(f): im for f, im in zip(frames, movie)}
    out = np.empty((len(ids), box, box), np.float32)
    for i, row in enumerate(ids.itertuples(index=False)):
        x0, y0 = int(row.x) - 3, int(row.y) - 3
        raw = by_frame[int(row.frame)][y0:y0 + box, x0:x0 + box]
        assert raw.shape == (box, box)
        out[i] = raw
        np.testing.assert_array_equal(out[i], raw)
    return out


@contextmanager
def fit_adapter(detection_box):
    """Process-local adapter only; preserve the original detector's zero halo."""
    old_cut, old_zero = pc.cut_raw_spots, pc.detection_zero_counts
    pc.cut_raw_spots = exact_cut
    pc.detection_zero_counts = lambda ids, frames, movie, fit_box: NATIVE_ZERO(ids, frames, movie, detection_box)
    try:
        yield
    finally:
        pc.cut_raw_spots, pc.detection_zero_counts = old_cut, old_zero


def synthetic_checks(config):
    checks = []
    yy, xx = np.mgrid[:65, :83]
    truth = [(40.17, 28.77), (40.51, 28.49), (39.70, 29.24)]
    movie = []
    for x, y in truth:
        e = 20 + 1000 / (2*np.pi*.9*1.1) * np.exp(-.5*((xx-x)**2/.9**2 + (yy-y)**2/1.1**2))
        movie.append(np.rint(e/.46 + 100).astype(np.uint16))
    movie = np.stack(movie)
    frames = np.arange(len(movie), dtype=np.uint32)
    ids = pd.DataFrame({'frame': frames, 'x': [40]*3, 'y': [29]*3,
                        'net_gradient': [1000.]*3, 'n_id': frames})
    empty = ids.iloc[:0].copy()
    config = dict(config, sample_segments=[[0,2]])
    for box, detector in [(5,5), (6,7)]:
        with fit_adapter(detector):
            table, theta, _ = pc.fit_candidates(ids, frames, movie, box, config)
            blank, _, _ = pc.fit_candidates(empty, frames, movie, box, config)
        np.testing.assert_array_equal(table.fit_state.to_numpy(), np.zeros(3))
        np.testing.assert_allclose(table[['x','y']].to_numpy(), truth, atol=.015, rtol=0)
        np.testing.assert_allclose(table[['sx','sy']].to_numpy(), np.tile([.9,1.1], (3,1)), atol=.02, rtol=0)
        np.testing.assert_allclose(table.photons, 1000., rtol=.02, atol=0)
        assert table.precision_model_finite.all() and len(blank) == 0
        for axis in ['x','y','lpx','lpy']:
            np.testing.assert_allclose(table[axis+'_nm'], table[axis]*117, atol=.001, rtol=0)
        crops = exact_cut(ids, frames, movie, box)
        for i in range(3):
            h = box//2
            np.testing.assert_array_equal(crops[i], movie[i,29-h:29-h+box,40-h:40-h+box])
        checks.append({'box': box, 'detector': detector, 'synthetic_position_max_error_px': float(np.max(np.abs(table[['x','y']].to_numpy()-truth))),
            'integer_and_fractional_positions_verified': True, 'exact_raw_crop_verified': True,
            'states_preserved': True, 'precision_model_finite': True, 'nm_conversion_verified': True,
            'empty_candidates_verified': True})
    try:
        NATIVE_CUT(ids.iloc[:1], frames, movie, 6)
    except ValueError as e:
        native_error = str(e)
    else:
        raise AssertionError('Native crop unexpectedly accepts Box6')
    # The detection support is the ORIGINAL 7x7 detector plus axial halo.
    support_movie = np.ones((1,65,83), np.uint16)
    support_movie[0,25,36] = 0  # halo corner: not read by the detector
    assert NATIVE_ZERO(ids.iloc[:1], frames[:1], support_movie, 7)[0] == 0
    support_movie[0,25,40] = 0  # axial halo: read by the detector
    with fit_adapter(7):
        assert pc.detection_zero_counts(ids.iloc[:1],frames[:1],support_movie,6)[0] == 1
    return {'passed': True, 'boxes': checks, 'native_even_crop_error': native_error,
            'original_detector_halo_preserved_for_even_fit': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config_box_trials.yaml')
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    c = yaml.safe_load(cfg_path.read_text())
    assert c['frame_range'] == [0,250] and not c['tardis'] and not c['mitochondrial_classification']
    source = (cfg_path.parent/c['source_roi_run']).resolve()
    fit_config_path = (cfg_path.parent/c['source_fit_config']).resolve()
    render_config_path = (cfg_path.parent/c['source_render_config']).resolve()
    fit_config = yaml.safe_load(fit_config_path.read_text())
    original_fit_config = json.loads(json.dumps(fit_config))
    fit_config['sample_segments'] = [[0,249]]
    fit_config['segment_frames'] = 250
    base_render = yaml.safe_load(render_config_path.read_text())
    data = Path(base_render['data_directory'])
    tiff = data/base_render['source_tiff']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = data/c['output_relative_directory']/('T1_C2_box5_box6_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    for dirname in ['audit','selected','fits','qc','reproducibility']:
        (out/dirname).mkdir()
    log=(out/'run.log').open('w')
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    start=time.perf_counter()
    print('FIT_OUTPUT_DIRECTORY:',out,flush=True)
    try:
        roi_info=json.loads((source/'roi_validation.json').read_text())
        protect=[tiff,source/'audit/box7_threshold1p00.csv',source/'audit/box7_threshold1p00.hdf5',
                 source/'roi_vertices.csv',source/'roi_validation.json',source/'COMPLETED.json',
                 Path(roi_info['roi_source']),Path(roi_info['mat_file']),fit_config_path,
                 PROJECT/'picasso_pilot/pilot_core.py',PROJECT/'picasso_full_sensitivity/full_core.py',
                 PROJECT/'picasso_roi_sensitivity/roi_core.py',PROJECT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py']
        before=[fingerprint(p) for p in protect]
        dump(out/'source_fingerprints_before.json',before)
        assert before[0]['sha256']==roi_info['source_tiff_metadata']['source_sha256']
        vertices = pd.read_csv(source/'roi_vertices.csv')[['x_px','y_px']].to_numpy(float)
        roi_vertices = load_imagej_roi(roi_info['roi_source'])
        mat_vertices, record = read_mat_roi(roi_info['mat_file'],roi_info['mat_record'])
        np.testing.assert_array_equal(vertices,roi_vertices)
        np.testing.assert_array_equal(vertices,mat_vertices)
        assert before[6]['sha256']==roi_info['roi_sha256']
        roi_info=dict(roi_info, vertices_exactly_equal=True, no_refitting=False,
            no_refitting_note='New Box5/Box6 fits ONLY frames0–249; old source cache remains read-only.',
            mat_hdf_record=record, formally_rechecked_utc=datetime.now(timezone.utc).isoformat())
        dump(out/'roi_validation.json',roi_info)
        shutil.copyfile(source/'roi_vertices.csv',out/'roi_vertices.csv')
        (out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
        (out/'fit_config.yaml').write_text(yaml.safe_dump(fit_config,sort_keys=False))
        dump(out/'software_versions.json',{n:importlib.metadata.version(n) for n in ['picassosr','numpy','pandas','scipy','numba','tifffile','h5py','Pillow','PyYAML']})
        (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(cfg_path)])+'\n')
        for path in [Path(__file__),ROOT/'render_frames.py',cfg_path,*protect[8:]]:
            shutil.copyfile(path,out/'reproducibility'/path.name)
        print('Checking synthetic Box5 and exact Box6 crop/coordinates/CRLB...',flush=True)
        dump(out/'synthetic_validation.json',synthetic_checks(fit_config))
        frames=np.arange(250,dtype=np.uint32)
        with tifffile.TiffFile(tiff) as tf:
            assert len(tf.pages)==2500 and tf.series[0].axes=='TYX'
            assert tuple(tf.series[0].shape)==(2500,684,428)
            movie=np.stack([tf.pages[int(f)].asarray() for f in frames])
            ref_ids=np.array(c['threshold_reference_frames'],np.uint32)
            ref_movie=np.stack([tf.pages[int(f)].asarray() for f in ref_ids])
            assert tf.imagej_metadata['finterval']==.02 and tf.imagej_metadata['spacing']==.117
            for i,f in enumerate(frames):
                assert f' t:{int(f)+1}/2500 ' in tf.imagej_metadata['Labels'][int(f)]
                np.testing.assert_array_equal(movie[i],tf.pages[int(f)].asarray())
        pd.DataFrame([{'frame':int(f),'time_s':float(f)*.02,'sha256':hashlib.sha256(im.tobytes()).hexdigest()}
                      for f,im in zip(frames,movie)]).to_csv(out/'input_frames.csv',index=False)
        summaries=[]; render_configs=[]
        for spec in c['groups']:
            name=spec['name'];box=spec['fitting_box'];detector=spec['detection_box']
            assert (box,detector) in [(5,5),(6,7)]
            print(f'Preparing {name}: fit{box}, detect{detector}',flush=True)
            if box==5:
                ref_candidates,_=pc.identify_candidates(ref_ids,ref_movie,detector)
                threshold,threshold_stats=pc.baseline_threshold(ref_candidates,original_fit_config)
                candidates,borders=pc.identify_candidates(frames,movie,detector)
                candidates=candidates.loc[candidates.net_gradient>threshold].reset_index(drop=True)
                borders.to_csv(out/'qc'/f'{name}_detector_borders.csv',index=False)
                ref_candidates.to_csv(out/'qc'/f'{name}_threshold_reference_candidates.csv',index=False,float_format='%.17g')
            else:
                old=pd.read_csv(source/'audit/box7_threshold1p00.csv',float_precision='round_trip')
                old=old.loc[old.frame.between(0,249)]
                threshold=spec['absolute_threshold']
                np.testing.assert_allclose(old.absolute_threshold,threshold,rtol=0,atol=1e-10)
                assert (old.box_size==7).all() and (old.threshold_factor==1.).all()
                candidates=pd.DataFrame({'frame':old.frame.to_numpy(),'x':old.candidate_x.to_numpy(),
                    'y':old.candidate_y.to_numpy(),'net_gradient':old.net_gradient.to_numpy(),'n_id':old.candidate_id.to_numpy()})
                threshold_stats={'baseline_threshold_adu_gradient':threshold,'source':'Original frozen Box7 factor1.00 detection threshold','candidate_cache':str(source/'audit/box7_threshold1p00.csv')}
                assert len(candidates)==18822 and candidates.n_id.is_unique
            candidates['n_id']=(candidates.frame.astype(np.int64)*(684*428)+candidates.y.astype(np.int64)*428+candidates.x.astype(np.int64)).astype(np.uint32)
            candidates.to_csv(out/'fits'/f'{name}_candidates.csv',index=False,float_format='%.17g')
            dump(out/'fits'/f'{name}_threshold.json',threshold_stats)
            print(f'{name}: threshold={threshold!r}, {len(candidates)} candidates before fit QC/ROI',flush=True)
            tables=[];timings=[]
            for s in range(0,250,c['fit_batch_frames']):
                ids=candidates.loc[candidates.frame.between(s,s+c['fit_batch_frames']-1)].reset_index(drop=True)
                with fit_adapter(detector):
                    table,theta,timing=pc.fit_candidates(ids,frames[s:s+c['fit_batch_frames']],movie[s:s+c['fit_batch_frames']],box,fit_config)
                np.save(out/'fits'/f'{name}_theta_f{s:04d}.npy',theta,allow_pickle=False)
                tables.append(table);timings.append(timing)
                print(f'{name}: fitted frames {s}–{s+c["fit_batch_frames"]-1}, {len(ids)} candidates',flush=True)
            table=annotate(pd.concat(tables,ignore_index=True),box,fit_config)
            table['threshold_factor']=1.;table['absolute_threshold']=threshold
            table['detection_box_size']=detector
            table['experimental_even_box']=spec['experimental_even_box']
            table['fit_crop_origin_x']=table.candidate_x-box//2
            table['fit_crop_origin_y']=table.candidate_y-box//2
            table['fit_crop_side_px']=box
            table['fit_method_note']='experimental exact6 crop [-3,+3); unchanged7 detector candidates' if box==6 else 'standard Picasso odd5 detection and fit'
            inside,status,selected=roi_flags(table,vertices)
            table['roi_inside']=inside;table['roi_geometry_status']=status;table['stats_selected_roi']=selected
            kept=table.loc[selected].copy()
            assert len(table)==len(candidates) and table.candidate_id.is_unique
            np.testing.assert_array_equal(table.candidate_id,candidates.n_id)
            assert kept.qc_unflagged_model_fit.all() and kept.roi_inside.all()
            metadata={'source_tiff':str(tiff),'source_sha256':before[0]['sha256'],'group':spec,
                'absolute_threshold':threshold,'camera':fit_config['camera'],'roi_sha256':roi_info['roi_sha256'],
                'original_frame_range':[0,249],'fit_box_origin_rule':'candidate - floor(box/2)',
                'detection_zero_support':'original detection box plus one-pixel axial halo; corners excluded',
                'calibration_status':'manufacturer_nominal','quantitatively_calibrated':False,'tardis':False}
            pc.save_locs(out/'audit'/f'{name}.hdf5',table,metadata)
            pc.save_locs(out/'selected'/f'{name}.hdf5',kept,metadata)
            perframe=pd.DataFrame({'frame':np.arange(250)}).set_index('frame')
            for label,mask in {'candidates':np.ones(len(table),bool),'qc_pass':table.qc_unflagged_model_fit,
                'roi_selected':table.stats_selected_roi,**{k:table[k] for k in EXCLUSIONS}}.items():
                perframe[label]=table.loc[mask].groupby('frame').size().reindex(perframe.index,fill_value=0)
            perframe.to_csv(out/'qc'/f'{name}_per_frame.csv')
            summary={'group':name,'fit_box':box,'detection_box':detector,'absolute_threshold':threshold,
                'candidates':len(table),'converged':int(table.fit_converged.sum()),'qc_pass':int(table.qc_unflagged_model_fit.sum()),
                'roi_selected':len(kept),'frames':250,'experimental_even_box':spec['experimental_even_box'],
                'median_lpx_nm_nominal':float(kept.lpx_nm.median()),'median_lpy_nm_nominal':float(kept.lpy_nm.median()),
                'fit_runtime_s':sum(t['fit_runtime_seconds'] for t in timings)}
            summaries.append(summary)
            render=dict(base_render,source_roi_run=str(out),source_group=name,expected_localizations=len(kept),
                        box_size=box,absolute_threshold=threshold,detection_box_size=detector,
                        experimental_even_box=spec['experimental_even_box'])
            render_path=out/f'render_config_{name}.yaml'
            render_path.write_text(yaml.safe_dump(render,sort_keys=False))
            render_configs.append(render_path)
        after=[fingerprint(p) for p in protect]
        assert before==after,'A read-only source changed during fitting.'
        dump(out/'source_fingerprints_after.json',after)
        pd.DataFrame(summaries).to_csv(out/'fit_summary.csv',index=False,float_format='%.17g')
        dump(out/'fit_validation.json',{'passed':True,'all250_source_pages_pixel_exact':True,'roi_mat_vertices_rechecked_equal':True,
            'all_candidates_preserved':True,'original_source_files_unchanged':True,'empty_frames_not_compressed':True,
            'original_candidate_detections_preserved_for_box6':True,'fit_frames_limited_to':[0,249],
            'boxes':summaries,'elapsed_seconds':time.perf_counter()-start})
        dump(out/'COMPLETED.json',{'fitting_completed':True,'groups':summaries,'output_directory':str(out)})
        for render_path in render_configs:
            print('Rendering PNGs/video for',render_path.name,flush=True)
            command=[sys.executable,'-B','-u',str(ROOT/'render_frames.py'),'--config',str(render_path)]
            with (out/f'{render_path.stem}_command.txt').open('w') as f:f.write(shlex.join(command)+'\n')
            subprocess.run(command,check=True)
        dump(out/'OVERLAYS_COMPLETED.json',{'all_checks_passed':True,'fit_groups':summaries,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start})
        print('ALL_COMPLETE:',out,flush=True)
    except Exception as exc:
        dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    main()
