"""Full 2500-frame extension, with exact cached detections/fits reused.

Native detector chunks may run concurrently. Frame numbers never compressed.
Only the two requested histogram figures are generated; no TARDIS or masks.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import yaml

import run_real as core
from run_preview import annotate_roi
from parallel_fit import fit_batch


def decorate(table, vertices, c):
    table = table.copy()
    finite = np.isfinite(table[['fit_x_px','fit_y_px']]).all(axis=1)
    inside = np.zeros(len(table),bool)
    inside[finite] = core.roi_lib.points_in_polygon(table.loc[finite,['fit_x_px','fit_y_px']].to_numpy(),vertices)
    table['fitted_center_inside_ROI'] = inside
    table['histogram_eligible'] = table.fit_usable_interior & table.finite_uncertainty & inside
    table['qc_unflagged_model_fit'] = table.histogram_eligible & ~table.suspect_overlap
    table['lpx_model_nm'] = table.x_se_nm; table['lpy_model_nm'] = table.y_se_nm
    table['lpx_model_px'] = table.x_se_nm/c['pixel_size_nm']; table['lpy_model_px'] = table.y_se_nm/c['pixel_size_nm']
    table['audit_status'] = np.select([~table.fit_attempted, ~table.fit_success, table.active_bound,
        ~table.finite_uncertainty, ~table.fitted_center_inside_ROI, table.suspect_overlap],
        ['unavailable_support','nonconverged','active_parameter_bound','unavailable_covariance',
         'refined_center_outside_ROI','overlap_review'],default='unflagged_model_fit')
    return table


def summary(table, backgrounds, total, first, last):
    clean = table[table.qc_unflagged_model_fit]
    eligible = table[table.histogram_eligible]
    return dict(frames_inclusive=[first,last],number_frames=last-first+1,FOV_candidates=total,ROI_candidates=len(table),
        counts={str(k):int(v) for k,v in table.audit_status.value_counts().items()},
        fit_attempted=int(table.fit_attempted.sum()),converged=int(table.fit_success.sum()),
        interior_model_fits=int(table.fit_usable_interior.sum()),finite_interior_ROI=len(eligible),unflagged=len(clean),
        unflagged_statistics={k:core.stats(clean[k]) for k in
            ['x_se_nm','y_se_nm','photons_e','N_se_e','bg_e_per_pixel','sx_px','sy_px','standardized_residual_rms']},
        eligible_statistics={k:core.stats(eligible[k]) for k in ['x_se_nm','y_se_nm','photons_e']},
        background_mean_ADU=core.stats(backgrounds.mean_adu),background_mean_e=core.stats(backgrounds.mean_e),
        background_variance_e2=core.stats(backgrounds.variance_e2),
        background_constraint_SE_e=core.stats(backgrounds.mean_constraint_se_e),
        variance_to_nominal_Poisson_ratio=core.stats(backgrounds.sample_variance_to_nominal_Poisson_ratio))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); started=time.perf_counter()
    f=yaml.safe_load(args.config.read_text()); ref=args.config.parent/f['reference_sample_config'];c=yaml.safe_load(ref.read_text())
    assert f['frames_inclusive']==[0,2499] and f['histogram_axes']=='linear'
    assert not any(f[k] for k in ['tracking','tardis','mitochondria_analysis'])
    out=args.output;assert out.exists() and not (out/'COMPLETED.json').exists()
    print('OUTPUT:',out,flush=True)
    preview=yaml.safe_load((Path(c['source_preview'])/'resolved_config.yaml').read_text())
    tiff=Path(preview['input_tiff']);camera_path=(ref.parent/c['nominal_camera_config']).resolve()
    camera=yaml.safe_load(camera_path.read_text())['camera']
    fit_cache=Path(f['fit_cache'])
    completed_checkpoints=sorted((out/'checkpoints').glob('fits_through_*.csv'))
    resume_path=completed_checkpoints[-1] if completed_checkpoints else None
    resume_fits=pd.read_csv(resume_path) if resume_path else None
    resume_last=int(resume_fits.frame.max()) if resume_fits is not None else -1
    if resume_path:
        print(f'Reusing checkpoint through frame {resume_last}: {len(resume_fits):,} fits',flush=True)
    cache_config=yaml.safe_load((fit_cache/'config.yaml').read_text())
    for name in ['box_px','annulus_inner_radius_px','annulus_outer_radius_px','neighbor_core_radius_px',
                 'minimum_background_pixels','minimum_jackknife_sectors','initial_sigma_multistart_px',
                 'width_bounds_px','fit_max_iterations','pixel_size_nm','frame_interval_s','noise_model']:
        assert c[name]==cache_config[name],name
    protected=[tiff,Path(preview['roi_file']),Path(preview['mat_file']),camera_path,
        fit_cache/'all_ROI_target_fits.csv',fit_cache/'annular_background_all_ROI_targets.csv',
        fit_cache/'retained_annular_pixels.npz',fit_cache/'summary.json',fit_cache/'config.yaml',
        Path(core.base.__file__),core.ROOT/'trackmate_python_bridge/picasso_refinement.py',
        Path(preview['fiji'])/'jars/TrackMate-7.14.0.jar']
    if resume_path:protected.append(resume_path)
    cache_groups={}; cache_hashes={};cache_manifests={}; detection_sources=[]
    for directory in f['detection_caches']:
        src=Path(directory);p=yaml.safe_load((src/'resolved_config.yaml').read_text())
        assert p['input_tiff']==str(tiff) and p['radius_um']==.28 and p['threshold']==6.5
        assert p['median_filter'] and p['subpixel_localization'] and p['trackmate_version']=='7.14.0'
        assert p['expected_tiff_sha256']==preview['expected_tiff_sha256']
        hashes=json.loads((src/'output_fingerprints.json').read_text())
        assert core.fingerprint(src/'all_detections.csv')['sha256']==hashes['all_detections_sha256']
        t=pd.read_csv(src/'all_detections.csv')
        manifest=pd.read_csv(src/'frame_manifest.csv')
        for frame in range(p['frame_start'],p['frame_stop_exclusive']):
            assert frame not in cache_groups
            cache_groups[frame]=t[t.frame.eq(frame)].copy();cache_hashes[frame]=str(src)
            cache_manifests[frame]=manifest.loc[manifest.frame.eq(frame),'source_pixel_sha256'].iloc[0]
        protected.extend(src/name for name in ['all_detections.csv','resolved_config.yaml','frame_manifest.csv','output_fingerprints.json'])
        detection_sources.append(dict(path=str(src),frames=[p['frame_start'],p['frame_stop_exclusive']-1],kind='existing_cache'))
    print('Fingerprinting read-only sources...',flush=True)
    before=[core.fingerprint(p) for p in protected];core.dump(out/'input_fingerprints_before.json',before)
    assert before[0]['sha256']==preview['expected_tiff_sha256'] and before[1]['sha256']==preview['expected_roi_sha256']
    vertices=core.roi_lib.load_imagej_roi(preview['roi_file'])
    mat_vertices,mat_record=core.read_mat_vertices(preview['mat_file'],preview['mat_record'])
    np.testing.assert_array_equal(vertices,mat_vertices)
    np.testing.assert_array_equal(vertices,pd.read_csv(Path(c['source_preview'])/'roi_vertices.csv').to_numpy())
    assert len(vertices)==454
    yy,xx=np.mgrid[:684,:428]
    roi_mask=core.roi_lib.points_in_polygon(np.column_stack([xx.ravel(),yy.ravel()]),vertices).reshape(684,428)
    pd.DataFrame(vertices,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    core.dump(out/'rules_before_run.json',dict(full=f,fit_rules=c,camera=camera,ROI_coordinate_transform='identity/native C2',
        native_TrackMate=dict(version='7.14.0',radius_um=.28,threshold=6.5,median_filter=True,subpixel=True),
        candidate_identity='unique combined candidate_index and source_detection_uid; original spot_id retained',
        no_TARDIS_or_temporal_pairing=True,formal_accuracy_not_available=True))
    core.dump(out/'mat_roi_validation.json',dict(vertices=454,MAT_exactly_equal=True,record=mat_record,
        boundary_included=True,boundary_tolerance_px=1e-9,coordinate_transform='identity/native C2'))
    shutil.copyfile(ref,out/'sample_fit_rules.yaml');shutil.copyfile(camera_path,out/'nominal_camera_source.yaml')
    core.dump(out/'math_validation.json',core.math_checks())
    previous_fit=pd.read_csv(fit_cache/'all_ROI_target_fits.csv')
    previous_bg=pd.read_csv(fit_cache/'annular_background_all_ROI_targets.csv')
    previous_masks=np.load(fit_cache/'retained_annular_pixels.npz')
    old_index={int(i):j for j,i in enumerate(previous_masks['candidate_index'])}
    job=json.loads((out/'native_detection_job.json').read_text())
    chunk_map={frame:(a,b) for a,b in job['chunks'] for frame in range(a,b)}
    assert set(chunk_map).isdisjoint(cache_groups) and set(chunk_map)|set(cache_groups)==set(range(2500))
    current_chunk=None;chunk_groups={};chunk_hashes={};total=0
    original_tables=[];fit_tables=[];bg_tables=[];all_masks=[];mask_indices=[];pages=[];frame_counts=[]
    reused_points=0;fresh_points=0;checkpoint_points=0;chunks_done=set();checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    assert f['fitting_execution_backend']=='process_groups' and f['workers']==4
    shutil.copyfile(args.config,out/'config_full2500.yaml')
    with tifffile.TiffFile(tiff) as tf, concurrent.futures.ProcessPoolExecutor(
            max_workers=f['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        assert len(tf.pages)==2500 and tf.series[0].shape==(2500,684,428) and tf.series[0].axes=='TYX'
        for frame in range(2500):
            if frame in cache_groups:
                table=cache_groups[frame].copy();expected_hash=cache_manifests[frame];src=cache_hashes[frame]
            else:
                a,b=chunk_map[frame];tag=f'frames{a:04d}_{b-1:04d}'
                if current_chunk!=(a,b):
                    p=out/'native_detection'/(tag+'.json')
                    while not p.exists():
                        if time.perf_counter()-started>7200: raise TimeoutError('Native detector completion not published')
                        time.sleep(1)
                    report=json.loads(p.read_text());csv=out/'native_detection'/(tag+'.csv')
                    assert core.fingerprint(csv)['sha256']==report['csv_sha256']
                    assert report['trackmate_version']=='7.14.0' and report['radius_um']==.28 and report['threshold']==6.5
                    assert report['median_filter'] and report['subpixel'] and not report['tracking']
                    native=pd.read_csv(csv)
                    assert len(native)==report['rows']
                    counts=native.groupby('frame').size().reindex(range(a,b),fill_value=0).to_numpy()
                    np.testing.assert_array_equal(counts,report['counts'])
                    chunk_groups={int(fr):g.copy() for fr,g in native.groupby('frame')}
                    chunk_hashes={r['frame']:r['pixel_sha256'] for r in report['page_hashes']}
                    current_chunk=(a,b);chunks_done.add(tag)
                    detection_sources.append(dict(path=str(csv),frames=[a,b-1],kind='new_native_detection',sha256=report['csv_sha256']))
                table=chunk_groups.get(frame,pd.DataFrame(columns=['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response','radius_setting_um']))
                table=annotate_roi(table,vertices);table['radius_setting_px']=table.radius_setting_um/.117
                expected_hash=chunk_hashes[frame];src=str(out/'native_detection'/(tag+'.csv'))
            table=table.reset_index(drop=True)
            table.insert(0,'candidate_index',np.arange(total,total+len(table),dtype=np.int64))
            table['source_detection_cache']=src
            table['source_detection_uid']=[f'{Path(src).name}:{int(i)}' for i in table.spot_id]
            total+=len(table);original_tables.append(table.copy())
            np.testing.assert_array_equal(table.roi_inside.eq(1),core.roi_lib.points_in_polygon(table[['x_px','y_px']].to_numpy(float),vertices))
            np.testing.assert_allclose(table.time_s,table.frame*.02,rtol=0,atol=1e-12)
            targets=table[table.roi_inside.eq(1)]
            raw=tf.pages[frame].asarray();pixel_hash=hashlib.sha256(raw.tobytes()).hexdigest()
            assert raw.shape==(684,428) and raw.dtype==np.uint16 and pixel_hash==expected_hash
            pages.append(dict(frame=frame,pixel_sha256=pixel_hash,detection_source=src))
            if not len(targets):
                frame_counts.append(dict(frame=frame,time_s=frame*.02,FOV_candidates=len(table),ROI_candidates=0,finite_interior_ROI=0,unflagged=0))
                continue
            if frame<=resume_last:
                result=resume_fits[resume_fits.frame.eq(frame)].copy().reset_index(drop=True)
                np.testing.assert_array_equal(result.spot_id,targets.spot_id)
                np.testing.assert_array_equal(result.candidate_index,targets.candidate_index)
                background,masks,movie=core.background_geometry(targets,table,raw,roi_mask,c,camera)
                checkpoint_points+=len(result)
            elif frame in range(1250,1301):
                cached=previous_fit[previous_fit.frame.eq(frame)].copy().reset_index(drop=True)
                background=previous_bg[previous_bg.frame.eq(frame)].copy().reset_index(drop=True)
                np.testing.assert_array_equal(targets.spot_id,cached.spot_id)
                np.testing.assert_array_equal(targets.spot_id,background.spot_id)
                masks=[]
                for i in cached.candidate_index:
                    j=old_index[int(i)];masks.append(previous_masks['xy_pixels'][previous_masks['offsets'][j]:previous_masks['offsets'][j+1]].copy())
                cached['candidate_index']=targets.candidate_index.to_numpy();background['candidate_index']=targets.candidate_index.to_numpy()
                result=cached;result['fit_result_source']=str(fit_cache);reused_points+=len(result)
            else:
                background,masks,movie=core.background_geometry(targets,table,raw,roi_mask,c,camera)
                items=[(int(r.candidate_index),int(r.spot_id),frame,float(r.x_px),float(r.y_px)) for r in targets.itertuples()]
                groups=np.array_split(np.arange(len(items)),f['workers'])
                payloads=[([items[j] for j in group],raw,movie,background.iloc[group].reset_index(drop=True),
                           [masks[j] for j in group],c) for group in groups if len(group)]
                processed=list(pool.map(fit_batch,payloads))
                result=pd.DataFrame([r for group in processed for r in group]);result['fit_result_source']='new_fit_same_fixed_rules';fresh_points+=len(result)
            result=decorate(result,vertices,c)
            result['resumed_from_checkpoint']=frame<=resume_last
            np.testing.assert_array_equal(result.spot_id,targets.spot_id)
            fit_tables.append(result);bg_tables.append(background);all_masks.extend(masks);mask_indices.extend(targets.candidate_index)
            d=dict(frame=frame,time_s=frame*.02,FOV_candidates=len(table),ROI_candidates=len(result),
                finite_interior_ROI=int(result.histogram_eligible.sum()),unflagged=int(result.qc_unflagged_model_fit.sum()))
            d.update({str(k):int(v) for k,v in result.audit_status.value_counts().items()});frame_counts.append(d)
            if frame%25==0 or frame==2499:
                print(f'Frame {frame}/2499: ROI fitted {fresh_points+reused_points+checkpoint_points:,}, unflagged this frame {d["unflagged"]}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
            if (frame+1)%f['checkpoint_frames']==0:
                checkpoint=pd.concat(fit_tables,ignore_index=True)
                checkpoint_path=checkpoints/f'fits_through_{frame:04d}.csv'
                if not checkpoint_path.exists():checkpoint.to_csv(checkpoint_path,index=False)
                pd.DataFrame(frame_counts).fillna(0).to_csv(out/'frame_counts_progress.csv',index=False)
    print('Writing combined audit tables and histograms...',flush=True)
    original=pd.concat(original_tables,ignore_index=True);fits=pd.concat(fit_tables,ignore_index=True);bg=pd.concat(bg_tables,ignore_index=True)
    assert original.candidate_index.is_unique and original.source_detection_uid.is_unique
    assert fits.candidate_index.is_unique and len(fits)==fresh_points+reused_points+checkpoint_points
    np.testing.assert_array_equal(fits.candidate_index,original.loc[original.roi_inside.eq(1),'candidate_index'])
    original.to_csv(out/'all_detections.csv',index=False)
    fits.to_csv(out/'all_ROI_target_fits.csv',index=False);bg.to_csv(out/'annular_background_all_ROI_targets.csv',index=False)
    np.savez_compressed(out/'retained_annular_pixels.npz',xy_pixels=np.concatenate(all_masks),
        offsets=np.r_[0,np.cumsum([len(p) for p in all_masks])],candidate_index=np.array(mask_indices))
    joined=original.join(fits.set_index('candidate_index').drop(columns=['spot_id','frame','time_s']).add_prefix('annular__'))
    pd.testing.assert_frame_equal(joined[original.columns],original)
    joined.to_csv(out/'all_original_detections_with_new_fits.csv',index=False)
    counts=pd.DataFrame(frame_counts).fillna(0);counts.to_csv(out/'frame_counts.csv',index=False)
    np.testing.assert_array_equal(counts.frame,np.arange(2500));assert counts.ROI_candidates.sum()==len(fits)
    s=summary(fits,bg,len(original),0,2499)
    s.update(reused_fit_points=reused_points,reused_checkpoint_fit_points=checkpoint_points,new_fit_points=fresh_points,cached_detection_frames=len(cache_groups),
        new_detection_frames=len(chunk_map),calibration_status='manufacturer_nominal',noise_model=c['noise_model'],
        model_is_native_Picasso=False,coverage_validated=False,truth_available=False,output_directory=str(out),elapsed_s=time.perf_counter()-started)
    core.dump(out/'summary.json',s);core.plots(out,fits,s)
    segments=[];flat=[]
    for a in range(0,2500,250):
        q=fits[fits.frame.between(a,a+249)];b=bg[bg.frame.between(a,a+249)]
        segment=summary(q,b,int(counts.loc[counts.frame.between(a,a+249),'FOV_candidates'].sum()),a,a+249)
        segments.append(segment)
        r=dict(frame_start=a,frame_end=a+249,ROI_candidates=len(q),unflagged=segment['unflagged'],finite_interior_ROI=segment['finite_interior_ROI'])
        for field,d in segment['unflagged_statistics'].items():
            for stat in ['median','p10','p90']:r[field+'_'+stat]=d[stat]
        flat.append(r)
    core.dump(out/'segment_summaries.json',segments);pd.DataFrame(flat).to_csv(out/'segment_summary.csv',index=False)
    core.dump(out/'source_page_manifest.json',pages);core.dump(out/'detection_sources.json',detection_sources)
    assert len(pages)==2500 and len(chunks_done)==len(job['chunks'])
    after=[core.fingerprint(p) for p in protected];core.dump(out/'input_fingerprints_after.json',after);assert before==after
    # Check that cache reuse changes ONLY the combined identifier/source marker.
    reused=fits[fits.frame.between(1250,1300)].reset_index(drop=True)
    for field in previous_fit:
        if field=='candidate_index':continue
        pd.testing.assert_series_equal(reused[field],previous_fit[field],check_names=False,check_dtype=False,check_exact=False,rtol=1e-13,atol=1e-13)
    assert reused_points==2113
    if resume_fits is not None:
        resumed=fits[fits.frame.le(resume_last)].reset_index(drop=True)
        for field in resume_fits:
            if field=='resumed_from_checkpoint':continue
            pd.testing.assert_series_equal(resumed[field],resume_fits[field],check_names=False,check_dtype=False,check_exact=False,rtol=1e-13,atol=1e-13)
    assert fits.loc[fits.active_bound|~fits.fit_success,['x_se_nm','y_se_nm','N_se_e']].isna().all().all()
    np.testing.assert_allclose(fits.lpx_model_px*117,fits.x_se_nm,equal_nan=True)
    core.dump(out/'validation.json',dict(all_2500_original_frames_preserved=True,all_source_pixels_match_detector_inputs=True,
        ROI_and_MAT_454_vertices_equal=True,all_candidates_retained=True,empty_frames_not_compressed=True,
        source_fingerprints_unchanged=True,prior_2113_fit_values_and_flags_unchanged=True,checkpoint_fit_values_unchanged=True,
        no_uncertainty_for_bound_fits=True,original_detector_parameters_fixed=True,no_TARDIS=True,histogram_axes='linear'))
    snap=out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(core.__file__),Path(__file__).with_name('detect_batches.py'),
              Path(core.base.__file__),core.ROOT/'background_9x9_simulation/local_background/run_local_background.py',
              Path(__file__).with_name('resume_detection.py'),Path(__file__).with_name('parallel_fit.py'),
              Path(__file__).with_name('benchmark_parallel.py'),core.ROOT/'trackmate_python_bridge/run_detection.py',core.ROOT/'trackmate_python_bridge/run_preview.py',
              core.ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py']:
        shutil.copyfile(p,snap/p.name)
    core.dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','numba','pandas','matplotlib','tifffile','picassosr','PyYAML']}))
    shutil.copyfile(core.ROOT/'trackmate_python_bridge/requirements.lock.txt',out/'native_detector_requirements.lock.txt')
    shutil.copyfile(Path(__file__).with_name('parallel_benchmark.json'),out/'parallel_backend_validation.json')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),'--output',str(out.resolve())])+'\n')
    (out/'REPORT_zh.txt').write_text(
        '完整 T1 C2 原始帧 0–2499；2500 时间帧，117 nm/像素、20 ms/帧。\n'
        'TrackMate 7.14.0 LoG radius 0.28 μm、threshold 6.5、3×3 median 与亚像素检测；只对缺少缓存的帧检测。\n'
        f'复用检测 {len(cache_groups)} 帧；补检测 {len(chunk_map)} 帧。复用 1250–1300 的 {reused_points} 个外圈约束拟合，拟合值与标记全部不变。\n'
        f'此外从中断前检查点复用 {checkpoint_points} 个拟合；仅重建相同背景支持像素，不重新拟合已保存的点。\n'
        f'全视野 {len(original)} 候选全部保留；共同细胞 ROI 内 {len(fits)} 个目标；未标记 {s["unflagged"]}；有限内点 ROI 拟合 {s["finite_interior_ROI"]}。\n'
        '独立像素积分二维椭圆 Gaussian + Poisson 信号/Gaussian 外圈经验背景近似；不是原生 Picasso 后端。\n'
        '7×7 拟合框；外圈 6<r≤12 px，限定同一 ROI，排除全部同帧候选 4 px 核心和原图零像素；至少 80 像素。\n'
        '背景使用全部保留像素的均值＋有限不确定度，约束包含本点 PSF 外翼。未选最暗 25% 或固定 b。\n'
        '背景均值方差取样本方差/n 与 8 扇区删一 jackknife 方差较大值；噪声方差取真实外圈方差，名义读噪声方差为下限。\n'
        '保留 (ADU−100)×0.46 的名义换算，EM gain 1，读噪声下限 1.6 电子 RMS。没有实测相机校准，不套用模拟方差 500。\n'
        'X/Y 和 N 的 ±1 标准误为含信号、背景、宽度的完整观测 Hessian 逆；背景方差估计、选点效应与误差覆盖率未完整校准。\n'
        'N 是完整 Gaussian 的光电子等效积分，不仅是 7×7 框内信号，也不是入射光子。\n'
        '直方图仅使用收敛、非边界、有有限正协方差且精修中心仍在 ROI 内的点；残差重叠复核组单独显示。\n'
        '原 spot_id 按检测来源保留；不同 Java 会话的 spot_id 不必全球唯一，以 candidate_index/source_detection_uid 联合审计。\n'
        '十个 250 帧时段仅做描述性数量/估计分布汇总，不是独立生物学重复，不做任何显著性或 TARDIS 检验。\n'
        '所有图使用线性坐标；N 左侧 p99 主体、右侧完整范围。真实图无真值，不能据此证明真实定位误差、光电子准确度或真值相关。\n'
        '没有轨迹链接、TARDIS、mask、误差扣除、去卷积、扩散或状态比例拟合。旧输入与结果只读。\n')
    core.dump(out/'COMPLETED.json',dict(all_checks_passed=True,frames=2500,elapsed_s=time.perf_counter()-started,output_directory=str(out)))
    print(json.dumps(s,indent=2),flush=True)


if __name__=='__main__':main()
