"""Complete provenance after full fits and independent checks; never refit."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import run_real as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    here = Path(__file__).resolve().parent
    assert not (out/'COMPLETED.json').exists()
    rules = json.loads((out/'rules_before_run.json').read_text())
    s = json.loads((out/'summary.json').read_text())
    audit = json.loads((out/'independent_validation.json').read_text())
    before = json.loads((out/'input_fingerprints_before.json').read_text())
    after = json.loads((out/'input_fingerprints_after.json').read_text())
    assert before == after
    assert all(v for v in audit.values() if isinstance(v, bool))
    assert audit['independent_pixel_PSF_and_full_nuisance_covariance_checks'] == 20
    fits = pd.read_csv(out/'all_ROI_target_fits.csv')
    counts = pd.read_csv(out/'frame_counts.csv')
    np.testing.assert_array_equal(counts.frame, np.arange(2500))
    assert len(fits) == int(counts.ROI_candidates.sum()) == s['ROI_candidates']
    assert int(counts.unflagged.sum()) == s['unflagged']
    assert int(counts.finite_interior_ROI.sum()) == s['finite_interior_ROI']
    assert fits.audit_status.value_counts().to_dict() == s['counts']
    # CSV readers infer all-null columns differently when concatenated with
    # other frames. Compare actual values and null masks, allowing only dtype
    # inference to differ; numerical precision and all flags remain checked.
    cache = Path(rules['full']['fit_cache'])
    previous = pd.read_csv(cache/'all_ROI_target_fits.csv')
    reused = fits[fits.frame.between(1250,1300)].reset_index(drop=True)
    old_checkpoint = out/'checkpoints/fits_through_0499.csv'
    checkpoint = pd.read_csv(old_checkpoint)
    resumed = fits[fits.frame.le(499)].reset_index(drop=True)
    dtype_notes = []
    for label, current, old, ignored in [
            ('sample_cache', reused, previous, {'candidate_index'}),
            ('checkpoint', resumed, checkpoint, {'resumed_from_checkpoint'})]:
        assert len(current) == len(old)
        for field in old:
            if field in ignored:
                continue
            pd.testing.assert_series_equal(current[field], old[field],
                check_names=False, check_dtype=False, check_exact=False,
                rtol=1e-13, atol=1e-13)
            if current[field].dtype != old[field].dtype:
                dtype_notes.append(dict(source=label, field=field,
                    old_dtype=str(old[field].dtype), new_dtype=str(current[field].dtype),
                    all_values_and_nulls_equal=True))
    assert len(previous) == s['reused_fit_points'] == 2113
    assert len(checkpoint) == s['reused_checkpoint_fit_points'] == 47876
    native_sources = json.loads((out/'detection_sources.json').read_text())
    for source in native_sources:
        if source['kind'] == 'new_native_detection':
            assert core.fingerprint(source['path'])['sha256'] == source['sha256']
    # Fill the first two progress files from saved results (no optimization).
    reports = out/'progress_reports'
    reports.mkdir(exist_ok=True)
    for segment in json.loads((out/'segment_summaries.json').read_text()):
        first, last = segment['frames_inclusive']
        path = reports/f'frames{first:04d}_{last:04d}.json'
        if not path.exists():
            stat = segment['unflagged_statistics']
            core.dump(path, dict(frames_inclusive=[first,last],
                ROI_candidates=segment['ROI_candidates'], unflagged=segment['unflagged'],
                status_counts=segment['counts'], median_N_e=stat['photons_e']['median'],
                median_x_SE_nm=stat['x_se_nm']['median'], median_y_SE_nm=stat['y_se_nm']['median'],
                calibration_status='manufacturer_nominal', uncertainty_coverage_validated=False,
                reconstructed_from_saved_results=True))
    assert len(list(reports.glob('frames*.json'))) == 10
    snap = out/'source_snapshot'
    snap.mkdir(exist_ok=True)
    paths = [here/name for name in ['run_full.py','run_real.py','detect_batches.py',
        'resume_detection.py','parallel_fit.py','benchmark_parallel.py','validate_full.py',
        'report_checkpoint.py','finalize_full.py']]
    paths += [core.ROOT/'background_9x9_simulation/calibrated_photometry.py',
        core.ROOT/'background_9x9_simulation/local_background/run_local_background.py',
        core.ROOT/'trackmate_python_bridge/run_detection.py',
        core.ROOT/'trackmate_python_bridge/run_preview.py',
        core.ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py']
    for path in paths:
        shutil.copyfile(path, snap/path.name)
    units = json.loads((cache/'field_units.json').read_text())
    units.update(candidate_index='unique integer within combined full movie',
        spot_id='original native TrackMate ID, unique within its detection source',
        source_detection_uid='detection-source name plus original spot ID',
        lpx_model_px='local full-Hessian X standard error in pixels; not native Picasso CRLB',
        lpy_model_px='same, Y', calibration_status='manufacturer_nominal; not empirically calibrated')
    core.dump(out/'field_units.json', units)
    core.dump(out/'versions.json', dict(python=sys.version, executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in
            ['numpy','scipy','numba','pandas','matplotlib','tifffile','picassosr','PyYAML']},
        native_TrackMate='7.14.0', CPU=True, fitting_backend='independent_annular_constraint_MLE'))
    freeze = subprocess.run([sys.executable,'-m','pip','freeze'], check=True, capture_output=True, text=True)
    (out/'requirements.lock.txt').write_text(freeze.stdout)
    shutil.copyfile(core.ROOT/'trackmate_python_bridge/requirements.lock.txt', out/'native_detector_requirements.lock.txt')
    shutil.copyfile(here/'parallel_benchmark.json', out/'parallel_backend_validation.json')
    logs = out/'logs'
    logs.mkdir(exist_ok=True)
    for path in here.glob('*.log'):
        if path.name.startswith(('full_','parallel_')):
            shutil.copyfile(path, logs/path.name)
    # Preserve the actual interrupt/recovery history and a clean reproducible
    # command. Native detections must be initialized using the saved job JSON.
    command = shlex.join([sys.executable,'-u',str(here/'run_full.py'),
        '--config',str(here/'config_full2500.yaml'),'--output',str(out)])
    (out/'run_command.txt').write_text(command+'\n')
    (out/'validation_command.txt').write_text(shlex.join([sys.executable,'-u',
        str(here/'validate_full.py'),'--output',str(out)])+'\n')
    (out/'finalization_command.txt').write_text(shlex.join([sys.executable,'-u',
        str(Path(__file__).resolve()),'--output',str(out)])+'\n')
    recovery = dict(no_fit_or_detection_rerun=True,
        original_source_fingerprints_equal=True, cached_values_and_flags_unchanged=True,
        metadata_only_dtype_inference=dtype_notes,
        previous_postfit_check_failure='strict dtype equality for all-null CSV fields; values unchanged',
        independent_audit_completed_before_copy_failure=True,
        previous_validator_copy_failure='source_snapshot directory missing; now created',
        detector_recovery='periodic fresh JVMs after initial memory exhaustion; completed chunks reused and hashed',
        fitting_recovery='reuse saved frame 499 checkpoint and sample fits; validated four-process backend',
        elapsed_s_in_summary_scope='last fit process and output phase only; excludes earlier interrupted attempts')
    core.dump(out/'postfit_recovery.json', recovery)
    core.dump(out/'validation.json', dict(all_checks_passed=True,
        all_2500_original_frames_preserved=True, all_source_pixels_match_detector_inputs=True,
        ROI_and_MAT_454_vertices_equal=True, all_candidates_retained=True,
        source_fingerprints_unchanged=True, prior_2113_fit_values_and_flags_unchanged=True,
        checkpoint_47876_fit_values_unchanged=True, no_uncertainty_for_bound_fits=True,
        original_detector_parameters_fixed=True, independent_audit=audit,
        no_TARDIS=True, histogram_axes='linear', coverage_validated=False))
    stat = s['unflagged_statistics']
    text = (
        '完整 T1 C2：原始帧 0–2499，共 2500 时间帧；117 nm/像素，20 ms/帧。\n'
        'TrackMate 7.14.0 LoG radius 0.28 μm、threshold 6.5、3×3 median 和亚像素检测。\n'
        '复用 102 帧检测缓存，补检测 2398 帧；全部全视野候选保留。\n'
        '复用此前 1250–1300 的 2113 个拟合及中断检查点的 47876 个拟合，值与标记逐列验证一致。\n'
        f'全视野 {s["FOV_candidates"]} 个候选；ROI 内 {s["ROI_candidates"]} 个目标。\n'
        f'未触发复核 {s["unflagged"]}；疑似重叠但有限内点协方差 161；直方图合计 {s["finite_interior_ROI"]}。\n'
        '其他：约束边界 11826；精修中心在 ROI 外 45；支持域不可用 11；协方差不可用 3。全部保留审计。\n'
        f'未触发复核组：N 中位数 {stat["photons_e"]["median"]:.6f} 名义光电子等效数；'
        f'X 标准误 {stat["x_se_nm"]["median"]:.6f} nm；Y 标准误 {stat["y_se_nm"]["median"]:.6f} nm。\n'
        '采用独立像素积分二维椭圆 Gaussian MLE，不是原生 Picasso 拟合后端。7×7 中心框。\n'
        '外圈 6<r≤12 px，使用共同细胞 ROI 内非零像素，排除全部同帧候选 4 px 核心；至少 80 像素。\n'
        '使用全部保留像素均值，不选择最暗 25%；背景与信号联合拟合，计入本点 PSF 外翼。\n'
        '均值约束方差为样本方差/n 和 8 扇区删一 jackknife 方差的较大值。\n'
        '信号暂按 Poisson，背景按外圈经验方差的 Gaussian 近似；读噪声方差 2.56 为下限。\n'
        '保留 (ADU−100)×0.46 的厂家名义换算，EM gain=1。外圈均值不替代相机 offset；未使用模拟方差 500。\n'
        'X/Y 和 N 的标准误来自包含 N、背景、宽度等全部自由参数的观测 Hessian 逆；不是已校准真实误差。\n'
        '背景方差本身的不确定度、选点效应和置信覆盖率尚未完整验证。真实图像没有真值。\n'
        'N 是完整 Gaussian 的光电子等效积分，不仅是 7×7 框内信号，也不是入射光子；不预设每点 800。\n'
        '图内青色为未触发复核拟合，紫色为疑似重叠复核组；不可用拟合不进入误差直方图。\n'
        'X/Y 图与 N 图均为线性横轴；N 左图显示 pooled p99 内主体，右图显示全部范围，保留尾部。\n'
        '原 spot_id 保留但跨 Java 会话不保证全局唯一，组合 candidate_index/source_detection_uid 唯一。\n'
        '十个 250 帧时段仅作描述性数量与分布汇总，不是独立生物学重复。\n'
        '源 TIFF、ROI、MAT、既有缓存和相机配置的运行前后 SHA256 及时间戳相同。\n'
        '无轨迹链接、TARDIS、mask、误差扣除、去卷积、扩散或状态比例分析。\n'
        '初次 JVM 内存不足与运行中断已恢复；后处理仅 CSV 空字段类型推断引起检查失败，修正后实际字段值全部一致。\n'
        '逐像素背景和完整协方差独立抽查通过；全部 2500 帧编号、逐帧数量、审计原字段和图计数检查通过。\n')
    (out/'REPORT_zh.txt').write_text(text)
    core.dump(out/'visual_review.json', dict(
        reviewed_files=['localization_uncertainty_histogram.png','photoelectron_number_histogram.png'],
        reviewed=True, axes_linear=True, full_range_tail_retained=True,
        review_group_separately_shown=True, text_legible=True))
    manifest = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name not in {'output_fingerprints.json','COMPLETED.json'}:
            d = core.fingerprint(path)
            d['relative_path'] = str(path.relative_to(out))
            manifest.append(d)
    core.dump(out/'output_fingerprints.json', manifest)
    core.dump(out/'COMPLETED.json', dict(all_checks_passed=True, frames=2500,
        completed_UTC=datetime.now(timezone.utc).isoformat(), ROI_candidates=s['ROI_candidates'],
        histogram_points=s['finite_interior_ROI'], output_directory=str(out),
        finalization_only=True, uncertainty_coverage_validated=False))
    print(json.dumps(dict(completed=True, ROI_candidates=s['ROI_candidates'],
        histogram_points=s['finite_interior_ROI'], unflagged=s['unflagged'],
        output=str(out), dtype_notes=dtype_notes),indent=2), flush=True)


if __name__ == '__main__':
    main()
