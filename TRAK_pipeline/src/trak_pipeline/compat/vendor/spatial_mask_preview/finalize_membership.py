"""Archive delivery checks and logs without changing computed membership scores."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

from run_preview import HERE, ROOT, dump, fingerprint


def main():
    out = Path((HERE/'latest_membership_output.txt').read_text().strip())
    summary = json.loads((out/'summary.json').read_text())
    validation = json.loads((out/'validation.json').read_text())
    assert validation['all_checks_passed']
    assert validation['independent_full_grid_points'] == 480
    assert summary['scored_points'] == 115013
    assert summary['maximum_absolute_partition_residual'] < 1e-11
    assert summary['unknown_mass_above_1e_minus_6'] == 0

    # Keep the executed runner snapshot; add its unchanged imported helpers.
    for name in ['run_full.py', 'run_preview.py', 'geometry.py', 'contracts.py']:
        shutil.copyfile(HERE/name, out/'source_snapshot'/name)
    shutil.copyfile(Path(__file__), out/'finalize_membership.py')
    shutil.copyfile(ROOT/'mitochondria_mask_pilot/envs/cleanup/requirements.lock.txt', out/'requirements.lock.txt')
    versions = json.loads((out/'versions.json').read_text())
    versions['packages']['shapely'] = importlib.metadata.version('shapely')
    dump(out/'versions.json', versions)
    (out/'replay_command.txt').write_text(shlex.join([
        sys.executable, str(out/'source_snapshot/run_membership_probability.py'),
        '--config', str(out/'config.json')])+'\n')

    checkpoints = [json.loads(p.read_text()) for p in sorted((out/'checkpoints').glob('checkpoint_*.json'))]
    assert [c['completed_frames'] for c in checkpoints] == list(range(250, 2501, 250))
    assert sum(c['block_scored'] for c in checkpoints) == 115013
    assert sum(c['block_QC_excluded'] for c in checkpoints) == 12046
    (out/'processing_checkpoints.log').write_text(
        'Processing records reconstructed from the saved checkpoint JSONs, not a captured stdout transcript.\n'
        + '\n'.join(json.dumps(c) for c in checkpoints)+'\n')

    now = datetime.now(timezone.utc).isoformat()
    entry = (
        '\n[2026-10-16] TDKd3 W5/min32：逐定位条件 Gaussian mask 概率质量\n'
        f'更新时间：{now}\n输出：{out}\n'
        '入口：spatial_mask_preview/run_membership_probability.py --config spatial_mask_preview/config_membership_probability.json\n'
        '原始帧0–2499；115013个入选点新增P(On)/P(Not-on)/P(unknown)；127059个ROI候选原字段保留，12046个原QC排除不赋分。\n'
        '均值为精修坐标；轴向位置标准差为x_se_nm/117、y_se_nm/117，非PSF sx/sy；XY协方差缺失，采用对角Gaussian。\n'
        '固定未裁切新mask，对闭合像素方块积分正态CDF乘积；C1原始零值和视野外质量单列unknown。\n'
        '原k=1椭圆相交标签不变，概率质量是在原mask内的位置质量，二者定义不同；没有新筛选或改标签。\n'
        '原On57615点：P(On)中位数0.991299；34749点(60.3124%)>=0.95；8362点(14.5136%)<0.5。\n'
        '原Not-on57398点：P(Not-on)中位数约1；53796点(93.7245%)>=0.95；0点<0.5。\n'
        '0.5/0.95仅为描述性分档，不是验收阈值；这些不是实测正确率、已校准后验或结合概率。\n'
        '当前SE为厂家名义标定下完整观测Hessian局部模型估计，非原生Picasso CRLB；mask与C1/C2恒等配准固定但未验证。\n'
        '5项测试(含200个随机完整枚举对照)及480个真实点完整网格独立复算通过；全部原字段/拟合/SE/QC/标签与入选ID逐字核验不变。\n'
        '最大质量守恒残差1.8874e-15；unknown质量>1e-6的点数0。输入SHA256前后相同，10个250帧检查点完整。\n'
        '直方图及CDF已视觉复核；源码、配置、依赖锁、来源/输出指纹与复现命令保存。未重新定位/分割，未做TARDIS、轨迹或显著性检验。\n'
    )
    base = out.parents[2]
    logs = [ROOT/'real_annular_background/LOG.txt',
            base/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']
    for path in logs:
        text = path.read_text()
        if str(out) not in text:
            with path.open('a') as stream:
                stream.write(entry)
    (out/'log_entry.txt').write_text(entry)

    dump(out/'delivery_QC.json', dict(
        checked_at_utc=now, histogram_and_CDF_visually_reviewed=True,
        checkpoint_sums_valid=True, original_input_files_untouched=True,
        unchanged_fits_QC_and_existing_labels=True,
        source_helpers_and_environment_lock_archived=True,
        logs_synced=[str(p) for p in logs],
        limitation='Fixed-mask nominal diagonal-Gaussian model, not empirically calibrated membership certainty'))
    complete = json.loads((out/'COMPLETED.json').read_text())
    complete.update(delivery_QC_passed=True, logs_synced=True)
    dump(out/'COMPLETED.json', complete)
    files = [p for p in sorted(out.rglob('*')) if p.is_file()
             and p.name not in ['output_fingerprints.json', 'COMPLETED.json']]
    records = [fingerprint(p) for p in files]
    dump(out/'output_fingerprints.json', records)
    # Re-read the major deliverables to verify the saved manifest.
    for name in ['all_ROI_candidates_with_model_probability.csv',
                 'selected_points_with_model_probability.csv', 'summary.json',
                 'membership_probability_histograms.png', 'membership_probability_CDF.png',
                 'run_status.json']:
        record = next(r for r in records if Path(r['path']).name == name)
        assert fingerprint(record['path']) == record
    print(json.dumps(dict(delivery_QC_passed=True, output=str(out), output_files=len(records)), indent=2))


if __name__ == '__main__':
    main()
