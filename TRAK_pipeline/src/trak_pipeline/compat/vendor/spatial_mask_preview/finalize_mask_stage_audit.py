"""Finalize an implementation-verified mask-stage audit after reviewing its figures."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mito_tardis'))
from render_point_stack import dump, fingerprint

REASONS = [
    'no_raw_broad_foreground_in_ellipse_in_block',
    'raw_broad_present_below_per_pixel_support_threshold',
    'per_frame_min32_lowered_below_threshold',
    'post_support_min32_removed',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    done = json.loads((out / 'COMPLETED.json').read_text())
    assert done['all_implementation_checks_passed'] and done['visual_review_pending']
    before = json.loads((out / 'input_fingerprints_before.json').read_text())
    assert before == json.loads((out / 'input_fingerprints_after.json').read_text())
    # Compare bytes again after pending-review rendering, without changing sources.
    for record in before:
        actual = fingerprint(Path(record['path']))
        assert actual == record, record['path']
    config = json.loads((out / 'config.json').read_text())
    prior = Path(config['prior_cluster_QC'])
    originals = pd.read_csv(prior / 'region_localizations.csv', float_precision='round_trip')
    originals = originals[originals.region_id.isin(config['regions']) &
                          originals.support50_pointwise_label.eq('non-mito-compatible')]
    points = pd.read_csv(out / 'point_stage_diagnostics.csv', float_precision='round_trip')
    assert len(points) == len(originals) == 8711 and points.candidate_index.is_unique
    reference = originals.set_index('candidate_index', drop=False).loc[points.candidate_index]
    for key in originals.columns:
        np.testing.assert_array_equal(points[key].to_numpy(), reference[key].to_numpy(), err_msg=key)
    assert not points.stage_final_hit.any() and not points.stage_post_min32_hit.any()
    assert points.stage_first_failure.isin(REASONS).all()
    # Frozen independent implementation findings, not counts inferred from plotting.
    expected = {
        'R4': {'window': [743, 3376, 615, 245], 'core': [501, 1313, 52, 0]},
        'R3': {'window': [341, 2348, 137, 12], 'core': [92, 791, 21, 0]},
        'R1': {'window': [490, 397, 7, 0], 'core': [202, 83, 0, 0]},
    }
    summary = pd.read_csv(out / 'stage_reason_summary.csv', float_precision='round_trip')
    compact = []
    for rid, scopes in expected.items():
        for scope, expected_counts in scopes.items():
            q = points[points.region_id.eq(rid)]
            if scope == 'core':
                q = q[q.in_core]
            counts = [int(q.stage_first_failure.eq(reason).sum()) for reason in REASONS]
            assert counts == expected_counts and sum(counts) == len(q)
            saved = summary[summary.region_id.eq(rid) & summary.scope.eq(scope)].set_index('reason')
            np.testing.assert_array_equal(saved.loc[REASONS, 'count'], counts)
            assert saved.total.eq(len(q)).all()
            compact.append(dict(region=rid, scope=scope, Not_on_observations=len(q),
                                no_raw_broad=counts[0], raw_per_pixel_support_1_to_9=counts[1],
                                first_min32_threshold_loss=counts[2], second_min32_loss=counts[3],
                                raw_support_already_below10_fraction=sum(counts[:2]) / len(q)))
    pd.DataFrame(compact).to_csv(out / 'compact_stage_summary.csv', index=False)
    frames = pd.read_csv(out / 'per_frame_reconstruction_checks.csv')
    np.testing.assert_array_equal(frames.frame, np.arange(2500))
    np.testing.assert_array_equal(frames.block_index, np.arange(2500) // 50)
    assert frames.stored_area_reproduced.all()
    for end in range(250, 2501, 250):
        checkpoint = json.loads((out / 'checkpoints' / f'frames_{end:04d}.json').read_text())
        assert checkpoint['completed_frames'] == end and checkpoint['original_support_and_final_masks_reproduced']

    reviewed = []
    for name in [f'regions/{rid}_all_mask_stages.png' for rid in config['regions']] + ['stage_support_and_point_reasons.png']:
        with Image.open(out / name) as picture:
            picture.load()
            assert picture.mode in ['RGB', 'RGBA'] and min(picture.size) > 800
            reviewed.append(dict(path=name, pixels=list(picture.size), sha256=fingerprint(out / name)['sha256']))
    dump(out / 'visual_review.json', dict(passed=True, images=reviewed,
        notes='All three contact sheets and time/stage overview visually reviewed. Same-frame raw/broad/min32 panels distinguished from block count/support panels. Counts, labels, contours and nominal SE visible. Fixed-pixel captions moved into titles to avoid row overlap.'))

    (out / 'FINDINGS_zh.txt').write_text(
        'TDKd3 T1 support50：原有 Not-on 结构的逐阶段只读诊断\n\n'
        '范围：R4/R3/R1 三个此前固定窗口的8711次原Not-on定位；核心为原固定半径2.5像素区域。不是全细胞抽样或独立分子数。\n'
        '未修改定位、误差、QC、原分类或任何mask参数，未重新检测/拟合/Nellie分割，未运行TARDIS。\n'
        '流程：原逐帧Nellie broad→完整图像8连通min32→50帧逐像素支持≥10→再次完整图像min32→4连通闭合孔<64。\n'
        '完整2500帧第一道min32、全部50块计数/支持阈值/第二道min32/最终补孔均与已保存结果逐像素一致。\n'
        '逐点使用其原中心、原X/Y标准误/117的轴对齐k1椭圆，闭合像素方块、相切包含、容差1e-12。原始所有21字段未变。\n\n'
        '核心区域：按“原broad完全无支持／有支持但每个相交像素均少于10帧／第一道min32使门槛跌破／第二道min32去除”计：\n'
        'R4：1866次 = 501 / 1313 / 52 / 0；原始逐像素支持已不足10为97.21%。\n'
        'R3：904次 = 92 / 791 / 21 / 0；原始逐像素支持已不足10为97.68%。\n'
        'R1：285次 = 202 / 83 / 0 / 0；原始逐像素支持已不足10为100%。\n'
        '整个窗口：R4 4979次 =743/3376/615/245；R3 2838次 =341/2348/137/12；R1 894次 =490/397/7/0。\n'
        '窗口外围确实有面积筛选影响，尤其R4第二道min32为245次；核心结论不应扩大到整个窗口或全片。\n\n'
        '关键定义：支持次数按每个像素在完整50帧块中的出现次数计算，然后取与该点固定椭圆相交像素的最大值。\n'
        '所有相交像素各自少于10帧，不等于整个椭圆少于10个不同帧接触过前景；前景可以在不同像素之间移动。\n'
        '原broad支持≥10只是在第一道min32之前的诊断性对照，不是本次生成或验证过的替代最终分割。\n'
        '固定峰值像素结果与逐点椭圆结果单独保存。R4固定峰(243,377)仅最初3块属于最终mask，后47块原broad逐像素支持已小于10。\n'
        '这提示：这些核心Not-on反复信号主要在已有前端分割/逐像素时间支持层面就没有形成合格区域，单独放宽第二道min32难以改变它们。\n'
        '不能判断：它们是否为线粒体漏分割、真实mask外信号、运动造成空间支持分散或通道对应偏差。原broad无支持不等于生物学上无结构。\n'
        '下一步宜结合这些C1原图核对弱结构和独立C1/C2配准；若探索参数，应预先固定少量对照并同步检查新增误分割，不以吸纳所有Not-on点为目标。\n'
        'C1/C2恒等配准仍未经独立验证，SE仍为厂家名义校准下模型标准误，不涵盖mask/配准误差。\n\n'
        '验证：100个真实点以独立完整像素枚举复核；几何/阶段分支10项测试通过；独立实现复核计数与全部原字段；十个250帧检查点；输入前后指纹一致。\n'
        '查看：stage_support_and_point_reasons.png；regions/R4、R3、R1_all_mask_stages.png；compact_stage_summary.csv；point_stage_diagnostics.csv。\n')
    versions = dict(python=sys.version, executable=sys.executable, platform=platform.platform())
    for package in ['numpy', 'scipy', 'pandas', 'matplotlib', 'tifffile', 'Pillow', 'shapely']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    dump(out / 'software_versions.json', versions)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (out / 'requirements.lock.txt').write_text(freeze.stdout)
    for name in ['run_mask_stage_audit.py', 'mask_stage_qc_core.py', 'test_mask_stage_qc_core.py',
                 'run_not_on_cluster_qc.py', 'cluster_qc_geometry.py', 'geometry.py',
                 'finalize_mask_stage_audit.py', 'publish_not_on_cluster_qc.py']:
        shutil.copyfile(HERE / name, out / 'source_snapshot' / name)
    (out / 'finalize_command.txt').write_text(shlex.join([sys.executable, '-B', str(Path(__file__).resolve()), '--output', str(out)]) + '\n')
    parent_config = json.loads((Path(config['source']) / 'resolved_config.json').read_text())
    destination = Path(parent_config['C1_file']).parent / 'analysis_results/support50_not_on_cluster_qc' / ('TDKd3_T1_' + out.name)
    (out / 'publication_command.txt').write_text(shlex.join([sys.executable, '-B', str(HERE / 'publish_not_on_cluster_qc.py'),
        '--source', str(out), '--destination', str(destination)]) + '\n')
    validation = json.loads((out / 'validation.json').read_text())
    assert len(validation['exact_ellipse_geometry_integration_checked_candidates']) == 100
    validation.update(visual_review_pending=False, visual_review_passed=True,
                      saved_original_fields_exactly_unchanged=True, saved_reason_counts_independently_reproduced=True,
                      all2500_frame_ids_and10_checkpoints_verified=True,
                      sources_unchanged_after_rendering=True, software_versions_recorded=True)
    dump(out / 'validation.json', validation)
    done.update(visual_review_pending=False, delivery_checks_passed=True, finished_utc=datetime.now(timezone.utc).isoformat())
    dump(out / 'COMPLETED.json', done)
    dump(out / 'output_fingerprints.json', [fingerprint(path) for path in sorted(out.rglob('*'))
         if path.is_file() and path.name not in ['output_fingerprints.json', 'COMPLETED.json']])
    print(json.dumps(dict(output=str(out), audited_observations=len(points), delivery_checks_passed=True), indent=2))


if __name__ == '__main__':
    main()
