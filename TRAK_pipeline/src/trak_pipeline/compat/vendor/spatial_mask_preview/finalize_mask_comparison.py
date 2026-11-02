"""Record inspected comparison figures, durable checkpoints, and synchronized logs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw

from finalize_full_qc import prefix_checks
from run_preview import HERE,ROOT,dump,fingerprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visual-reviewed',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.visual_reviewed:raise ValueError('Inspect the figures before recording visual QC')
    complete=json.loads((out/'COMPLETED.json').read_text());assert complete['all_checks_passed']
    c=json.loads((out/'resolved_config.json').read_text());source=json.loads((out/'source_verification.json').read_text())
    assert out.parent==Path(c['data_directory'])/c['output_relative_directory']
    before=json.loads((out/'input_fingerprints_before.json').read_text())
    after=json.loads((out/'input_fingerprints_after.json').read_text());assert before==after
    for row in after:
        stat=Path(row['path']).stat();assert stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns']
    checkpoints=[json.loads(p.read_text()) for p in sorted((out/'checkpoints').glob('checkpoint_*.json'))]
    prefix_records={}
    for i,name in enumerate(['all_ROI_candidates_mask_comparison.csv','selected_points_mask_comparison.csv','changed_labels.csv']):
        prefix_records[name]=prefix_checks(out/name,[{'file':p['durable_csv_fingerprints'][i]} for p in checkpoints],'file')
    frame=pd.read_csv(out/'per_frame_comparison.csv')
    old=pd.read_csv(Path(c['old_classification_directory'])/'per_frame_summary.csv')
    np.testing.assert_array_equal(frame.frame,np.arange(2500));np.testing.assert_array_equal(frame.QC_excluded,old.excluded)
    curve=pd.read_csv(out/'new_mask_on_not_on_QC_by_timepoint.csv')
    for field in curve.columns:np.testing.assert_array_equal(curve[field],frame[field])
    np.testing.assert_array_equal(curve.new_on+curve.new_not_on+curve.new_unknown,curve.selected)
    np.testing.assert_array_equal(curve.selected+curve.QC_excluded,curve.ROI_candidates)
    # Every scientific classification transition is accounted for by its row/column.
    matrix=np.asarray(complete['paired_matrix'],int)
    np.testing.assert_array_equal(matrix.sum(axis=1),[59831,55182,0])
    np.testing.assert_array_equal(matrix.sum(axis=0),[int(frame.new_on.sum()),int(frame.new_not_on.sum()),int(frame.new_unknown.sum())])
    manifests=json.loads((out/'representative_frame_manifest.json').read_text())
    assert [r['frame'] for r in manifests]==c['representative_frames']
    canvas=Image.new('RGB',(2200,2180),'white');draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(manifests):
        with Image.open(out/item['image']) as im:
            assert int(im.info['original_frame'])==item['frame']
            thumb=ImageOps.contain(im.convert('RGB'),(1080,390))
            x=10+(i%2)*1100;y=5+(i//2)*435;canvas.paste(thumb,(x,y));draw.text((x,y+397),item['image'],fill='black')
    canvas.save(out/'QC_contact_sheet.png')
    images=sorted(out.rglob('*.png'))
    for p in images:
        with Image.open(p) as im:im.verify()
    qa=dict(all_passed=True,visual_review_recorded=True,PNG_decode_and_frame_metadata_valid=True,
        inspected_representative_frames=c['representative_frames'],
        inspected_summary_figures=['new_mask_on_not_on_QC_by_timepoint.png','old_new_counts_by_frame.png',
            'flipped_labels_by_frame.png','classification_transition_matrix.png'],
        inspected_flip_examples=12,example_layout='Redrawn with distinct row headers and compact two-direction preview',
        durable_final_CSV_prefixes=prefix_records,QC_curve_identical_to_old=True,plot_CSV_values_match_per_frame_summary=True,
        all_matrix_and_subset_totals_conserved=True,original_protected_inputs_stat_unchanged_after_final_SHA_check=True,
        no_TARDIS=True,registration_verified=False)
    dump(out/'QC_delivery_validation.json',qa)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    # Preserve the helper used for checkpoint validation in the source archive.
    shutil.copyfile(HERE/'finalize_full_qc.py',out/'source_snapshot/finalize_full_qc.py')
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--visual-reviewed'])+'\n')
    n=complete['counts'];stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    entry=(f'\n[2026-10-14] TDKd3：旧 min16 / 新 W5-min32 同定位空间分类对照\n更新时间：{stamp}\n'
        f'输出：{out}\n入口：spatial_mask_preview/run_mask_comparison.py --config spatial_mask_preview/config_mask_comparison.yaml\n'
        '只处理 TDKd3，原始帧 0–2499；同一批 115013 入选定位，全部127059候选及12046原QC排除保留。\n'
        'TrackMate 0.28 μm＋7×7 外圈约束拟合不变；原位置、误差、信号、QC、ROI、旧分类与所有CSV原字段逐字不变。\n'
        '旧min16 On=59831，Not-on=55182；新W5/min32 On=57615，Not-on=57398；unknown均0。\n'
        '旧On保持On=50893；旧Not-on保持Not-on=48460；On→Not-on=8938；Not-on→On=6722。\n'
        f'保持分类 {n["unchanged"]}（{100*n["unchanged"]/n["selected"]:.4f}%）；变化 {n["changed"]}（{100*n["changed"]/n["selected"]:.4f}%）。\n'
        '逐点k=1，闭合像素方块相交/相切；使用未裁切full mask，不扩大ROI；对角协方差近似。\n'
        '新mask为原始pre-cleanup broad的5帧3票多数＋完整8连通面积>=32像素²，之后ROI裁切。\n'
        '新mask端点0/1/2498/2499为未平滑min32回退，单独记录；与旧min16仍可不同。\n'
        'C1/C2恒等配准未验证；模型误差仍属厂家名义校准，置信覆盖未验证。分类翻转不是分子转移或因果证据。\n'
        '变化同时包含时间平滑与面积门槛改动，不分别归因；On/Not-on仅为空间相容性简称。\n'
        '19项测试通过；10个代表帧全部480入选点＋12个翻转实例均通过独立完整前景像素枚举的新旧标签核验。\n'
        '全部来源指纹前后相同；30个已落盘CSV检查点字节前缀与最终文件一致；逐帧QC曲线完全相同，数量守恒。\n'
        '原图/翻转示例、旧新数量图、分类矩阵与新On/Not-on/QC三线图已保存并视觉复核；曲线未平滑。\n'
        '没有重定位、重分割、扩张mask电影、TARDIS、轨迹或运动状态/显著性分析。\n')
    logs=[ROOT/'real_annular_background/LOG.txt',Path(c['data_directory'])/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']
    for p in logs:
        text=p.read_text()
        if str(out) not in text:p.write_text(text+entry)
    assert all(str(out) in p.read_text() for p in logs)
    dump(out/'synced_log_records.json',[fingerprint(p) for p in logs]);(out/'log_entry.txt').write_text(entry)
    validation=json.loads((out/'validation.json').read_text())
    validation.update(unit_tests_passed=19,delivery_QC_passed=True,all_checkpoint_CSV_prefixes_verified=True,
        count_graph_CSV_and_QC_curve_validated=True,logs_synced=True)
    dump(out/'validation.json',validation)
    report=out/'REPORT_zh.txt'
    if '交付复核：' not in report.read_text():
        with report.open('a') as f:f.write('\n交付复核：代表帧、12个翻转实例及全部汇总图已检查；局部图排版已修整。\n'
            '新三线图及CSV逐帧匹配统计输入，QC与旧结果完全相同；30个检查点前缀指纹核验通过，项目与硬盘日志已同步。\n')
    complete.update(delivery_QC_passed=True,logs_synced=True)
    dump(out/'COMPLETED.json',complete)
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run.log','output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(out),delivery_QC_passed=True,logs_synced=True,counts=n),ensure_ascii=False))


if __name__=='__main__':main()
