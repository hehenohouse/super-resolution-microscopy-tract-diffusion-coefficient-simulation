"""Delivery audit and synchronized logs after representative visual review."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from run_mean50_outputs import HERE,PROJECT,TIFF_NAME,dump,fingerprint,pixel_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or Path((HERE/'latest_mean50_output.txt').read_text().strip())
    c=json.loads((out/'resolved_config.json').read_text())
    complete=json.loads((out/'COMPLETED.json').read_text());assert complete['all_implementation_checks_passed']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    summary=json.loads((out/'classification/summary.json').read_text())
    assert summary['On']+summary['Not_on']+summary['unknown']==c['expected_selected']
    assert summary['ROI_candidates']==c['expected_ROI_candidates'] and summary['QC_excluded']==c['expected_QC_excluded']
    frame=pd.read_csv(out/'classification/per_frame_counts.csv')
    np.testing.assert_array_equal(frame.frame,np.arange(2500))
    old=pd.read_csv(Path(c['localization_cache'])/'frame_counts.csv' if c.get('input_kind')=='annular_cache'
                    else Path(c['comparison_directory'])/'per_frame_comparison.csv')
    for field in ['ROI_candidates','selected','QC_excluded']:
        np.testing.assert_array_equal(frame[field],old[field])
    for field in ['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']:
        assert int(frame[field].sum())==summary[field]
    assert (frame.On+frame.Not_on+frame.unknown==frame.selected).all()
    block=pd.read_csv(out/'classification/per_block_counts.csv')
    np.testing.assert_array_equal(block.block_index,np.arange(50))
    for field in ['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']:
        np.testing.assert_array_equal(block[field],frame.groupby('block_index')[field].sum())
    for pattern,key in [('classified_*.json','completed_frames'),('rendered_*.json','rendered_frames')]:
        actual=[json.loads(p.read_text())[key] for p in sorted((out/'checkpoints').glob(pattern))]
        assert actual==list(range(250,2501,250)),(pattern,actual)
    readback=pd.read_csv(out/'ROI_C1_stack/readback_validation.csv')
    np.testing.assert_array_equal(readback.frame,np.arange(2500))
    np.testing.assert_array_equal(readback.block_index,np.arange(2500)//50)
    assert readback.decoded_pixels_equal.all() and readback.outside_ROI_black.all()
    manifest=pd.read_csv(out/'ROI_C1_stack/frame_manifest.csv')
    for f in [0,49,50,1199,1200,1999,2000,2499]:
        r=manifest.iloc[f];b=f//50
        assert r.frame==f and r.mean50_block_index==b
        assert r.mean50_original_frame_start==50*b and r.mean50_original_frame_end_inclusive==50*b+49
    reviewed=json.loads((out/'previews/visual_review.json').read_text());assert reviewed['passed']
    assert reviewed['image_manifest']==json.loads((out/'previews/image_manifest.json').read_text())
    for row in reviewed['image_manifest']:
        with Image.open(out/'previews'/f'frame_{row["frame"]:04d}_C1_ROI.png') as png:
            assert pixel_hash(np.asarray(png))==row['rendered_sha256']
    assert json.loads((out/'projections/visual_review.json').read_text())['passed']
    for name in ['run_full.py','run_preview.py','contracts.py','finalize_mean50_outputs.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    frozen=subprocess.run([c['cleanup_python'],'-B','-c',
        'import importlib.metadata; print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in importlib.metadata.distributions())))'],
        capture_output=True,text=True,check=True)
    (out/'cleanup_requirements.lock.txt').write_text(frozen.stdout)
    now=datetime.now(timezone.utc).isoformat();target=out/'ROI_C1_stack'/TIFF_NAME;base=Path(c['C1_file']).parent
    geom=json.loads((out/'classification/validation.json').read_text())
    entry=(f'\n[2026-10-16] {c.get("sample_name","TDKd3 T1")}：mean50/min32/holes<64分类、叠加图与ROI内C1多页TIFF\n更新时间：{now}\n输出：{out}\n'
        f'新mask：{c["mask_directory"]}\nOn {summary["On"]}；Not-on {summary["Not_on"]}；unknown {summary["unknown"]}；入选{summary["selected"]}，QC排除{summary["QC_excluded"]}，全部{summary["ROI_candidates"]}行原字段保留。\n'
        'TrackMate0.28um+7x7外圈背景约束缓存、原位置/SE/信号/QC/ROI不变。只追加mean50标签及块来源，原有标签如有则保留。\n'
        '每帧f使用mask[f//50]；50个非重叠1秒平均图mask，不插值、不加W5、不再分割/min32/补孔。min32去小前景，holes<64补小闭合背景，二者不混淆。\n'
        'k1逐点SE_nm/117轴对齐椭圆接触完整未被ROI裁切的mask，闭合像素方块/相切包含/q容差1e-12；原始单帧C1零区域及FOV未知规则保留。\n'
        f'ROI/MAT{c["expected_roi_vertices"]}顶点和{c["expected_roi_pixels"]}像素一致，最终补孔数组、50块与映射、ROI交集、原前景及新增{c["expected_added_ROI_pixels"]}ROI像素核验。\n'
        f'复用7项几何测试、新增3项块映射测试通过；代表帧独立全像素枚举复核{len(geom["independent_checked_candidates"])}个定位。\n'
        f'三张静态PNG/RGB TIFF+并排预览，黑底、绿On/紫Not-on；{summary["selected"]}观察不去重，全点计数网格与来源定位完全一致。\n'
        f'ROI内C1 TIFF：{target}\n'
        f'2500页原始20ms C1，407–651ADU固定灰度，mask每50帧更新；ROI外黑、红色完整mask轮廓先提取后裁切；2倍显示、裁切{c["crop_xyxy_px"]}、64像素标题。\n'
        'TYXS RGB8 OME-BigTIFF,T2500/Z1,20ms/页,Deflate无损。页眉注明原始帧/时间/mask块/1秒范围，无旧W5回退标签。\n'
        '两类各10个250帧检查点完整；全部原始C1页SHA及2500输出页读回匹配；ROI外全黑；来源指纹前后完全相同。\n'
        '200/1200/2000预览与三栏叠加图已视觉核验；完整审计/帧与块汇总/映射/版本/源码/配置/命令/日志/指纹保存。\n'
        '这是1秒平均mask下探索性空间相容性，C1/C2恒等配准未验证、SE仍名义模型估计；不是瞬时边界或结合/胞质证据。\n'
        '未重定位、拟合、分割、运行TARDIS/轨迹/显著性；旧结果、pilot默认标记、暂停47样本批次均未改写。\n')
    for p in [PROJECT/'real_annular_background/LOG.txt',base/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or str(out) not in p.read_text():
            with p.open('a') as f:f.write(entry)
    (out/'log_entry.txt').write_text(entry)
    v=json.loads((out/'validation.json').read_text());v.update(visual_review_pending=False,
        all20_checkpoints_present=True,per_frame_QC_and_selected_counts_unchanged=True,
        representative_previews_and_projection_visually_reviewed=True,logs_synced=True)
    dump(out/'validation.json',v)
    complete.update(visual_review_pending=False,delivery_checks_passed=True,logs_synced=True,finished_utc=now)
    dump(out/'COMPLETED.json',complete);dump(out/'run_status.json',dict(status='complete'))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(out),counts=summary,TIFF=str(target),TIFF_MB=target.stat().st_size/1e6,
        delivery_checks_passed=True),indent=2))


if __name__=='__main__':main()
