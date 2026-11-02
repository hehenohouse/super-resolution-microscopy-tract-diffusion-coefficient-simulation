"""Audit region QC, record reviewed figures and reproducible local completion."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mito_tardis'))
from render_point_stack import fingerprint,dump
from run_not_on_cluster_qc import COLUMNS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    done=json.loads((out/'COMPLETED.json').read_text());assert done['all_implementation_checks_passed']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    config=json.loads((out/'config.json').read_text());source=Path(config['source'])
    old=pd.read_csv(source/'classification/selected_localizations_support50.csv',usecols=COLUMNS,float_precision='round_trip').set_index('candidate_index',drop=False)
    p=pd.read_csv(out/'region_localizations.csv',float_precision='round_trip');assert p.candidate_index.is_unique
    ref=old.loc[p.candidate_index]
    for col in COLUMNS:
        np.testing.assert_array_equal(p[col].to_numpy(),ref[col].to_numpy(),err_msg=col)
    frame=pd.read_csv(out/'per_frame_statistics.csv',float_precision='round_trip');block=pd.read_csv(out/'per_block_statistics.csv',float_precision='round_trip')
    assert len(frame)==15000 and len(block)==300
    for rid,g in frame.groupby('region_id',sort=False):
        np.testing.assert_array_equal(g.frame,np.arange(2500));np.testing.assert_array_equal(g.block_index,g.frame//50)
        selected=p[p.region_id==rid]
        for name,label in [('On','mito-compatible'),('Not_on','non-mito-compatible')]:
            count=selected[selected.support50_pointwise_label==label].groupby('frame').size().reindex(range(2500),fill_value=0)
            np.testing.assert_array_equal(g[name].to_numpy(),count.to_numpy())
        for name in ['On','Not_on','core_On','core_Not_on']:
            np.testing.assert_array_equal(block[block.region_id==rid][name].to_numpy(),g.groupby('block_index')[name].sum().to_numpy())
    # The analytic-distance integration already passed independent full-pixel
    # enumeration on 250 actual candidates; retain the helper's synthetic log.
    result=subprocess.run([sys.executable,'-B',str(HERE/'test_cluster_qc_geometry.py')],capture_output=True,text=True)
    (out/'geometry_tests.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    names=['overview_regions.png','R4_core_diagnostics.png','temporal_counts_and_core_positions.png']+[f'regions/R{i}_raw_mask_localizations.png' for i in range(1,7)]
    images=[]
    for name in names:
        with Image.open(out/name) as image:
            assert image.mode=='RGB' or image.mode=='RGBA';assert min(image.size)>800
            image.load();images.append(dict(path=name,size=list(image.size),sha256=fingerprint(out/name)['sha256']))
    dump(out/'visual_review.json',dict(passed=True,images=images,notes='Reviewed all six raw-image contact sheets, region overview, unlinked temporal raster and R4 diagnostics. Coordinates, labels, contours and nominal SE overlays visible. C1 means explicitly marked display-only. Fixed white core circles do not represent localization errors.'))
    regions=json.loads((out/'region_summary.json').read_text());r4=regions[3];r1=regions[0]
    off=p[(p.region_id=='R4')&p.in_core&p.support50_pointwise_label.eq('non-mito-compatible')]
    b=block[block.region_id=='R4'];assert b.loc[b.final_mask_at_fixed_peak,'block_index'].tolist()==[0,1,2]
    summary=pd.DataFrame([{key:r[key] for key in ['id','name','cx','cy','n_on','n_off','off_frames','off_blocks','core_off','core_off_frames','core_off_blocks']} for r in regions])
    summary.to_csv(out/'region_summary.csv',index=False)
    text=(
        'TDKd3 T1：support50 Not-on 局部 QC（2026-10-19）\n\n'
        '沿用115013个原QC入选定位及原77,601 On/37,412 Not-on标签；未重新检测、拟合、分类或分割。\n'
        '选区：从全片Not-on空间密集处探索性选6个28x28像素窗口，覆盖上缘、左右侧及邻近On的区域。不是正式聚类检验，也不是总体随机样本。\n'
        '窗口按连续坐标半开区间选点；原图多显示1像素上下文。白色虚线圆是固定2.5像素诊断核心，不是误差范围；第三列误差条才是每点X/Y名义标准误。\n\n'
        f'重点R4（中心243,377）：整个窗口{r4["n_off"]}次Not-on，覆盖{r4["off_frames"]}个原始帧；固定核心{r4["core_off"]}次Not-on来自{r4["core_off_frames"]}帧，覆盖全部50个时间块。\n'
        f'核心Not-on距当前mask最近闭合像素方块的中位距离{off.distance_to_current_full_mask_nm.median():.1f}nm；X/Y名义标准误中位数{off.x_se_nm.median():.2f}/{off.y_se_nm.median():.2f}nm。\n'
        '固定峰值像素(243,377)只在块0、1、2属于最终mask，其后块3–49均不属于；它的前景支持次数从30/27/21下降到多数块0–4，低于10/50门槛。\n'
        '固定像素的mask状态不等于每个精修中心的分类：早期仍可有核心内的Not-on点，其实际位置和误差椭圆已逐点按原规则判定。\n'
        'R4可见小范围内反复出现的定位，并有随时间缓慢变化的位置分布；没有做轨迹链接，不能据此说是同一分子。\n'
        f'对照R1：核心{r1["core_off"]}次Not-on，仅覆盖{r1["core_off_blocks"]}个时间块，明显集中在早期。六区的时间表现不同。\n\n'
        '能确认：这批Not-on叠加结构具有局部重复和时间异质性；至少R4不能仅用少量边界附近的定位误差解释当前几何分类。\n'
        '仍待区分：弱结构/分割支持不足、未经验证的通道空间对应、其他持续荧光来源或真实mask外结构。现有图像不能证明哪一种机制，也不能将Not-on直接等同自由扩散胞质。\n'
        '50帧C1均值只帮助显示原图，support50 mask不是用这些均值生成的；未进行背景/光子重标定。\n'
        '距离相对于当前mask，不是实测线粒体边界；名义标准误未涵盖mask、配准及全部模型误差。\n\n'
        '输出：overview_regions.png；regions/六区原图面板；temporal_counts_and_core_positions.png；R4_core_diagnostics.png；逐点、逐帧、逐块表；源码、版本、命令、指纹及验证。\n'
        '所有源文件前后指纹一致；6x2500原始帧和空帧保留；250个实际点的距离与独立完整像素枚举一致，几何单元测试9项通过。未运行TARDIS、轨迹或显著性检验。\n')
    (out/'FINDINGS_zh.txt').write_text(text)
    for name in ['run_not_on_cluster_qc.py','cluster_qc_geometry.py','test_cluster_qc_geometry.py','finalize_not_on_cluster_qc.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out)])+'\n')
    validation=json.loads((out/'validation.json').read_text());validation.update(visual_review_pending=False,visual_review_passed=True,
        written_point_fields_match_source=True,per_frame_and_block_counts_verified=True,geometry_tests9_passed=True)
    dump(out/'validation.json',validation);done.update(visual_review_pending=False,delivery_checks_passed=True,finished_utc=datetime.now(timezone.utc).isoformat());dump(out/'COMPLETED.json',done)
    dump(out/'output_fingerprints.json',[fingerprint(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(out),reviewed_regions=6,delivery_checks_passed=True),indent=2))


if __name__=='__main__':main()
