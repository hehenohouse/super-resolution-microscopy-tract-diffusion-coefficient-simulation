"""Delivery audit/logs for a completed, all-page-verified point viewing stack."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from render_point_stack import HERE,PROJECT,TIFF_NAME,EXAMPLES,dump,fingerprint,pixel_hash


def main():
    out=Path((HERE/'latest_point_stack_output.txt').read_text().strip())
    complete=json.loads((out/'COMPLETED.json').read_text());assert complete['all_implementation_checks_passed']
    check=json.loads((out/'TIFF_validation.json').read_text());assert check['all2500_pages_readback_equal']
    extension=json.loads((out/'panel_extension_validation.json').read_text())
    assert extension['all_first_five_panel_pixels_unchanged']
    assert extension['original_raw_C1_C2_mask_page_hashes_and_counts_unchanged']
    recovery_path=out/'validation_recovery.json'
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        output_manifest={r['path']:r for r in json.loads((out/'output_fingerprints.json').read_text())}
        assert output_manifest[str(out/TIFF_NAME)]==recovery['TIFF_before']
        recovery.update(resolved=True,all2500_pages_readback_passed=True,TIFF_bytes_mtime_and_SHA256_unchanged=True)
        dump(recovery_path,recovery)
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    manifest=pd.read_csv(out/'frame_manifest.csv');decoded=pd.read_csv(out/'readback_validation.csv')
    np.testing.assert_array_equal(manifest.frame,np.arange(2500))
    np.testing.assert_array_equal(decoded.frame,manifest.frame)
    np.testing.assert_array_equal(decoded.decoded_pixel_sha256,manifest.rendered_sha256)
    assert decoded.equals_rendered.all()
    assert int(manifest.On.sum())==57615 and int(manifest.Not_on.sum())==57398
    assert int(manifest.combined.sum())==115013
    np.testing.assert_array_equal(manifest.loc[~manifest.mask_temporal_applied,'frame'],[0,1,2498,2499])
    checkpoints=[json.loads(p.read_text()) for p in sorted((out/'checkpoints').glob('rendered_*.json'))]
    assert [c['rendered_frames'] for c in checkpoints]==list(range(250,2501,250))
    with tifffile.TiffFile(out/TIFF_NAME) as tf:
        assert len(tf.pages)==2500 and tf.series[0].axes=='TYXS'
        for f in EXAMPLES:
            with Image.open(out/'previews'/f'frame_{f:04d}_six_panels.png') as png:
                np.testing.assert_array_equal(tf.pages[f].asarray(),np.asarray(png))
        # Retain endpoint samples for the explicitly labelled fallback pages.
        for f in [0,2499]:
            page=tf.pages[f].asarray();assert pixel_hash(page)==manifest.rendered_sha256.iloc[f]
            Image.fromarray(page).save(out/'previews'/f'frame_{f:04d}_endpoint_QC.png',compress_level=6)
    shutil.copyfile(__file__,out/'source_snapshot/finalize_point_stack.py')
    now=datetime.now(timezone.utc).isoformat()
    target=out/TIFF_NAME;size=target.stat().st_size
    entry=(f'\n[2026-10-16] TDKd3 W5/min32：完整2500帧六栏RGB时间TIFF\n更新时间：{now}\n输出：{out}\nTIFF：{target}\n'
        '入口：mito_tardis/render_point_stack.py --run <当前TARDIS结果目录>；先--prepare-only视觉复核，再--output续接导出。\n'
        '六栏为On点、Not-on点、合并点、C2原图+圈、C1原图+完整mask红色轮廓、C1原图+mask+定位圈；均叠加细灰色共同细胞ROI。\n'
        '原五栏在200/1200/2000代表帧中与上一版逐像素完全一致；第六栏使用同一C1灰度底图、mask轮廓及浮点圈中心，不混合C1/C2原图。\n'
        '原始帧0–2499，每页同一真实时间点；On57615、Not-on57398，共115013；原QC、ROI、精修坐标和分类不变，无概率重筛选。\n'
        '固定裁切x[82,347),y[232,430)，2倍显示；浮点中心共用映射，x向右y向下；点半径1.5原像素、圆半径3原像素，仅为显示。\n'
        'C2固定406–554ADU、C1固定407–651ADU，原16-bit像素不变；显示栈为RGB8查看图像，没有整页物理像素标定。\n'
        'Mask轮廓由完整原生前景像素方块边缘提取，先计算边界再裁切，保留孔洞，不画ROI切口假边界；回退0/1/2498/2499标记。\n'
        'OME-BigTIFF，2500页，TYXS，SizeT2500/SizeZ1，20ms/frame，Deflate无损；六栏合成视野3292×530像素。\n'
        '6项坐标/灰度/空点/混色/非裁切边界/合并栏/流式TIFF测试通过；全部2500页已解码像素SHA256逐页吻合；代表PNG与对应TIFF页像素完全一致。\n'
        '10个250帧检查点、逐帧数量与源页hash、坐标映射、原始点源CSV、源码/环境/命令/OME元数据与指纹保存。\n'
        '读回校验修正了tifffile TiffFrame标签访问，复用已写好的完整TIFF；恢复前后文件字节数、mtime与SHA256完全不变，原报错与恢复命令留档。\n'
        '源输入指纹前后相同。C1/C2恒等配准仍未独立验证；On/Not-on为空间相容性，不做结合或显著性判断。\n'
        '未重新检测、拟合、分割、分类、运行TARDIS或轨迹分析；所有旧输入和输出保留。\n')
    base=out.parents[2]
    for path in [PROJECT/'real_annular_background/LOG.txt',base/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        if str(out) not in path.read_text():
            with path.open('a') as f:f.write(entry)
    (out/'log_entry.txt').write_text(entry)
    v=json.loads((out/'validation.json').read_text())
    v.update(final_visual_review_pending=False,representative_frames_visually_reviewed=EXAMPLES,
        logs_synced=True,original_frame_mapping_preserved=True,all_checkpoints_present=True,
        first_five_panels_unchanged=True,sixth_panel_C1_mask_and_circles=True)
    dump(out/'validation.json',v)
    complete.update(final_visual_review_pending=False,delivery_checks_passed=True,logs_synced=True,
                    TIFF_bytes=size,finished_utc=now)
    dump(out/'COMPLETED.json',complete)
    dump(out/'run_status.json',dict(status='complete',frames=2500))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(out),TIFF=str(target),TIFF_MB=size/1e6,delivery_checks_passed=True),indent=2))


if __name__=='__main__':main()
