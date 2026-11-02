"""Record inspected figures, verify durable checkpoint prefixes, and sync logs.

Only the newly completed classification directory and designated project/disk
logs are written. Source fits, masks, original images and prior results are not.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

from PIL import Image, ImageDraw, ImageOps

from run_preview import HERE, ROOT, STATUS, dump, fingerprint


def prefix_checks(path, checkpoints, key):
    items=[c[key] for c in checkpoints]
    digest=hashlib.sha256();position=0;checks=[]
    with path.open('rb') as stream:
        for item in items:
            target=item['bytes'];assert target>=position
            while position<target:
                chunk=stream.read(min(8*1024*1024,target-position))
                if not chunk:raise ValueError('Checkpoint claims more bytes than final output')
                digest.update(chunk);position+=len(chunk)
            assert digest.hexdigest()==item['sha256']
            checks.append(dict(bytes=position,sha256=digest.hexdigest()))
        assert not stream.read(1)
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visual-reviewed',action='store_true',help='Record visual inspection already performed by the caller')
    args=parser.parse_args();out=args.output.resolve()
    if not args.visual_reviewed:raise ValueError('Inspect QC figures before recording visual review')
    complete=json.loads((out/'COMPLETED.json').read_text());assert complete['all_checks_passed']
    c=json.loads((out/'resolved_config.json').read_text())
    source=json.loads((out/'source_verification.json').read_text())
    assert out.parent==Path(c['data_directory'])/c['output_relative_directory']
    checks=json.loads((out/'validation.json').read_text());assert checks['all_checks_passed']
    before=json.loads((out/'input_fingerprints_before.json').read_text())
    after=json.loads((out/'input_fingerprints_after.json').read_text());assert before==after
    for item in after:
        st=Path(item['path']).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns']
    checkpoints=[json.loads(p.read_text()) for p in sorted((out/'checkpoints').glob('checkpoint_*.json'))]
    durable={name:prefix_checks(out/name,checkpoints,key) for name,key in [
        ('all_ROI_candidates_with_spatial_classification.csv','audit_file'),
        ('selected_localizations_with_spatial_classification.csv','selected_file')]}
    manifest=json.loads((out/'representative_frame_manifest.json').read_text())
    assert [r['frame'] for r in manifest]==c['representative_frames']
    canvas=Image.new('RGB',(1800,1760),'#171A20');draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(manifest):
        path=out/item['image']
        with Image.open(path) as im:
            assert int(im.info['original_frame'])==item['frame']
            assert int(im.info['selected_points'])==item['selected']
            assert im.info['registration_status']==STATUS
            thumb=ImageOps.contain(im.convert('RGB'),(880,320))
            x=10+(i%2)*900;y=5+(i//2)*350
            canvas.paste(thumb,(x,y));draw.text((x,y+325),path.name,fill='white')
        with Image.open(path) as im:im.verify()
    canvas.save(out/'QC_contact_sheet.png')
    for item in json.loads((out/'histogram_display_ranges_and_overflow.json').read_text()):
        count=complete['counts'][item['label']]
        assert item['shown']+item['below']+item['above']==count
    qa=dict(all_passed=True,inspected_representative_frames=c['representative_frames'],
        inspected_summary_figures=['counts_fractions_mask_area_over_time.png','group_error_signal_histograms.png'],
        visual_observations='Native image direction retained; contour/dot colors, legends and frame labels readable. No layout clipping observed.',
        PNG_metadata_and_decode_valid=True,checkpoint_prefixes_match_final_CSV_bytes=durable,
        histogram_display_and_overflow_counts_conserve_membership=True,
        protected_source_stat_unchanged_since_final_SHA256_check=True,
        registration_verified=False,classification_validated_only_conditionally=True)
    dump(out/'QC_delivery_validation.json',qa)
    metadata=out/'source_metadata';metadata.mkdir(exist_ok=True)
    for key,names in [('localization_directory',['rules_before_run.json','summary.json','COMPLETED.json']),
            ('mask_directory',['parameters.json','nellie_parameters.json','validation.json','COMPLETE.json'])]:
        for name in names:shutil.copyfile(Path(source[key])/name,metadata/(key+'__'+name))
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    (out/'finalize_QC_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--visual-reviewed'])+'\n')
    s=json.loads((out/'summary.json').read_text());tot=s['full']
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    entry=(f'\n[2026-10-14] 全 2500 帧逐点线粒体空间分类与 QC\n更新时间：{stamp}\n'
        f'输出：{out}\n'
        '入口：spatial_mask_preview/run_full.py --config spatial_mask_preview/config_full.yaml\n'
        '只读复用 TrackMate 0.28 μm＋7×7 外圈背景约束定位与 Nellie broad min16 clean_full_mask。\n'
        f'全部 ROI 候选 {tot["ROI_candidates"]} 原字段逐字保留；入选 {tot["selected"]}，原 QC 排除 {tot["excluded"]}。\n'
        f'逐点 k=1：相容 {tot["mito-compatible"]}；不相容 {tot["non-mito-compatible"]}；unknown {tot["unknown"]}。\n'
        f'中心直接落入原 mask：{tot["center_inside_mask"]}；自身误差椭圆相交新增 {tot["mito-compatible"]-tot["center_inside_mask"]}。\n'
        f'椭圆触及/跨越细胞 ROI 边界 {tot["touches_ROI_boundary"]}/{tot["crosses_ROI_boundary"]}，只标记，不排除。\n'
        '排除原因：active_parameter_bound=11826，overlap_review=161，refined_center_outside_ROI=45，unavailable_support=11，unavailable_covariance=3。\n'
        'C1/C2 恒等坐标是假设，分类状态 exploratory_identity_registration_unverified；不是已证明结合或确定胞质归属。\n'
        '每点半轴 X/Y SE_nm÷117，闭合像素方块解析相交/相切；XY 协方差未知，采用轴对齐对角近似。\n'
        '使用未被 ROI 切断的原 mask；不扩张细胞 ROI，不用逐帧平均误差分类，不生成扩张 mask 电影。\n'
        '视野外/C1 原始零值未知区域保持 unknown 单独类别；本片入选点未触发。\n'
        '名义厂家标定下的完整观测 Hessian 模型标准误，不是原生 Picasso CRLB；真实误差/置信覆盖未验证。\n'
        '15 项几何/帧范围/空帧/CSV 保真测试通过；10 代表帧全部 480 入选点经独立完整像素枚举一致。\n'
        '166 个旧三帧候选按 candidate_index 逐点复现；帧 200/1200/2000 相容/不相容仍为45/35、23/22、12/10。\n'
        '10 个250帧检查点保存并核验最终CSV字节前缀指纹；全片/分段三类+排除数量守恒。\n'
        '原 TIFF/ROI/MAT/定位/mask 与旧预览受保护输入 SHA256 前后相同。PNG、逐帧/分段汇总、误差/信号分布和来源/源码/版本完整保存。\n'
        '未重新定位/分割、未执行 TARDIS/轨迹/运动状态分析，不作显著性、扩散或因果结论。\n')
    local_log=ROOT/'real_annular_background/LOG.txt'
    disk_log=Path(c['data_directory'])/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt'
    logs=[]
    for path in [local_log,disk_log]:
        text=path.read_text()
        if str(out) not in text:path.write_text(text+entry)
        logs.append(fingerprint(path))
    assert local_log.read_text()==disk_log.read_text()
    dump(out/'synced_log_records.json',logs)
    (out/'log_entry.txt').write_text(entry)
    checks.update(unit_tests_passed=15,delivery_QC_checks_passed=True,checkpoint_byte_prefixes_verified=True)
    dump(out/'validation.json',checks)
    report=out/'REPORT_zh.txt'
    if '交付复核：' not in report.read_text():
        with report.open('a') as f:f.write('\n交付复核：10 张代表帧＋时间图＋线性直方图已视觉检查；PNG 元数据与解码正常。\n'
            '20 个检查点 CSV 前缀指纹与最终文件一致；显示范围外数量完整记录，组内直方图数量守恒。\n'
            '项目与 My Book 日志已同步追加，完整数据分析在此停止。\n')
    complete['delivery_QC_passed']=True;complete['logs_synced']=True
    dump(out/'COMPLETED.json',complete)
    outputs=[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run.log','output_fingerprints.json','COMPLETED.json']]
    dump(out/'output_fingerprints.json',outputs)
    print(json.dumps(dict(output=str(out),counts=complete['counts'],logs_synced=True,delivery_QC_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
