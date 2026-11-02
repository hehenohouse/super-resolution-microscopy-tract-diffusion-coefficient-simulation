"""Record completed visual review and delivery of support50 pooled point views."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import sys

from run_mean50_outputs import dump,fingerprint,HERE,PROJECT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    done=json.loads((out/'COMPLETED.json').read_text());assert done['all_implementation_checks_passed']
    review=json.loads((out/'projections/visual_review.json').read_text());assert review['passed']
    assert fingerprint(out/'projections/all_timepoints_triptych.png')['sha256']==review['triptych_sha256']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    counts=json.loads((out/'classification/summary.json').read_text());assert counts['On']+counts['Not_on']+counts['unknown']==115013
    assert counts['ROI_candidates']==127059 and counts['QC_excluded']==12046
    c=json.loads((out/'resolved_config.json').read_text());checked=json.loads((out/'classification/validation.json').read_text())
    assert checked['all_original_rows_and_cells_preserved_verbatim'] and checked['independent_full_pixel_enumeration_passed']
    cps=sorted((out/'checkpoints').glob('classified_*.json'));assert [json.loads(p.read_text())['completed_frames'] for p in cps]==list(range(250,2501,250))
    entry=(f'\n[2026-10-16] TDKd3 T1：support50>=10/min32/holes<64新mask逐点分类与全时间叠加图\n输出：{out}\n'
        f'Mask：{c["mask_directory"]}\nOn {counts["On"]}；Not-on {counts["Not_on"]}；unknown {counts["unknown"]}；115013原QC入选，12046排除，127059审计行原字段与旧标签完整保留。\n'
        '同一TrackMate0.28um+7x7外圈背景约束定位缓存；k1每点SE_nm/117轴对齐椭圆与完整mask闭合像素方块接触，相切包含/q容差1e-12。\n'
        'mask为逐帧broad→逐帧min32→50帧support>=10→再次min32→补闭合holes<64；无图像平均或额外W5；帧f使用块f//50。\n'
        f'ROI/MAT454顶点与32179像素、完整帧映射、50块完成、来源前后指纹、精确计数网格核验通过；独立完整像素枚举复核{len(checked["independent_checked_candidates"])}个定位。\n'
        '三栏绿色On/紫色Not-on/全部有效定位，黑底灰ROI、4倍、0.20原像素点半径、2um标尺；三张PNG和RGB TIFF/并排PNG/计数网格/帧与块汇总已保存。\n'
        'C1/C2恒等配准未验证，标准误为名义模型估计；相对于1秒support区域的空间相容性，不证明结合或确定胞质归属。\n'
        '原始C1单帧零值与视野unknown规则保留；未重新检测/拟合/分割，未运行TARDIS/轨迹/显著性；未恢复47样本队列。\n')
    for p in [PROJECT/'real_annular_background/LOG.txt',Path(c['C1_file']).parent/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        if str(out) not in p.read_text():
            with p.open('a') as stream:stream.write(entry)
    (out/'log_entry.txt').write_text(entry)
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out)])+'\n')
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    validation=json.loads((out/'validation.json').read_text());validation.update(visual_review_pending=False,visual_review_passed=True,logs_synced=True)
    dump(out/'validation.json',validation)
    done.update(visual_review_pending=False,delivery_checks_passed=True,finished_utc=datetime.now(timezone.utc).isoformat());dump(out/'COMPLETED.json',done)
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(out),counts=counts,delivery_checks_passed=True),indent=2))


if __name__=='__main__':main()
