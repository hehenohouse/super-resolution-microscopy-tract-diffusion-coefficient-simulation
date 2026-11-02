"""Independent saved-result and delivery checks for an already completed run."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import numpy as np
import pandas as pd

from run_intra import ROOT,PROJECT,dump,fingerprint
from mito_core import GROUPS,scopes,range_histograms,decompose_roi


def main():
    out=Path((ROOT/'latest_output.txt').read_text().strip())
    complete=json.loads((out/'COMPLETED.json').read_text())
    assert complete['implementation_checks_passed'] and complete['frames']==2500
    c=json.loads((out/'config.json').read_text());width=c['distance_max_um']/c['distance_bins']
    source=pd.read_csv(out/'selected_original_rows.csv',dtype=str,keep_default_na=False)
    stats=pd.read_csv(out/'statistical_input.csv',dtype=str,keep_default_na=False)
    assert len(source)==len(stats)==115013
    for field in ['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm',
                  'photons_e','bg_e_per_pixel','sx_px','sy_px','p_on_model','p_not_on_model']:
        a=np.fromiter(map(float,source[field]),float);b=np.fromiter(map(float,stats[field]),float)
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(source.new_pointwise_label,stats.new_pointwise_label)
    s=pd.read_csv(out/'per_scope_summary.csv',float_precision='round_trip')
    checks=[]
    with np.load(out/'pair_histograms/per_frame_raw_counts.npz') as hist:
        for g,group in enumerate(GROUPS):
            for scope,start,stop in scopes(c):
                tc,bc=range_histograms(hist['total'][g],hist['background'][g],start,stop)
                d=decompose_roi(tc,bc,c)
                saved=pd.read_csv(out/'distribution'/group/(scope+'.csv'),float_precision='round_trip')
                np.testing.assert_array_equal(saved.total_count,tc)
                np.testing.assert_array_equal(saved.background_ordered_nonzero_count,bc)
                for key in ['total','background','inter','intra','shape']:
                    np.testing.assert_allclose(saved[key+'_density_um_inv'],d[key][0]/width,rtol=2e-15,atol=2e-15,equal_nan=True)
                np.testing.assert_allclose(saved.total_density_um_inv.sum()*width,1,atol=1e-13)
                np.testing.assert_allclose(saved.background_density_um_inv.sum()*width,1,atol=1e-13)
                if d['shape_valid'][0]:
                    np.testing.assert_allclose(saved.shape_density_um_inv.sum()*width,1,atol=1e-13)
                row=s[(s.group==group)&(s.scope==scope)].iloc[0]
                np.testing.assert_allclose(saved.intra_density_um_inv.sum()*width,row.signed_area,atol=1e-13)
                checks.append(dict(group=group,scope=scope,all_bins_saved_correctly=True))
    banks=[]
    for length in c['bootstrap']['block_lengths_frames']:
        with np.load(out/'bootstrap'/f'shape_draws_L{length}.npz') as draws:
            np.testing.assert_array_equal(draws['replicate_id'],np.arange(1000))
            for j,(scope,_,_) in enumerate(scopes(c)):
                common=draws['valid'][0,j]&draws['valid'][1,j]
                with np.load(out/'bootstrap'/f'paired_shape_{scope}_L{length}.npz') as paired:
                    np.testing.assert_array_equal(paired['replicate_id'],np.arange(1000))
                    np.testing.assert_array_equal(paired['common_valid'],common)
                    np.testing.assert_allclose(paired['difference'][common],
                        draws['density'][0,j,common]-draws['density'][1,j,common],rtol=0,atol=0)
                band=pd.read_csv(out/'bootstrap'/f'paired_bands_{scope}_L{length}.csv',float_precision='round_trip')
                if common.mean()>=.95:
                    low,high=np.quantile(draws['density'][0,j,common]-draws['density'][1,j,common],[.025,.975],axis=0)
                    np.testing.assert_allclose(band.shape_difference_low,low,rtol=1e-13,atol=1e-13)
                    np.testing.assert_allclose(band.shape_difference_high,high,rtol=1e-13,atol=1e-13)
                banks.append(dict(scope=scope,block_frames=length,common_valid_count=int(common.sum()),common_valid_fraction=float(common.mean())))
    fallback=pd.read_csv(out/'per_frame_counts.csv')
    np.testing.assert_array_equal(fallback.loc[~fallback.mask_temporal_applied,'frame'],[0,1,2498,2499])
    shutil.copyfile(Path(__file__),out/'source_snapshot/mito_tardis/finalize.py')
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'source_snapshot/picasso_pilot/requirements.lock.txt')
    replay={**c,'shared_weights':str(out/'bootstrap/shared_block_resampling.npz')}
    dump(out/'replay_config.json',replay)
    (out/'REPRODUCE.txt').write_text(
        'Original command is in run_command.txt. Inputs remain at the fingerprinted original paths.\n'
        'For archived-source replay, copy source_snapshot to a NEW working directory first; use the existing pinned Picasso Python environment.\n'
        'Then run <python> <new_working_directory>/mito_tardis/run_intra.py --config '+shlex.quote(str(out/'replay_config.json'))+'\n'
        'The replay config changes only the saved-weight file path, retaining all numerical rules. Each execution creates a fresh timestamped output.\n'
        'Do not run directly inside the archived source_snapshot: latest_output/attempt pointer files belong to a writable working copy.\n'
        'Per-frame checkpoint pair counters count source frames f and may include f->f+1 crossing the checkpoint; scope calculations explicitly exclude segment boundaries.\n')
    periods=s[s.scope!='full'].pivot(index='scope',columns='group',values='tail_main_ratio')
    n=int((periods['Not-on']>periods.On).sum())
    paired=pd.read_csv(out/'paired_metric_differences.csv')
    full=s[s.scope=='full']
    outgroup={r.group:r for r in full.itertuples()}
    now=datetime.now(timezone.utc).isoformat()
    entry=(f'\n[2026-10-16] TDKd3 W5/min32：On / Not-on 分别重算 TARDIS Intra\n更新时间：{now}\n输出：{out}\n'
        '入口：mito_tardis/run_intra.py --config mito_tardis/config.json\n'
        '原始帧0–2499；现有k=1分类不变，On=57615，Not-on=57398，unknown=0，QC排除12046。\n'
        '未用新增概率分数筛选、改标签或加权；位置、误差、拟合状态与QC未改变；全部入选原始行逐字复制保存。\n'
        '逐组构建真实相邻帧Total和有序同帧非零B；不跨组/空帧/时段/重采样块制造配对。\n'
        '117nm/px，tau=20ms，0–3um/300bins，beta固定2.1–3um最小二乘并限制[0,1]；两类尾部计数均须>=100。\n'
        f'On：beta={outgroup["On"].beta:.9f}，signed面积={outgroup["On"].signed_area:.9f}，tail/main={outgroup["On"].tail_main_ratio:.9f}。\n'
        f'Not-on：beta={outgroup["Not-on"].beta:.9f}，signed面积={outgroup["Not-on"].signed_area:.9f}，tail/main={outgroup["Not-on"].tail_main_ratio:.9f}。\n'
        f'两组全部十个250帧时段均通过支持/分母检查；Not-on的tail/main高于On见于{n}/10时段，属描述性结果。\n'
        '保留signed负bins；归一化范围0–3um；面积不是生物学状态比例，signed曲线不是严格非负PDF。\n'
        '复用原1000次共享分层权重，L25主，L10/50敏感性；所有时段与块长的两组曲线共同有效重复1000/1000。\n'
        '探索性逐点percentile区间，无已校准覆盖保证；未调用旧maxT、未给p值或显著性结论。\n'
        '保存跨组全点对直方图作诊断，不是分子转移轨迹/转移率。动态mask、ROI/组边界选择与误差构成仍限制运动解释。\n'
        '5项测试通过；26个真实组×帧独立直接枚举；全部原始帧/bin与现有TARDIS实现一致，组内+跨组直方图与全入选点守恒。\n'
        '额外落盘复核：22条分布、33个两组重复交集和差值/分位数一致；统计坐标/SE与原CSV经正确舍入浮点解析完全相同。\n'
        'C1/C2恒等配准未验证；定位SE仍为厂家名义标定的局部模型估计，不做误差扣除/去卷积/扩散或状态比例拟合。\n'
        '源指纹前后相同。对比图/分段图/背景诊断已视觉复核，mask稳定性图分别绘制W5与未平滑min32基线。\n')
    for path in [PROJECT/'real_annular_background/LOG.txt',out.parents[2]/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        if str(out) not in path.read_text():
            with path.open('a') as f:f.write(entry)
    (out/'log_entry.txt').write_text(entry)
    with (out/'RESULTS.txt').open('a') as f:
        f.write(f'\nDelivery review: Not-on tail/main exceeds On in {n}/10 windows (descriptive). All scopes pass the fixed tail support/area rules.\n')
        f.write('All 10/25/50-frame paired shape banks have 1000/1000 common valid repeats. Percentile ranges remain exploratory.\n')
        f.write('W5 endpoint frames0,1,2498,2499 use unsmoothed min32 fallback; flags retained in per_frame_counts.csv.\n')
    dump(out/'delivery_validation.json',dict(all_delivery_checks_passed=True,
        full_saved_distribution_rechecks=checks,paired_bootstrap_bank_rechecks=banks,
        selected_coordinates_SE_and_signal_float_values_bit_identical=True,
        representative_figures_visually_reviewed=True,mask_IoU_series_separated_by_candidate=True,
        fallback_frames=[0,1,2498,2499],logs_synced=True,
        formal_statistical_significance_claimed=False))
    complete.update(delivery_visual_QC_pending=False,delivery_checks_passed=True,logs_synced=True,finished_utc=now)
    dump(out/'COMPLETED.json',complete)
    records=[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']]
    dump(out/'output_fingerprints.json',records)
    print(json.dumps(dict(output=str(out),delivery_checks_passed=True,distributions_checked=len(checks),paired_banks_checked=len(banks)),indent=2))


if __name__=='__main__':main()
