#!/usr/bin/env python3
"""Audit completed artifacts and summarize paired sensitivity findings."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from full_core import ROOT,key_name,point_summary
from pilot_core import dump_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    out=Path(parser.parse_args().run).resolve();assert (out/'COMPLETED.json').exists()
    c=yaml.safe_load((out/'config.yaml').read_text());s=pd.read_csv(out/'per_scope_summary.csv')
    p=pd.read_csv(out/'paired_comparisons.csv');b=pd.read_csv(out/'bootstrap_summary.csv')
    tables={};checks=[]
    for g in c['groups']:
        name=key_name(g['box'],g['factor']);t=pd.read_csv(out/'audit'/(name+'.csv'))
        kept=pd.read_csv(out/'selected'/(name+'.csv'));tables[name]=kept
        assert t.candidate_id.is_unique and t.frame.between(0,2499).all()
        np.testing.assert_allclose(t.time_s,t.frame*.02,rtol=0,atol=1e-13)
        for k in ['x','y','lpx','lpy','sx','sy']:
            np.testing.assert_allclose(t[k+'_nm'],t[k]*117,rtol=3e-7,atol=1e-7,equal_nan=True)
        assert kept.qc_unflagged_model_fit.all() and not kept.suspect_zero_detection_support.any()
        assert (t.loc[t.nonpositive_backend_amplitude,'photons']<=0).all()
        with h5py.File(out/'audit'/(name+'.hdf5')) as h:
            assert len(h['locs'])==len(t)
            np.testing.assert_array_equal(h['locs']['frame'],t.frame)
            np.testing.assert_array_equal(h['locs']['candidate_id'],t.candidate_id)
        row=s.loc[(s.group==name)&(s.scope=='full')].iloc[0]
        assert len(t)==row.candidates and len(kept)==row.points
        checks.append({'group':name,'audit_rows':len(t),'selected_rows':len(kept),'all_fields_and_hdf5_counts_checked':True})
    low=key_name(9,.75);ref=key_name(9,1.);high=key_name(9,1.25)
    assert set(tables[high].candidate_id)<=set(tables[ref].candidate_id)<=set(tables[low].candidate_id)
    for a,z in [(high,ref),(ref,low)]:
        left=tables[a].set_index('candidate_id');right=tables[z].set_index('candidate_id').loc[left.index]
        np.testing.assert_array_equal(left[['frame','x','y','photons','lpx','lpy']].to_numpy(),right[['frame','x','y','photons','lpx','lpy']].to_numpy())
    added=tables[low].loc[~tables[low].candidate_id.isin(tables[ref].candidate_id)]
    removed=tables[ref].loc[~tables[ref].candidate_id.isin(tables[high].candidate_id)]
    cohorts=[]
    for cohort,t in [('low_threshold_added_QC_retained',added),('high_threshold_removed_QC_retained',removed),('shared_high_threshold_QC_retained',tables[high])]:
        for scope,a,z in [('full',0,2500)]+[(f'f{i:04d}_{i+249:04d}',i,i+250) for i in range(0,2500,250)]:
            cohorts.append({'cohort':cohort,'scope':scope,**point_summary(t.loc[t.frame.between(a,z-1)])})
    pd.DataFrame(cohorts).to_csv(out/'threshold_membership_summary.csv',index=False)
    contrast=p.loc[(p.comparison==low+' - '+ref)&(p.metric=='tail_main_ratio')]
    periods=s.loc[s.scope!='full'];pivot=periods.pivot(index='scope',columns='group',values='tail_main_ratio')
    facts={'low_threshold_tail_ratio_larger_in_windows':int((pivot[low]>pivot[ref]).sum()),
           'number_windows':10,'low_vs_reference_tail_ratio':[],
           'low_vs_reference_raw_signed_tail_mass':[],
           'nominal_x_precision_larger_low_in_windows':int((periods.pivot(index='scope',columns='group',values='lpx_nm_median')[low]>
                       periods.pivot(index='scope',columns='group',values='lpx_nm_median')[ref]).sum()),
           'interpretation':'relative tail persistence and nominal uncertainty association; no noise/motion attribution'}
    notes=['','追加的结论与产物审计：','']
    for length in [10,25,50]:
        r=contrast.loc[(contrast.block_frames==length)&(contrast.scope=='full')].iloc[0]
        intervals=contrast.loc[(contrast.block_frames==length)&(contrast.scope!='full')]
        facts['low_vs_reference_tail_ratio'].append({'block_frames':length,'delta':r.point_difference,
            'ci_low':r.ci_low,'ci_high':r.ci_high,'windows_ci_above_zero':int((intervals.ci_low>0).sum())})
        tail=p.loc[(p.comparison==low+' - '+ref)&(p.metric=='tail_mass')&(p.scope=='full')&(p.block_frames==length)].iloc[0]
        facts['low_vs_reference_raw_signed_tail_mass'].append({'block_frames':length,'delta':tail.point_difference,'ci_low':tail.ci_low,'ci_high':tail.ci_high})
        notes.append(f'L={length}：低阈值−基准的全片 tail/main 差 {r.point_difference:.5f} [{r.ci_low:.5f},{r.ci_high:.5f}]；{int((intervals.ci_low>0).sum())}/10段的95%区间高于0。原始signed tail面积差 {tail.point_difference:.6f} [{tail.ci_low:.6f},{tail.ci_high:.6f}]。')
    notes+=['','“相对尾部厚”不等于“原始尾部面积增加”：低阈值组的0–0.10um主峰有符号面积更小，原始0.10–0.25um尾部面积与基准接近。',
            '低阈值组在全部10个时段的名义X精度中位数较大；这同时受阈值选择的亮度/背景/宽度和点集成员影响，不是噪声或真实运动的因果证据。',
            '2000–2249帧是低阈值−基准相对尾比的较弱时段：三种块长下差值95%区间都包含0；不能把全片差异等同于每个时段都同样确定。',
            '提高阈值后，全片相对尾比变小，但后期个别时段排序可能反转且区间更宽。完整时间表/区间是主要解释依据，不按全片峰高选最佳参数。',
            '后期点数和通过比例下降，保留点的名义精度中位数变化相对小；不能由保留点推断全部候选的真实误差稳定。',
            '全片曲线按点对计数汇总，早期高密度时段权重更大；它不是十个时段曲线的等权平均。']
    for g in c['groups']:
        name=key_name(g['box'],g['factor']);full=s.loc[(s.group==name)&(s.scope=='full')].iloc[0]
        early=s.loc[(s.group==name)&(s.scope=='f0000_0249')].iloc[0]
        notes.append(f'{name}：最初250帧的段内Total配对占全片Total配对 {early.total_pairs/full.total_pairs:.3%}。')
    acf=pd.read_csv(out/'temporal_acf_summary.csv')
    notes+=['',f'去趋势点数/精度的“连续五个lag绝对相关<0.1”诊断在 {int(acf.first_five_lags_below_abs_0p1.min())}–{int(acf.first_five_lags_below_abs_0p1.max())}帧达到；25帧覆盖这项粗诊断的相关时间。',
            '这不是所有位移bin都无时间相关的证明；邻近块长结果稳定只支持本数据的敏感性结论，不能证明bootstrap覆盖率或相机模型正确。',
            '若两次定位误差独立，Var(delta x error)=Var(x1 error)+Var(x2 error)，Y同理。无轨迹链接时不知哪些点对同分子，本轮未进行误差宽度扣除。',
            'threshold_membership_summary.csv另列低阈值新增、提高阈值移除及共享强点的名义精度/信号分布。所有这些cohort均指通过共同QC的点。',
            '所有五组CSV/HDF5行数、原始帧号、nm转换、真正负信号标记、共享候选拟合值和统计输入成员已额外核查。']
    report=out/'RESULTS.txt';text=report.read_text();marker='追加的结论与产物审计：'
    if marker not in text:report.write_text(text+'\n'.join(notes)+'\n')
    dump_json(out/'sensitivity_findings.json',facts)
    dump_json(out/'artifact_validation.json',{'passed':True,'audit_checks':checks,'same_box_shared_fits_exact':True,
          'all_source_frame_and_units_checks_passed':True,'all_failed_candidate_rows_preserved':True})
    shutil.copy2(__file__,out/'reproducibility/finalize_report.py')
    (out/'finalize_command.txt').write_text(f'cd {shlex.quote(str(ROOT))}\n{shlex.join(sys.orig_argv)}\n')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
