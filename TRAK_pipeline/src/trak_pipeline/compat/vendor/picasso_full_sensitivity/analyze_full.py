#!/usr/bin/env python3
"""Full-film/250-frame TARDIS, paired block uncertainty and diagnostics."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml

from full_core import (ROOT,EXCLUSIONS,key_name,frame_histograms,decompose_counts,range_histograms,
                       point_summary,shared_resampling_plan,bootstrap_configuration)
from pilot_core import dump_json,sha256_file
from tardis_tau1.core import decompose_distribution
from tardis_tau1 import Tau1Config
from run_full import Tee


def scopes(config):
    return [('full',0,config['frames'])]+[(f'f{s:04d}_{s+249:04d}',s,s+250) for s in range(0,config['frames'],250)]


def finite_ci(values,c,minimum=.95):
    values=np.asarray(values);finite=values[np.isfinite(values)];n=len(finite)
    if n/len(values)<minimum:return np.nan,np.nan,n
    lo,hi=np.percentile(finite,c['bootstrap']['confidence_percentiles']);return float(lo),float(hi),n


def scalar_row(d,index=0):
    keys=['beta','beta_raw','signed_area','main_mass','tail_mass','tail_main_ratio','negative_mass','tail_background_count',
          'valid','shape_valid','ratio_valid']
    return {k:float(d[k][index]) if d[k].dtype.kind!='b' else bool(d[k][index]) for k in keys}


def distribution_table(tc,bc,d,c):
    width=c['distance_max_um']/c['distance_bins'];left=np.arange(c['distance_bins'])*width
    data={'distance_left_um':left,'distance_right_um':left+width,'distance_mid_um':left+width/2,
          'total_count':tc,'background_ordered_nonzero_count':bc}
    for k in ['total','background','inter','intra','shape']:data[k+'_density_um_inv']=d[k][0]/width
    return pd.DataFrame(data)


def acf_diagnostics(perframe,c,out):
    rows=[];decay=[]
    for (box,factor),part in perframe.groupby(['box','factor'],sort=False):
        part=part.sort_values('frame')
        for field in ['retained','retained_median_lpx_nm','retained_median_lpy_nm']:
            array=part[field].to_numpy(float).reshape(10,250);residual=np.full_like(array,np.nan)
            for s,a in enumerate(array):
                valid=np.isfinite(a);x=np.arange(250)
                if valid.sum()>=20:residual[s,valid]=a[valid]-np.polyval(np.polyfit(x[valid],a[valid],1),x[valid])
            acf=[]
            for lag in range(1,101):
                a=residual[:,:-lag];b=residual[:,lag:];valid=np.isfinite(a)&np.isfinite(b)
                a=a[valid];b=b[valid];den=np.sqrt(np.sum(a*a)*np.sum(b*b))
                r=float(np.sum(a*b)/den) if den else np.nan;acf.append(r)
                rows.append({'box':box,'factor':factor,'series':field,'lag_frames':lag,'lag_s':lag*.02,
                             'detrended_acf':r,'within_stratum_pairs':len(a)})
            stable=[i+1 for i in range(96) if np.isfinite(acf[i:i+5]).all() and (np.abs(acf[i:i+5])<.1).all()]
            decay.append({'box':box,'factor':factor,'series':field,'first_five_lags_below_abs_0p1':stable[0] if stable else np.nan,
                          'acf_10':acf[9],'acf_25':acf[24],'acf_50':acf[49]})
    pd.DataFrame(rows).to_csv(out/'temporal_acf.csv',index=False)
    pd.DataFrame(decay).to_csv(out/'temporal_acf_summary.csv',index=False)


def report(out,c,summary,bootstrap,paired,elapsed):
    full=summary.loc[summary.scope=='full'];lines=[
        'T1 C2 full 2500-frame sensitivity analysis','',
        '五组配置使用原 pilot 绝对阈值。使用最新 review 的相同 QC；未选择最佳参数。',
        '厂家名义校准，CRLB 不是真实误差测量；不 masking、不链接、不去卷积、不拟合扩散或状态比例。','',
        '全片汇总（signed Intra 面积 = 1−beta；tail/main 为0.10–0.25um与0–0.10um有符号面积比）：',
        'box\tfactor\tcandidates\tkept\tbeta\tsigned_area\ttail/main\tmedian_lpx/lpy_nm']
    for r in full.itertuples():lines.append(f'{r.box}\t{r.factor:g}\t{r.candidates}\t{r.points}\t{r.beta:.6f}\t{r.signed_area:.6f}\t{r.tail_main_ratio:.6f}\t{r.lpx_nm_median:.2f}/{r.lpy_nm_median:.2f}')
    lines+=['','主要问题与敏感性检验：']
    low=key_name(9,.75);ref=key_name(9,1.)
    pairs=paired.loc[(paired.comparison==low+' - '+ref)&(paired.metric=='tail_main_ratio')]
    for length in c['bootstrap']['block_lengths_frames']:
        v=pairs.loc[(pairs.block_frames==length)&(pairs.scope=='full')].iloc[0]
        lines.append(f'低阈值−基准全片 tail/main 差，block={length}：{v.point_difference:.6f}，paired percentile 95%区间 [{v.ci_low:.6f}, {v.ci_high:.6f}]，有效重采样 {v.finite_replicates}。')
    periods=summary.loc[summary.scope!='full']
    pivot=periods.pivot(index='scope',columns='group',values='tail_main_ratio')
    delta=pivot[low]-pivot[ref]
    lines.append(f'低阈值 tail/main 大于基准的时段：{int((delta>0).sum())}/{int(delta.notna().sum())}；每段只用其内部249个可能相邻帧对。')
    for r in full.itertuples():
        epoch=periods.loc[periods.group==r.group].sort_values('start_frame');first=epoch.iloc[0];last=epoch.iloc[-1]
        lines.append(f'{r.group}：首/末250帧保留点 {int(first.points)}/{int(last.points)}；通过比例 {first.retained_fraction:.3f}/{last.retained_fraction:.3f}；名义X精度中位数 {first.lpx_nm_median:.2f}/{last.lpx_nm_median:.2f}nm。')
    lines += ['',
        '统计区间：1000次共享的时间块重采样，主25帧，邻近10/50帧；每个250帧段内抽非重叠块。全片保留10个时间分层后合并原始计数。',
        '每次重新计算Total、归一化背景B、beta及Intra；复用原始逐帧距离直方图作为精确充分统计量，不把点对独立重采样。',
        'Total只加每次抽到的原始块内部f→f+1，不在重复块或拼接块的边界加配对。共享抽样权重保存在shared_block_resampling.npz。',
        '非重采样主点估计使用全部真实相邻帧；bootstrap排除块边界，因此block_restricted_controls.csv明确记录边界排除带来的点估计差异。',
        '25帧是预先选择的0.5秒折中，每段有10块；10/50帧为0.2/1秒检验。不是数据推导的最优长度。',
        'temporal_acf.csv使用每250帧段内线性去趋势的点数与名义精度，检查残余时间相关；不跨时段计算ACF。',
        '区间是单个电影的时间抽样不确定度，不能代替跨细胞/跨实验误差，也不包括相机标定不确定度。曲线带是逐bin百分位区间，不是同时置信带。',
        'signed Intra保留负bins；signed-area归一化也保留负值，不是严格非负PDF。abs(area)<=0.01标记不可用；main area<=0.01则tail/main不可用。',
        '曲线shape带或tail/main区间需要至少95%的重采样有可用分母，否则标记不可用；表中保留实际有效次数。',
        '逐点lpx/lpy是另一类量：字段单位nm/pixel明确，per_scope_summary.csv与precision_histograms.csv给出通过QC点的分布；没有用中位CRLB扣除曲线宽度。',
        '无轨迹链接，不能确认任意跨帧点对属于同分子；即使独立定位误差方差可相加，本轮也不做未经验证的位移误差修正。',
        '若低阈值尾部与较大CRLB一起变化，仍只是关联；阈值也改变检测成员、信号、背景、空间密度和配对结构。不能归因为噪声或真实运动。',
        '参数失败原因、负宽度符号诊断、负信号、负背景与QC原因分别保存。排除类别可重叠，qc_exclusion_pattern_counts.csv记录组合；不报告假阳性率。',
        '相机实际offset/sensitivity/逐像素噪声、曝光/模式和TIFF导出校正历史依然未核实。原始源像素、已有pilot与TARDIS源码保持不变。','',
        '输出：per_scope_summary.csv；per_frame_qc.csv；bootstrap_summary.csv；paired_comparisons.csv；block_restricted_controls.csv；',
        'distribution/*；bootstrap/bands/*；precision_histograms.csv；temporal_acf*.csv；figures/*；audit/*；selected/*；',
        'RULES_BEFORE_RUN.json、config.yaml、software_versions.json、fit_command.txt、analysis_command.txt、日志、测试、校验及源码快照。',
        f'统计/QC图阶段耗时 {elapsed:.1f}s；定位耗时另见fit_runtime.json。','']
    (out/'RESULTS.txt').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    out=Path(parser.parse_args().run).resolve();assert (out/'FIT_COMPLETED.json').exists()
    c=yaml.safe_load((out/'config.yaml').read_text());started=time.perf_counter()
    log=(out/'analysis.log').open('w');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    assert sha256_file(ROOT/'full_core.py')==sha256_file(out/'reproducibility/full_core.py')
    for name in ['distribution','pair_histograms','bootstrap','bootstrap/bands','figures']:(out/name).mkdir(exist_ok=True)
    for script in ['analyze_full.py','full_plots.py']:shutil.copy2(ROOT/script,out/'reproducibility'/script)
    (out/'analysis_command.txt').write_text(f'cd {shlex.quote(str(ROOT))}\n{shlex.join(sys.orig_argv)}\n')
    perframe=pd.read_csv(out/'per_frame_qc.csv');hist={};point_rows=[];precision_rows=[];normalization_checks=[]
    scope_list=scopes(c);mids=(np.arange(c['distance_bins'])+.5)*c['distance_max_um']/c['distance_bins']
    for g in c['groups']:
        name=key_name(g['box'],g['factor']);print('Full/segmented distributions:',name,flush=True)
        selected=pd.read_csv(out/'selected'/(name+'.csv'))
        th,bh,adj,zeros,edges=frame_histograms(selected,c);hist[name]=(th,bh)
        np.savez_compressed(out/'pair_histograms'/(name+'.npz'),total_by_source_frame=th,background_by_frame=bh,
                  original_frames=np.arange(c['frames']),adjacent_occupied=adj,same_frame_zeros_removed=zeros,edges_m=edges)
        q=perframe.loc[(perframe.box==g['box'])&(perframe.factor==g['factor'])]
        pieces=[range_histograms(th,bh,s,s+250)[0] for s in range(0,2500,250)]
        np.testing.assert_array_equal(np.sum(pieces,axis=0)+th[np.arange(249,2499,250)].sum(axis=0),th[:-1].sum(axis=0))
        for scope,a,b in scope_list:
            tc,bc=range_histograms(th,bh,a,b);d=decompose_counts(tc,bc,c)
            # Existing TARDIS decomposition is a second check of beta/Intra.
            if d['valid'][0]:
                expected=decompose_distribution(tc/tc.sum(),bc/bc.sum(),Tau1Config(bootstrap_count=0,save_plots=False))
                np.testing.assert_allclose(d['beta'][0],expected['beta'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(d['intra'][0],expected['intra_contribution'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(d['signed_area'][0],1-d['beta'][0],atol=1e-12)
            if d['shape_valid'][0]:np.testing.assert_allclose(d['shape'][0].sum(),1,atol=1e-12)
            normalization_checks.append({'group':name,'scope':scope,'valid':bool(d['valid'][0]),'shape_valid':bool(d['shape_valid'][0])})
            subset=selected.loc[selected.frame.between(a,b-1)];qc=q.loc[q.frame.between(a,b-1)]
            row={'group':name,'box':g['box'],'factor':g['factor'],'absolute_threshold':g['threshold'],'scope':scope,
                'start_frame':a,'end_frame':b-1,'time_mid_s':(a+b-1)/2*.02,'frames':b-a,
                'possible_adjacent_frame_pairs':b-a-1,'occupied_adjacent_frame_pairs':int(adj[a:b-1].sum()),
                'total_pairs':int(tc.sum()),'background_pairs':int(bc.sum()),'candidates':int(qc.candidates.sum()),
                'retained_fraction':len(subset)/max(1,int(qc.candidates.sum())),**point_summary(subset),**scalar_row(d)}
            for k in [*EXCLUSIONS,'raw_negative_width','nonpositive_backend_amplitude','negative_background']:
                row[k+'_count']=int(qc[k].sum())
            row['sparse_background_tail']=d['tail_background_count'][0]<100
            row['beta_clipped']=bool(d['valid'][0] and d['beta'][0]!=d['beta_raw'][0])
            row['shape_status']='available_nominal_signed' if d['shape_valid'][0] else 'unavailable_invalid_or_abs_area_le_0p01'
            point_rows.append(row);folder=out/'distribution'/name;folder.mkdir(exist_ok=True)
            distribution_table(tc,bc,d,c).to_csv(folder/(scope+'.csv'),index=False)
            for axis in ['lpx_nm','lpy_nm']:
                values=subset[axis].to_numpy();pe=np.arange(0,505,5);counts,_=np.histogram(values,bins=pe)
                row[axis+'_above_500_nm']=int((values>500).sum())
                for lo,hi,count in zip(pe[:-1],pe[1:],counts):precision_rows.append({'group':name,'scope':scope,'axis':axis,
                    'left_nm':lo,'right_nm':hi,'count':int(count),'population':'QC retained; nominal CRLB'})
    summary=pd.DataFrame(point_rows);summary.to_csv(out/'per_scope_summary.csv',index=False)
    pd.DataFrame(precision_rows).to_csv(out/'precision_histograms.csv',index=False)
    acf_diagnostics(perframe,c,out)
    weights_all={};bootstrap_rows=[];controls_rows=[];paired_rows=[]
    for length in c['bootstrap']['block_lengths_frames']:
        weights=shared_resampling_plan(c,length);weights_all[f'L{length}']=weights
        draws={};print(f'Shared paired bootstrap: block={length}, repetitions={c["bootstrap"]["replicates"]}',flush=True)
        for g in c['groups']:
            name=key_name(g['box'],g['factor']);results,controls=bootstrap_configuration(*hist[name],c,length,weights)
            for key in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                draws[name+'__'+key]=np.stack([d[key] for d in results])
            for scope_index,((scope,a,b),d,control) in enumerate(zip(scope_list,results,controls)):
                if d['shape_valid'].any():np.testing.assert_allclose(d['shape'][d['shape_valid']].sum(axis=1),1,atol=1e-12)
                row={'group':name,'scope':scope,'block_frames':length,'block_seconds':length*.02,
                     'replicates':c['bootstrap']['replicates'],'shape_valid_fraction':float(d['shape_valid'].mean()),
                     'beta_clipped_fraction':float(np.mean(d['beta']!=d['beta_raw'])),
                     'source_blocks':(b-a)//length if scope!='full' else c['frames']//length}
                for key in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                    lo,hi,n=finite_ci(d[key],c);row.update({key+'_ci_low':lo,key+'_ci_high':hi,key+'_finite_replicates':n})
                bootstrap_rows.append(row)
                point=summary.loc[(summary.group==name)&(summary.scope==scope)].iloc[0]
                cr={'group':name,'scope':scope,'block_frames':length,**scalar_row(control),
                    'original_beta':point.beta,'original_tail_main_ratio':point.tail_main_ratio,
                    'beta_boundary_exclusion_delta':control['beta'][0]-point.beta,
                    'tail_main_boundary_exclusion_delta':control['tail_main_ratio'][0]-point.tail_main_ratio,
                    'possible_block_internal_pairs':c['frames']-c['frames']//length if scope=='full' else (b-a)-(b-a)//length}
                controls_rows.append(cr)
                band={'distance_mid_um':mids}
                width=c['distance_max_um']/c['distance_bins']
                for key in ['total','background','inter','intra','shape']:
                    good=np.isfinite(d[key]).all(axis=1)
                    enough=good.mean()>=c['bootstrap']['minimum_finite_fraction_for_shape_band']
                    if enough:lower,upper=np.quantile(d[key][good],np.array(c['bootstrap']['confidence_percentiles'])/100,axis=0)/width
                    else:lower=upper=np.full(c['distance_bins'],np.nan)
                    band[key+'_ci_low']=lower;band[key+'_ci_high']=upper;band[key+'_finite_replicates']=int(good.sum())
                pd.DataFrame(band).to_csv(out/'bootstrap/bands'/f'{name}_{scope}_L{length}.csv',index=False)
        names=[key_name(g['box'],g['factor']) for g in c['groups']];ref=key_name(9,1.)
        for other in [name for name in names if name!=ref]:
            for scope_index,(scope,a,b) in enumerate(scope_list):
                for metric in ['beta','signed_area','tail_main_ratio','tail_mass']:
                    difference=draws[other+'__'+metric][scope_index]-draws[ref+'__'+metric][scope_index]
                    lo,hi,n=finite_ci(difference,c)
                    arow=summary.loc[(summary.group==other)&(summary.scope==scope)].iloc[0]
                    brow=summary.loc[(summary.group==ref)&(summary.scope==scope)].iloc[0]
                    paired_rows.append({'comparison':other+' - '+ref,'scope':scope,'block_frames':length,'metric':metric,
                        'point_difference':arow[metric]-brow[metric],'ci_low':lo,'ci_high':hi,'finite_replicates':n,
                        'paired_resampling':True})
        np.savez_compressed(out/'bootstrap'/f'draw_metrics_L{length}.npz',**draws)
    np.savez_compressed(out/'bootstrap/shared_block_resampling.npz',**weights_all)
    bootstrap=pd.DataFrame(bootstrap_rows);paired=pd.DataFrame(paired_rows)
    bootstrap.to_csv(out/'bootstrap_summary.csv',index=False)
    paired.to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(controls_rows).to_csv(out/'block_restricted_controls.csv',index=False)
    from full_plots import make_figures
    print('Rendering full-film, time-window and uncertainty figures...',flush=True)
    make_figures(out,c,summary,bootstrap,paired,perframe)
    for p,h in json.loads((out/'protected_source_hashes.json').read_text()).items():assert sha256_file(p)==h
    source=Path(c['input_tiff']);meta=json.loads((out/'input_metadata.json').read_text())
    assert source.stat().st_size==meta['source_bytes'] and source.stat().st_mtime_ns==meta['source_mtime_ns']
    elapsed=time.perf_counter()-started;report(out,c,summary,bootstrap,paired,elapsed)
    dump_json(out/'analysis_validation.json',{'all_five_full_histograms_match_existing_tardis_binwise':True,
         'all_55_decompositions_checked':True,'full_vs_segments_plus_real_boundary_pairs_conserved':True,
         'no_resampled_block_boundary_pairs':True,'all_configurations_share_weights':True,
         'near_zero_area_guard_applied':True,'negative_bins_retained':True,'no_uncertainty_deconvolution':True,
         'protected_inputs_unchanged':True,'normalization_checks':normalization_checks})
    dump_json(out/'analysis_runtime.json',{'seconds':elapsed,'finished_utc':datetime.now(timezone.utc).isoformat()})
    dump_json(out/'COMPLETED.json',{'full_frames':2500,'configurations':5,'segments':10,'block_lengths':[10,25,50],
                                  'bootstrap_replicates':1000,'masking':False,'state_fitting':False})
    print('Completed full sensitivity analysis:',out,flush=True)


if __name__=='__main__':main()
