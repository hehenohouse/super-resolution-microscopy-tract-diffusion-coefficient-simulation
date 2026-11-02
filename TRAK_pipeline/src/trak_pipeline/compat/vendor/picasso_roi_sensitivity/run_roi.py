#!/usr/bin/env python3
"""Immutable-cache ROI statistics and centered full-curve inference. No refitting."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import tifffile
import yaml

from roi_core import (ROOT,EXCLUSIONS,key_name,load_imagej_roi,read_mat_roi,points_in_polygon,
    frame_histograms,range_histograms,frame_qc,point_summary,decompose_roi,bootstrap_roi,
    scalar_row,distribution_table,scopes,finite_ci,paired_difference,dump_json,sha256_file)
from full_core import shared_resampling_plan
from cache_annotation import annotate_group
from curve_inference import centered_max_test,effect_sizes,continuous_intervals,mc_interval
import roi_plots


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,value):self.stream.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.stream.flush();self.log.flush()


def jsonable(value):
    if isinstance(value,np.ndarray):return [jsonable(x) for x in value]
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def dump(path,data):
    path.write_text(json.dumps(jsonable(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def histogram_independent_check(selected,th,bh,c):
    # Actual ROI observations, including an empty frame if present.
    counts=selected.groupby('frame').size().reindex(range(c['frames']),fill_value=0)
    frames=list(range(8))+[1250,2450,2498]
    empty=counts[counts==0].index.tolist()
    if empty:frames.extend(empty[:2])
    edges=np.linspace(0,c['distance_max_um'],c['distance_bins']+1)
    groups={int(f):part[['x','y']].to_numpy()*c['pixel_size_nm']/1000 for f,part in selected.groupby('frame')}
    for f in sorted(set(frames)):
        a=groups.get(f,np.empty((0,2)));b=groups.get(f+1,np.empty((0,2)))
        distances=np.array([np.hypot(*(x-y)) for x in a for y in b])
        expected=np.histogram(distances,bins=edges)[0]
        np.testing.assert_array_equal(expected,th[f])
        distances=np.array([np.hypot(*(x-y)) for x in a for y in a if np.any(x!=y)])
        np.testing.assert_array_equal(np.histogram(distances,bins=edges)[0],bh[f])
    return len(set(frames))


def per_frame_table(table,c,g):
    # QC reasons refer to finite fitted centers inside the ROI, not to a false-positive rate.
    q=frame_qc(table.loc[table.roi_inside==1],c,g['box'],g['factor'],g['threshold'])
    def counts(mask):return table.loc[mask].groupby('frame').size().reindex(range(c['frames']),fill_value=0).to_numpy()
    q['whole_candidates']=counts(np.ones(len(table),bool))
    q['whole_qc_pass']=counts(table.qc_unflagged_model_fit)
    q['invalid_center_candidates']=counts(table.roi_inside<0)
    q['qc_pass_outside_roi']=counts(table.qc_unflagged_model_fit&(table.roi_inside!=1))
    q['roi_fraction_of_qc_pass']=q.retained/q.whole_qc_pass.replace(0,np.nan)
    q['group']=key_name(g['box'],g['factor'])
    return q


def report(out,c,summary,boot,paired,tests,pair_results,sim):
    full=summary.loc[summary.scope=='full'];primary=tests.loc[tests.block_frames==25].iloc[0]
    lines=['T1 C2: common cell ROI sensitivity and full-curve test','',
        '本目录复用已完成的定位缓存：未重新检测、拟合或改变原 QC/CRLB。',
        '原 ROI 与 MAT 顶点完全一致；C2 局部像素坐标直接使用，不应用 LOCB 变换。',
        '选择规则：原 QC 通过，且拟合中心在同一细胞 ROI 内或边界上。不是线粒体 mask。','',
        '全片 0–2499；下表 tail/main 指原 signed Intra 在0.10–0.25与0–0.10um的面积比。',
        'box\tfactor\tcandidates_in_ROI\tQC_kept_ROI\tbeta\tsigned_area\ttail/main\tmedian_nominal_lpx/lpy_nm']
    if not sim['nominal_alpha_in_null_rate_interval']:
        lines[2:2]=['统计校准警告：预设maxT方法在玩具零假设模拟中拒绝率高于5%。',
            '下面p值和95%家族带是按预设规则计算的名义近似量，不能声称已验证5%错误率或95%覆盖。',
            '不通过调整随机种子、删除模拟或更改主检验规则使验证通过；尤其不把临界组对作为可靠显著结论。','']
    for r in full.itertuples():lines.append(f'{r.box}\t{r.factor:g}\t{r.candidates}\t{r.points}\t{r.beta:.6f}\t{r.signed_area:.6f}\t{r.tail_main_ratio:.6f}\t{r.lpx_nm_median:.2f}/{r.lpy_nm_median:.2f}')
    lines+=['','主要全曲线问题：',
        'H0：五条面积归一化 signed Intra 曲线在0–3um全部300个固定bins上相同。',
        '归一化范围0–3um；不平滑、不删负bins、不使用普通KS；曲线不是严格非负PDF。',
        f'L25主检验：status={primary.status}，Tobs={primary.statistic:.6f} (1/um)，整体p={primary.p_global:.9g}。',
        f'五组共同有效重复 {int(primary.joint_valid_count)}/{c["bootstrap"]["replicates"]}，比例 {primary.joint_valid_fraction:.3%}。',
        f'加一p分辨率 {primary.p_resolution:.9g}；超越次数 {int(primary.exceedances)}；条件Monte Carlo超越概率95%区间 [{primary.mc_ci_low:.6g},{primary.mc_ci_high:.6g}]。',
        '每个重复重新计算 ROI 的Total/B/beta/shape，再减各组自身在五组交集上的bootstrap平均曲线。',
        '零分布为中心化误差的十组对/300bins最大绝对差。没有把未中心化组差作为零分布。',
        'maxT校正p与同时带都使用同一个五组共同有效重复集合；描述性两组区间使用两组交集。',
        f'名义家族95%同时带半宽 {primary.critical:.6f} (1/um)，目标覆盖域为全部十个组对及300bins；实际覆盖未通过本次模拟校准。',
        '名义同时带排除零的连续区间保存在 significant_distance_intervals.csv；原图逐点阴影不用于显著性判断。','',
        '十组对数值结果（前组−后组；L25名义整体p通过预设门槛，以下仍受上述校准限制）：',
        'comparison\tmaxT_adjusted_p\tsup_difference_1_per_um\tintegrated_abs_difference\tmain_mass_difference\ttail_mass_difference']
    for r in pair_results.itertuples():lines.append(f'{r.comparison}\t{r.adjusted_p:.9g}\t{r.sup_abs_density_difference:.6f}\t{r.integrated_abs_difference:.6f}\t{r.normalized_main_mass_difference:.6f}\t{r.normalized_tail_mass_difference:.6f}')
    lines+=['','块长度与边界敏感性：']
    for r in tests.itertuples():lines.append(f'L={r.block_frames}：p={r.p_global:.9g}；块内点估计对照p={r.boundary_restricted_p:.9g}；共同有效比例{r.joint_valid_fraction:.3%}；最大组对中心化偏移{r.max_pair_centering_offset:.6g}，边界删除变化{r.max_pair_boundary_change:.6g} (1/um)。')
    lines+=['L25是唯一主要检验；L10/L50是敏感性检查，不选择最小p。',
        '主点估计使用全片全部真实相邻帧；bootstrap只用原始块内部相邻帧，两者构造存在近似差异。',
        'block_restricted_controls.csv及centering_offsets.csv披露该差异；有限电影和分层块抽样下的p是近似校准，非精确检验。','',
        'ROI内低阈值尾部与不确定度：']
    low=key_name(9,.75);ref=key_name(9,1.)
    for length in [10,25,50]:
        r=paired.loc[(paired.group_a==low)&(paired.group_b==ref)&(paired.scope=='full')&(paired.block_frames==length)&(paired.metric=='tail_main_ratio')].iloc[0]
        lines.append(f'L{length}：低阈值−基准 tail/main 差 {r.point_difference:.6g}，95%区间 [{r.ci_low:.6g},{r.ci_high:.6g}]，两组共同有效比例{r.common_valid_fraction:.3%}。')
    periods=summary.loc[summary.scope!='full'];pivot=periods.pivot(index='scope',columns='group',values='tail_main_ratio')
    delta=pivot[low]-pivot[ref];lines.append(f'可用时段中低阈值尾比更大 {int((delta>0).sum())}/{int(delta.notna().sum())}；缺项不算作无差异。')
    full_pair=paired.loc[(paired.group_a==low)&(paired.group_b==ref)&(paired.scope=='full')&(paired.block_frames==25)]
    for metric,description in [('tail_mass','原始 signed 尾部面积'),('main_mass','原始 signed 主峰面积')]:
        selected=full_pair.loc[full_pair.metric==metric]
        if len(selected):
            r=selected.iloc[0]
            lines.append(f'低阈值−基准 {description}差 {r.point_difference:.6g}，名义描述性95%区间 [{r.ci_low:.6g},{r.ci_high:.6g}]。')
    lines+=['尾/主比升高不能等同于原始尾部面积增加：低阈值的主峰面积下降更多。',
        '低阈值组名义 CRLB 较大，但这不能识别噪声、定位误差与真实运动各自的因果贡献。',
        '全片曲线最大差异主要在峰附近；与远尾部逐bin差异或生物学运动增加不是同一个结论。',
        '基准 Box9 与 Box7 的名义校正p接近0.05；邻近块长及有限块方差诊断会改变其门槛判断，不能作为稳定显著组对。',
        '基准 Box9 与 Box11 的名义校正p大于0.05；这不能证明两条曲线相同。',
        '后期点数下降；ROI内可判定拟合中心的全片原QC通过比例约93–97%，不等同于可信分子的准确率。']
    lines+=['','稀疏尾部/不可用时段：']
    for r in periods.loc[~periods.shape_valid].itertuples():lines.append(f'{r.group} {r.scope}：{r.shape_status}；B有序尾部点对{r.background_tail_count}，Total尾部点对{r.total_tail_count}，B非空bins {r.background_tail_nonempty_bins}/90。')
    lines+=['100点对是最低警戒线，不代表独立样本量或充分统计支持；B的有序点对有两个方向。',
        '有效重复不足95%时相应区间不发布；原点估计不可用时，不因某些bootstrap偶然通过而发布结果。',
        '十个250帧时段仅描述，不额外做整体p检验；空帧和原始帧号完整保留。','',
        '定位不确定度：manufacturer_nominal CRLB 是另一类量，不是曲线置信带。未验证约50nm真实误差。',
        'ROI选择会排除跨边界运动；统计代表两个端点在ROI内的空间选择，不是整细胞无偏运动分布。',
        '所有结论限于这一个电影中分析参数造成的差异，不解释成生物学组间差异或噪声/运动因果。',
        '背景系数beta和signed面积不是生物学状态比例。未做masking、轨迹、去卷积、漂移或扩散校正。',
        '相机实测offset/gain/逐像素噪声、曝光及TIFF导出处理仍未确认。','',
        '实现校准模拟：',
        f'零假设拒绝 {sim["null_rejections"]}/{sim["null_movies"]}，拒绝率{sim["null_rejection_rate"]:.3%}，二项95%区间{sim["null_rejection_rate_95_binomial_ci"]}；同时带覆盖{sim["family_band_null_coverage"]:.3%}。',
        f'已知局部差异检出率{sim["known_alternative_rejection_rate"]:.3%}；错误未中心化参考的检出率{sim["incorrect_uncentered_alternative_rejection_rate"]:.3%}。',
        '这是具有共享组噪声/距离相关性的线性signed曲线玩具验证；不证明真实电影时间混合、相机校准或非线性beta的覆盖率正确。','',
        '输出索引见 README.txt；所有固定规则、完整命令、源指纹、代码快照与验证在本目录。','']
    (out/'RESULTS.txt').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config.yaml');parser.add_argument('--output')
    args=parser.parse_args();extra=yaml.safe_load(Path(args.config).read_text())
    source=(ROOT/extra['source_run']).resolve();assert (source/'COMPLETED.json').exists()
    inherited=yaml.safe_load((source/'config.yaml').read_text());c={**inherited,**extra}
    assert c['frames']==2500 and c['segment_frames']==250 and c['distance_bins']==300 and c['distance_max_um']==3
    assert c['normalization_range_um']==c['test_range_um']==[0.,3.]
    assert c['bootstrap']['replicates']==1000 and c['bootstrap']['primary_block_frames']==25
    assert not any(c[k] for k in ['tracking','masking','drift_correction','uncertainty_deconvolution','diffusion_correction','biological_state_fitting'])
    assert importlib.metadata.version('picassosr')=='0.11.3'
    assert sha256_file(ROOT.parent/'picasso_full_sensitivity/full_core.py')==sha256_file(source/'reproducibility/full_core.py')
    out=Path(args.output).resolve() if args.output else ROOT/'runs'/('T1_C2_ROI_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    assert out.is_relative_to(ROOT/'runs');out.mkdir(parents=True,exist_ok=False)
    for folder in ['audit','selected','distribution','pair_histograms','bootstrap','bootstrap/bands','inference','inference/bands','figures','reproducibility']:(out/folder).mkdir()
    log=(out/'run.log').open('w');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    start=time.perf_counter();print('ROI output:',out,flush=True)
    shutil.copy2(args.config,out/'requested_config.yaml');(out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True))
    for p in ROOT.glob('*.py'):shutil.copy2(p,out/'reproducibility'/p.name)
    shutil.copytree(source/'reproducibility',out/'reproducibility/source_full_snapshot')
    shutil.copy2(source/'reproducibility/requirements.lock.txt',out/'reproducibility/requirements.lock.txt')
    if (ROOT/'README.txt').exists():shutil.copy2(ROOT/'README.txt',out/'README.txt')
    (out/'run_command.txt').write_text(f'cd {shlex.quote(str(ROOT))}\n{shlex.join(sys.orig_argv)}\n')
    dump(out/'RULES_BEFORE_RUN.json',{'configuration':c,'source_run':str(source),
        'roi_geometry':'native C2 pixels, unchanged ImageJ/MAT vertices, inclusive boundary; no LOCB transform',
        'pairing':'replicate_id intersection, never independently compact groups',
        'primary_inference':'full movie, L25, alpha.05, 0-3um/300 bins, joint five-group centered maxT',
        'family':'ten pairs and all bins; no posthoc changes of test range or block length',
        'minimum_support':'both beta-tail counts>=100; only a minimum count diagnostic, not independent sample size',
        'preservation':'original numeric fields and QC remain unchanged; only ROI membership flags appended'})
    software={'python':sys.version,'platform':platform.platform(),'executable':sys.executable,
              'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'refitting':False}
    dump(out/'software_versions.json',software)
    protected=list((source/'audit').glob('*'))+list((source/'selected').glob('*'))+list((source/'pair_histograms').glob('*'))
    protected += [source/p for p in ['config.yaml','RESULTS.txt','bootstrap/shared_block_resampling.npz','input_metadata.json']]
    protected += list((ROOT.parent/'TARDIS/path_irrelevant_python/tardis_tau1').glob('*.py'))
    protected += [Path(p) for p in json.loads((source/'protected_source_hashes.json').read_text())]
    locked={str(p):sha256_file(p) for p in set(protected) if p.is_file()};dump(out/'protected_source_hashes.json',locked)
    tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_roi'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('ROI preflight tests failed')
    print('ROI preflight tests passed.',flush=True)
    roi_path=Path(c['roi_file']);assert sha256_file(roi_path)==c['roi_sha256']
    roi=load_imagej_roi(roi_path);mat_roi,mat_record_path=read_mat_roi(c['mat_file'],c['mat_record'])
    np.testing.assert_array_equal(roi,mat_roi)
    assert np.isfinite(roi).all() and (roi>=0).all() and (roi[:,0]<c['width']).all() and (roi[:,1]<c['height']).all()
    inputmeta=json.loads((source/'input_metadata.json').read_text());tiff=Path(c['input_tiff']);stat=tiff.stat()
    assert stat.st_size==inputmeta['source_bytes'] and stat.st_mtime_ns==inputmeta['source_mtime_ns']
    with tifffile.TiffFile(tiff) as f:
        assert len(f.pages)==2500 and f.pages[0].shape==(684,428) and f.pages[0].dtype==np.dtype('uint16')
        a=np.mean([f.pages[k].asarray().astype(float) for k in range(50)],axis=0)
    shutil.copy2(roi_path,out/'cell_original.roi')
    pd.DataFrame(roi,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    roi_meta={'roi_source':str(roi_path),'roi_sha256':c['roi_sha256'],'vertices':len(roi),'mat_file':c['mat_file'],
        'mat_record':c['mat_record'],'mat_hdf_record':mat_record_path,'vertices_exactly_equal':True,
        'mat_vertex_fingerprint':__import__('hashlib').sha256(mat_roi.astype('<f8').tobytes()).hexdigest(),
        'x_range_px':roi[:,0].tolist() and [roi[:,0].min(),roi[:,0].max()],
        'y_range_px':[roi[:,1].min(),roi[:,1].max()],
        'coordinate_status':'native C2 coordinates; ImageJ/MAT equal and original C2 overlay checked in prior read-only inspection',
        'boundary_included':True,'boundary_tolerance_px':1e-9,'source_tiff_metadata':inputmeta,'no_refitting':True}
    dump(out/'roi_validation.json',roi_meta)
    original_units=json.loads((source/'field_units.json').read_text());original_units.update({
        'roi_inside':'CSV true/false/empty; HDF5 int8: 1 inside/boundary, 0 outside, -1 invalid center',
        'roi_geometry_status':'inside / outside / invalid_center','stats_selected_roi':'original QC AND roi_inside',
        'curve_density':'1/um; signed, normalized over0-3um','integrated_abs_difference':'dimensionless signed-density L1 distance, not PDF total variation',
        'replicate_id':'original row in shared temporal bootstrap weights; never reindexed'})
    dump(out/'field_units.json',original_units)
    annotated=[];perframes=[];summary_rows=[];hist={};observed={};precision=[];coverage_checks=[];patterns=[]
    width=c['distance_max_um']/c['distance_bins'];mids=(np.arange(c['distance_bins'])+.5)*width;scope_list=scopes(c)
    for g in c['groups']:
        name=key_name(g['box'],g['factor']);print('Annotating cached candidates and ROI distances:',name,flush=True)
        table,kept,check=annotate_group(source,out,name,roi,roi_meta);annotated.append(check)
        q=per_frame_table(table,c,g);perframes.append(q)
        th,bh,adj,zeros,edges=frame_histograms(kept,c);hist[name]=(th,bh)
        real_checks=histogram_independent_check(kept,th,bh,c)
        previous=pd.read_csv(source/'selected'/(name+'.csv'))
        cover=np.array([[-1,-1],[c['width']+1,-1],[c['width']+1,c['height']+1],[-1,c['height']+1]])
        assert points_in_polygon(previous[['x','y']],cover).all()
        wt,wb,wa,wz,we=frame_histograms(previous,c)
        with np.load(source/'pair_histograms'/(name+'.npz')) as old:
            np.testing.assert_array_equal(wt,old['total_by_source_frame']);np.testing.assert_array_equal(wb,old['background_by_frame'])
        coverage_checks.append({'group':name,'full_cover_reproduces_original_histograms':True,'independent_ROI_frames_checked':real_checks})
        np.savez_compressed(out/'pair_histograms'/(name+'.npz'),total_by_source_frame=th,background_by_frame=bh,
            original_frames=np.arange(2500),adjacent_occupied=adj,same_frame_zeros_removed=zeros,edges_m=edges)
        pieces=[range_histograms(th,bh,s,s+250)[0] for s in range(0,2500,250)]
        np.testing.assert_array_equal(sum(pieces)+th[np.arange(249,2499,250)].sum(0),th[:-1].sum(0))
        for scope,s,e in scope_list:
            tc,bc=range_histograms(th,bh,s,e);d=decompose_roi(tc,bc,c);observed[name,scope]=d
            if d['shape_valid'][0]:np.testing.assert_allclose(d['shape'].sum(),1,atol=1e-12)
            subset=kept.loc[kept.frame.between(s,e-1)];qs=q.loc[q.frame.between(s,e-1)]
            tail=(mids>=2.1)&(mids<3.)
            block_b=[int(bh[k:k+25,tail].sum()) for k in range(s,e,25)]
            block_t=[int(th[k:k+24,tail].sum()) for k in range(s,e,25)]
            row={'group':name,'box':g['box'],'factor':g['factor'],'scope':scope,'start_frame':s,'end_frame':e-1,
                'time_mid_s':(s+e-1)/2*.02,'absolute_threshold':g['threshold'],'frames':e-s,
                'possible_adjacent_frame_pairs':e-s-1,'occupied_adjacent_frame_pairs':int(adj[s:e-1].sum()),
                'total_pairs':int(tc.sum()),'background_pairs':int(bc.sum()),'candidates':int(qs.candidates.sum()),
                'whole_candidates':int(qs.whole_candidates.sum()),'whole_qc_pass':int(qs.whole_qc_pass.sum()),
                'retained_fraction':len(subset)/max(1,int(qs.candidates.sum())),
                'roi_fraction_of_whole_qc':len(subset)/max(1,int(qs.whole_qc_pass.sum())),
                'empty_roi_qc_frames':int((qs.retained==0).sum()),'invalid_center_candidates':int(qs.invalid_center_candidates.sum()),
                'B_tail_frames':int((bh[s:e,tail].sum(1)>0).sum()),'Total_tail_source_frames':int((th[s:e-1,tail].sum(1)>0).sum()),
                'B_tail_supported_L25_blocks':sum(x>0 for x in block_b),'Total_tail_supported_L25_blocks':sum(x>0 for x in block_t),
                'L25_blocks':len(block_b),**point_summary(subset),**scalar_row(d)}
            for flag in [*EXCLUSIONS,'raw_negative_width','nonpositive_backend_amplitude','negative_background']:row[flag+'_count']=int(qs[flag].sum())
            summary_rows.append(row)
            folder=out/'distribution'/name;folder.mkdir(exist_ok=True);distribution_table(tc,bc,d,c).to_csv(folder/(scope+'.csv'),index=False)
            for axis in ['lpx_nm','lpy_nm']:
                pe=np.arange(0,505,5);values=subset[axis].to_numpy();ct,_=np.histogram(values,bins=pe)
                for lo,hi,n in zip(pe[:-1],pe[1:],ct):precision.append({'group':name,'scope':scope,'axis':axis,
                    'left_nm':lo,'right_nm':hi,'count':int(n),'above_500_nm':int((values>500).sum()),'population':'ROI and original QC; manufacturer_nominal'})
            reason=table.loc[table.frame.between(s,e-1)]
            for (geo,pattern),n in reason.groupby(['roi_geometry_status','qc_exclusion_reasons'],dropna=False).size().items():
                patterns.append({'group':name,'scope':scope,'roi_geometry_status':geo,'qc_exclusion_reasons':pattern,'count':int(n)})
        if g['box']==9 and g['factor']==1:
            roi_plots.roi_overlay(out,a,roi,mat_roi,pd.read_csv(source/'selected'/(name+'.csv')))
        print(f'{name}: original candidates {len(table)}, ROI original-QC points {len(kept)}',flush=True)
    perframe=pd.concat(perframes,ignore_index=True);summary=pd.DataFrame(summary_rows)
    perframe.to_csv(out/'per_frame_qc.csv',index=False);summary.to_csv(out/'per_scope_summary.csv',index=False)
    pd.DataFrame(precision).to_csv(out/'precision_histograms.csv',index=False)
    pd.DataFrame(patterns).to_csv(out/'qc_exclusion_pattern_counts.csv',index=False)
    dump(out/'cache_annotation_validation.json',annotated)
    dump(out/'histogram_validation.json',coverage_checks)
    roi_plots.count_figure(out,c,perframe);roi_plots.full_shapes(out,c,summary)
    print('ROI overlay, original-frame counts and normalized curves saved before formal inference.',flush=True)
    names=[key_name(g['box'],g['factor']) for g in c['groups']];pair_order=list(combinations(range(5),2))
    with np.load(source/'bootstrap/shared_block_resampling.npz') as w:
        weights={length:w[f'L{length}'].copy() for length in c['bootstrap']['block_lengths_frames']}
    for length,w in weights.items():np.testing.assert_array_equal(w,shared_resampling_plan(c,length))
    shutil.copy2(source/'bootstrap/shared_block_resampling.npz',out/'bootstrap/shared_block_resampling.npz')
    bootstrap_rows=[];paired_rows=[];control_rows=[];test_rows=[];offset_rows=[];pair_results=[];intervals=[];inferences={}
    # Point shape density is over the entire 0-3um domain, including negative bins.
    point=np.stack([observed[name,'full']['shape'][0]/width for name in names])
    for length,w in weights.items():
        print('ROI paired block bootstrap and centered inference:',length,flush=True)
        scalar_draws={};shape_draws=[];shape_valids=[];block_points=[]
        for name in names:
            draws,controls=bootstrap_roi(*hist[name],c,length,w)
            shape_draws.append(np.stack([d['shape']/width for d in draws]))
            shape_valids.append(np.stack([d['shape_valid'] for d in draws]))
            block_points.append(controls[0]['shape'][0]/width)
            for metric in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                scalar_draws[name+'__'+metric]=np.stack([d[metric] for d in draws])
            for (scope,s,e),d,control in zip(scope_list,draws,controls):
                point_d=observed[name,scope]
                r={'group':name,'scope':scope,'block_frames':length,'replicates':1000,
                   'valid_fraction':float(d['valid'].mean()),'shape_valid_fraction':float(d['shape_valid'].mean()),
                   'ratio_valid_fraction':float(d['ratio_valid'].mean()),'tail_supported_fraction':float(d['tail_supported'].mean())}
                for metric in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                    lo,hi,n,f=finite_ci(d[metric],c['minimum_joint_valid_fraction'],eligible=np.isfinite(point_d[metric][0]))
                    r.update({metric+'_ci_low':lo,metric+'_ci_high':hi,metric+'_finite_replicates':n,metric+'_finite_fraction':f})
                bootstrap_rows.append(r)
                ctl={'group':name,'scope':scope,'block_frames':length,**scalar_row(control),
                    'original_beta':point_d['beta'][0],'original_tail_main_ratio':point_d['tail_main_ratio'][0],
                    'possible_block_internal_pairs':(e-s)-(e-s)//length,
                    'beta_boundary_exclusion_delta':control['beta'][0]-point_d['beta'][0],
                    'tail_main_boundary_exclusion_delta':control['tail_main_ratio'][0]-point_d['tail_main_ratio'][0]}
                control_rows.append(ctl)
                band={'distance_mid_um':mids}
                for key in ['total','background','inter','intra','shape']:
                    good=np.isfinite(d[key]).all(1)
                    enough=good.mean()>=c['minimum_joint_valid_fraction'] and np.isfinite(point_d[key]).all()
                    lo,hi=np.quantile(d[key][good],[.025,.975],axis=0)/width if enough else (np.full(300,np.nan),np.full(300,np.nan))
                    band[key+'_ci_low']=lo;band[key+'_ci_high']=hi;band[key+'_finite_replicates']=int(good.sum())
                pd.DataFrame(band).to_csv(out/'bootstrap/bands'/f'{name}_{scope}_L{length}.csv',index=False)
        sd=np.stack(shape_draws);sv=np.stack(shape_valids);bp=np.stack(block_points)
        # Keep full-film arrays with their original replicate IDs and validity flags.
        np.savez_compressed(out/'bootstrap'/f'full_curve_draws_L{length}.npz',density=sd[:,0],valid=sv[:,0],
            group_order=np.array(names),replicate_id=np.arange(1000),distance_mid_um=mids,block_point_density=bp,full_point_density=point)
        np.savez_compressed(out/'bootstrap'/f'draw_metrics_L{length}.npz',**scalar_draws)
        for pair_index,(i,j) in enumerate(pair_order):
            for scope_index,(scope,s,e) in enumerate(scope_list):
                curve_common=sv[i,scope_index]&sv[j,scope_index]
                for metric in ['beta','signed_area','main_mass','tail_mass','tail_main_ratio']:
                    av=scalar_draws[names[i]+'__'+metric][scope_index];bv=scalar_draws[names[j]+'__'+metric][scope_index]
                    ap=observed[names[i],scope][metric][0];bpoint=observed[names[j],scope][metric][0]
                    difference,good,(lo,hi,n,f)=paired_difference(av,bv,c['minimum_joint_valid_fraction'],np.isfinite(ap)&np.isfinite(bpoint))
                    paired_rows.append({'pair_index':pair_index,'group_a':names[i],'group_b':names[j],
                        'comparison':names[i]+' - '+names[j],'scope':scope,'block_frames':length,'metric':metric,
                        'point_difference':ap-bpoint,'ci_low':lo,'ci_high':hi,'common_valid_count':n,'common_valid_fraction':f,
                        'curve_common_valid_count':int(curve_common.sum()),'curve_common_valid_fraction':float(curve_common.mean()),
                        'paired_by_original_replicate_id':True})
        inference=centered_max_test(point,sd[:,0],sv[:,0],c['alpha'],c['minimum_joint_valid_fraction'])
        control_inference=centered_max_test(bp,sd[:,0],sv[:,0],c['alpha'],c['minimum_joint_valid_fraction'])
        inferences[length]=inference
        if inference['available']:
            means=inference['bootstrap_mean'];offset=means-point;boundary=bp-point
            pair_offsets=np.stack([offset[i]-offset[j] for i,j in pair_order])
            pair_boundary=np.stack([boundary[i]-boundary[j] for i,j in pair_order])
            ci=inference['mc_exceedance_ci']
            row={'block_frames':length,'primary':length==25,'status':inference['status'],
                'statistic':inference['statistic'],'p_global':inference['p_global'],'critical':inference['critical'],
                'critical_rank_1based':inference['critical_rank_1based'],'joint_valid_count':inference['joint_valid_count'],
                'joint_valid_fraction':inference['joint_valid_fraction'],'p_resolution':inference['p_resolution'],
                'exceedances':inference['exceedances'],'mc_ci_low':ci[0],'mc_ci_high':ci[1],
                'boundary_restricted_p':control_inference.get('p_global',np.nan),
                'max_pair_centering_offset':float(np.abs(pair_offsets).max()),
                'max_pair_boundary_change':float(np.abs(pair_boundary).max())}
            np.savez_compressed(out/'inference'/f'null_calibration_L{length}.npz',
                null_statistics=inference['null_statistics'],joint_replicate_ids=inference['joint_replicate_ids'],
                group_order=np.array(names),bootstrap_mean=inference['bootstrap_mean'],pair_centering_offsets=pair_offsets,
                pair_boundary_changes=pair_boundary)
            for i,name in enumerate(names):
                for k,x in enumerate(mids):offset_rows.append({'group':name,'block_frames':length,'distance_mid_um':x,
                    'bootstrap_mean_minus_full_density':offset[i,k],'bootstrap_mean_minus_block_control':means[i,k]-bp[i,k],
                    'block_control_minus_full_density':boundary[i,k]})
        else:
            row={'block_frames':length,'primary':length==25,'status':inference['status'],'p_global':np.nan,
                'joint_valid_count':inference['joint_valid_count'],'joint_valid_fraction':inference['joint_valid_fraction'],
                'statistic':np.nan,'critical':np.nan,'p_resolution':np.nan,'exceedances':0,
                'mc_ci_low':np.nan,'mc_ci_high':np.nan,'boundary_restricted_p':np.nan,
                'max_pair_centering_offset':np.nan,'max_pair_boundary_change':np.nan}
        test_rows.append(row)
        dump(out/'inference'/f'omnibus_L{length}.json',{k:v for k,v in inference.items() if k not in ['lower','upper','differences','bootstrap_mean','null_statistics']})
    primary=inferences[25];posthoc=bool(primary['available'] and primary['p_global']<=c['alpha'])
    for k,(i,j) in enumerate(pair_order):
        difference=point[i]-point[j]
        row={'pair_index':k,'group_a':names[i],'group_b':names[j],'comparison':names[i]+' - '+names[j],
            'posthoc_gate_passed':posthoc,'adjusted_p':primary['adjusted_p'][k] if posthoc else np.nan,
            'family_reject':bool(primary['adjusted_p'][k]<=c['alpha']) if posthoc else False,
            'pair_common_valid_count':int(primary['pair_valid_counts'][k]),
            'pair_common_valid_fraction':float(primary['pair_valid_fractions'][k]),
            'actual_family_valid_count':primary['joint_valid_count'],'actual_family_valid_fraction':primary['joint_valid_fraction'],
            **effect_sizes(difference,mids,width)}
        pair_results.append(row)
        band={'distance_mid_um':mids,'difference_density_um_inv':difference,
              'simultaneous_lower':primary['lower'][k] if posthoc else np.full(300,np.nan),
              'simultaneous_upper':primary['upper'][k] if posthoc else np.full(300,np.nan)}
        pd.DataFrame(band).to_csv(out/'inference/bands'/f'pair{k:02d}_L25.csv',index=False)
        if posthoc:
            for lo,hi,sign in continuous_intervals(primary['lower'][k],primary['upper'][k],np.linspace(0,3,301)):
                intervals.append({'pair_index':k,'comparison':row['comparison'],'left_um':lo,'right_um':hi,
                    'direction':'group_a > group_b' if sign>0 else 'group_a < group_b','family_band_confidence':.95})
    boot=pd.DataFrame(bootstrap_rows);paired=pd.DataFrame(paired_rows);tests=pd.DataFrame(test_rows);pairs=pd.DataFrame(pair_results)
    boot.to_csv(out/'bootstrap_summary.csv',index=False);paired.to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(control_rows).to_csv(out/'block_restricted_controls.csv',index=False)
    tests.to_csv(out/'omnibus_tests.csv',index=False);pairs.to_csv(out/'ten_pair_curve_results.csv',index=False)
    pd.DataFrame(offset_rows).to_csv(out/'centering_offsets.csv',index=False)
    pd.DataFrame(intervals,columns=['pair_index','comparison','left_um','right_um','direction','family_band_confidence']).to_csv(out/'significant_distance_intervals.csv',index=False)
    roi_plots.distributions(out,c,summary,'full')
    for scope,s,e in scope_list[1:]:roi_plots.distributions(out,c,summary,scope)
    roi_plots.summaries(out,c,summary,boot)
    roi_plots.qc_diagnostics(out,c,perframe,summary)
    if primary['available']:roi_plots.simultaneous_figures(out,c,pairs,primary,primary)
    if all(d['available'] for d in inferences.values()):roi_plots.block_comparison(out,tests)
    from validate_inference import simulate
    sim=simulate(c,weights[25],progress=lambda value:print(value,flush=True));dump(out/'simulation_validation.json',sim)
    calibration_status='toy_size_check_passed' if sim['nominal_alpha_in_null_rate_interval'] else 'caution_toy_null_size_inflation'
    if calibration_status.startswith('caution'):
        print('Statistical calibration caution: null toy rejection rate exceeds nominal alpha; publish nominal numerical outputs with explicit caution, not validated FWER claims.',flush=True)
    for frame in [tests,pairs]:frame['inference_validation_status']=calibration_status
    pairs=pairs.rename(columns={'family_reject':'nominal_family_reject'})
    tests.to_csv(out/'omnibus_tests.csv',index=False);pairs.to_csv(out/'ten_pair_curve_results.csv',index=False)
    if primary['available']:roi_plots.simultaneous_figures(out,c,pairs,primary,primary)
    # Fail rather than mark complete if upstream inputs/outputs changed during this run.
    for p,h in locked.items():assert sha256_file(p)==h, p
    assert sha256_file(roi_path)==c['roi_sha256']
    assert tiff.stat().st_size==stat.st_size and tiff.stat().st_mtime_ns==stat.st_mtime_ns
    report(out,c,summary,boot,paired,tests,pairs,sim)
    dump(out/'validation.json',{'implementation_checks_passed':True,'statistical_calibration_status':calibration_status,
        'nominal_alpha_in_toy_interval':sim['nominal_alpha_in_null_rate_interval'],'preflight_tests':12,'cached_original_fields_preserved':True,
        'same_roi_all_five_groups':True,'no_refitting':True,'protected_sources_unchanged':True,
        'full_cover_reproduces_original':True,'independent_roi_pair_checks':True,'no_gap_or_block_join_pairs':True,
        'shared_weights_exactly_reused':True,'paired_replicate_id_intersections':True,
        'null_centered_before_testing':True,'negative_bins_preserved':True,'toy_null_calibration_checked':True})
    dump(out/'runtime.json',{'seconds':time.perf_counter()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})
    dump(out/'COMPLETED.json',{'source_run':str(source),'roi_frames':2500,'configurations':5,'windows':10,
        'primary_omnibus_p_nominal':primary.get('p_global'),'ten_pairs_nominal_outputs':posthoc,
        'statistical_calibration_status':calibration_status,'refitting':False})
    manifest={str(p.relative_to(out)):sha256_file(p) for p in (out/'reproducibility').rglob('*') if p.is_file()}
    dump(out/'reproducibility_manifest.json',manifest)
    print('ROI analysis complete:',out,flush=True)
    print(tests.to_string(index=False),flush=True)
    print(pairs[['comparison','adjusted_p','sup_abs_density_difference','integrated_abs_difference']].to_string(index=False),flush=True)


if __name__=='__main__':main()
