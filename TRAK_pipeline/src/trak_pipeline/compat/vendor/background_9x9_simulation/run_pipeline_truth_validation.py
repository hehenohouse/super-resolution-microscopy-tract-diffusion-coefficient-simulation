"""End-to-end native TrackMate -> unchanged Picasso pipeline on known simulation.

All candidate positions come from TrackMate. Truth is read for scoring only
after detection/fitting. No threshold, fit box, noise or QC parameters tuned.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import tifffile
import yaml

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'trackmate_python_bridge'))
from picasso_refinement import refine_candidates,model_images
from run_preview import fingerprint,roi_lib
from run_simulation import pixel_gaussian


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')


def assign_one_to_one(truth_xy,candidate_xy,radius):
    """Unique pairs, maximize cardinality before minimizing distance."""
    n=len(truth_xy);m=len(candidate_xy)
    dist=np.linalg.norm(truth_xy[:,None,:]-candidate_xy[None,:,:],axis=2)
    # 1000 is greater than all possible sums of accepted distances in this 50-point scene.
    cost=np.full((n,m+n),1000.0)
    cost[:,:m]=np.where(np.isfinite(dist)&(dist<=radius),dist,1e6)
    r,c=linear_sum_assignment(cost)
    good=c<m
    r=r[good];c=c[good]
    return r,c,dist[r,c]


def metric(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,median=None,mean=None,p5=None,p95=None)
    return dict(n=len(a),median=float(np.median(a)),mean=float(a.mean()),p5=float(np.percentile(a,5)),p95=float(np.percentile(a,95)))


def subset_metrics(t):
    out=dict(n=len(t))
    for c in ['TM_position_error_nm','MLE_position_error_nm','signal_ratio_to_expected','relative_signal_error',
              'width_x_ratio','width_y_ratio','bg_error_to_expectation_e_per_pixel','lpx_nm','lpy_nm',
              'normalized_x_error','normalized_y_error']:
        out[c]=metric(t[c])
    if len(t):
        out['signal_RMSE_e']=float(np.sqrt(np.mean((t.photons-t.truth_expected_total_electrons)**2)))
        out['position_RMSE_nm']=float(np.sqrt(np.mean(t.MLE_position_error_nm**2)))
    return out


def plots(out,movie,truth,table,matched,theta,patches,used,cfg):
    image=(movie[0].astype(float)-100)*.46
    low,high=np.percentile(image,[1,99.9])
    fig,axs=plt.subplots(1,3,figsize=(16,7))
    fig.subplots_adjust(top=.76,bottom=.1,left=.045,right=.985,wspace=.22)
    for a in axs:
        a.imshow(image,cmap='gray',vmin=low,vmax=high,origin='upper');a.grid(False)
        a.set(xlabel='x (pixel)',ylabel='y (pixel)',xlim=(-.5,299.5),ylim=(299.5,-.5))
    for t in truth.itertuples():axs[0].add_patch(Circle((t.x_px,t.y_px),t.hwhm_px,fill=False,color='lime',lw=.6))
    axs[0].set_title('Observed simulated image\n50 true emitters: green HWHM circles')
    for t in table.itertuples():
        col='cyan' if t.truth_matched_from_TrackMate else '#FF9E42'
        axs[1].add_patch(Circle((t.x_px,t.y_px),3,fill=False,color=col,lw=.5,alpha=.9 if col=='cyan' else .6))
    axs[1].set_title(f'TrackMate: {len(table)} candidates\ncyan=one-to-one truth match; orange=unmatched')
    fitmatched=table.truth_matched_from_TrackMate&table.nominal_precision_valid.eq(True)
    for i,t in table.iterrows():
        if pd.notna(t.nominal_precision_valid) and bool(t.nominal_precision_valid) and np.isfinite([t.fit_x_px,t.fit_y_px]).all():
            axs[2].plot(t.fit_x_px,t.fit_y_px,'+',color='lime' if t.truth_matched_from_TrackMate else '#FF9E42',ms=4,alpha=.85 if t.truth_matched_from_TrackMate else .5)
    axs[2].set_title(f'Picasso 7x7 MLE\n{int(fitmatched.sum())} truth-matched fits with finite model bounds')
    fig.suptitle('Existing pipeline: native TrackMate 7.14.0, radius 0.28 um / threshold 6.5 / median + subpixel\nPicasso 0.11.3, free bg, no extra detection gate; 117 nm/pixel',fontsize=13,y=.97)
    fig.savefig(out/'simulation_trackmate_picasso_overlay.png',dpi=130);plt.close(fig)
    good=matched[matched.nominal_precision_valid.eq(True)]
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    if len(good):
        col=np.where(good.neighbor_contaminated,'#DC752D','#197AB1')
        a=axs[0,0];a.scatter(good.truth_expected_total_electrons,good.photons,c=col,s=25)
        lim=max(1600,float(np.nanmax(np.r_[good.truth_expected_total_electrons,good.photons]))*1.03)
        a.plot([0,lim],[0,lim],'k--');a.set(xlabel='True expected total e-',ylabel='Fitted total e-',xlim=(0,lim),ylim=(0,lim),title='Blue: isolated; orange: neighbor >1%')
        a=axs[0,1]
        for f,label in [('sx','sx'),('sy','sy')]:a.scatter(good.truth_gaussian_sigma_px,good[f],s=22,label=label)
        a.plot([.4,3],[.4,3],'k--');a.set(xlabel='True Gaussian sigma (px)',ylabel='Fitted sigma (px)',title='Source: pixel-integrated PSF\nModel: pixel-center Gaussian');a.legend()
        a=axs[0,2];a.scatter(good.truth_expected_total_electrons,good.bg,c=col,s=25);a.axhline(100,color='k',ls='--');a.set(xlabel='True expected total e-',ylabel='Fitted bg (e-/px)',title='True background expectation = 100')
        a=axs[1,0];a.scatter(good.TM_position_error_nm,good.MLE_position_error_nm,c=col,s=25)
        lim=max(100,float(np.nanmax(np.r_[good.TM_position_error_nm,good.MLE_position_error_nm]))*1.05)
        a.plot([0,lim],[0,lim],'k--');a.set(xlabel='TrackMate distance to truth (nm)',ylabel='MLE distance to truth (nm)',title='Below line: MLE center closer to truth')
        a=axs[1,1];a.hist(good.relative_signal_error*100,bins=15);a.axvline(0,color='k',ls='--');a.set(xlabel='Signal relative error (%)',ylabel='Matched fits')
        a=axs[1,2]
        a.scatter(good.lpx_nm,np.abs(good.fit_x_nm-good.truth_x_nm),s=22,label='x')
        a.scatter(good.lpy_nm,np.abs(good.fit_y_nm-good.truth_y_nm),s=22,label='y')
        a.set(xlabel='Current model CRLB SD (nm)',ylabel='Actual absolute axis error (nm)',title='Actual error vs model bound\nOne realization: coverage not validated');a.legend()
    fig.suptitle(f'Known truth vs unchanged current pipeline | {len(good)} computationally admissible matched fits\nSource: Poisson signal + Gaussian bg SD 20 + read SD 10; current MLE read SD remains 1.6',fontsize=13)
    fig.savefig(out/'truth_recovery_comparison.png',dpi=130);plt.close(fig)
    # Examples are chosen by deterministic quantiles, not best-looking fits.
    choices=[]
    if len(good):
        for label,target in [('median signal error',np.median(np.abs(good.relative_signal_error))),('large signal error',np.quantile(np.abs(good.relative_signal_error),.9))]:
            i=int(good.index[np.argmin(np.abs(np.abs(good.relative_signal_error.to_numpy())-target))]);choices.append((label,i))
    unmatched=table[~table.truth_matched_from_TrackMate & table.fit_attempted]
    if len(unmatched): choices.append(('unmatched candidate',int(unmatched.index[len(unmatched)//2])))
    fig,axs=plt.subplots(max(1,len(choices)),3,figsize=(12,4*max(1,len(choices))),squeeze=False,layout='constrained')
    rows=[]
    for j,(label,i) in enumerate(choices):
        t=table.loc[i];k=int(np.flatnonzero(used==i)[0]);data=(patches[k]-100)*.46;pred=model_images(theta[k:k+1],7)[0]
        vmin=float(data.min());vmax=float(data.max())
        for q,(arr,title) in enumerate([(data,'Raw 7x7, e-/px'),(pred,'Fitted Gaussian, same scale'),(data-pred,'Residual, e-/px')]):
            a=axs[j,q];a.grid(False)
            if q<2:im=a.imshow(arr,cmap='gray',vmin=vmin,vmax=vmax)
            else:
                limit=max(1,float(np.abs(arr).max()));im=a.imshow(arr,cmap='coolwarm',vmin=-limit,vmax=limit)
            a.set_title(title);fig.colorbar(im,ax=a,shrink=.7)
            if q==0:a.set_ylabel(f'{label}\nTM spot {int(t.spot_id)}\nNfit={t.photons:.0f}, bg={t.bg:.1f}')
        rows.append(dict(category=label,candidate_spot_id=int(t.spot_id),truth_id=None if pd.isna(t.truth_id) else int(t.truth_id)))
    fig.suptitle('Representative pipeline fits: original pixels, model and residual\nAll candidate and failure rows retained; no truth coordinates used to fit',fontsize=13)
    fig.savefig(out/'representative_pipeline_fits.png',dpi=130);plt.close(fig);dump(out/'representative_examples.json',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('pipeline_truth_config.yaml')));args=parser.parse_args()
    cfgpath=Path(args.config).resolve();cfg=yaml.safe_load(cfgpath.read_text());started=time.perf_counter()
    src=Path(cfg['source_simulation']);out=Path(cfg['output_root'])/('trackmate_picasso_truth_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);print('OUTPUT:',out,flush=True)
    protected=[src/'scene_components.npz',src/'truth_spots.csv',src/'config.yaml',ROOT/'trackmate_python_bridge/run_detection.py',ROOT/'trackmate_python_bridge/picasso_refinement.py',ROOT/'picasso_pilot/config_T1_C2.yaml']
    before=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_before.json',before)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    snap=out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),cfgpath,ROOT/'trackmate_python_bridge/run_detection.py',ROOT/'trackmate_python_bridge/picasso_refinement.py',ROOT/'picasso_pilot/pilot_core.py',ROOT/'background_9x9_simulation/run_simulation.py']:
        shutil.copyfile(p,snap/p.name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--config',str(cfgpath)])+'\n')
    with np.load(src/'scene_components.npz') as z:arrays={k:z[k] for k in z.files}
    observed=arrays['observed_electrons'];assert observed.shape==(300,300)
    camera=cfg['camera'];scale=camera['sensitivity_e_per_adu']/camera['em_gain']
    floatadu=camera['baseline_adu']+observed/scale
    assert floatadu.min()>=0 and floatadu.max()<65535
    adu=np.rint(floatadu).astype(np.uint16);movie=adu[None,:,:]
    recovered=(adu.astype(float)-camera['baseline_adu'])*scale
    np.testing.assert_allclose(recovered,observed,rtol=0,atol=scale/2+1e-10)
    tifffile.imwrite(out/'simulation_input_uint16_ADU.tiff',adu,metadata={'axes':'YX','pixel_size_nm':117,'source_units':'converted from known electrons'},photometric='minisblack')
    np.save(out/'original_observed_electrons.npy',observed)
    tm=cfg['trackmate'];python=(ROOT/tm['detector_python']).absolute()
    command=[str(python),'-B','-u',str(ROOT/'trackmate_python_bridge/run_detection.py'),'--input',str(out/'simulation_input_uint16_ADU.tiff'),
        '--start','0','--count','1','--pixel-size-um',str(cfg['pixel_size_nm']/1000),'--frame-interval-s',str(cfg['frame_interval_s']),
        '--radius-um',str(tm['radius_um']),'--threshold',str(tm['threshold']),'--fiji',tm['fiji'],'--java-home',tm['java_home'],'--output-root',str(out/'native_trackmate')]
    (out/'trackmate_command.txt').write_text(shlex.join(command)+'\n')
    print('Running native TrackMate on converted original simulation, no truth seeds...',flush=True)
    with (out/'trackmate.log').open('w') as log:subprocess.run(command,stdout=log,stderr=log,check=True)
    reports=list((out/'native_trackmate').glob('*/report.json'));assert len(reports)==1
    report=json.loads(reports[0].read_text());assert report['versions']['trackmate']==tm['version'] and not report['tracking_executed']
    native=reports[0].parent
    np.testing.assert_array_equal(np.load(native/'input_frames.npy'),movie)
    table=pd.read_csv(native/'spots.csv',float_precision='round_trip');assert table.spot_id.is_unique
    table['roi_inside']=1;table['roi_status']='full_synthetic_field';table['radius_setting_px']=tm['radius_um']/.117
    print(f'TrackMate returned {len(table)} candidates; fitting all with existing Picasso function...',flush=True)
    result,theta,patches,used=refine_candidates(table,np.array([0]),movie,cfg['picasso']|{'pixel_size_nm':cfg['pixel_size_nm']},camera)
    table=pd.concat([table,result.drop(columns=['spot_id','frame'])],axis=1)
    assert len(table)==report['total_spots'];assert np.array_equal(table.spot_id.to_numpy(),pd.read_csv(native/'spots.csv').spot_id.to_numpy())
    table['uncertainty_model_matches_generator']=False
    table['precision_interpretation']='current_pipeline_CRLB_noise_mismatched_to_generator;not_validated_coverage'
    # Only now read emitter truth, after native detection and pipeline fitting.
    truth=pd.read_csv(src/'truth_spots.csv');assert len(truth)==50
    r,c,d=assign_one_to_one(truth[['x_px','y_px']].to_numpy(),table[['x_px','y_px']].to_numpy(),cfg['matching']['maximum_distance_px'])
    table['truth_matched_from_TrackMate']=False;table['truth_id']=pd.Series([pd.NA]*len(table),dtype='Int64')
    table.loc[c,'truth_matched_from_TrackMate']=True;table.loc[c,'truth_id']=truth.iloc[r].spot_id.to_numpy()
    # Match membership is fixed by TrackMate centers, not revised after MLE.
    lookup=truth.add_prefix('truth_').set_index('truth_spot_id')
    table=table.join(lookup,on='truth_id')
    table['TM_position_error_nm']=np.hypot(table.x_px-table.truth_x_px,table.y_px-table.truth_y_px)*117
    table['MLE_position_error_nm']=np.hypot(table.fit_x_px-table.truth_x_px,table.fit_y_px-table.truth_y_px)*117
    table['signal_ratio_to_expected']=table.photons/table.truth_expected_total_electrons
    table['signal_ratio_to_realized']=table.photons/table.truth_realized_total_electrons
    table['relative_signal_error']=table.signal_ratio_to_expected-1
    table['width_x_ratio']=table.sx/table.truth_gaussian_sigma_px
    table['width_y_ratio']=table.sy/table.truth_gaussian_sigma_px
    table['bg_error_to_expectation_e_per_pixel']=table.bg-100
    table['normalized_x_error']=(table.fit_x_px-table.truth_x_px)/table.lpx
    table['normalized_y_error']=(table.fit_y_px-table.truth_y_px)/table.lpy
    truth['TM_matched']=False;truth.loc[r,'TM_matched']=True
    truth['matched_candidate_id']=pd.Series([pd.NA]*len(truth),dtype='Int64');truth.loc[r,'matched_candidate_id']=table.iloc[c].spot_id.to_numpy()
    expected_fields=[]
    for t in truth.itertuples():expected_fields.append(t.expected_total_electrons*pixel_gaussian((300,300),t.x_px,t.y_px,t.gaussian_sigma_px)[0])
    expected_fields=np.asarray(expected_fields)
    np.testing.assert_allclose(expected_fields.sum(axis=0),arrays['signal_expected_electrons'],rtol=1e-12,atol=1e-10)
    table['neighbor_expected_fraction_in_fit_box']=np.nan;table['true_local_background_mean_e_per_pixel']=np.nan
    table['true_own_expected_signal_in_fit_box']=np.nan
    for i in c:
        if not table.loc[i,'fit_support_valid']:continue
        x=int(table.loc[i,'fit_anchor_x_px']);y=int(table.loc[i,'fit_anchor_y_px']);ss=(slice(y-3,y+4),slice(x-3,x+4))
        k=int(np.flatnonzero(truth.spot_id.to_numpy()==table.loc[i,'truth_id'])[0])
        own=float(expected_fields[k][ss].sum());neighbor=float((expected_fields.sum(axis=0)[ss]-expected_fields[k][ss]).sum())
        table.loc[i,'neighbor_expected_fraction_in_fit_box']=neighbor/own
        table.loc[i,'true_local_background_mean_e_per_pixel']=arrays['background_electrons'][ss].mean()
        table.loc[i,'true_own_expected_signal_in_fit_box']=own
    table['neighbor_contaminated']=table.neighbor_expected_fraction_in_fit_box>0.01
    table['bg_error_to_realized_local_mean_e_per_pixel']=table.bg-table.true_local_background_mean_e_per_pixel
    table.to_csv(out/'all_trackmate_picasso_candidates.csv',index=False)
    truth.to_csv(out/'all_50_truth_match_status.csv',index=False)
    table[table.truth_matched_from_TrackMate].to_csv(out/'matched_truth_comparison.csv',index=False)
    np.savez_compressed(out/'picasso_backend_audit.npz',fitted_candidate_index=used,spot_id=table.iloc[used].spot_id.to_numpy(),
        canonical_theta=theta,original_ADU_patches=patches)
    matched=table[table.truth_matched_from_TrackMate].copy()
    good=matched[matched.nominal_precision_valid.eq(True)]
    clean=matched[matched.qc_unflagged_model_fit.eq(True)]
    summary=dict(output_directory=str(out),n_true=50,n_TM_candidates=len(table),n_matched=len(r),n_missed=50-len(r),
        n_unmatched_candidates=len(table)-len(r),matching_radius_px=cfg['matching']['maximum_distance_px'],
        detection_recall=len(r)/50,detection_match_fraction=len(r)/len(table) if len(table) else None,
        n_all_admissible=int(table.nominal_precision_valid.sum()),n_matched_admissible=len(good),n_matched_unflagged=len(clean),
        n_all_unflagged=int(table.qc_unflagged_model_fit.eq(True).sum()),
        n_unmatched_unflagged=int((~table.truth_matched_from_TrackMate & table.qc_unflagged_model_fit.eq(True)).sum()),
        source_expected_signal=metric(truth.expected_total_electrons),matched_admissible=subset_metrics(good),
        matched_unflagged=subset_metrics(clean),matched_isolated_admissible=subset_metrics(good[~good.neighbor_contaminated]),
        noise=dict(simulated_background_mean_e=100,simulated_background_SD_e=20,simulated_read_SD_e=10,
            simulated_background_plus_read_variance_e2=500,
            pipeline_background_plus_read_variance_at_true_bg_e2=100+camera['read_noise_e_rms']**2,
            statistical_model_match=False,
            note='Simulation has Gaussian background and no background Poisson; current shifted-Poisson model assumes mean-dependent shot noise plus read variance 2.56.'),
        quantization_max_abs_e=float(np.abs(recovered-observed).max()),truth_used_for_detection_or_fitting=False,
        no_parameter_tuning=True,elapsed_s=time.perf_counter()-started)
    dump(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['n_TM_candidates','n_matched','n_missed','n_unmatched_candidates','n_matched_admissible','n_matched_unflagged']},indent=2),flush=True)
    plots(out,movie,truth,table,matched,theta,patches,used,cfg)
    # Matching, unit conversion, row preservation and input integrity checks.
    assert len(np.unique(r))==len(r) and len(np.unique(c))==len(c) and (d<=cfg['matching']['maximum_distance_px']).all()
    ar,ac,ad=assign_one_to_one(np.array([[0.,0.],[1.,0.]]),np.array([[.1,0.]]),1.5)
    assert len(ar)==len(ac)==1
    np.testing.assert_allclose(table.x_um,table.x_px*.117,rtol=0,atol=1e-12)
    np.testing.assert_allclose(table.fit_x_nm,table.fit_x_px*117,equal_nan=True,rtol=1e-6)
    np.testing.assert_array_equal(table.loc[~table.fit_attempted,'fit_support_valid'],False)
    after=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_after.json',after);assert before==after
    versions=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['picassosr','numpy','scipy','pandas','tifffile','matplotlib','numba','PyYAML']},native_TrackMate=report['versions'])
    dump(out/'software_versions.json',versions)
    dump(out/'validation.json',dict(all_passed=True,converted_pixels_roundtrip_max_error_e=float(np.abs(recovered-observed).max()),
        TrackMate_native_input_pixels_exact=True,all_detection_rows_preserved=True,one_to_one_matching_unique=True,
        truth_loaded_for_scoring_after_fit=True,units_checked=True,original_simulation_and_pipeline_sources_unchanged=True,
        caveat='These implementation checks do not validate the mismatched noise model or real-data calibration.'))
    report_text=f'''# 原模拟图：现有 TrackMate + Picasso pipeline 真值验证\n\n使用原始 300x300、50 个光点的同一模拟，没有重新生成场景。TrackMate 原生 {tm['version']}，radius=0.28 µm、threshold=6.5、3x3 median、亚像素定位；Picasso {cfg['picasso']['version']}，7x7、二维椭圆 Gaussian MLE、自由背景，tolerance=1e-4，1000 次迭代上限。\n\nTrackMate 独立生成全部候选，Picasso 只以其浮点位置初始化；真值在检测和拟合结束后才读取作评估。全模拟视野参与，不套真实细胞 ROI，不做轨迹或 TARDIS，也不按真值筛候选或调阈值。\n\n模拟光电子转换为 uint16 ADU：round(100 + e/0.46)。100 在此是添加的电子 offset；模拟光学背景的 100 e/px 换成 raw 背景约 317.39 ADU。再按现有公式换回，最大量化误差≤0.23 e。没有把模拟光学背景扣作电子 offset，也没有把已是电子单位的图再次乘0.46。\n\n逐个真值 N 使用实际抽取的 expected_total_electrons；800 是50点总体分布均值，单点并不全为800。realized_total_electrons 另列。宽度由HWHM转换为σ；源PSF是像素积分Gaussian，原Picasso模型在像素中心求值，存在像素积分模型差异。\n\n真值匹配半径预先固定1.5px=175.5nm，使用最大匹配数、最小距离的一对一分配；身份以原TrackMate中心确定，不按MLE结果重新挑配对。未匹配候选、漏检及失败拟合全部保留，不把未匹配候选在真实数据中类推为假分子。逐点同时标记邻居在7x7内的期望信号比例，>1%作模型污染诊断，不改变候选/QC。\n\n检测出 {len(table)} 候选，匹配 {len(r)}/50，未匹配候选 {len(table)-len(r)}，漏检 {50-len(r)}。匹配中 {len(good)} 通过参数与有限模型CRLB数值检查，{len(clean)} 通过原无复核标记QC。量化结果见 summary.json；all_trackmate_picasso_candidates.csv 保留全部行。\n\n**噪声不匹配**：保留现有拟合读噪声1.6 e RMS，源模拟读噪声10 e RMS，加Gaussian背景SD20 e，且背景未抽Poisson。真实背景+读噪声方差500 e²/px，当前模型在b=100时约102.56。因此该轮CRLB不能被视为符合生成器的误差下界或已校准覆盖率；一次图像的50个点也不能验证每点置信覆盖。信号和位置恢复可检查，但噪声匹配和误差覆盖须另做验证，不据此推断真实电影已标定。\n\n本轮严格不调检测/拟合/QC设置。原模拟与原pipeline代码指纹前后不变。\n'''
    (out/'REPORT.md').write_text(report_text)
    (out/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    dump(out/'COMPLETED.json',dict(status='complete',output_directory=str(out),elapsed_s=time.perf_counter()-started))
    print('COMPLETE:',out,flush=True)


if __name__=='__main__':main()
