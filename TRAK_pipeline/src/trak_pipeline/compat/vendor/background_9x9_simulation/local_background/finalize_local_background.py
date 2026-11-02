"""Independent output/mask checks and fixed-subset figures for the trial."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd,tifffile,yaml
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_local_background import rings,metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();out=a.output;c=yaml.safe_load((out/'config.yaml').read_text())
    fingerprints=json.loads((out/'input_fingerprints.json').read_text());assert all(sha(p)==v for p,v in fingerprints.items())
    source=Path(c['source_run']);prev=Path(c['free_fit_run']);baseline=pd.read_csv(source/'all_trackmate_picasso_candidates.csv')
    free=pd.read_csv(prev/'box7_all_candidates.csv');old11=pd.read_csv(prev/'box11_all_candidates.csv')
    original=tifffile.imread(source/'simulation_input_uint16_ADU.tiff');movie=(original.astype(float)-100)*.46
    features=baseline[['spot_id','frame','time_s','x_px','y_px','quality_log_response']].copy();features.insert(0,'candidate_index',np.arange(len(features)))
    geometry={}
    for label,threshold in [('primary',25.),('all_candidates',6.5)]:
        bg=pd.read_csv(out/f'background_{label}_all_candidates.csv')
        with np.load(out/f'annulus_pixels_{label}.npz') as z:
            offsets=z['offsets'];flat=z['xy_pixels'];np.testing.assert_array_equal(z['spot_id'],baseline.spot_id)
        recomputed,masks=rings(features,movie,c,threshold)
        np.testing.assert_array_equal(offsets,np.concatenate([[0],np.cumsum([len(p) for p in masks])]))
        np.testing.assert_array_equal(flat,np.concatenate(masks))
        for key in ['mean_e','median_e','q25_e','mean_nominal_se_e','median_nominal_se_e']:
            np.testing.assert_allclose(bg[key],recomputed[key],rtol=1e-12,atol=1e-12,equal_nan=True)
        tree=cKDTree(features.loc[features.quality_log_response>=threshold,['x_px','y_px']])
        for i,pixels in enumerate(masks):
            r=features.iloc[i];distance=np.hypot(pixels[:,0]-r.x_px,pixels[:,1]-r.y_px)
            assert ((distance>6)&(distance<=12)).all()
            assert (tree.query(pixels)[0]>4).all()
            ax,ay=np.floor([r.x_px+.5,r.y_px+.5]).astype(int)
            assert not ((np.abs(pixels[:,0]-ax)<=3)&(np.abs(pixels[:,1]-ay)<=3)).any()
        geometry[label]=(bg,masks)
    methods=['free_7x7']+c['methods'];tables={'free_7x7':free}
    for method in c['methods']:
        t=pd.read_csv(out/f'{method}_all_candidates.csv');assert len(t)==1653
        for key in ['spot_id','frame','time_s','x_px','y_px']:
            np.testing.assert_array_equal(t[key],baseline[key])
        assert not (t.active_bound&t.fit_usable_interior).any()
        assert t.loc[t.active_bound,'N_se_e'].isna().all()
        assert not t.coverage_validated.any()
        assert (t.loc[t.fit_attempted,'bg_e_per_pixel']>=0).all()
        if method=='mean_constraint_all_candidates':
            no_bg=~t.background_supported
            assert not t.loc[no_bg,'fit_attempted'].any() and t.loc[no_bg,'photons_e'].isna().all()
        if method.endswith('_init'):
            key={'mean_init':'mean_e','median_init':'median_e','q25_init':'q25_e'}[method]
            selected=t.fit_attempted.to_numpy()
            np.testing.assert_allclose(t.loc[selected,'background_initial_e_per_pixel'],geometry['primary'][0].loc[selected,key],rtol=1e-12)
        tables[method]=t
    fixed=baseline.truth_matched_from_TrackMate&baseline.qc_unflagged_model_fit&free.fit_usable_interior&old11.fit_usable_interior
    assert fixed.sum()==24
    primary=['free_7x7','mean_init','median_init','q25_init','mean_constraint','median_constraint']
    assert all(tables[k].loc[fixed,'fit_usable_interior'].all() for k in primary)
    fixed_summary=[]
    for method in primary+['free_11x11_reference']:
        t=old11 if method=='free_11x11_reference' else tables[method]
        m=metrics(t[fixed]);m['method']=method
        q=t[fixed];errors=np.abs(q.photons_e/q.truth_expected_total_electrons-1)*100
        m['p90_absolute_relative_error_pct']=float(np.percentile(errors,90))
        fixed_summary.append(m)
    pd.DataFrame(fixed_summary).to_csv(out/'same24_primary_comparison.csv',index=False)
    # A neighbor-core fixture verifies bright-core exclusion without truth-based selection.
    fixture=np.full((45,45),100.)
    yy,xx=np.mgrid[:45,:45];fixture[np.hypot(xx-30,yy-22)<=4]=1000.
    candidates=pd.DataFrame(dict(candidate_index=[0,1],spot_id=[1,2],frame=[0,0],time_s=[0.,0.],x_px=[22.,30.],y_px=[22.,22.],quality_log_response=[40.,40.]))
    fstats,fmasks=rings(candidates,fixture,c,25.)
    assert fstats.background_supported.all() and np.all(fstats.mean_e==100.)
    # Preserve original fields alongside trial values in a separate complete audit table.
    audit=baseline.copy()
    for method in c['methods']:
        t=tables[method]
        columns=['background_supported','fit_attempted','fit_success','active_bound','fit_usable_interior',
            'fit_x_px','fit_y_px','photons_e','bg_e_per_pixel','sx_px','sy_px','N_se_e','bg_se_e_per_pixel','x_se_nm','y_se_nm','uncertainty_status']
        for key in columns:audit[f'trial_{method}__{key}']=t[key].to_numpy()
    audit.to_csv(out/'all_original_candidates_with_trial_fields.csv',index=False)
    labels={'free_7x7':'Free background, 7x7','mean_init':'Mean: initial only','median_init':'Median: initial only','q25_init':'Q25: initial only',
        'mean_constraint':'Mean + uncertainty','median_constraint':'Median + uncertainty','free_11x11_reference':'Free background, 11x11'}
    fig,axs=plt.subplots(1,3,figsize=(16,5.4),layout='constrained')
    bg=geometry['primary'][0];matched=baseline.truth_matched_from_TrackMate
    for key,label in [('mean_e','Annulus mean'),('median_e','Annulus median'),('q25_e','Annulus Q25')]:
        axs[0].scatter(baseline.loc[matched,'truth_id'],bg.loc[matched,key],s=18,label=label)
    axs[0].axhline(100,color='k',ls='--',label='True population mean');axs[0].set(xlabel='Original matched emitter ID',ylabel='Background (e- / pixel)',title='Annulus: 6 < r <= 12 px');axs[0].legend(fontsize=8)
    for method in ['free_7x7','mean_constraint','median_constraint']:
        t=tables[method].loc[fixed];axs[1].scatter(t.truth_expected_total_electrons,t.photons_e,s=24,alpha=.8,label=labels[method])
    axs[1].plot([0,1800],[0,1800],'k--');axs[1].set(xlabel='True expected total e-',ylabel='Fitted total e-',title='Same preceding 24 candidates');axs[1].legend(fontsize=8)
    summary=pd.DataFrame(fixed_summary);colors=['#7b8c99' if k.endswith('init') or k.startswith('free') else '#368fb0' for k in summary.method]
    axs[2].barh(np.arange(len(summary)),summary.median_abs_relative_error_pct,color=colors)
    axs[2].set_yticks(np.arange(len(summary)),labels=[labels[k] for k in summary.method],fontsize=8);axs[2].invert_yaxis()
    for j,e in enumerate(summary.median_abs_relative_error_pct):axs[2].text(e+.3,j,f'{e:.1f}%',va='center',fontsize=9)
    axs[2].set(xlabel='Median absolute relative N error (%)',title='Identical 24-point comparison',xlim=(0,30))
    fig.suptitle('Local-background trial on the saved 50-emitter simulation\nAll 1653 targets retained; neighbor exclusions use cached quality >= 25; error coverage NOT validated',fontsize=12)
    fig.savefig(out/'primary_methods_same24.png',dpi=130);plt.close(fig)
    chosen=[]
    for truth_id in [0,2,4]:
        indices=np.flatnonzero(baseline.truth_id.eq(truth_id).to_numpy())
        if len(indices):chosen.append(int(indices[0]))
    fig,axs=plt.subplots(len(chosen),2,figsize=(10.8,3.7*len(chosen)),layout='constrained',squeeze=False)
    global_lo,global_hi=np.percentile(movie,[1,99.9])
    for row,i in enumerate(chosen):
        r=baseline.iloc[i];ax,ay=np.floor([r.x_px+.5,r.y_px+.5]).astype(int);ox,oy=ax-16,ay-16
        raw=movie[max(0,oy):min(300,oy+33),max(0,ox):min(300,ox+33)];left=max(0,ox);top=max(0,oy)
        a=axs[row,0];a.imshow(raw,cmap='gray',vmin=global_lo,vmax=global_hi,origin='upper',interpolation='nearest')
        p=geometry['primary'][1][i];a.scatter(p[:,0]-left,p[:,1]-top,s=3,c='#5ad47c',alpha=.45,label='Retained background pixels')
        a.add_patch(Circle((r.x_px-left,r.y_px-top),6,fill=False,color='#efc84a',lw=1));a.add_patch(Circle((r.x_px-left,r.y_px-top),12,fill=False,color='#efc84a',lw=1))
        a.add_patch(Rectangle((ax-3.5-left,ay-3.5-top),7,7,fill=False,color='#f69b35',lw=1.2));a.plot(r.x_px-left,r.y_px-top,'+',color='#44e3f0')
        a.set(title=f'True emitter {int(r.truth_id)}; {len(p)} background pixels',xlabel='Local X (pixels)',ylabel='Local Y (pixels)')
        q=geometry['primary'][0].iloc[i];v=movie[p[:,1],p[:,0]];b=axs[row,1]
        b.hist(v,bins=22,color='#a4afb8');b.axvline(q.mean_e,color='#2d80aa',label=f'Mean {q.mean_e:.2f} +/- {q.mean_nominal_se_e:.2f}')
        b.axvline(q.median_e,color='#db8b35',label=f'Median {q.median_e:.2f}');b.axvline(q.q25_e,color='#5eac70',ls='--',label=f'Q25 {q.q25_e:.2f}')
        new=tables['mean_constraint'].iloc[i];old=free.iloc[i]
        b.set(title=f'True N {r.truth_expected_total_electrons:.0f}; free {old.photons_e:.0f}; constrained {new.photons_e:.0f}',xlabel='Background pixel value (e-)',ylabel='Pixels');b.legend(fontsize=8)
    fig.suptitle('Background support examples: orange = 7x7 fit; green = retained ring pixels\nMean SE assumes known synthetic noise; masking-selection effects are not included',fontsize=12)
    fig.savefig(out/'background_support_examples.png',dpi=130);plt.close(fig)
    validations=dict(all_passed=True,original_and_previous_hashes_unchanged=True,all_1653_original_rows_preserved_with_original_fields=True,
        masks_recomputed_from_coordinates_and_quality_only=True,annulus_disjoint_from_central_pixels=True,
        neighbor_core_exclusion_verified=True,mean_median_q25_and_SE_recomputed=True,
        unsupported_background_not_fitted_or_silently_filled=True,active_bound_covariance_not_reported=True,
        primary_comparison_same_previous_24_points=True,bright_neighbor_fixture_excluded=True,
        observed_Hessian_and_pixel_selection_coverage_not_validated=True)
    dump(out/'independent_validation.json',validations)
    shutil.copyfile(__file__,out/'source_snapshot'/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('run.log'),out/'run.log')
    dump(out/'figure_scopes.json',dict(primary_methods_same24='Identical preceding 24-point subset; primary methods all valid. Includes unchanged 11x11 reference.',
        local_background_comparison='19-point intersection additionally includes all-candidate-exclusion sensitivity.',
        background_support_examples='Representative true IDs 0,2,4 selected for auditing after fitting; never used to select pixels or fit.'))
    print(json.dumps(dict(validation=validations,same24=fixed_summary),indent=2))

if __name__=='__main__':main()
