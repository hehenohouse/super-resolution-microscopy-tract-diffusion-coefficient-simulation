"""Annular background experiment on cached TrackMate detections, synthetic only.

All target candidates remain. No native TrackMate/Picasso source is edited.
Background masks are determined from candidate coordinates/quality, never truth.
"""
import argparse,concurrent.futures,hashlib,json,shlex,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,tifffile,yaml
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'background_9x9_simulation'))
import calibrated_photometry as base

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rings(candidates,movie,c,threshold):
    centers=candidates.loc[candidates.quality_log_response>=threshold,['x_px','y_px']].to_numpy()
    tree=cKDTree(centers) if len(centers) else None
    masks=[];records=[];height,width=movie.shape;outer=c['annulus_outer_radius_px'];inner=c['annulus_inner_radius_px']
    for r in candidates.itertuples():
        x,y=r.x_px,r.y_px
        yy,xx=np.mgrid[max(0,int(np.floor(y-outer))):min(height,int(np.ceil(y+outer))+1),
                       max(0,int(np.floor(x-outer))):min(width,int(np.ceil(x+outer))+1)]
        rad=np.hypot(xx-x,yy-y);geometric=(rad>inner)&(rad<=outer)
        px,py=xx[geometric],yy[geometric];n_geom=len(px)
        d=tree.query(np.column_stack([px,py]))[0] if tree is not None else np.full(n_geom,np.inf)
        keep=d>c['neighbor_core_radius_px'];px,py=px[keep],py[keep]
        data=movie[py,px];n=len(data);supported=n>=c['minimum_background_pixels']
        masks.append(np.column_stack([px,py]).astype(np.int16))
        v=float(c['noise_variance_e2'])
        records.append(dict(candidate_index=r.candidate_index,spot_id=r.spot_id,frame=r.frame,
            neighbor_quality_threshold=threshold,annulus_geometric_pixels=n_geom,annulus_retained_pixels=n,
            background_supported=supported,mean_e=float(np.mean(data)) if n else np.nan,
            median_e=float(np.median(data)) if n else np.nan,q25_e=float(np.percentile(data,25)) if n else np.nan,
            observed_sd_e=float(np.std(data,ddof=1)) if n>1 else np.nan,
            mean_nominal_se_e=float(np.sqrt(v/n)) if n else np.nan,
            median_nominal_se_e=float(np.sqrt(np.pi*v/(2*n))) if n else np.nan,
            SE_status='known_synthetic_noise_iid_approximation;selection_bias_not_included'))
    return pd.DataFrame(records),masks

def arbitrary_psf(xy,x,y,sx,sy):
    dx=xy[:,0]-x;dy=xy[:,1]-y
    def axis(d,s):
        lo=(d-.5)/s;hi=(d+.5)/s
        fl=np.exp(-lo*lo/2)/np.sqrt(2*np.pi);fh=np.exp(-hi*hi/2)/np.sqrt(2*np.pi)
        return ndtr(hi)-ndtr(lo),(fl-fh)/s,(lo*fl-hi*fh)/s
    px,gx,gsx=axis(dx,sx);py,gy,gsy=axis(dy,sy)
    return px*py,gx*py,px*gy,gsx*py,px*gsy

def joint_objective(theta,observed,box,auxiliary):
    loss,grad=base.objective(theta,observed,box)
    if auxiliary is None:return loss,grad
    kind,xy,measurement,se=auxiliary
    N=theta[0]*1000;x,y=theta[1:3];sx,sy=np.exp(theta[3:5]);bg=theta[5]*100
    if kind=='mean':
        P,dx,dy,dsx,dsy=arbitrary_psf(xy,x,y,sx,sy);n=len(xy)
        extra,g,gb=base.marginal_noise_loss(np.array([measurement]),np.array([N*P.sum()]),n*bg,500.*n)
        grad=grad+np.array([1000*g[0]*P.sum(),N*g[0]*dx.sum(),N*g[0]*dy.sum(),N*sx*g[0]*dsx.sum(),N*sy*g[0]*dsy.sum(),100*n*gb])
        return loss+extra,grad
    residual=(bg-measurement)/se
    grad=grad.copy();grad[5]+=100*(bg-measurement)/(se*se)
    return loss+.5*residual*residual+np.log(se*np.sqrt(2*np.pi)),grad

def covariance(theta,data,box,aux):
    steps=np.array([1e-5,1e-4,1e-4,1e-5,1e-5,1e-5]);H=np.empty((6,6))
    for j in range(6):
        d=np.zeros(6);d[j]=steps[j]
        H[:,j]=(joint_objective(theta+d,data,box,aux)[1]-joint_objective(theta-d,data,box,aux)[1])/(2*steps[j])
    H=(H+H.T)/2;ev=np.linalg.eigvalsh(H)
    if not np.isfinite(ev).all() or ev[0]<=1e-8:return None
    S=np.diag([1000,1,1,np.exp(theta[3]),np.exp(theta[4]),100]);return S@np.linalg.inv(H)@S

def fit_one(item,movie,box,c,method,stats,masks):
    i,spot_id,x,y=item;stat=stats.iloc[i];pixels=masks[i]
    out=dict(candidate_index=i,spot_id=spot_id,x_px=x,y_px=y,frame=0,time_s=0.,box=box,method=method,
        fit_attempted=False,fit_success=False,active_bound=False,fit_usable_interior=False,
        background_supported=bool(stat.background_supported),background_neighbor_threshold=float(stat.neighbor_quality_threshold),
        background_pixels=len(pixels),uncertainty_status='unavailable',coverage_validated=False,
        calibration_status='known_synthetic_conversion',estimator='independent_pixel_integrated_Poisson_Gaussian;not_native_Picasso')
    ax,ay=int(np.floor(x+.5)),int(np.floor(y+.5));h=box//2
    if ax<h or ay<h or ax+h>=movie.shape[1] or ay+h>=movie.shape[0]:
        out['unavailable_reason']='central_fitting_box_outside_image';return out
    if not stat.background_supported:
        out['unavailable_reason']='too_few_retained_background_pixels';return out
    ox,oy=ax-h,ay-h;data=movie[oy:oy+box,ox:ox+box].ravel()
    field='q25_e' if method=='q25_init' else 'median_e' if method.startswith('median') else 'mean_e'
    bg=max(float(stat[field]),0.);out['background_initial_e_per_pixel']=bg
    aux=None
    if 'constraint' in method:
        if method.startswith('mean'):
            observed=movie[pixels[:,1],pixels[:,0]]
            aux=('mean',pixels.astype(float)-[ox,oy],float(observed.sum()),float(stat.mean_nominal_se_e))
        else:aux=('median',pixels.astype(float)-[ox,oy],float(stat.median_e),float(stat.median_nominal_se_e))
    bounds=[(0,None),(0,box-1),(0,box-1),(np.log(.5),np.log(3.5)),(np.log(.5),np.log(3.5)),(0,None)]
    opts=[]
    for sigma in c['initial_sigma_multistart_px']:
        P=base.psf_and_derivatives(box,x-ox,y-oy,sigma,sigma)[0]
        N=max(30.,float(np.sum(data-bg))/P.sum())
        initial=np.array([N/1000,x-ox,y-oy,np.log(sigma),np.log(sigma),bg/100])
        opts.append(minimize(joint_objective,initial,args=(data,box,aux),jac=True,method='L-BFGS-B',bounds=bounds,
            options=dict(maxiter=c['fit_max_iterations'],ftol=1e-10,gtol=1e-5,maxls=40)))
    opt=min(opts,key=lambda v:v.fun);theta=opt.x;N=theta[0]*1000;sx,sy=np.exp(theta[3:5]);bg=theta[5]*100
    active=N<=1e-3 or bg<=1e-3 or min(sx,sy)<=.50001 or max(sx,sy)>=3.49999 or min(theta[1:3])<=1e-5 or max(theta[1:3])>=box-1-1e-5
    usable=bool(opt.success and np.isfinite(theta).all() and N>0 and not active)
    out.update(fit_attempted=True,fit_success=bool(opt.success),fit_message=str(opt.message),iterations=int(opt.nit),
        negative_log_likelihood=float(opt.fun),fit_x_px=float(theta[1]+ox),fit_y_px=float(theta[2]+oy),
        photons_e=float(N),bg_e_per_pixel=float(bg),sx_px=float(sx),sy_px=float(sy),active_bound=bool(active),fit_usable_interior=usable)
    if usable:
        C=covariance(theta,data,box,aux)
        if C is not None:
            out.update(N_se_e=float(np.sqrt(C[0,0])),bg_se_e_per_pixel=float(np.sqrt(C[5,5])),
                x_se_nm=float(np.sqrt(C[1,1])*117),y_se_nm=float(np.sqrt(C[2,2])*117),
                corr_N_bg=float(C[0,5]/np.sqrt(C[0,0]*C[5,5])),
                uncertainty_status='full_observed_Hessian_with_nuisance_bg;selection_and_coverage_not_validated')
        else:out['uncertainty_status']='nonpositive_or_singular_observed_Hessian'
    elif active:out['uncertainty_status']='active_bound;covariance_not_reported'
    return out

def mathematical_checks():
    checks=base.validate_math();rng=np.random.default_rng(18);data=rng.normal(110,22.36,49)
    yy,xx=np.mgrid[-12:13,-12:13];keep=(np.hypot(xx,yy)>6)&(np.hypot(xx,yy)<=12)
    xy=np.column_stack([xx[keep]+3,yy[keep]+3]).astype(float)
    theta=np.array([.8,3.2,2.8,np.log(1.45),np.log(1.6),1.])
    for aux in [('mean',xy,float(100*len(xy)),np.sqrt(500/len(xy))),('median',xy,100.,np.sqrt(np.pi*500/(2*len(xy))))]:
        gradient=joint_objective(theta,data,7,aux)[1];finite=[]
        for j in range(6):
            d=np.zeros(6);d[j]=1e-5
            finite.append((joint_objective(theta+d,data,7,aux)[0]-joint_objective(theta-d,data,7,aux)[0])/(2e-5))
        np.testing.assert_allclose(gradient,finite,rtol=1e-5,atol=1e-5)
    checks['auxiliary_gradients_verified']=True
    # A zero-signal annulus sum supplies finite, not infinite, background precision.
    n=len(xy);expected=500/n
    aux=('mean',xy,float(100*n),np.sqrt(expected));test=np.array([0,3,3,np.log(1.5),np.log(1.5),1.])
    d=np.zeros(6);d[5]=1e-5
    total=(joint_objective(test+d,data,7,aux)[1][5]-joint_objective(test-d,data,7,aux)[1][5])/(2e-5)
    central=(base.objective(test+d,data,7)[1][5]-base.objective(test-d,data,7)[1][5])/(2e-5)
    np.testing.assert_allclose(total-central,10000/expected,rtol=1e-6)
    checks['annular_uncertainty_not_zero_or_fixed_background']=True
    return checks

def metrics(t):
    if not len(t):return dict(n=0)
    ratio=t.photons_e/t.truth_expected_total_electrons
    return dict(n=len(t),median_abs_relative_error_pct=float(np.median(np.abs(ratio-1))*100),
        median_ratio=float(np.median(ratio)),within20pct=int((np.abs(ratio-1)<=.2).sum()),
        median_bg_e_per_pixel=float(t.bg_e_per_pixel.median()),median_bg_abs_error_to100=float(np.median(np.abs(t.bg_e_per_pixel-100))),
        median_position_error_nm=float(t.position_error_nm.median()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text());checks=mathematical_checks();started=time.perf_counter()
    source=Path(cfg['source_run']);previous=Path(cfg['free_fit_run']);simulation=Path(cfg['simulation_run'])
    out=source.parent/('local_background_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    print('OUTPUT:',out,flush=True)
    protected=[source/'all_trackmate_picasso_candidates.csv',source/'simulation_input_uint16_ADU.tiff',
        previous/'box7_all_candidates.csv',previous/'summary.json',simulation/'scene_components.npz',
        ROOT/'trackmate_python_bridge/picasso_refinement.py',Path(base.__file__)]
    before={str(p):sha(p) for p in protected};dump(out/'input_fingerprints.json',before)
    shutil.copyfile(a.config,out/'config.yaml');snap=out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(base.__file__),Path(__file__).with_name('config.yaml')]:shutil.copyfile(p,snap/p.name)
    dump(out/'rules_before_run.json',cfg)
    baseline=pd.read_csv(source/'all_trackmate_picasso_candidates.csv');features=['spot_id','frame','time_s','x_px','y_px','quality_log_response']
    candidates=baseline[features].copy();candidates.insert(0,'candidate_index',np.arange(len(candidates)))
    raw=tifffile.imread(source/'simulation_input_uint16_ADU.tiff');movie=(raw.astype(float)-cfg['camera_offset_adu'])*cfg['sensitivity_e_per_adu']
    geometry={}
    for label,threshold in [('primary',cfg['primary_neighbor_quality_threshold']),('all_candidates',cfg['neighbor_threshold_sensitivity'])]:
        table,masks=rings(candidates,movie,cfg,threshold);geometry[label]=(table,masks)
        table.to_csv(out/f'background_{label}_all_candidates.csv',index=False)
        offsets=np.concatenate([[0],np.cumsum([len(p) for p in masks])]);flat=np.concatenate(masks)
        np.savez_compressed(out/f'annulus_pixels_{label}.npz',offsets=offsets,xy_pixels=flat,spot_id=candidates.spot_id.to_numpy())
        print('Background mask:',label,'supported',int(table.background_supported.sum()),'median n',table.annulus_retained_pixels.median(),flush=True)
    free=pd.read_csv(previous/'box7_all_candidates.csv');results={'free_7x7':free}
    items=[(i,int(r.spot_id),float(r.x_px),float(r.y_px)) for i,r in enumerate(candidates.itertuples())]
    for method in cfg['methods']:
        stats,masks=geometry['all_candidates' if method.endswith('all_candidates') else 'primary'];rows=[]
        print('Fitting:',method,flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            for j,row in enumerate(pool.map(lambda item:fit_one(item,movie,cfg['box_px'],cfg,method,stats,masks),items),1):
                rows.append(row)
                if j%400==0:print(method,j,'/',len(items),flush=True)
        t=pd.DataFrame(rows);np.testing.assert_array_equal(t.spot_id,baseline.spot_id)
        for key in ['truth_id','truth_matched_from_TrackMate','truth_expected_total_electrons','truth_x_px','truth_y_px','qc_unflagged_model_fit']:
            t[key]=baseline[key].to_numpy()
        t['ratio_to_expected']=t.photons_e/t.truth_expected_total_electrons
        t['position_error_nm']=np.hypot(t.fit_x_px-t.truth_x_px,t.fit_y_px-t.truth_y_px)*117
        t.to_csv(out/f'{method}_all_candidates.csv',index=False);results[method]=t
    # Synthetic truth is consulted only for diagnostic scoring, not mask construction.
    with np.load(simulation/'scene_components.npz') as z:
        true_bg=z['background_electrons'];noise=z['read_noise_electrons'];signal=z['signal_realized_electrons']
    bg_eval=[]
    for label,(table,masks) in geometry.items():
        table=table.copy();table['truth_id']=baseline.truth_id.to_numpy();table['truth_matched_from_TrackMate']=baseline.truth_matched_from_TrackMate.to_numpy()
        for field,image in [('true_ring_background_mean_e',true_bg),('true_ring_signal_mean_e',signal),('true_ring_read_noise_mean_e',noise)]:
            table[field]=[float(image[p[:,1],p[:,0]].mean()) if len(p) else np.nan for p in masks]
        table.to_csv(out/f'background_{label}_truth_audit.csv',index=False)
        q=table[table.truth_matched_from_TrackMate&table.background_supported]
        for field in ['mean_e','median_e','q25_e']:
            bg_eval.append(dict(mask=label,estimator=field,n=len(q),median_bg_estimate=float(q[field].median()),
                mean_error_to_population100=float((q[field]-100).mean()),median_abs_error_to_population100=float(np.median(np.abs(q[field]-100))),
                median_n=float(q.annulus_retained_pixels.median()),median_signal_contamination_e_per_pixel=float(q.true_ring_signal_mean_e.median())))
    pd.DataFrame(bg_eval).to_csv(out/'background_estimator_summary.csv',index=False)
    # A fixed preceding subset and a shared valid subset both prevent misleading comparisons.
    prev11=pd.read_csv(previous/'box11_all_candidates.csv')
    fixed=baseline.truth_matched_from_TrackMate&baseline.qc_unflagged_model_fit&free.fit_usable_interior&prev11.fit_usable_interior
    common=fixed.copy()
    for t in results.values():common &= t.fit_usable_interior
    summary={}
    pairrows=[]
    for name,t in results.items():
        valid=t.truth_matched_from_TrackMate&t.fit_usable_interior
        summary[name]=dict(all_candidates=len(t),attempted=int(t.fit_attempted.sum()),interior_matched=int(valid.sum()),
            fixed_previous_subset_n=int(fixed.sum()),valid_in_fixed_subset_n=int((fixed&t.fit_usable_interior).sum()),
            all_fixed_finite_points=metrics(t[fixed&t.photons_e.notna()]),valid_matched=metrics(t[valid]),common_valid_subset=metrics(t[common]))
        for name_other,other in [('free_7x7',free)]:
            pair=fixed&t.fit_usable_interior&other.fit_usable_interior
            pairrows.append(dict(method=name,n_common=int(pair.sum()),base_error_pct=metrics(other[pair]).get('median_abs_relative_error_pct'),
                method_error_pct=metrics(t[pair]).get('median_abs_relative_error_pct')))
    dump(out/'summary.json',summary);pd.DataFrame(pairrows).to_csv(out/'paired_to_free_summary.csv',index=False)
    # Background-only masking controls reveal selection bias from excluding noise candidates.
    bg_only=(true_bg+noise);control_dir=Path(cfg['background_only_controls']);control_rows=[]
    for threshold in [6.5,25.]:
        f=pd.read_csv(control_dir/f'background_only_threshold_{threshold:g}.csv');f.insert(0,'candidate_index',np.arange(len(f)))
        for neighbor_th in [6.5,25.]:
            allpoints=pd.read_csv(control_dir/'background_only_threshold_6.5.csv');allpoints.insert(0,'candidate_index',np.arange(len(allpoints)))
            # Probe centers use the specified detector list, neighbors use the complete cached list.
            controls={k:cfg[k] for k in cfg}
            # Compute masks with combined features, then choose the requested matching probe coordinates.
            st,pm=rings(allpoints,bg_only,controls,neighbor_th)
            dist,idx=cKDTree(allpoints[['x_px','y_px']]).query(f[['x_px','y_px']])
            assert (dist<1e-8).all();q=st.iloc[idx];q=q[q.background_supported]
            for estimator in ['mean_e','median_e','q25_e']:
                control_rows.append(dict(probe_threshold=threshold,neighbor_threshold=neighbor_th,estimator=estimator,n=len(q),
                    mean_estimate_e=float(q[estimator].mean()) if len(q) else None,median_estimate_e=float(q[estimator].median()) if len(q) else None))
    pd.DataFrame(control_rows).to_csv(out/'background_only_selection_bias.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    primary=geometry['primary'][0];q=primary[baseline.truth_matched_from_TrackMate&primary.background_supported]
    for key,label in [('mean_e','Annulus mean'),('median_e','Annulus median'),('q25_e','Annulus q25')]:axs[0].scatter(baseline.loc[q.index,'truth_id'],q[key],s=20,label=label)
    axs[0].axhline(100,color='k',ls='--');axs[0].set(xlabel='Original matched true emitter ID',ylabel='Background estimate (e- / pixel)',title='No truth used to choose background pixels');axs[0].legend(fontsize=8)
    chosen=['free_7x7','mean_constraint','median_constraint','mean_constraint_all_candidates']
    for name in chosen:
        t=results[name];q=t[common];axs[1].scatter(q.truth_expected_total_electrons,q.photons_e,s=22,alpha=.7,label=name)
    axs[1].plot([0,1800],[0,1800],'k--');axs[1].set(xlabel='True expected total e-',ylabel='Fitted total e-',title=f'Same common subset, n={int(common.sum())}');axs[1].legend(fontsize=7)
    names=list(results);errors=[summary[n]['common_valid_subset'].get('median_abs_relative_error_pct',np.nan) for n in names]
    axs[2].barh(np.arange(len(names)),errors,color='#458aa8');axs[2].set_yticks(np.arange(len(names)),labels=names,fontsize=8)
    axs[2].set(xlabel='Median absolute relative N error (%)',title='Same shared subset; no auto selection')
    fig.suptitle('Local background controls: cached TrackMate centers + independent 7x7 photometry\nBackground remains uncertain; neighbor-mask selection and precision coverage are not calibrated',fontsize=12)
    fig.savefig(out/'local_background_comparison.png',dpi=130);plt.close(fig)
    checks.update(source_fingerprints_unchanged=before=={str(p):sha(p) for p in protected},all_target_candidates_preserved=True,
        truth_not_used_for_background_selection_or_fitting=True,annulus_disjoint_from_central_7x7=True,
        q25_used_as_initialization_only=True,real_pipeline_unchanged=True,coverage_not_validated=True)
    assert checks['source_fingerprints_unchanged'];dump(out/'validation.json',checks)
    import importlib.metadata
    dump(out/'versions.json',dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','numba','pandas','tifffile','matplotlib','pyyaml']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__)),'--config',str(a.config)])+'\n')
    (out/'REPORT.txt').write_text('''Synthetic-only local background test; original data and real pipeline are read-only.
Same 1653 TrackMate target candidates, radius0.28um/threshold6.5; no re-detection.
The primary neighbor-core mask uses cached quality>=25 candidates; sensitivity uses
all quality>=6.5 candidates. Threshold25 only chooses background exclusions; it
does not remove target candidates. Both choices and their biases are reported.
The ring is 6<r<=12px around the original floating TM center, excluding neighbors
within4px; it is disjoint from the central7x7 fitting pixels. Minimum80 retained pixels.
Every retained pixel enters the mean/median/percentile; no dark-tail trimming.
Initial-only fits keep b free. Mean-constrained fits combine the central likelihood
with the likelihood of the disjoint annulus SUM, including this emitter's fitted
PSF wings. Its Gaussian variance is500*n, not a zero-uncertainty fixed background.
Median-constrained fits use a Gaussian auxiliary observation with asymptotic SE
sqrt(pi/2)*sqrt(500/n); ring signal should be negligible. This is approximate.
q25 is an initialization-only comparator and is not fixed as a background mean.
The likelihood, camera conversion and variance500 are specific to the known synthetic
generator. They do not calibrate the real TIFF. Original calibrated7x7 is reused.
Background-only controls expose potential bias from masking noise-peak candidates.
Neighbor PSF wings can remain, and the single-emitter model does not solve overlap.
Geometry and masks are data-selected; quoted local errors do not account for all
selection effects, and no CRLB/interval coverage is claimed validated.
Fixed prior24 points and common-valid-subset results are both retained; different
QC membership must not be confused with improved estimation on identical points.
No automatic best method, native-package modification, TARDIS, or real-image refit.
''')
    dump(out/'COMPLETED.json',dict(status='complete',elapsed_s=time.perf_counter()-started,output=str(out)))
    print(json.dumps(summary,indent=2),flush=True);print('BACKGROUND:',json.dumps(bg_eval,indent=2),flush=True);print('COMPLETE:',out,flush=True)

if __name__=='__main__':main()
