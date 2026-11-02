"""Independent synthetic photometry control, not the native Picasso backend.

Pixel-integrated elliptical Gaussian signal with total amplitude N; marginal
Poisson(signal) + Gaussian(background mean b, variance 500) likelihood.
This noise law is specific to the saved synthetic generator, NOT to the camera.
The existing TrackMate detector, original fit cache, and real pipeline are read-only.
"""
import argparse,concurrent.futures,hashlib,json,math,shlex,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba,numpy as np,pandas as pd,tifffile
from scipy.optimize import minimize
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[1]

@numba.njit(cache=True,nogil=True)
def marginal_noise_loss(observed,mean_signal,background,variance):
    """Poisson--Gaussian convolution, evaluated by a stable truncated sum.

    Poisson support uses 12 SD plus 30 counts. This is a numerical convolution,
    not the shifted-Poisson approximation. ADU quantization is not modeled.
    """
    loss=0.;gm=np.empty(observed.size);gb=0.
    constant=.5*math.log(2*math.pi*variance)
    for i in range(observed.size):
        lam=mean_signal[i];z=observed[i]-background
        if lam<=1e-12:
            loss+=constant+z*z/(2*variance)
            gm[i]=1-math.exp(min(500.,(2*z-1)/(2*variance)))
            gb-=z/variance
            continue
        sd=math.sqrt(lam+1)
        lower=max(0,int(math.floor(lam-12*sd-30)))
        upper=int(math.ceil(lam+12*sd+30))
        loglam=math.log(lam);logpois=-lam+lower*loglam-math.lgamma(lower+1)
        largest=-math.inf;sw=0.;sn=0.
        for n in range(lower,upper+1):
            logterm=logpois-constant-(z-n)*(z-n)/(2*variance)
            if logterm>largest:
                rescale=math.exp(largest-logterm)
                sw=sw*rescale+1.;sn=sn*rescale+n;largest=logterm
            else:
                weight=math.exp(logterm-largest);sw+=weight;sn+=weight*n
            logpois+=loglam-math.log(n+1)
        posterior_n=sn/sw
        loss-=largest+math.log(sw)
        gm[i]=1-posterior_n/lam
        gb+=(posterior_n-z)/variance
    return loss,gm,gb

def psf_and_derivatives(box,x,y,sx,sy):
    center=np.arange(box,dtype=float)
    def axis(c,s):
        lo=(center-.5-c)/s;hi=(center+.5-c)/s
        flo=np.exp(-lo*lo/2)/np.sqrt(2*np.pi);fhi=np.exp(-hi*hi/2)/np.sqrt(2*np.pi)
        return ndtr(hi)-ndtr(lo),(flo-fhi)/s,(lo*flo-hi*fhi)/s
    px,dx,dsx=axis(x,sx);py,dy,dsy=axis(y,sy)
    return tuple(a.ravel() for a in [np.outer(py,px),np.outer(py,dx),np.outer(dy,px),np.outer(py,dsx),np.outer(dsy,px)])

def objective(theta,observed,box):
    N=1000*theta[0];x,y=theta[1:3];sx,sy=np.exp(theta[3:5]);bg=100*theta[5]
    P,dx,dy,dsx,dsy=psf_and_derivatives(box,x,y,sx,sy)
    loss,g,gb=marginal_noise_loss(observed,N*P,bg,500.)
    gradient=np.array([1000*np.dot(g,P),N*np.dot(g,dx),N*np.dot(g,dy),N*sx*np.dot(g,dsx),N*sy*np.dot(g,dsy),100*gb])
    return loss,gradient

def local_covariance(theta,observed,box):
    """Full observed Hessian; nuisance N, widths and bg remain in its inverse.

    Only an interior local-model diagnostic; NOT a coverage-validated CRLB.
    """
    steps=np.array([1e-5,1e-4,1e-4,1e-5,1e-5,1e-5]);H=np.empty((6,6))
    for i in range(6):
        d=np.zeros(6);d[i]=steps[i]
        H[:,i]=(objective(theta+d,observed,box)[1]-objective(theta-d,observed,box)[1])/(2*steps[i])
    H=(H+H.T)/2
    eig=np.linalg.eigvalsh(H)
    if not np.isfinite(eig).all() or eig[0]<=1e-8:return None
    C=np.linalg.inv(H)
    scaling=np.diag([1000,1,1,np.exp(theta[3]),np.exp(theta[4]),100])
    return scaling@C@scaling

def fit_one(candidate,movie,box):
    i,spot_id,x,y=candidate
    result=dict(candidate_index=i,spot_id=spot_id,x_px=x,y_px=y,frame=0,time_s=0.,box=box,
        fit_attempted=False,fit_success=False,active_bound=False,fit_usable_interior=False,
        uncertainty_status='unavailable',noise_model='synthetic_Poisson_signal_plus_Gaussian_variance500',
        estimator='independent_pixel_integrated_Poisson_Gaussian;not_native_Picasso',
        calibration_status='known_synthetic_conversion',coverage_validated=False)
    ax,ay=int(np.floor(x+.5)),int(np.floor(y+.5));half=box//2
    if ax<half or ay<half or ax+half>=movie.shape[1] or ay+half>=movie.shape[0]:return result
    ox,oy=ax-half,ay-half;data=movie[oy:oy+box,ox:ox+box].astype(float).ravel()
    patch=data.reshape(box,box);edge=np.concatenate([patch[0],patch[-1],patch[1:-1,0],patch[1:-1,-1]])
    bg=max(float(np.median(edge)),0.)
    bounds=[(0,None),(0,box-1),(0,box-1),(np.log(.5),np.log(3.5)),(np.log(.5),np.log(3.5)),(0,None)]
    starts=[]
    for sigma in [1.,2.]:
        P=psf_and_derivatives(box,x-ox,y-oy,sigma,sigma)[0]
        N=max(30.,float(np.sum(data-bg))/P.sum())
        initial=np.array([N/1000,x-ox,y-oy,np.log(sigma),np.log(sigma),bg/100])
        starts.append(minimize(objective,initial,args=(data,box),jac=True,method='L-BFGS-B',bounds=bounds,
            options=dict(maxiter=350,ftol=1e-10,gtol=1e-5,maxls=40)))
    opt=min(starts,key=lambda z:z.fun);theta=opt.x
    N=theta[0]*1000;sx,sy=np.exp(theta[3:5]);bg=theta[5]*100
    active= N<=1e-3 or bg<=1e-3 or min(sx,sy)<=.50001 or max(sx,sy)>=3.49999 or min(theta[1:3])<=1e-5 or max(theta[1:3])>=box-1-1e-5
    usable=bool(opt.success and np.isfinite(theta).all() and N>0 and not active)
    P=psf_and_derivatives(box,*theta[1:3],sx,sy)[0]
    result.update(fit_attempted=True,fit_success=bool(opt.success),fit_message=str(opt.message),nfev=int(opt.nfev),iterations=int(opt.nit),
        negative_log_likelihood=float(opt.fun),fit_x_px=float(theta[1]+ox),fit_y_px=float(theta[2]+oy),
        photons_e=float(N),bg_e_per_pixel=float(bg),sx_px=float(sx),sy_px=float(sy),
        captured_psf_fraction=float(P.sum()),signal_inside_box_e=float(N*P.sum()),
        active_bound=bool(active),fit_usable_interior=usable,fit_origin_x=ox,fit_origin_y=oy)
    if usable:
        C=local_covariance(theta,data,box)
        if C is not None:
            result.update(N_se_e=float(np.sqrt(C[0,0])),x_se_nm=float(np.sqrt(C[1,1])*117),y_se_nm=float(np.sqrt(C[2,2])*117),
                bg_se_e_per_pixel=float(np.sqrt(C[5,5])),corr_N_bg=float(C[0,5]/np.sqrt(C[0,0]*C[5,5])),
                corr_N_sx=float(C[0,3]/np.sqrt(C[0,0]*C[3,3])),corr_N_sy=float(C[0,4]/np.sqrt(C[0,0]*C[4,4])),
                uncertainty_status='observed_Hessian_full_nuisance_covariance;coverage_not_validated')
        else:result['uncertainty_status']='nonpositive_or_singular_observed_Hessian'
    elif active:result['uncertainty_status']='active_bound;unconstrained_covariance_not_reported'
    return result

def validate_math():
    from scipy.special import gammaln,logsumexp
    for lam,z in [(0.,12.),(.5,-20.),(25.,10.),(80.,115.)]:
        loss,g,gb=marginal_noise_loss(np.array([100.+z]),np.array([lam]),100.,500.)
        if lam:
            n=np.arange(1500);reference=-logsumexp(-lam+n*np.log(lam)-gammaln(n+1)-.5*np.log(2*np.pi*500)-(z-n)**2/1000)
        else:reference=.5*np.log(2*np.pi*500)+z*z/1000
        assert abs(loss-reference)<1e-10
    theta=np.array([.8,3.2,2.8,np.log(1.45),np.log(1.6),1.])
    rng=np.random.default_rng(17);data=rng.normal(110,22.4,49)
    analytic=objective(theta,data,7)[1]
    numerical=[]
    for i in range(6):
        d=np.zeros(6);d[i]=1e-5
        numerical.append((objective(theta+d,data,7)[0]-objective(theta-d,data,7)[0])/(2e-5))
    np.testing.assert_allclose(analytic,numerical,rtol=1e-5,atol=1e-5)
    P=psf_and_derivatives(101,50.2,49.8,1.45,1.6)[0]
    assert abs(P.sum()-1)<1e-12
    return dict(convolution_matches_direct_sum=True,analytic_gradient_verified=True,pixel_PSF_normalization_verified=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    start=time.perf_counter();checks=validate_math()
    source=a.source;out=source.parent/('calibrated_photometry_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    print('OUTPUT:',out,flush=True)
    protected=[source/'all_trackmate_picasso_candidates.csv',source/'simulation_input_uint16_ADU.tiff',ROOT/'trackmate_python_bridge/picasso_refinement.py']
    def fingerprints():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before=fingerprints();(out/'input_fingerprints.json').write_text(json.dumps(before,indent=2))
    baseline=pd.read_csv(source/'all_trackmate_picasso_candidates.csv')
    raw=tifffile.imread(source/'simulation_input_uint16_ADU.tiff');movie=(raw.astype(float)-100)*.46
    candidates=[(i,int(r.spot_id),float(r.x_px),float(r.y_px)) for i,r in enumerate(baseline.itertuples())]
    result={}
    for box in [7,11]:
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for j,r in enumerate(pool.map(lambda candidate:fit_one(candidate,movie,box),candidates),1):
                rows.append(r)
                if j%200==0:print('Box',box,':',j,'/',len(candidates),flush=True)
        table=pd.DataFrame(rows);np.testing.assert_array_equal(table.spot_id,baseline.spot_id)
        # Truth metadata is consulted only AFTER all fits, for scoring.
        for key in ['truth_id','truth_matched_from_TrackMate','truth_expected_total_electrons','truth_x_px','truth_y_px','neighbor_contaminated','qc_unflagged_model_fit']:
            table[key]=baseline[key].to_numpy()
        table['ratio_to_expected']=table.photons_e/table.truth_expected_total_electrons
        table['position_error_nm']=np.hypot(table.fit_x_px-table.truth_x_px,table.fit_y_px-table.truth_y_px)*117
        table.to_csv(out/f'box{box}_all_candidates.csv',index=False);result[box]=table
    common=baseline.truth_matched_from_TrackMate &baseline.qc_unflagged_model_fit
    for t in result.values():common &= t.fit_usable_interior
    summary={}
    for box,t in result.items():
        mask=t.truth_matched_from_TrackMate &t.fit_usable_interior
        def metrics(q):
            return dict(n=len(q),median_abs_relative_error_pct=float(np.median(np.abs(q.ratio_to_expected-1))*100) if len(q) else None,
                median_ratio=float(q.ratio_to_expected.median()) if len(q) else None,within20pct=int((np.abs(q.ratio_to_expected-1)<=.2).sum()),
                median_position_error_nm=float(q.position_error_nm.median()) if len(q) else None,
                median_N_relative_local_se_pct=float(np.nanmedian(q.N_se_e/q.photons_e)*100) if len(q) else None)
        summary[f'box{box}']=dict(all_candidates=len(t),attempted=int(t.fit_attempted.sum()),converged=int(t.fit_success.sum()),
            active_bound=int(t.active_bound.sum()),usable_matched=metrics(t[mask]),common_unchanged_subset=metrics(t[common]))
    baseline_ratio=baseline.photons/baseline.truth_expected_total_electrons
    summary['original_same_common_subset']=dict(n=int(common.sum()),median_abs_relative_error_pct=float(np.median(np.abs(baseline_ratio[common]-1))*100))
    fig,axs=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    for box,t in result.items():
        q=t[t.truth_matched_from_TrackMate];valid=q.fit_usable_interior
        axs[0].scatter(q.truth_id,q.ratio_to_expected,s=22,alpha=.75,label=f'Box {box}; all original matches')
        q=t[common];axs[1].scatter(q.truth_expected_total_electrons,q.photons_e,s=25,alpha=.8,label=f'Box {box}')
        q=t[t.truth_matched_from_TrackMate&t.fit_usable_interior]
        axs[2].scatter(q.corr_N_bg,np.abs(q.ratio_to_expected-1)*100,s=25,alpha=.8,label=f'Box {box}')
    axs[0].axhline(1,color='k',ls='--');axs[0].set(yscale='log',xlabel='True emitter ID',ylabel='Fitted N / true N',title='Same original TM matches; includes boundary fits');axs[0].legend(fontsize=8)
    axs[1].plot([0,1800],[0,1800],'k--');axs[1].set(xlabel='True expected total e-',ylabel='Fitted total e-',title=f'Same common original-QC/interior subset: {common.sum()}');axs[1].legend()
    axs[2].set(xlabel='Local fitted correlation: N versus background',ylabel='Absolute relative N error (%)',title='Parameter coupling remains a diagnostic');axs[2].legend()
    fig.suptitle('Independent synthetic photometry: pixel-integrated PSF + Poisson/Gaussian convolution\nKnown synthetic variance 500; every candidate retained; uncertainty coverage NOT validated',fontsize=13)
    fig.savefig(out/'photometry_comparison.png',dpi=130);plt.close(fig)
    checks.update(all_original_candidates_preserved=True,original_files_unchanged=before==fingerprints(),truth_not_used_to_initialize_or_fit=True,
        same_prior_original_TrackMate_matches_used=True,model_specific_to_synthetic_generator=True,real_camera_calibration_not_inferred=True,uncertainty_coverage_not_validated=True)
    assert checks['original_files_unchanged']
    (out/'validation.json').write_text(json.dumps(checks,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'config.json').write_text(json.dumps(dict(boxes=[7,11],bounds=dict(N_e=[0,None],sx_sy_px=[.5,3.5],background_e=[0,None]),
        variance_e2=500,variance_source='synthetic_Gaussian_background_SD20_plus_read_SD10;no_background_Poisson',
        camera_offset_ADU=100,sensitivity_e_per_ADU=.46,psf='pixel_integrated_elliptical_Gaussian',
        original_TrackMate_threshold=6.5,original_TrackMate_radius_um=.28,position_initialization='original_TrackMate_floating_center',
        multistart_sigma_px=[1,2],ADC_quantization='density evaluated at converted values; +/-0.23 e quantization neglected',
        N_se_interpretation='observed_full_Hessian_nuisance_covariance;not_validated_CRLB',
        width_bounds_source='existing_7x7_QC_range;not_individual_truth_widths',real_data_run=False),indent=2))
    shutil.copyfile(__file__,out/Path(__file__).name)
    import importlib.metadata
    (out/'software_versions.json').write_text(json.dumps(dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','numba','pandas','matplotlib','tifffile']}),indent=2))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__))]+sys.argv[1:])+'\n')
    (out/'REPORT.txt').write_text('''This is an independent simulation-specific fitting control, not a native Picasso result.
Uses the identical original 1653 TrackMate candidates. Fits raw converted camera pixels,
without pixel clipping, with a normalized pixel-integrated Gaussian signal of total N.
Gaussian mean background b is fitted with N,x,y,sx,sy. Variance 500 comes solely from
the saved generator: Normal background SD20 plus read SD10, with Poisson signal only.
The Poisson/Gaussian convolution is summed numerically. Very small ADC quantization
is ignored; its maximum rounding error is 0.23 e against noise SD22.36 e.
No individual true N, widths, positions or true local background enter the fitting.
Width and positivity constraints are declared, with active-bound solutions flagged.
All candidates survive; unsupported boxes and failed fits are unavailable, not dropped.
Full observed-Hessian covariance retains nuisance parameter coupling. This is a
local diagnostic, not a coverage-validated CRLB. No claim of calibrated real precision.
Known synthetic noise calibration cannot be transplanted to the real TIFF.
''')
    (out/'COMPLETED.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-start,status='complete'),indent=2))
    print(json.dumps(summary,indent=2),flush=True);print('COMPLETE:',out,flush=True)

if __name__=='__main__':main()
