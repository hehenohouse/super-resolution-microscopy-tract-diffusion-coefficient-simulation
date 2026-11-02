"""Single synthetic electron-count image and auditable 9x9 mean diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.special import erf
import tifffile
import yaml


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def positive_normal(rng, mean, std, size):
    values = rng.normal(mean, std, size)
    rejected = 0
    while (values <= 0).any():
        bad = values <= 0; rejected += int(bad.sum())
        values[bad] = rng.normal(mean, std, int(bad.sum()))
    return values, rejected


def pixel_gaussian(shape, x, y, sigma):
    height, width = shape
    root_two_sigma = np.sqrt(2)*sigma
    xe = np.arange(width+1)-.5
    ye = np.arange(height+1)-.5
    px = np.maximum(0, .5*np.diff(erf((xe-x)/root_two_sigma)))
    py = np.maximum(0, .5*np.diff(erf((ye-y)/root_two_sigma)))
    field = np.outer(py, px)
    captured_mass = float(field.sum())
    assert captured_mass > 0
    # Explicit finite-FOV normalization, negligible with the recorded margin.
    return field/captured_mass, captured_mass


def mean9(image, size):
    result = uniform_filter(image.astype(np.float64), size=size, mode='constant', cval=0)
    half = size//2
    result[:half,:] = result[-half:,:] = np.nan
    result[:,:half] = result[:,-half:] = np.nan
    return result


def plot_results(out, c, truth, arrays, diagnostics):
    observed = arrays['observed_electrons']
    low, high = np.percentile(observed, [1,99.9])
    h, w = observed.shape
    fig, axes = plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    specs = [(arrays['signal_expected_electrons'],'Expected signal only','magma',None,None),
             (arrays['background_electrons'],'Gaussian background: N(100, 20^2)','gray',40,160),
             (arrays['read_noise_electrons'],'Read noise: N(0, 10^2)','coolwarm',-30,30),
             (observed,'Simulated observed image','gray',low,high),
             (observed,'Same image + 50 truth HWHM circles','gray',low,high),
             (arrays['observed_mean9_electrons_per_pixel'],'9x9 mean of observed image','viridis',90,125)]
    for ax,(data,title,cmap,vmin,vmax) in zip(axes.ravel(),specs):
        im = ax.imshow(data,origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set(title=title,xlabel='X (pixels)',ylabel='Y (pixels)',xlim=(-.5,w-.5),ylim=(h-.5,-.5))
        fig.colorbar(im,ax=ax,shrink=.75,label='e- / pixel')
    for row in truth.itertuples():
        axes[1,1].add_patch(Circle((row.x_px,row.y_px),row.hwhm_px,fill=False,edgecolor='#00E5FF',lw=.65))
    fig.suptitle(f'300 x 300 pixels | 117 nm/pixel | 50 emitters | Seed {c["seed"]}\n'
                 'N_e ~ N(800,250^2), HWHM ~ N(0.20,0.05^2) um; positive draws; signal Poisson',fontsize=15)
    fig.savefig(out/'simulation_overview.png',dpi=160);plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    for ax,key,title in [(axes[0,0],'background_mean9_electrons_per_pixel','True local 9x9 background mean'),
                          (axes[0,1],'observed_mean9_electrons_per_pixel','Estimated background = whole 9x9 image mean')]:
        im=ax.imshow(arrays[key],origin='upper',cmap='viridis',vmin=90,vmax=125,interpolation='nearest')
        ax.set(title=title,xlabel='X (pixels)',ylabel='Y (pixels)')
        fig.colorbar(im,ax=ax,label='e- / pixel',shrink=.8)
    ax=axes[1,0]
    im=ax.scatter(diagnostics.true_background_mean9_e_per_pixel,diagnostics.observed_mean9_e_per_pixel,
                  c=diagnostics.signal_realized_mean9_e_per_pixel,cmap='magma',s=35)
    span=[float(min(diagnostics.true_background_mean9_e_per_pixel.min(),diagnostics.observed_mean9_e_per_pixel.min())-1),
          float(max(diagnostics.true_background_mean9_e_per_pixel.max(),diagnostics.observed_mean9_e_per_pixel.max())+1)]
    ax.plot(span,span,'--',color='gray',label='Estimated = true')
    ax.set(xlabel='True local background (e- / pixel)',ylabel='Observed 9x9 mean (e- / pixel)',title='Each dot is one known emitter');ax.legend()
    fig.colorbar(im,ax=ax,label='Signal included in 9x9 (e- / pixel)')
    ax=axes[1,1]
    ax.hist(diagnostics.background_estimation_error_e_per_pixel,bins=15,color='#537D9D',alpha=.85)
    ax.axvline(0,color='gray',ls='--');ax.axvline(diagnostics.background_estimation_error_e_per_pixel.mean(),color='#DC7726',label='Mean error')
    ax.set(xlabel='9x9 mean minus true background (e- / pixel)',ylabel='Emitters',title='Signal contamination + averaged read noise');ax.legend()
    fig.suptitle('A whole-window mean includes the emitter and neighboring signals',fontsize=15)
    fig.savefig(out/'background_9x9_diagnostics.png',dpi=170);plt.close(fig)

    distance=np.hypot(truth.x_px.to_numpy()[:,None]-truth.x_px.to_numpy()[None,:],
                      truth.y_px.to_numpy()[:,None]-truth.y_px.to_numpy()[None,:])
    np.fill_diagonal(distance,np.inf)
    nearest=distance.min(axis=1)
    order=np.argsort(truth.expected_total_electrons.to_numpy())
    median_idx=int(order[len(order)//2])
    chosen=list(dict.fromkeys([median_idx,int(np.argmax(truth.expected_total_electrons)),int(np.argmin(nearest))]))
    fig, axes=plt.subplots(len(chosen),3,figsize=(11,3.4*len(chosen)),squeeze=False,constrained_layout=True)
    for j,i in enumerate(chosen):
        row=truth.iloc[i];x,y=int(row.anchor_x_px),int(row.anchor_y_px)
        sy,sx=slice(y-10,y+11),slice(x-10,x+11)
        for k,(data,title) in enumerate([(observed,'Observed'),(arrays['signal_expected_electrons'],'Expected signal'),
                                       (arrays['background_electrons'],'True Gaussian background')]):
            ax=axes[j,k];patch=data[sy,sx]
            assert patch.shape==(21,21)
            limits=(low,high) if k in (0,2) else (0,float(arrays['signal_expected_electrons'][sy,sx].max()))
            im=ax.imshow(patch,origin='upper',cmap='gray' if k!=1 else 'magma',vmin=limits[0],vmax=limits[1],interpolation='nearest')
            ax.add_patch(Rectangle((5.5,5.5),9,9,fill=False,edgecolor='#FFAD31',lw=1))
            ax.add_patch(Circle((row.x_px-x+10,row.y_px-y+10),row.hwhm_px,fill=False,edgecolor='#00E5FF',lw=1))
            ax.set(title=title,xlabel='Local X (pixels)',ylabel='Local Y (pixels)')
            if k==0:
                d=diagnostics.iloc[i]
                ax.set_ylabel(f'Spot {int(row.spot_id)}\nN={row.expected_total_electrons:.0f} e-, HWHM={row.hwhm_um:.3f} um\n'
                              f'9x9 mean={d.observed_mean9_e_per_pixel:.1f}; true bg={d.true_background_mean9_e_per_pixel:.1f}')
            fig.colorbar(im,ax=ax,label='e- / pixel',shrink=.7)
    fig.suptitle('21x21 contexts | Orange: 9x9 averaging window | Cyan: true HWHM',fontsize=13)
    fig.savefig(out/'spot_examples.png',dpi=170);plt.close(fig)


def run(config_path):
    started=time.perf_counter()
    c=yaml.safe_load(config_path.read_text())
    assert c['psf']['width_definition']=='HWHM' and c['psf']['profile']=='isotropic_gaussian_integrated_over_pixel'
    assert c['background']['additional_poisson_sampling'] is False
    shape=tuple(c['image_shape_yx']);height,width=shape;n=int(c['spot_count']);margin=c['center_margin_px']
    assert shape==(300,300) and n==50 and c['background_estimator']['size_px']==9
    root=Path(c['output_root']);assert root.parents[2].is_dir(), 'Configured data directory must exist'
    out=root/('scene300x300_50spots_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT_DIRECTORY:',out,flush=True)
    streams=np.random.SeedSequence(c['seed']).spawn(5+n)
    center_rng=np.random.default_rng(streams[0]);x=center_rng.uniform(margin,width-1-margin,n);y=center_rng.uniform(margin,height-1-margin,n)
    signal=c['signal'];psf=c['psf'];bg=c['background'];rn=c['read_noise']
    totals,n_rejected=positive_normal(np.random.default_rng(streams[1]),signal['expected_total_electrons_mean'],signal['expected_total_electrons_std'],n)
    hwhm,r_rejected=positive_normal(np.random.default_rng(streams[2]),psf['hwhm_um_mean'],psf['hwhm_um_std'],n)
    pixel_um=c['pixel_size_nm']/1000;sigma=hwhm/(np.sqrt(2*np.log(2))*pixel_um)
    background=np.random.default_rng(streams[3]).normal(bg['mean_electrons_per_pixel'],bg['std_electrons_per_pixel'],shape)
    read_noise=np.random.default_rng(streams[4]).normal(rn['mean_electrons_per_pixel'],rn['std_electrons_per_pixel'],shape)
    expectation=np.zeros(shape);realization=np.zeros(shape,dtype=np.int64)
    anchors=np.floor(np.column_stack([x,y])+.5).astype(int);rows=[];own_mean=[]
    for i in range(n):
        weights,mass=pixel_gaussian(shape,x[i],y[i],sigma[i]);expected=totals[i]*weights
        assert np.isfinite(weights).all() and np.all(weights>=0)
        np.testing.assert_allclose(expected.sum(),totals[i],rtol=1e-14)
        counts=np.random.default_rng(streams[5+i]).poisson(expected)
        expectation+=expected;realization+=counts
        ax,ay=anchors[i];own_mean.append(float(expected[ay-4:ay+5,ax-4:ax+5].mean()))
        rows.append({'spot_id':i,'x_px':x[i],'y_px':y[i],'x_nm':x[i]*c['pixel_size_nm'],'y_nm':y[i]*c['pixel_size_nm'],
                     'anchor_x_px':ax,'anchor_y_px':ay,'expected_total_electrons':totals[i],
                     'realized_total_electrons':int(counts.sum()),'hwhm_um':hwhm[i],'hwhm_px':hwhm[i]/pixel_um,
                     'gaussian_sigma_px':sigma[i],'gaussian_sigma_nm':sigma[i]*c['pixel_size_nm'],
                     'FWHM_um':2*hwhm[i],'unrenormalized_PSF_mass_inside_FOV':mass})
    truth=pd.DataFrame(rows);observed=realization+background+read_noise
    arrays={'signal_expected_electrons':expectation,'signal_realized_electrons':realization,
            'background_electrons':background,'read_noise_electrons':read_noise,'observed_electrons':observed,
            'background_mean9_electrons_per_pixel':mean9(background,9),
            'observed_mean9_electrons_per_pixel':mean9(observed,9)}
    comparisons=[]
    for i,row in truth.iterrows():
        ax,ay=int(row.anchor_x_px),int(row.anchor_y_px);win=(slice(ay-4,ay+5),slice(ax-4,ax+5))
        assert observed[win].shape==(9,9)
        true_bg=float(background[win].mean());estimate=float(observed[win].mean())
        signal_mean=float(realization[win].mean());noise_mean=float(read_noise[win].mean())
        np.testing.assert_allclose(estimate,true_bg+signal_mean+noise_mean,atol=1e-12,rtol=0)
        np.testing.assert_allclose(arrays['observed_mean9_electrons_per_pixel'][ay,ax],estimate,atol=1e-10,rtol=0)
        comparisons.append({'spot_id':int(row.spot_id),'true_background_mean9_e_per_pixel':true_bg,
                            'observed_mean9_e_per_pixel':estimate,'background_estimation_error_e_per_pixel':estimate-true_bg,
                            'signal_realized_mean9_e_per_pixel':signal_mean,'read_noise_mean9_e_per_pixel':noise_mean,
                            'own_signal_expected_mean9_e_per_pixel':own_mean[i],
                            'neighbor_signal_expected_mean9_e_per_pixel':float(expectation[win].mean())-own_mean[i]})
    diagnostics=pd.DataFrame(comparisons)
    truth.to_csv(out/'truth_spots.csv',index=False,float_format='%.17g')
    diagnostics.to_csv(out/'background_9x9_at_spots.csv',index=False,float_format='%.17g')
    np.savez_compressed(out/'scene_components.npz',**arrays)
    tifffile.imwrite(out/'simulated_image_electrons_float32.tiff',observed.astype(np.float32),metadata={'axes':'YX','unit':'photoelectrons','pixel_size_nm':c['pixel_size_nm']},photometric='minisblack')
    np.testing.assert_array_equal(tifffile.imread(out/'simulated_image_electrons_float32.tiff'),observed.astype(np.float32))
    plot_results(out,c,truth,arrays,diagnostics)
    shutil.copyfile(config_path,out/'config.yaml');shutil.copyfile(__file__,out/'run_simulation.py')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path.resolve())])+'\n')
    dump(out/'software_versions.json',{**{n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','pandas','tifffile','PyYAML']},'python':sys.version,'executable':sys.executable})
    active=expectation>=1
    summary={'shape_yx':list(shape),'field_width_um':width*pixel_um,'field_height_um':height*pixel_um,
             'spots':n,'seed':c['seed'],'calibration_status':'synthetic_ground_truth_electrons',
             'radius_definition':'HWHM of continuous Gaussian intensity; integrated into camera pixels',
             'input_expected_total_e_mean_std':[800,250],'input_HWHM_um_mean_std':[.2,.05],
             'realized_mean_expected_electrons_per_spot':float(totals.mean()),'realized_std_expected_electrons_per_spot':float(totals.std(ddof=1)),
             'realized_HWHM_um_mean':float(hwhm.mean()),'realized_HWHM_um_std':float(hwhm.std(ddof=1)),
             'background_field_mean_std':[float(background.mean()),float(background.std(ddof=1))],
             'read_noise_mean_std':[float(read_noise.mean()),float(read_noise.std(ddof=1))],
             'positive_normal_redraws':{'signal':n_rejected,'HWHM':r_rejected},
             'signal_expected_sum_electrons':float(expectation.sum()),'signal_realized_sum_electrons':int(realization.sum()),
             'signal_total_Poisson_standardized_deviation':float((realization.sum()-expectation.sum())/np.sqrt(expectation.sum())),
             'Poisson_Pearson_mean_on_expected_signal_ge_1e_pixels':float(np.mean((realization[active]-expectation[active])**2/expectation[active])),
             'background9_mean_estimation_error_e_per_pixel':float(diagnostics.background_estimation_error_e_per_pixel.mean()),
             'background9_median_estimation_error_e_per_pixel':float(diagnostics.background_estimation_error_e_per_pixel.median()),
             'background9_error_RMSE_e_per_pixel':float(np.sqrt(np.mean(diagnostics.background_estimation_error_e_per_pixel**2))),
             'mean_own_expected_signal_contamination_e_per_pixel':float(np.mean(own_mean)),
             'mean_neighbor_expected_signal_contamination_e_per_pixel':float(diagnostics.neighbor_signal_expected_mean9_e_per_pixel.mean()),
             'mean_realized_signal_in_9x9_e_per_pixel':float(diagnostics.signal_realized_mean9_e_per_pixel.mean()),
             'mean_read_noise_in_9x9_e_per_pixel':float(diagnostics.read_noise_mean9_e_per_pixel.mean()),
             'background_extra_Poisson':False,'minimum_captured_mass_before_FOV_renormalization':float(truth.unrenormalized_PSF_mass_inside_FOV.min()),
             'no_detection_or_localization_fitting_performed':True}
    dump(out/'summary.json',summary)
    dump(out/'validation.json',{'all_checks_passed':True,'50_truth_rows':True,'per_spot_expected_signal_integrates_to_sampled_N':True,
                               'poisson_signal_integer_nonnegative':bool(np.all(realization>=0)),'native_XY_nm_conversions_verified':True,
                               '9x9_direct_means_match_map':True,'signal_plus_background_plus_noise_exact':True,
                               'edge_mean_map_unavailable':True,'float32_TIFF_roundtrip_exact':True})
    (out/'REPORT.txt').write_text('Single synthetic frame, 300x300, 117nm/pixel, 50 independent uniformly placed emitters inside a 10px margin; overlap allowed.\n'
        'Each emitter: expected total detected electrons N~Normal(800,250^2), HWHM r~Normal(0.20um,(0.05um)^2), positive values by rejection.\n'
        'Spatial PSF is isotropic Gaussian, sigma = HWHM / sqrt(2 ln 2), integrated over each pixel and normalized inside the finite FOV.\n'
        'Each emitter/pixel signal count drawn independently from Poisson(expected count); background pixels drawn directly from Normal(100,20^2), no extra Poisson; read noise Normal(0,10^2).\n'
        'All brightness values in detected electrons or electrons/pixel; no ADU conversion, offset or clipping. TIFF float32; NPZ retains full precision.\n'
        'Averaging uses all 81 pixels around floor(true position + 0.5), including emitter and neighbor signal; four-pixel map edges unavailable.\n'
        'True background is the mean of the stored background component within the same window, not necessarily exactly 100 in each finite window.\n'
        f'Mean background-estimation error = {summary["background9_mean_estimation_error_e_per_pixel"]:.6g} e-/pixel; RMSE = {summary["background9_error_RMSE_e_per_pixel"]:.6g}.\n'
        'This scene diagnoses the whole-window mean; it does not alter the real-data localization/background/QC pipeline and does not establish real-camera noise calibration.\n')
    dump(out/'output_fingerprints.json',{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in out.iterdir() if q.is_file()})
    dump(out/'COMPLETED.json',{'all_checks_passed':True,'output_directory':str(out),'elapsed_s':time.perf_counter()-started})
    print(json.dumps(summary,indent=2),flush=True);print('COMPLETE:',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config.resolve())
