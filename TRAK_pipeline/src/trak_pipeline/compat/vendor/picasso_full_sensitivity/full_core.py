"""Frozen-pilot localization rules and frame-aware sufficient statistics."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'picasso_pilot'))
sys.path.insert(0,str(ROOT.parent/'TARDIS/path_irrelevant_python'))
from pilot_core import fit_candidates,identify_candidates
from pilot_qc import key_name
from tardis_tau1 import Tau1Config
from tardis_tau1.core import pair_histogram


EXCLUSIONS={
    'exclude_backend_state':lambda t:~t.fit_converged,
    'exclude_model_parameters':lambda t:~t.usable_model_fit,
    'exclude_precision':lambda t:~t.precision_model_finite,
    'exclude_psf_width':lambda t:t.suspect_psf_width,
    'exclude_residual_overlap':lambda t:t.suspect_overlap,
    'exclude_fit_zeros':lambda t:t.suspect_zero_pixel_support,
    'exclude_detector_zeros':lambda t:t.suspect_zero_detection_support,
}


def annotate(table,box,config):
    table=table.copy()
    table['box_size']=box
    table['analysis_segment']=table.frame//config['segment_frames']
    table['raw_negative_sx']=table.backend_sx_signed_px<0
    table['raw_negative_sy']=table.backend_sy_signed_px<0
    table['raw_negative_width']=table.raw_negative_sx|table.raw_negative_sy
    table['nonpositive_backend_amplitude']=table.backend_amplitude_e_per_pixel<=0
    table['negative_background']=table.bg<0
    table['center_outside_fitting_box']=table.fit_parameter_issues.str.contains('center_outside_fitting_box')
    for name,fn in EXCLUSIONS.items():table[name]=fn(table)
    table['qc_exclusion_reasons']=[';'.join(k for k in EXCLUSIONS if getattr(r,k)) for r in table.itertuples()]
    table['qc_parameter_reasons']=table.fit_parameter_issues
    np.testing.assert_array_equal(table.qc_unflagged_model_fit,~table[list(EXCLUSIONS)].any(axis=1))
    return table


def frame_histograms(selected,config):
    """T[f]=hist(frame f, f+1); B[f]=ordered nonzero same-frame hist."""
    frames=config['frames'];bins=config['distance_bins']
    edges=np.linspace(0,config['distance_max_um']*1e-6,bins+1)
    groups={int(f):g[['x','y']].to_numpy(dtype=float)*config['pixel_size_nm']*1e-9 for f,g in selected.groupby('frame')}
    total=np.zeros((frames,bins),np.int64);background=np.zeros_like(total)
    adjacent=np.zeros(frames,bool);zero_removed=np.zeros(frames,np.int64)
    for f,a in groups.items():
        d=cdist(a,a).ravel();zero_removed[f]=(d==0).sum()
        background[f]=np.histogram(d[(d>0)&(d<=edges[-1])],bins=edges)[0]
        if f+1 in groups:
            adjacent[f]=True;d=cdist(a,groups[f+1]).ravel()
            total[f]=np.histogram(d[d<=edges[-1]],bins=edges)[0]
    # Independent existing implementation must agree for all real full-film bins.
    fv=np.array(sorted(groups));fp=[groups[int(f)] for f in fv]
    tau=Tau1Config(n_bins=bins,max_distance_m=edges[-1],bootstrap_count=0,save_plots=False)
    np.testing.assert_array_equal(total.sum(axis=0),pair_histogram(fv,fp,edges,'total',tau).histogram)
    np.testing.assert_array_equal(background.sum(axis=0),pair_histogram(fv,fp,edges,'inter',tau).histogram)
    return total,background,adjacent,zero_removed,edges


def decompose_counts(total_count,background_count,config):
    """Vectorized exact same normalization and tail LS, with explicit area guard."""
    tc=np.atleast_2d(total_count).astype(float);bc=np.atleast_2d(background_count).astype(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        total=tc/tc.sum(axis=1,keepdims=True);background=bc/bc.sum(axis=1,keepdims=True)
    mids=(np.arange(config['distance_bins'])+.5)*config['distance_max_um']/config['distance_bins']
    fit=(mids>=config['background_fit_um'][0])&(mids<config['background_fit_um'][1])
    numerator=(total[:,fit]*background[:,fit]).sum(axis=1)
    denominator=(background[:,fit]**2).sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):raw=numerator/denominator
    valid=np.isfinite(raw)&(denominator>0)&(tc.sum(axis=1)>0)&(bc.sum(axis=1)>0)
    beta=np.where(valid,np.clip(raw,0,1),np.nan)
    inter=beta[:,None]*background;intra=total-inter;area=intra.sum(axis=1)
    shape_valid=valid&np.isfinite(area)&(np.abs(area)>config['signed_area_min_abs'])
    shape=np.full_like(intra,np.nan);shape[shape_valid]=intra[shape_valid]/area[shape_valid,None]
    main=(mids>=config['main_region_um'][0])&(mids<config['main_region_um'][1])
    tail=(mids>=config['tail_region_um'][0])&(mids<config['tail_region_um'][1])
    main_mass=intra[:,main].sum(axis=1);tail_mass=intra[:,tail].sum(axis=1)
    ratio=np.full(len(tc),np.nan)
    ratio_valid=valid&np.isfinite(main_mass)&(main_mass>config['main_area_min_positive'])
    ratio[ratio_valid]=tail_mass[ratio_valid]/main_mass[ratio_valid]
    return dict(total=total,background=background,inter=inter,intra=intra,shape=shape,beta=beta,beta_raw=raw,
                signed_area=area,main_mass=main_mass,tail_mass=tail_mass,tail_main_ratio=ratio,
                valid=valid,shape_valid=shape_valid,ratio_valid=ratio_valid,
                negative_mass=-np.minimum(intra,0).sum(axis=1),tail_background_count=bc[:,fit].sum(axis=1))


def range_histograms(total,background,start,stop):
    """Half-open time range: final frame may not pair outside that range."""
    return total[start:stop-1].sum(axis=0),background[start:stop].sum(axis=0)


def block_histograms(total,background,starts,length):
    return (np.stack([total[s:s+length-1].sum(axis=0) for s in starts]),
            np.stack([background[s:s+length].sum(axis=0) for s in starts]))


def shared_resampling_plan(config,length):
    nseg=config['frames']//config['segment_frames'];nblock=config['segment_frames']//length
    assert config['segment_frames']%length==0
    rng=np.random.default_rng(config['bootstrap']['seed']+length)
    weights=np.zeros((nseg,config['bootstrap']['replicates'],nblock),np.int16)
    for s in range(nseg):
        draws=rng.integers(0,nblock,size=(len(weights[s]),nblock))
        for r,d in enumerate(draws):weights[s,r]=np.bincount(d,minlength=nblock)
    assert (weights.sum(axis=2)==nblock).all()
    return weights


def bootstrap_configuration(total,background,config,length,weights):
    starts=np.arange(0,config['frames'],length)
    th,bh=block_histograms(total,background,starts,length)
    nseg=config['frames']//config['segment_frames'];nblock=config['segment_frames']//length
    th=th.reshape(nseg,nblock,-1);bh=bh.reshape(nseg,nblock,-1)
    # Repeated block occurrence multiplicities aggregate raw counts; no new adjacency.
    rt=np.einsum('srb,sbk->srk',weights,th,optimize=True)
    rb=np.einsum('srb,sbk->srk',weights,bh,optimize=True)
    results=[decompose_counts(rt.sum(axis=0),rb.sum(axis=0),config)]
    results.extend(decompose_counts(rt[s],rb[s],config) for s in range(nseg))
    controls=[decompose_counts(th.sum(axis=(0,1)),bh.sum(axis=(0,1)),config)]
    controls.extend(decompose_counts(th[s].sum(axis=0),bh[s].sum(axis=0),config) for s in range(nseg))
    return results,controls


def point_summary(table):
    result={'points':len(table)}
    for key in ['lpx_nm','lpy_nm','photons','bg','sx','sy','deviance_per_dof']:
        values=table[key].to_numpy();values=values[np.isfinite(values)]
        for label,q in [('q10',.10),('q25',.25),('median',.5),('q75',.75),('q90',.90)]:
            result[key+'_'+label]=float(np.quantile(values,q)) if len(values) else np.nan
    return result


def frame_qc(table,config,box,factor,threshold):
    index=pd.Index(np.arange(config['frames']),name='frame')
    counts=table.groupby('frame').size().reindex(index,fill_value=0).to_frame('candidates')
    flags=['qc_unflagged_model_fit','fit_converged','raw_negative_sx','raw_negative_sy','raw_negative_width',
           'nonpositive_backend_amplitude','negative_background','center_outside_fitting_box',*EXCLUSIONS]
    for flag in flags:counts[flag]=table.groupby('frame')[flag].sum().reindex(index,fill_value=0)
    kept=table.loc[table.qc_unflagged_model_fit]
    for key in ['lpx_nm','lpy_nm','photons','bg','sx','sy','deviance_per_dof']:
        counts['retained_median_'+key]=kept.groupby('frame')[key].median().reindex(index)
    counts['retained']=counts.qc_unflagged_model_fit
    counts['retained_fraction']=counts.retained/counts.candidates.replace(0,np.nan)
    counts['time_s']=index.to_numpy()*config['frame_interval_s']
    counts['segment']=index.to_numpy()//config['segment_frames']
    counts['box']=box;counts['factor']=factor;counts['absolute_threshold']=threshold
    return counts.reset_index()
