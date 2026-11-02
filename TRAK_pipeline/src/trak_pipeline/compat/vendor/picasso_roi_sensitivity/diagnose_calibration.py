"""Diagnose finite-stratum bootstrap variance using the SAME predeclared H0 toys.

Variance rescaling is diagnostic only; never changes actual primary p or bands.
"""
import numpy as np
from curve_inference import centered_max_test


def diagnose(c,weights):
    s=c['monte_carlo_validation'];rng=np.random.default_rng(s['seed'])
    ns,nr,nb=weights.shape;n=ns*nb;k=s['toy_bins']
    x=np.linspace(0,3,k,endpoint=False)+1.5/k;base=np.exp(-x/.13);base/=base.sum()*3/k
    matrix=weights.transpose(1,0,2).reshape(nr,n)/n
    sigma=np.array([1.1,1.,.9,1.05,1.])*0.10
    raw=[];rescaled=[];variance_ratios=[]
    factor=np.sqrt(nb/(nb-1))
    for movie in range(s['null_movies']):
        z=rng.normal(size=(6,n,k))
        for q in range(1,k):z[:,:,q]=.75*z[:,:,q-1]+np.sqrt(1-.75**2)*z[:,:,q]
        z-=z.mean(2,keepdims=True)
        noise=(.75*z[0][None]+.65*z[1:])*sigma[:,None,None]
        blocks=base[None,None,:]+noise
        observed=blocks.mean(1);draws=np.einsum('rn,gnk->grk',matrix,blocks,optimize=True)
        d=centered_max_test(observed,draws,np.ones((5,nr),bool),c['alpha'])
        raw.append(d['p_global']<=c['alpha'])
        rescaled.append((1+(d['null_statistics']*factor>=d['statistic']).sum())/(nr+1)<=c['alpha'])
        contrast=noise[0]-noise[1]
        unbiased=contrast.reshape(ns,nb,k).var(axis=1,ddof=1).sum(0)*nb/(n*n)
        boot=(draws[0]-draws[1]).var(0,ddof=1)
        variance_ratios.extend((boot/unbiased)[unbiased>0])
    return {'diagnostic_only':True,'primary_real_movie_p_and_bands_unchanged':True,
        'predeclared_seed_and_same_null_movies_reused':True,'source_blocks_per_stratum':nb,
        'theoretical_iid_stratum_bootstrap_variance_ratio':(nb-1)/nb,
        'observed_mean_bootstrap_to_unbiased_within_stratum_variance_ratio':float(np.mean(variance_ratios)),
        'raw_null_rejections':int(sum(raw)),'raw_null_rate':float(np.mean(raw)),
        'diagnostic_variance_factor':float(factor),'diagnostic_rescaled_null_rejections':int(sum(rescaled)),
        'diagnostic_rescaled_null_rate':float(np.mean(rescaled)),
        'interpretation':'finite ten-block strata underestimate variance; this explains part of permissive calibration, but is not a validated correction for nonlinear real-movie statistics'}
