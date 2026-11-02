"""Toy calibration with joint correlated signed curves and known local alternatives.

Independent temporal blocks have shared group noise and correlated distance bins.
This validates the maxT implementation, not the TIFF camera model or real block mixing.
"""
import numpy as np
from scipy.stats import binomtest
from curve_inference import centered_max_test


def simulate(c,weights,progress=None):
    settings=c['monte_carlo_validation'];rng=np.random.default_rng(settings['seed'])
    k=settings['toy_bins'];ns,nr,nb=weights.shape;n=ns*nb
    assert nr==settings['replicates']
    x=np.linspace(0,3,k,endpoint=False)+1.5/k
    base=np.exp(-x/.13);base/=base.sum()*3/k
    pattern=np.exp(-((x-.12)/.06)**2);pattern-=pattern.mean();pattern/=np.abs(pattern).max()
    matrix=weights.transpose(1,0,2).reshape(nr,n)/n
    sigma=np.array([1.1,1.,.9,1.05,1.])*0.10
    null_p=[];coverage=[];alternative=[];bad_uncentered=[]
    count=settings['null_movies']+settings['alternative_movies']
    for movie in range(count):
        z=rng.normal(size=(6,n,k))
        for bin_index in range(1,k):z[:,:,bin_index]=.75*z[:,:,bin_index-1]+np.sqrt(1-.75**2)*z[:,:,bin_index]
        z-=z.mean(2,keepdims=True)
        noise=(.75*z[0][None]+.65*z[1:])*sigma[:,None,None]
        block_curves=base[None,None,:]+noise
        if movie>=settings['null_movies']:block_curves[0]+=0.12*pattern
        observed=block_curves.mean(1)
        draws=np.einsum('rn,gnk->grk',matrix,block_curves,optimize=True)
        result=centered_max_test(observed,draws,np.ones((5,nr),bool),c['alpha'])
        if movie<settings['null_movies']:
            null_p.append(result['p_global']);coverage.append(bool((np.abs(result['differences'])<=result['critical']).all()))
        else:
            alternative.append(result['p_global'])
            # Deliberately wrong uncentered reference: observed shifts contaminate the null.
            wrong=np.ptp(draws,axis=0).max(1)
            bad_uncentered.append((1+(wrong>=result['statistic']).sum())/(nr+1))
        if progress and (movie+1)%50==0:progress(f'Calibration toy movies: {movie+1}/{count}')
    reject=int(np.sum(np.array(null_p)<=c['alpha']));total=len(null_p)
    interval=binomtest(reject,total).proportion_ci(confidence_level=.95,method='exact')
    return {'model':'independent temporal blocks; shared group noise; spatially correlated bins; signed unit-area curves',
        'settings':settings,'null_rejections':reject,'null_movies':total,'null_rejection_rate':reject/total,
        'null_rejection_rate_95_binomial_ci':[interval.low,interval.high],
        'nominal_alpha_in_null_rate_interval':bool(interval.low<=c['alpha']<=interval.high),
        'family_band_null_coverage':float(np.mean(coverage)),
        'known_alternative_rejection_rate':float(np.mean(np.array(alternative)<=c['alpha'])),
        'incorrect_uncentered_alternative_rejection_rate':float(np.mean(np.array(bad_uncentered)<=c['alpha'])),
        'null_p_values':null_p,'alternative_p_values':alternative,
        'limitation':'linear signed-curve toy validation; not a validation of camera calibration, nonlinear beta regularity, or real temporal dependence'}
