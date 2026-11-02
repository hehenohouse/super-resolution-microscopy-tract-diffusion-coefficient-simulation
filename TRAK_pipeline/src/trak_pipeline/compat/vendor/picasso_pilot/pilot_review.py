"""Read-only review of cached fits with sign-canonical Gaussian parameters."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from picasso import localize
from scipy.ndimage import maximum_filter

from pilot_core import canonical_gaussian_parameters, model_images,detection_zero_counts

TARDIS_ROOT=Path(__file__).resolve().parent.parent/'TARDIS/path_irrelevant_python'
sys.path.insert(0,str(TARDIS_ROOT))
from tardis_tau1 import Tau1Config, analyze_positions


def review_cached_fit(original,signed_theta,box,config,frames,movie):
    """Reparameterize cached fits; no new candidate detection or optimization."""
    signed=np.asarray(signed_theta,dtype=np.float64)
    backend=signed.copy()
    with np.errstate(divide='ignore',invalid='ignore'):
        backend[:,0]=signed[:,0]/(2*np.pi*signed[:,3]*signed[:,4])
    canonical=canonical_gaussian_parameters(backend)
    ids=pd.DataFrame({'frame':original.frame.to_numpy(),'x':original.candidate_x.to_numpy(),
                      'y':original.candidate_y.to_numpy(),'net_gradient':original.net_gradient.to_numpy(),
                      'n_id':original.candidate_id.to_numpy()})
    variance=np.full((len(ids),box,box),config['camera']['read_noise_e_rms']**2,np.float32)
    converted=localize.locs_from_fits_gauss(ids,canonical,box,em=False,mle=True,
              log_likelihood=original.log_likelihood.to_numpy(),iterations=original.iterations.to_numpy(),variance=variance)
    np.testing.assert_array_equal(converted.n_id,original.candidate_id)
    revised=original.copy()
    for field in ['photons','sx','sy','lpx','lpy','ellipticity','photons_unc','bg_unc','sx_unc','sy_unc']:
        revised['v1_'+field]=original[field].to_numpy()
        revised[field]=converted[field].to_numpy()
    for field in ['usable_model_fit','qc_unflagged_model_fit','fit_parameter_issues','precision_status']:
        revised['v1_'+field]=original[field].to_numpy()
    # These coordinates and fit statistics are NOT changed by sign convention.
    np.testing.assert_array_equal(converted.x,original.x.to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(converted.y,original.y.to_numpy(dtype=np.float32))
    revised['backend_amplitude_e_per_pixel']=backend[:,0]
    revised['backend_sx_signed_px']=backend[:,3];revised['backend_sy_signed_px']=backend[:,4]
    revised['backend_signed_integral_e']=signed[:,0]
    sign=(backend[:,3]<0)|(backend[:,4]<0)
    revised['width_sign_canonicalized']=sign
    finite=np.isfinite(canonical).all(axis=1)
    center=(canonical[:,1]>=0)&(canonical[:,1]<=box-1)&(canonical[:,2]>=0)&(canonical[:,2]<=box-1)
    admissible=(revised.fit_state.to_numpy()==0)&finite&(backend[:,0]>0)&(canonical[:,3]>0)&(canonical[:,4]>0)&(canonical[:,5]>=0)&center
    revised['usable_model_fit']=admissible
    revised['precision_model_finite']=np.isfinite(revised[['lpx','lpy']]).all(axis=1)&(revised.lpx>0)&(revised.lpy>0)
    revised['suspect_psf_width']=(revised.sx<.5)|(revised.sy<.5)|(revised.sx>box/2)|(revised.sy>box/2)
    revised['sign_only_parameter_issue']=sign&admissible
    # Model invariance is checked for every cached candidate, including rejected ones.
    old_model=model_images(signed,box);new_model=model_images(canonical,box)
    np.testing.assert_allclose(old_model,new_model,rtol=1e-10,atol=1e-8,equal_nan=True)
    max_change=np.nanmax(np.abs(old_model-new_model)) if len(ids) else 0.
    image_index={int(frame):i for i,frame in enumerate(frames)};half=box//2
    raw=np.stack([movie[image_index[int(r.frame)],int(r.candidate_y)-half:int(r.candidate_y)+half+1,
                    int(r.candidate_x)-half:int(r.candidate_x)+half+1] for r in original.itertuples()]).astype(float)
    data=(raw-config['camera']['baseline_adu'])*config['camera']['sensitivity_e_per_adu']/config['camera']['em_gain']
    residual=data-new_model
    yy,xx=np.mgrid[:box,:box]
    distance=np.sqrt((xx-canonical[:,1,None,None])**2+(yy-canonical[:,2,None,None])**2)
    peaks=(residual==maximum_filter(residual,size=(1,3,3)))&(distance>2)&(residual>4*np.sqrt(np.maximum(new_model+variance,1)))
    peaks[:,0,:]=peaks[:,-1,:]=False;peaks[:,:,0]=peaks[:,:,-1]=False
    revised['suspect_overlap']=peaks.any(axis=(1,2))
    revised['qc_unflagged_model_fit']=admissible&revised.precision_model_finite&~revised.suspect_psf_width&~revised.suspect_overlap&~revised.suspect_zero_pixel_support
    revised['unflagged_before_detection_padding_check']=revised.qc_unflagged_model_fit
    revised['rescued_by_sign_canonicalization']=revised.qc_unflagged_model_fit&~revised.v1_qc_unflagged_model_fit
    revised['source_zero_pixels_detection_support']=detection_zero_counts(ids,frames,movie,box)
    revised['suspect_zero_detection_support']=revised.source_zero_pixels_detection_support>0
    revised['qc_unflagged_model_fit']&=~revised.suspect_zero_detection_support
    revised['precision_status']=np.select([~admissible,~revised.precision_model_finite,~revised.qc_unflagged_model_fit],
                    ['unavailable_or_invalid_fit','nonfinite_model_bound','requires_image_review'],default='nominal_model_only')
    flags={'backend_not_converged':revised.fit_state.to_numpy()!=0,'nonfinite_parameters':~finite,
           'nonpositive_backend_amplitude':backend[:,0]<=0,'zero_width':(canonical[:,3]<=0)|(canonical[:,4]<=0),
           'negative_background':canonical[:,5]<0,'center_outside_fitting_box':~center}
    revised['fit_parameter_issues']=[';'.join(k for k,v in flags.items() if v[i]) for i in range(len(ids))]
    for key in ['photons','sx','sy','lpx','lpy']:
        if key!='photons':revised[key+'_nm']=revised[key]*config['pixel_size_nm']
    stats={'raw_rows':len(original),'raw_negative_width':int(sign.sum()),
           'raw_nonpositive_signed_integral':int((signed[:,0]<=0).sum()),
           'nonpositive_backend_amplitude':int((backend[:,0]<=0).sum()),
           'negative_background':int((canonical[:,5]<0).sum()),
           'center_outside_fitting_box':int((~center).sum()),
           'old_inadmissible':int((~original.usable_model_fit).sum()),
           'revised_inadmissible':int((~admissible).sum()),
           'sign_only_parameter_issue':int(revised.sign_only_parameter_issue.sum()),
           'sign_only_still_width_flagged':int((revised.sign_only_parameter_issue&revised.suspect_psf_width).sum()),
           'canonical_unflagged_before_padding_check':int(revised.unflagged_before_detection_padding_check.sum()),
           'revised_unflagged':int(revised.qc_unflagged_model_fit.sum()),
           'sign_rescued_unflagged':int(revised.rescued_by_sign_canonicalization.sum()),
           'sign_rescued_with_zero_detection_support':int((revised.rescued_by_sign_canonicalization&revised.suspect_zero_detection_support).sum()),
           'newly_flagged_detection_padding':int((revised.unflagged_before_detection_padding_check&revised.suspect_zero_detection_support).sum()),
           'model_max_abs_change_nominal_e_per_pixel':float(max_change),'coordinates_unchanged':True,
           'reason_counts_overlap':True}
    return revised,canonical,stats


def brute_pair_histograms(positions,edges):
    """Independent explicit pair enumerator; original frame labels define lag."""
    group={int(f):positions[positions[:,0]==f,1:] for f in np.unique(positions[:,0])}
    total=[];background=[];source_pairs=[]
    for frame,a in group.items():
        if frame+1 in group:
            b=group[frame+1];source_pairs.append((frame,frame+1))
            for p in a:
                for q in b:
                    d=float(np.sqrt(np.sum((p-q)**2)))
                    if d<=edges[-1]:total.append(d)
        for p in a:
            for q in a:
                d=float(np.sqrt(np.sum((p-q)**2)))
                if 0<d<=edges[-1]:background.append(d)
    return np.histogram(total,bins=edges)[0],np.histogram(background,bins=edges)[0],source_pairs


def analyze_continuous_subset(table,config,frame_start=0,frame_end=49):
    if (frame_start,frame_end)!=(0,49):raise ValueError('This review is restricted to the approved continuous 0–49 segment')
    selected=table.loc[table.frame.between(frame_start,frame_end)&table.qc_unflagged_model_fit].copy()
    pos=np.column_stack([selected.frame,selected.x*config['pixel_size_nm']*1e-9,selected.y*config['pixel_size_nm']*1e-9])
    tauconfig=Tau1Config(save_plots=False,bootstrap_count=0)
    result=analyze_positions(pos,ids={'condition':'TDKd3','target':'TRAK1','cell':'T1_C2_0_49'},
                frame_time_s=config['frame_interval_s'],config=tauconfig,
                metadata={'calibration_status':'manufacturer_nominal','quantitatively_calibrated':False,
                          'selection':'canonical Gaussian parameters, heuristic unflagged, original frames 0–49, full TIFF field; no ROI or mitochondrial mask',
                          'uncertainty_correction':'none; lpx/lpy and empirical 50 nm are not used to deconvolve the distribution'})
    edges=result.parameters['edges_m']
    total,background,pairs=brute_pair_histograms(pos,edges)
    np.testing.assert_array_equal(total,result.counts['total_histogram'])
    np.testing.assert_array_equal(background,result.counts['inter_histogram'])
    assert all(b==a+1 and 0<=a<b<=49 for a,b in pairs)
    assert len(pairs)==result.counts['total_source_frame_pairs']
    return result,selected,pos,pairs,tauconfig
