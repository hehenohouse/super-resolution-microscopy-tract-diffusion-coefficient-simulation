"""Picasso MLE at existing TrackMate candidates; no detection or score threshold."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from picasso import localize
from picasso.fitting import gaussfit, seeds
from scipy.ndimage import maximum_filter

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'picasso_pilot'))
from pilot_core import canonical_gaussian_parameters, model_images, FIT_STATES


def refine_candidates(table, frames, movie, config, camera):
    """Return one row per candidate, preserving unsupported and failed rows.

    Integer anchors extract native pixels; the MLE position seed remains the
    floating TrackMate center. Only width signs are canonicalized. Negative
    amplitude or background is never made positive. Per-axis bounds come from
    Picasso's Fisher information for the fitted model and supplied variance.
    """
    table=table.reset_index(drop=True)
    box=int(config['box']);half=box//2
    assert box>=3
    right=box-half
    frame_index={int(f):i for i,f in enumerate(frames)}
    xy=table[['x_px','y_px']].to_numpy(float)
    finite=np.isfinite(xy).all(axis=1)
    anchors=np.zeros(xy.shape,np.int64)
    anchors[finite]=np.floor(xy[finite]+.5).astype(np.int64)
    valid=finite & table.frame.isin(frame_index).to_numpy()
    valid &= (anchors[:,0]>=half)&(anchors[:,1]>=half)
    valid &= (anchors[:,0]+right<=movie.shape[2])&(anchors[:,1]+right<=movie.shape[1])
    result=pd.DataFrame({'spot_id':table.spot_id,'frame':table.frame,
                         'fit_anchor_x_px':anchors[:,0],'fit_anchor_y_px':anchors[:,1],
                         'fit_box_origin_x_px':anchors[:,0]-half,'fit_box_origin_y_px':anchors[:,1]-half,
                         'fit_box_geometric_center_x_px':anchors[:,0]-half+(box-1)/2,
                         'fit_box_geometric_center_y_px':anchors[:,1]-half+(box-1)/2,
                         'fit_box_size_px':box,
                         'fit_support_valid':valid,'fit_attempted':valid})
    used=np.flatnonzero(valid)
    if not len(used):
        result['precision_status']='unavailable_fitting_support_or_position'
        return result,np.empty((0,6)),np.empty((0,box,box),np.float32),used
    raw=np.stack([movie[frame_index[int(table.frame.iloc[i])],
                 anchors[i,1]-half:anchors[i,1]-half+box,
                 anchors[i,0]-half:anchors[i,0]-half+box] for i in used]).astype(np.float32)
    assert raw.shape==(len(used),box,box)
    converted=localize._to_photons(raw,{'Baseline':camera['baseline_adu'],
                'Sensitivity':camera['sensitivity_e_per_adu'],'Gain':camera['em_gain']})
    variance=np.full(converted.shape,camera['read_noise_e_rms']**2,np.float32)
    data=np.maximum(converted,-variance)
    initial=seeds.initial_parameters_gauss(np.maximum(data,0),box).astype(np.float64)
    initial[:,1:3]=xy[used]-anchors[used]+half
    assert np.isfinite(initial).all()
    outputs=[]
    batch=int(config.get('fit_batch_size',512))
    for start in range(0,len(used),batch):
        stop=min(start+batch,len(used))
        values=gaussfit.fit_spots(gaussfit.ELLIPTIC,data[start:stop],initial[start:stop],
                     mle=True,tolerance=config['fit_tolerance'],
                     max_iterations=config['fit_max_iterations'],variance=variance[start:stop])
        outputs.append(values)
        print(f'Picasso MLE: {stop}/{len(used)} TrackMate ROI candidates fitted',flush=True)
    backend,objective,states,iterations=[np.concatenate([v[i] for v in outputs]) for i in range(4)]
    canonical=canonical_gaussian_parameters(backend)
    ids=pd.DataFrame({'frame':table.frame.iloc[used].to_numpy(),
          'x':anchors[used,0],'y':anchors[used,1],
          # This placeholder is not computed or tested: native converter expects
          # the column, but there is no Picasso identification in this pipeline.
          'net_gradient':np.full(len(used),np.nan),
          'n_id':used.astype(np.uint32)})
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        locs=localize.locs_from_fits_gauss(ids,canonical,box,em=False,mle=True,
              log_likelihood=-.5*objective,iterations=iterations,variance=variance)
    np.testing.assert_array_equal(locs.n_id,used)
    np.testing.assert_allclose(model_images(canonical,box),
       model_images(np.column_stack([backend[:,0]*2*np.pi*backend[:,3]*backend[:,4],backend[:,1:]]),box),
       rtol=1e-10,atol=1e-7,equal_nan=True)
    fitted=locs.drop(columns=['frame','net_gradient','n_id']).rename(columns={'x':'fit_x_px','y':'fit_y_px'})
    fitted.index=used
    # Native converter subtracts floor(box/2). Verify both odd and even boxes
    # restore the source pixel grid from the recorded top-left origin.
    np.testing.assert_array_equal(fitted.fit_x_px.to_numpy(),
        (canonical[:,1]+anchors[used,0]-half).astype(np.float32))
    np.testing.assert_array_equal(fitted.fit_y_px.to_numpy(),
        (canonical[:,2]+anchors[used,1]-half).astype(np.float32))
    for field in ['fit_x_px','fit_y_px','sx','sy','lpx','lpy']:
        fitted[field.replace('_px','')+'_nm']=fitted[field]*config['pixel_size_nm']
    fitted['fit_state']=states
    fitted['fit_state_label']=[FIT_STATES.get(int(s),'unknown') for s in states]
    fitted['fit_converged']=states==0
    fitted['fit_deviance']=objective
    fitted['deviance_per_dof']=objective/(box*box-6)
    fitted['backend_amplitude_e_per_pixel']=backend[:,0]
    fitted['backend_sx_signed_px']=backend[:,3]
    fitted['backend_sy_signed_px']=backend[:,4]
    fitted['raw_negative_width']=(backend[:,3]<0)|(backend[:,4]<0)
    fitted['width_sign_canonicalized']=fitted.raw_negative_width
    fitted['backend_signed_integral_e']=backend[:,0]*2*np.pi*backend[:,3]*backend[:,4]
    finite_parameters=np.isfinite(canonical).all(axis=1)
    outside=(canonical[:,1]<0)|(canonical[:,1]>box-1)|(canonical[:,2]<0)|(canonical[:,2]>box-1)
    flags={'backend_not_converged':states!=0,'nonfinite_parameters':~finite_parameters,
           'nonpositive_amplitude':backend[:,0]<=0,
           'zero_width':(canonical[:,3]<=0)|(canonical[:,4]<=0),
           'negative_background':canonical[:,5]<0,'center_outside_fitting_box':outside}
    fitted['fit_parameter_issues']=[';'.join(k for k,v in flags.items() if v[i]) for i in range(len(used))]
    fitted['usable_model_fit']=~np.column_stack(list(flags.values())).any(axis=1)
    fitted['precision_model_finite']=np.isfinite(fitted[['lpx','lpy']]).all(axis=1)&(fitted.lpx>0)&(fitted.lpy>0)
    fitted['nominal_precision_valid']=fitted.usable_model_fit&fitted.precision_model_finite
    fitted['suspect_psf_width']=(fitted.sx<.5)|(fitted.sy<.5)|(fitted.sx>box/2)|(fitted.sy>box/2)
    predicted=model_images(canonical,box)
    residual=converted-predicted
    fitted['model_residual_rms_e']=np.sqrt(np.mean(residual**2,axis=(1,2)))
    yy,xx=np.mgrid[:box,:box]
    distance=np.hypot(xx-canonical[:,1,None,None],yy-canonical[:,2,None,None])
    peaks=(residual==maximum_filter(residual,size=(1,3,3)))&(distance>2)
    peaks &= residual>4*np.sqrt(np.maximum(predicted+variance,1))
    peaks[:,0,:]=peaks[:,-1,:]=False;peaks[:,:,0]=peaks[:,:,-1]=False
    fitted['suspect_overlap']=peaks.any(axis=(1,2))
    fitted['source_zero_pixels_fit_box']=(raw==0).sum(axis=(1,2))
    fitted['source_pixels_below_assumed_offset']=(raw<camera['baseline_adu']).sum(axis=(1,2))
    fitted['mle_pixels_clipped_to_noise_floor']=(converted < -variance).sum(axis=(1,2))
    fitted['qc_unflagged_model_fit']=fitted.nominal_precision_valid&~fitted.suspect_psf_width&~fitted.suspect_overlap&(fitted.source_zero_pixels_fit_box==0)
    review={'suspect_psf_width':fitted.suspect_psf_width.to_numpy(),
            'suspect_residual_overlap':fitted.suspect_overlap.to_numpy(),
            'source_fit_box_contains_zeros':(fitted.source_zero_pixels_fit_box>0).to_numpy()}
    fitted['fit_review_flags']=[';'.join(k for k,v in review.items() if v[i]) for i in range(len(used))]
    fitted['precision_status']=np.select([~fitted.usable_model_fit,~fitted.precision_model_finite,
                     ~fitted.qc_unflagged_model_fit],
             ['unavailable_or_invalid_fit','nonfinite_model_bound','nominal_model_requires_review'],
             default='nominal_model_only')
    fitted['refinement_shift_px']=np.hypot(fitted.fit_x_px.to_numpy()-xy[used,0],fitted.fit_y_px.to_numpy()-xy[used,1])
    # All input rows survive the left join, including unsupported candidates.
    result=result.join(fitted)
    result.loc[~valid,'precision_status']='unavailable_fitting_support_or_position'
    result['calibration_status']=camera['calibration_status']
    result['quantitatively_calibrated']=False
    return result,canonical,raw,used
