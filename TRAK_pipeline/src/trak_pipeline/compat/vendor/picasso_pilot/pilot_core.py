"""Small-sample Picasso adapters; no tracking, masking, or TARDIS imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from picasso import localize
from picasso.fitting import gaussfit, seeds
from scipy.ndimage import maximum_filter
import tifffile
import yaml


FIT_STATES = {0: 'converged', 1: 'maximum_iterations', 2: 'singular_hessian',
              3: 'negative_curvature_mle'}

FIELD_UNITS = {
    'frame': 'original zero-based TIFF page (time)', 'time_s': 's',
    'x': 'camera pixel, column, center of first pixel = 0',
    'y': 'camera pixel, row, center of first pixel = 0',
    'x_nm': 'nm, relative to TIFF image origin', 'y_nm': 'nm, relative to TIFF image origin',
    'photons': 'detected photoelectron-equivalent count, manufacturer-nominal conversion',
    'bg': 'detected photoelectron equivalents / pixel, excluding assumed camera offset',
    'sx': 'Gaussian standard deviation, camera pixels', 'sy': 'Gaussian standard deviation, camera pixels',
    'sx_nm': 'Gaussian standard deviation, nm', 'sy_nm': 'Gaussian standard deviation, nm',
    'lpx': 'sqrt(model Fisher CRLB x variance), camera pixels; uncalibrated',
    'lpy': 'sqrt(model Fisher CRLB y variance), camera pixels; uncalibrated',
    'lpx_nm': 'nm; uncalibrated model estimate', 'lpy_nm': 'nm; uncalibrated model estimate',
    'photons_unc': 'detected photoelectron equivalents; model bound',
    'bg_unc': 'detected photoelectron equivalents / pixel; model bound',
    'sx_unc': 'camera pixels; model bound', 'sy_unc': 'camera pixels; model bound',
    'net_gradient': 'ADU gradient sum, detection on unmodified raw TIFF',
    'candidate_x': 'integer camera pixel, column', 'candidate_y': 'integer camera pixel, row',
    'candidate_id': 'identifier within box-size candidate table',
    'fit_state': 'Picasso backend integer status; see fit_state_labels',
    'fit_state_label': 'Picasso/Gpufit backend status name',
    'calibration_status': 'manufacturer_nominal; no acquisition-specific measured calibration',
    'iterations': 'count', 'fit_deviance': 'dimensionless shifted-Poisson deviance',
    'log_likelihood': 'Picasso convention: -0.5 * shifted-Poisson deviance (relative to saturated model)',
    'model_residual_rms_e': 'detected photoelectron equivalents / pixel',
    'deviance_per_dof': 'deviance / (box_pixels - 6); descriptive, not a p-value',
    'ellipticity': 'dimensionless', 'reference_segment': '0=start, 1=middle, 2=end',
    'fit_converged': 'boolean, backend state==0', 'usable_model_fit': 'converged and physically finite within fit box',
    'precision_model_finite': 'finite positive CRLB square roots under nominal model',
    'quantitatively_calibrated': 'false for all rows in this pilot',
    'suspect_overlap': 'heuristic flag: distant positive residual peak; not confirmed overlap',
    'suspect_psf_width': 'heuristic flag, sigma<0.5 px or sigma>half box',
    'source_pixels_below_assumed_offset': 'count in fitting box; clips are recorded',
    'mle_pixels_clipped_to_noise_floor': 'count clipped to -nominal read-noise variance for MLE',
    'source_zero_pixels': 'count of exactly zero original TIFF pixels in fitting box',
    'suspect_zero_pixel_support': 'boolean: fitting box contains zeros; see persistent-zero source diagnostic',
    'qc_unflagged_model_fit': 'admissible fit with finite precision and no width, residual-overlap, fitting-patch-zero or detector-support-zero flags; not calibrated or experimentally validated',
    'fit_parameter_issues': 'semicolon-separated reasons fitted parameters are inadmissible; empty if admissible',
    'precision_status': 'unavailable_or_invalid_fit / nonfinite_model_bound / requires_image_review / nominal_model_only; never acquisition-calibrated',
    'backend_amplitude_e_per_pixel': 'raw Gaussian peak-height above background, nominal detected electrons/pixel',
    'backend_sx_signed_px': 'raw signed backend width parameter in pixels; sign is model-degenerate',
    'backend_sy_signed_px': 'raw signed backend width parameter in pixels; sign is model-degenerate',
    'backend_signed_integral_e': 'raw amplitude * 2*pi * signed sx * signed sy; not physical signal if width signs differ',
    'width_sign_canonicalized': 'boolean, abs(sx/sy) reparameterization with unchanged model image and fitted coordinates',
    'source_zero_pixels_detection_support': 'count of original zero pixels read by local-maximum/finite-difference detector; box plus one-pixel axial halo, no corners',
    'suspect_zero_detection_support': 'detection reads original zero pixels even if fit patch does not; cause of zeros unverified',
}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def sample_ids(config):
    segments = config['sample_segments']
    ids = np.concatenate([np.arange(int(a),int(b)+1) for a,b in segments])
    if not (0<len(ids)<=200) or len(np.unique(ids))!=len(ids) or np.any(np.diff(ids)<=0):
        raise ValueError('Pilot requires 1–200 sorted, distinct source frames; full-movie processing is disabled')
    if ids[0]<0 or ids[-1]>=int(config['expected_frames']):
        raise ValueError('Requested source frame is out of bounds')
    if not set(config['threshold_reference_frames']).issubset(set(ids)):
        raise ValueError('Threshold reference frames must belong to the selected sample')
    return ids.astype(np.uint32)


def inspect_and_sample(path, config, out):
    path = Path(path)
    stat = path.stat()
    print('Hashing input TIFF (read-only)...', flush=True)
    source_hash = sha256_file(path)
    ids = sample_ids(config)
    arrays = []
    per_frame = []
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        metadata = {'source_path': str(path.resolve()), 'source_sha256': source_hash,
                    'source_bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,
                    'page_count':len(tif.pages), 'series_shape':list(series.shape), 'series_axes':series.axes,
                    'imagej_metadata':tif.imagej_metadata, 'first_page_description':tif.pages[0].description,
                    'bits_per_sample':int(tif.pages[0].bitspersample),
                    'samples_per_pixel':int(tif.pages[0].samplesperpixel),
                    'compression':str(tif.pages[0].compression), 'photometric':str(tif.pages[0].photometric),
                    'dtype':np.dtype(tif.pages[0].dtype).name,
                    'interpretation':'TIFF pages are time; ImageJ spacing does not make this a Z stack',
                    'channel':'C2 target; separate grayscale TIFF; LOCB channel 1 identity supplied by user',
                    'pixel_size_nm':config['pixel_size_nm'],'pixel_size_source':'117 nm/pixel confirmed by user; consistent with XY TIFF resolution',
                    'actual_exposure_time_s':None, 'frame_interval_s':config['frame_interval_s'],
                    'calibration_status':'manufacturer_nominal'}
        assert len(tif.pages)==int(config['expected_frames'])
        assert tuple(series.shape)==(config['expected_frames'],config['expected_height'],config['expected_width'])
        assert series.axes=='TYX', f'Unexpected axes: {series.axes}'
        assert metadata['dtype']==config['expected_dtype'] and metadata['samples_per_pixel']==1
        assert metadata['bits_per_sample']==16
        ij=tif.imagej_metadata or {}
        assert int(ij.get('frames',0))==config['expected_frames']
        assert abs(float(ij.get('finterval',0))-config['frame_interval_s'])<1e-10
        for axis,tag in [('x','XResolution'),('y','YResolution')]:
            numerator,denominator=tif.pages[0].tags[tag].value
            pixel_nm=1000*denominator/numerator
            metadata[f'{axis}_resolution_pixel_nm']=pixel_nm
            assert abs(pixel_nm-config['pixel_size_nm'])<.001
        for frame in ids:
            a=tif.pages[int(frame)].asarray()
            assert a.shape==(config['expected_height'],config['expected_width'])
            arrays.append(a.astype(np.uint16,copy=False))
            per_frame.append({'frame':int(frame), 'time_s':int(frame)*config['frame_interval_s'],
                              'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                              'min_adu':int(a.min()),'median_adu':float(np.median(a)), 'max_adu':int(a.max()),
                              'saturated_pixels':int((a==65535).sum()),
                              'pixels_below_nominal_offset':int((a<config['camera']['baseline_adu']).sum())})
    movie=np.stack(arrays)
    persistent_zeros=np.all(movie==0,axis=0)
    metadata['persistent_zero_pixel_count']=int(persistent_zeros.sum())
    metadata['persistent_zero_rows']={str(i):int(v) for i,v in enumerate(persistent_zeros.sum(axis=1)) if v}
    metadata['persistent_zero_columns']={str(i):int(v) for i,v in enumerate(persistent_zeros.sum(axis=0)) if v}
    metadata['persistent_zero_interpretation']='Static exact-zero pixels, predominantly image edges; possible export/calibration padding. Cause not established. No pixels are masked or changed.'
    # Store pixels without rescaling, with an explicit original-page mapping.
    with h5py.File(out/'sample_frames.hdf5','w') as h:
        h.create_dataset('images',data=movie,chunks=(1,*movie.shape[1:]),compression='gzip',compression_opts=1)
        h.create_dataset('original_frame_indices',data=ids)
        h.create_dataset('persistent_zero_pixel_flags',data=persistent_zeros,compression='gzip')
        h.attrs['axes']='TYX'; h.attrs['intensity_unit']='ADU (assumed; acquisition/export provenance unverified)'
        h.attrs['source_sha256']=source_hash
    pd.DataFrame(per_frame).to_csv(out/'sample_frames.csv',index=False)
    metadata['selected_frame_count']=len(ids)
    metadata['selected_segments_inclusive']=config['sample_segments']
    dump_json(out/'input_metadata.json',metadata)
    print(f'Loaded only {len(ids)} source frames; axes={series.axes}, shape={movie.shape}.',flush=True)
    return ids,movie,metadata


def verify_sample(path, out, metadata):
    with h5py.File(out/'sample_frames.hdf5','r') as saved, tifffile.TiffFile(path) as tif:
        for i,frame in enumerate(saved['original_frame_indices'][()]):
            assert np.array_equal(saved['images'][i],tif.pages[int(frame)].asarray()), f'Pixel mismatch at {frame}'
    stat=Path(path).stat()
    assert stat.st_size==metadata['source_bytes'] and stat.st_mtime_ns==metadata['source_mtime_ns']
    return {'all_selected_frames_pixel_exact':True,'source_size_and_mtime_unchanged':True,
            'selected_frame_count':metadata['selected_frame_count']}


def identify_candidates(ids, movie, box):
    rows=[]; borders=[]
    margin=box//2+1  # Extra pixel for finite-difference detection gradients.
    for i,(frame,image) in enumerate(zip(ids,movie)):
        y,x,ng=localize.identify_in_image(image.astype(np.int32),-np.inf,box)
        inside=(x>=margin)&(y>=margin)&(x<image.shape[1]-margin)&(y<image.shape[0]-margin)
        borders.append({'frame':int(frame),'box':box,'boundary_candidates_excluded':int((~inside).sum()),
                        'excluded_margin_pixels':margin})
        rows.extend((int(frame),int(xi),int(yi),float(g)) for xi,yi,g in zip(x[inside],y[inside],ng[inside]) if np.isfinite(g))
        if (i+1)%25==0: print(f'Box {box}: detected candidates in {i+1}/{len(ids)} sampled frames.',flush=True)
    candidates=pd.DataFrame(rows,columns=['frame','x','y','net_gradient'])
    candidates['n_id']=np.arange(len(candidates),dtype=np.uint32)
    return candidates,pd.DataFrame(borders)


def baseline_threshold(candidates, config):
    ng=candidates.loc[candidates.frame.isin(config['threshold_reference_frames']),'net_gradient'].to_numpy()
    if len(ng)==0: raise ValueError('No threshold-reference candidates; inspect the raw sample')
    median=float(np.median(ng)); mad=float(np.median(np.abs(ng-median)))
    robust_sd=1.4826*mad
    threshold=median+config['threshold_mad_multiplier']*robust_sd
    if not np.isfinite(threshold) or threshold<=0: raise ValueError('Degenerate reference net-gradient distribution')
    return threshold,{'candidate_count':len(ng),'median':median,'mad':mad,'robust_sd':robust_sd,
                      'baseline_threshold_adu_gradient':threshold}


def cut_raw_spots(identifications, ids, movie, box):
    images={int(frame):image for frame,image in zip(ids,movie)}
    half=box//2
    result=np.empty((len(identifications),box,box),np.float32)
    for i,row in enumerate(identifications.itertuples(index=False)):
        result[i]=images[int(row.frame)][int(row.y)-half:int(row.y)+half+1,int(row.x)-half:int(row.x)+half+1]
    return result


def model_images(theta, box):
    yy,xx=np.mgrid[:box,:box]
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        amplitude=theta[:,0]/(2*np.pi*theta[:,3]*theta[:,4])
        return theta[:,5,None,None]+amplitude[:,None,None]*np.exp(-.5*((xx-theta[:,1,None,None])**2/theta[:,3,None,None]**2+
                                                        (yy-theta[:,2,None,None])**2/theta[:,4,None,None]**2))


def canonical_gaussian_parameters(backend_theta):
    """Peak-height backend -> positive-width integrated-signal parameters.

    This is a model-preserving reparameterization, not a new fit or abs(signal).
    A genuinely negative amplitude remains negative after conversion.
    """
    backend=np.asarray(backend_theta,dtype=np.float64)
    canonical=backend.copy()
    canonical[:,3:5]=np.abs(backend[:,3:5])
    canonical[:,0]=backend[:,0]*(2*np.pi*canonical[:,3]*canonical[:,4])
    return canonical


def detection_zero_counts(identifications,ids,movie,box):
    image_index={int(f):i for i,f in enumerate(ids)}
    margin=box//2+1;support=np.ones((box+2,box+2),bool)
    support[0,0]=support[0,-1]=support[-1,0]=support[-1,-1]=False
    return np.array([int(((movie[image_index[int(r.frame)],int(r.y)-margin:int(r.y)+margin+1,
                  int(r.x)-margin:int(r.x)+margin+1]==0)&support).sum()) for r in identifications.itertuples()],dtype=np.int32)


def fit_candidates(identifications, ids, movie, box, config):
    started=time.perf_counter()
    cam=config['camera']; pixel=config['pixel_size_nm']
    raw=cut_raw_spots(identifications,ids,movie,box)
    # Pass explicit identity of the nominal conversion; do not load global camera defaults.
    spots=localize._to_photons(raw,{'Baseline':cam['baseline_adu'], 'Sensitivity':cam['sensitivity_e_per_adu'], 'Gain':cam['em_gain']})
    original_converted_spots=spots.copy()
    nominal_variance=cam['read_noise_e_rms']**2
    variance=np.full(spots.shape,nominal_variance,dtype=np.float32)
    clipped=spots < -variance
    below=(raw<cam['baseline_adu']).sum(axis=(1,2))
    spots=np.maximum(spots,-variance)
    if len(spots):
        initial=seeds.initial_parameters_gauss(np.maximum(spots,0),box).astype(np.float64)
        theta,objective,states,iterations=gaussfit.fit_spots(gaussfit.ELLIPTIC,spots,initial,mle=True,
                         tolerance=config['fit']['tolerance'],max_iterations=config['fit']['max_iterations'],variance=variance)
        backend_theta=theta.astype(np.float64)
        theta=canonical_gaussian_parameters(backend_theta)
        locs=localize.locs_from_fits_gauss(identifications,theta,box,em=False,mle=True,
                                         log_likelihood=-.5*objective,iterations=iterations,variance=variance)
        assert np.array_equal(locs.n_id.to_numpy(),identifications.n_id.to_numpy()),'Backend rows reordered'
    else:
        theta=np.empty((0,6));backend_theta=np.empty((0,6));objective=np.empty(0);states=np.empty(0,np.int32);iterations=np.empty(0,np.int32)
        locs=pd.DataFrame({k:pd.Series(dtype=np.float64) for k in ['frame','x','y','photons','bg','sx','sy','lpx','lpy','photons_unc','bg_unc','sx_unc','sy_unc','net_gradient','log_likelihood','iterations','n_id','ellipticity']})
    locs=locs.rename(columns={'n_id':'candidate_id'})
    locs['frame']=locs.frame.astype(np.uint32)
    locs['candidate_x']=identifications.x.to_numpy(dtype=np.int32)
    locs['candidate_y']=identifications.y.to_numpy(dtype=np.int32)
    locs['backend_amplitude_e_per_pixel']=backend_theta[:,0]
    locs['backend_sx_signed_px']=backend_theta[:,3]
    locs['backend_sy_signed_px']=backend_theta[:,4]
    locs['backend_signed_integral_e']=backend_theta[:,0]*2*np.pi*backend_theta[:,3]*backend_theta[:,4]
    locs['width_sign_canonicalized']=(backend_theta[:,3]<0)|(backend_theta[:,4]<0)
    locs['fit_state']=states
    locs['fit_state_label']=[FIT_STATES.get(int(v),'unknown') for v in states]
    locs['fit_converged']=states==0
    locs['fit_deviance']=objective
    locs['deviance_per_dof']=objective/(box*box-6)
    locs['time_s']=locs.frame*config['frame_interval_s']
    locs['reference_segment']=np.array([next(i for i,(a,b) in enumerate(config['sample_segments']) if a<=int(f)<=b) for f in locs.frame],dtype=np.int32)
    for key in ('x','y','sx','sy','lpx','lpy'): locs[f'{key}_nm']=locs[key]*pixel
    locs['calibration_status']=cam['calibration_status']
    locs['quantitatively_calibrated']=False
    finite=np.isfinite(locs[['x','y','photons','bg','sx','sy']].to_numpy()).all(axis=1)
    physical=(theta[:,0]>0)&(theta[:,3]>0)&(theta[:,4]>0)&(theta[:,5]>=0)&(theta[:,1]>=0)&(theta[:,1]<=box-1)&(theta[:,2]>=0)&(theta[:,2]<=box-1)
    locs['usable_model_fit']=(states==0)&finite&physical
    locs['precision_model_finite']=np.isfinite(locs[['lpx','lpy']]).all(axis=1)&(locs.lpx>0)&(locs.lpy>0)
    locs['suspect_psf_width']=(locs.sx<.5)|(locs.sy<.5)|(locs.sx>box/2)|(locs.sy>box/2)
    predicted=model_images(theta,box)
    residual=original_converted_spots-predicted
    locs['model_residual_rms_e']=np.sqrt(np.mean(residual**2,axis=(1,2)))
    yy,xx=np.mgrid[:box,:box]
    distance=np.sqrt((xx-theta[:,1,None,None])**2+(yy-theta[:,2,None,None])**2)
    secondary=(residual==maximum_filter(residual,size=(1,3,3)))&(distance>2)&(residual>4*np.sqrt(np.maximum(predicted+nominal_variance,1)))
    secondary[:,0,:]=secondary[:,-1,:]=False;secondary[:,:,0]=secondary[:,:,-1]=False
    locs['suspect_overlap']=secondary.any(axis=(1,2))
    locs['source_pixels_below_assumed_offset']=below
    locs['mle_pixels_clipped_to_noise_floor']=clipped.sum(axis=(1,2))
    locs['source_zero_pixels']=(raw==0).sum(axis=(1,2))
    locs['suspect_zero_pixel_support']=locs.source_zero_pixels>0
    locs['source_zero_pixels_detection_support']=detection_zero_counts(identifications,ids,movie,box)
    locs['suspect_zero_detection_support']=locs.source_zero_pixels_detection_support>0
    locs['qc_unflagged_model_fit']=locs.usable_model_fit&locs.precision_model_finite&~locs.suspect_psf_width&~locs.suspect_overlap&~locs.suspect_zero_pixel_support&~locs.suspect_zero_detection_support
    reasons={'backend_not_converged':states!=0,'nonfinite_parameters':~finite,
             'nonpositive_signal':theta[:,0]<=0,'nonpositive_sx':theta[:,3]<=0,
             'nonpositive_sy':theta[:,4]<=0,'negative_background':theta[:,5]<0,
             'center_outside_fitting_box':(theta[:,1]<0)|(theta[:,1]>box-1)|(theta[:,2]<0)|(theta[:,2]>box-1)}
    locs['fit_parameter_issues']=[';'.join(name for name,flag in reasons.items() if flag[i]) for i in range(len(locs))]
    locs['precision_status']=np.select([~locs.usable_model_fit,~locs.precision_model_finite,~locs.qc_unflagged_model_fit],
                 ['unavailable_or_invalid_fit','nonfinite_model_bound','requires_image_review'],default='nominal_model_only')
    return locs,theta,{'fit_runtime_seconds':time.perf_counter()-started,'nominal_variance_e2':nominal_variance}


def save_locs(path, locs, metadata):
    """Picasso-compatible /locs compound table, plus status/units metadata."""
    arrays=[]
    for name in locs.columns:
        a=locs[name].to_numpy()
        if a.dtype.kind in ('O','U'):
            encoded=[str(v).encode('utf-8') for v in a]
            size=max([1]+[len(v) for v in encoded]);a=np.asarray(encoded,dtype=f'S{size}')
        arrays.append((name,a))
    structured=np.empty(len(locs),dtype=[(name,a.dtype) for name,a in arrays])
    for name,a in arrays: structured[name]=a
    with h5py.File(path,'w') as f:
        f.create_dataset('locs',data=structured,compression='gzip')
        f.create_dataset('metadata',data=json.dumps([metadata],ensure_ascii=False))
        f.attrs['metadata']=json.dumps(metadata,ensure_ascii=False)
        f.attrs['field_units']=json.dumps(FIELD_UNITS,ensure_ascii=False)
        f.attrs['fit_state_labels']=json.dumps(FIT_STATES)
        f.attrs['calibration_status']='manufacturer_nominal'
        f.attrs['quantitatively_calibrated']=False
    locs.to_csv(Path(path).with_suffix('.csv'),index=False,float_format='%.17g')
    with h5py.File(path,'r') as f:
        reread=f['locs'][()]
        assert len(reread)==len(locs)
        for name in ('frame','candidate_id','x','y','lpx','lpy','fit_state'):
            np.testing.assert_equal(reread[name],locs[name].to_numpy())


def result_summary(locs, box, factor, threshold, timings):
    good=locs.loc[locs.usable_model_fit]
    finite_precision=good.loc[good.precision_model_finite]
    def med(frame,key):
        v=frame[key].to_numpy();v=v[np.isfinite(v)]
        return None if len(v)==0 else float(np.median(v))
    unflagged=locs.loc[locs.qc_unflagged_model_fit]
    return {'box':box,'threshold_factor':factor,'threshold_net_gradient':threshold,
            'candidates':len(locs),'converged':int(locs.fit_converged.sum()),'usable_model_fits':len(good),
            'converged_fraction':None if len(locs)==0 else float(locs.fit_converged.mean()),
            'suspect_overlap_count':int(good.suspect_overlap.sum()),'suspect_width_count':int(good.suspect_psf_width.sum()),
            'source_zero_support_count':int(locs.suspect_zero_pixel_support.sum()),
            'qc_unflagged_model_fit_count':len(unflagged),
            'unflagged_median_photons_nominal':med(unflagged,'photons'),
            'unflagged_median_sx_px':med(unflagged,'sx'),'unflagged_median_sy_px':med(unflagged,'sy'),
            'unflagged_median_lpx_nm_nominal':med(unflagged,'lpx_nm'),'unflagged_median_lpy_nm_nominal':med(unflagged,'lpy_nm'),
            'median_photons_nominal':med(good,'photons'),'median_bg_nominal_e_per_pixel':med(good,'bg'),
            'median_sx_px':med(good,'sx'),'median_sy_px':med(good,'sy'),
            'median_lpx_nm_nominal':med(finite_precision,'lpx_nm'),'median_lpy_nm_nominal':med(finite_precision,'lpy_nm'),
            'median_deviance_per_dof':med(good,'deviance_per_dof'),
            'backend_state_counts':{FIT_STATES.get(int(k),str(k)):int(v) for k,v in locs.fit_state.value_counts().items()},
            **timings}
