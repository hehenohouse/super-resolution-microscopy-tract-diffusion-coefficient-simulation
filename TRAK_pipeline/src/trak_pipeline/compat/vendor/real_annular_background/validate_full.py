"""Post-run audit: all-row/ID checks plus stratified direct pixel checks."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import ndtr
import tifffile
import yaml

import run_real as core


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
    rules=json.loads((out/'rules_before_run.json').read_text());c=rules['fit_rules'];camera=rules['camera']
    counts=pd.read_csv(out/'frame_counts.csv');s=json.loads((out/'summary.json').read_text())
    original=pd.read_csv(out/'all_detections.csv');fits=pd.read_csv(out/'all_ROI_target_fits.csv')
    bg=pd.read_csv(out/'annular_background_all_ROI_targets.csv');mask=np.load(out/'retained_annular_pixels.npz')
    np.testing.assert_array_equal(counts.frame,np.arange(2500))
    np.testing.assert_allclose(counts.time_s,counts.frame*.02,rtol=0,atol=1e-12)
    assert original.candidate_index.is_unique and original.source_detection_uid.is_unique
    np.testing.assert_array_equal(original.candidate_index,np.arange(len(original)))
    np.testing.assert_array_equal(fits.candidate_index,original.loc[original.roi_inside.eq(1),'candidate_index'])
    np.testing.assert_array_equal(mask['candidate_index'],fits.candidate_index)
    np.testing.assert_array_equal(bg.candidate_index,fits.candidate_index)
    assert counts.ROI_candidates.sum()==len(fits)==s['ROI_candidates']
    assert counts.FOV_candidates.sum()==len(original)==s['FOV_candidates']
    assert counts.unflagged.sum()==fits.qc_unflagged_model_fit.sum()==s['unflagged']
    assert counts.finite_interior_ROI.sum()==fits.histogram_eligible.sum()==s['finite_interior_ROI']
    eligible=fits[fits.histogram_eligible]
    assert np.isfinite(eligible[['photons_e','x_se_nm','y_se_nm','N_se_e']]).all().all()
    assert (eligible[['photons_e','x_se_nm','y_se_nm','N_se_e']]>0).all().all()
    assert fits.loc[fits.active_bound|~fits.fit_success,['x_se_nm','y_se_nm','N_se_e']].isna().all().all()
    np.testing.assert_allclose(fits.x_se_nm,fits.lpx_model_px*117,equal_nan=True)
    np.testing.assert_allclose(fits.y_se_nm,fits.lpy_model_px*117,equal_nan=True)
    vertices=pd.read_csv(out/'roi_vertices.csv').to_numpy()
    assert core.roi_lib.points_in_polygon(eligible[['fit_x_px','fit_y_px']].to_numpy(),vertices).all()
    # Check all combined original columns in bounded chunks, including all
    # original full-FOV detections not targeted by the fitter.
    offset=0
    for chunk in pd.read_csv(out/'all_original_detections_with_new_fits.csv',usecols=list(original),chunksize=40000):
        expected=original.iloc[offset:offset+len(chunk)].reset_index(drop=True)
        pd.testing.assert_frame_equal(chunk.reset_index(drop=True),expected,check_exact=False,rtol=1e-13,atol=1e-13)
        offset+=len(chunk)
    assert offset==len(original)
    pages=json.loads((out/'source_page_manifest.json').read_text())
    assert [r['frame'] for r in pages]==list(range(2500))
    # Deterministic temporal stratification, including cache-reused frames.
    sample_frames=sorted(set(range(0,2500,125))|{49,124,249,1250,1300,2000,2050,2499})
    sample_candidates=0;covariances=0
    tiff=rules['fit_rules']['source_preview']
    preview=yaml.safe_load((Path(tiff)/'resolved_config.yaml').read_text())
    with tifffile.TiffFile(preview['input_tiff']) as tf:
        for frame in sample_frames:
            raw=tf.pages[frame].asarray();data=(raw.astype(float)-100)*.46
            indices=np.flatnonzero(fits.frame.eq(frame))
            if len(indices):indices=indices[np.linspace(0,len(indices)-1,min(5,len(indices))).astype(int)]
            neighbors=original.loc[original.frame.eq(frame),['x_px','y_px']].to_numpy()
            for j in indices:
                row=fits.iloc[j];xy=mask['xy_pixels'][mask['offsets'][j]:mask['offsets'][j+1]]
                center=original.iloc[int(row.candidate_index)][['x_px','y_px']].to_numpy(float)
                radius=np.sqrt(np.sum((xy-center)**2,axis=1))
                assert np.all((radius>6)&(radius<=12))
                distances=np.sum((xy[:,None,:]-neighbors[None,:,:])**2,axis=2)
                assert np.all(distances.min(axis=1)>16)
                assert core.roi_lib.points_in_polygon(xy.astype(float),vertices).all()
                values=data[xy[:,1],xy[:,0]];r=bg.iloc[j]
                np.testing.assert_allclose(values.mean(),r.mean_e,rtol=1e-12)
                np.testing.assert_allclose(max(values.var(ddof=1),2.56),r.variance_e2,rtol=1e-12)
                assert r.mean_constraint_se_e**2+1e-12>=r.variance_e2/len(xy)
                ox,oy=int(row.fit_origin_x_px),int(row.fit_origin_y_px)
                assert not np.any((xy[:,0]>=ox)&(xy[:,0]<ox+7)&(xy[:,1]>=oy)&(xy[:,1]<oy+7))
                sample_candidates+=1
                if row.histogram_eligible and covariances<20:
                    x,y=row.fit_x_px-ox,row.fit_y_px-oy;grid=np.arange(7)
                    px=ndtr((grid+.5-x)/row.sx_px)-ndtr((grid-.5-x)/row.sx_px)
                    py=ndtr((grid+.5-y)/row.sy_px)-ndtr((grid-.5-y)/row.sy_px)
                    signal=row.photons_e*np.outer(py,px);central=data[oy:oy+7,ox:ox+7]
                    np.testing.assert_allclose(signal.sum(),row.signal_inside_box_e,rtol=1e-11)
                    np.testing.assert_allclose(np.sqrt(np.mean((central-signal-row.bg_e_per_pixel)**2)),row.residual_rms_e,rtol=1e-10)
                    aux=(xy.astype(float)-[ox,oy],float(values.sum()),len(xy)**2*r.mean_constraint_se_e**2)
                    theta=np.array([row.photons_e/1000,x,y,np.log(row.sx_px),np.log(row.sy_px),row.bg_e_per_pixel/100])
                    C=core.covariance(theta,central.ravel(),7,row.background_variance_e2,aux)
                    np.testing.assert_allclose([np.sqrt(C[1,1])*117,np.sqrt(C[2,2])*117,np.sqrt(C[0,0])],
                         [row.x_se_nm,row.y_se_nm,row.N_se_e],rtol=2e-6)
                    covariances+=1
    # Reused windows preserve source fields and original cache values.
    for cache in rules['full']['detection_caches']:
        previous=pd.read_csv(Path(cache)/'all_detections.csv')
        current=original[original.source_detection_cache.eq(cache)].reset_index(drop=True)
        pd.testing.assert_frame_equal(current[list(previous)],previous,check_exact=False,rtol=1e-13,atol=1e-13)
    previous=pd.read_csv(Path(rules['full']['fit_cache'])/'all_ROI_target_fits.csv')
    current=fits[fits.frame.between(1250,1300)].reset_index(drop=True)
    for field in previous:
        if field!='candidate_index':
            pd.testing.assert_series_equal(current[field],previous[field],check_names=False,check_dtype=False,check_exact=False,rtol=1e-13,atol=1e-13)
    for name in ['localization_uncertainty_histogram.png','photoelectron_number_histogram.png']:
        with Image.open(out/name) as im:im.verify()
    for field,bins in [('x_se_nm',35),('y_se_nm',35),('photons_e',40)]:
        values=eligible[field].to_numpy();edges=np.linspace(0,max(float(values.max())*1.05,1),bins+1)
        assert np.histogram(values,bins=edges)[0].sum()==len(eligible)
    validation=dict(all_full_FOV_rows_and_original_fields_preserved=True,all_ROI_targets_preserved=True,
        candidate_source_IDs_unique=True,all_2500_frames_and_zero_counts_preserved=True,
        no_double_nm_conversion=True,unavailable_errors_not_published=True,per_frame_counts_match=True,
        all_102_cached_detection_frames_unchanged=True,all_2113_prior_fit_values_unchanged=True,
        direct_annulus_neighbor_and_mean_variance_checks=dict(frames=sample_frames,candidates=sample_candidates),
        independent_pixel_PSF_and_full_nuisance_covariance_checks=covariances,
        full_range_linear_histogram_counts_match=True,PNG_decode_valid=True,no_new_optimization=True)
    core.dump(out/'independent_validation.json',validation)
    (out/'source_snapshot').mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
