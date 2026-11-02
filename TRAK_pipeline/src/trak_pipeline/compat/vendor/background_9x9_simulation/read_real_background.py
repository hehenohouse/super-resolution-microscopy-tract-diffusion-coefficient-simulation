"""Inspect TIFF pixels directly and inject shared known signals into real low-signal patches."""
import argparse
from datetime import datetime,timezone
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
from matplotlib.patches import Circle,Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt,minimum_filter
import tifffile
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'trackmate_python_bridge'))
from run_preview import roi_lib,read_mat_vertices,fingerprint,draw_lib
from run_simulation import dump,mean9


def seed_mask(table,shape):
    xy=table[['x_px','y_px']].to_numpy(float)
    xy=xy[np.isfinite(xy).all(axis=1)];xy=np.floor(xy+.5).astype(int)
    xy=xy[(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])]
    mask=np.zeros(shape,bool);mask[xy[:,1],xy[:,0]]=True
    return mask


def seam_diagnostics(base,size):
    # A difference crosses a seam iff its second pixel begins a new tile.
    dx=np.diff(base.astype(float),axis=1);dy=np.diff(base.astype(float),axis=0)
    xseam=np.arange(1,base.shape[1])%size==0
    yseam=np.arange(1,base.shape[0])%size==0
    return {'tile_boundary_X_difference_RMS_ADU':float(np.sqrt(np.mean(dx[:,xseam]**2))),
            'interior_X_difference_RMS_ADU':float(np.sqrt(np.mean(dx[:,~xseam]**2))),
            'tile_boundary_Y_difference_RMS_ADU':float(np.sqrt(np.mean(dy[yseam,:]**2))),
            'interior_Y_difference_RMS_ADU':float(np.sqrt(np.mean(dy[~yseam,:]**2)))}


def figures(out,c,frames,movie,vertices,table,stats,quiet_pixels,tiles,patches,base,observed,truth,comparison):
    lo,hi=c['raw_display_range_ADU'];x0,y0,x1,y1=c['raw_display_crop_XYXY']
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    for ax,f in zip(axes[0],[2000,2025,2050]):
        raw=movie[f-frames[0]];g=table[table.frame==f]
        ax.imshow(raw[y0:y1,x0:x1],origin='upper',cmap='gray',vmin=lo,vmax=hi,
                  extent=(x0-.5,x1-.5,y1-.5,y0-.5),interpolation='nearest')
        ax.plot(np.r_[vertices[:,0],vertices[0,0]],np.r_[vertices[:,1],vertices[0,1]],color='#EAB82A',lw=.8)
        for row in g.itertuples():ax.add_patch(Circle((row.x_px,row.y_px),3,fill=False,edgecolor='cyan',lw=.6))
        ax.set(title=f'Raw TIFF frame {f} | {len(g)} ROI detections',xlabel='Original X (pixels)',ylabel='Original Y (pixels)',
               xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5))
    ax=axes[1,0];ax.hist(quiet_pixels,bins=np.arange(380,481),density=True,color='#527D96')
    ax.set(xlabel='Raw intensity (ADU)',ylabel='Density (1/ADU)',title='ROI low-signal pixels, pooled 51 frames')
    ax=axes[1,1];ax.plot(stats.frame,stats.quiet_mean_ADU,label='Mean');ax.plot(stats.frame,stats.quiet_median_ADU,label='Median')
    ax.set(xlabel='Original frame',ylabel='ADU',title='Low-signal proxy over time');ax.legend()
    ax=axes[1,2]
    good=table.local_reference_n_pixels>=c['raw_spot_diagnostics']['minimum_reference_pixels']
    for label,mask,color in [('Unflagged',table.qc_unflagged_model_fit,'#228F71'),
                             ('Review',table.nominal_precision_valid&~table.qc_unflagged_model_fit,'#C059C2'),
                             ('Unavailable',~table.nominal_precision_valid,'#DB8F21')]:
        g=table[good&mask];ax.scatter(g.local_reference_mean_ADU,g.raw_mean9_ADU,s=10,alpha=.55,label=label,color=color)
    lim=[float(table.loc[good,'local_reference_mean_ADU'].min()-2),float(table.loc[good,['local_reference_mean_ADU','raw_mean9_ADU']].max().max()+2)]
    ax.plot(lim,lim,'--',color='gray')
    ax.set(xlabel='Nearby low-signal proxy mean (ADU)',ylabel='Whole 9x9 mean at detected point (ADU)',title='Real-image estimates; neither is known truth');ax.legend(fontsize=8)
    fig.suptitle('Direct raw TIFF inspection | Native C2 coordinates | 117 nm/pixel\nLow-signal proxy excludes pixels <7 px from ANY cached TrackMate detection',fontsize=14)
    fig.savefig(out/'raw_image_inspection.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,5.3),constrained_layout=True)
    for ax,a,title in zip(axes,[base,observed,observed],['Raw-pixel background mosaic','Same background + 50 known signals','Known added signal centers / HWHM']):
        im=ax.imshow(a,origin='upper',cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        ax.set(title=title,xlabel='X (pixels)',ylabel='Y (pixels)');fig.colorbar(im,ax=ax,label='ADU',shrink=.65)
    for row in truth.itertuples():axes[2].add_patch(Circle((row.x_px,row.y_px),row.hwhm_px,fill=False,edgecolor='cyan',lw=.6))
    fig.suptitle('300x300: real 21x21 low-signal patches + exactly the previous 50 simulated signals\n'
                 'Original background noise retained. Added signal mapped with nominal 0.46 e-/ADU; no extra read noise.',fontsize=13)
    fig.savefig(out/'empirical_background_and_injection.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(10,10),constrained_layout=True)
    for ax,i in zip(axes.ravel(),range(9)):
        row=tiles.iloc[i];p=patches[i]
        ax.imshow(p,origin='upper',cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        ax.add_patch(Rectangle((5.5,5.5),9,9,fill=False,edgecolor='orange',lw=1))
        ax.set(title=f'Frame {int(row.source_frame)}, center ({int(row.source_center_x_px)}, {int(row.source_center_y_px)})\n'
                     f'Mean {p.mean():.1f}, SD {p.std():.1f} ADU',xlabel='Local X (pixels)',ylabel='Local Y (pixels)')
    fig.suptitle('Unmodified real 21x21 pixel patches | Orange: central 9x9\nSelected uniformly from eligible low-signal patches, not by darkest intensity',fontsize=13)
    fig.savefig(out/'real_background_patch_examples.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    for ax,data,title in [(axes[0,0],mean9(base,9),'9x9 mean of the raw-pixel base'),
                          (axes[0,1],mean9(observed,9),'9x9 mean after signal injection')]:
        im=ax.imshow(data,origin='upper',cmap='viridis',vmin=420,vmax=485,interpolation='nearest')
        ax.set(title=title,xlabel='X (pixels)',ylabel='Y (pixels)');fig.colorbar(im,ax=ax,label='ADU',shrink=.8)
    ax=axes[1,0];ax.scatter(comparison.base_mean9_ADU,comparison.injected_mean9_ADU,c=comparison.added_signal_mean9_e_per_pixel,cmap='magma')
    span=[float(comparison.base_mean9_ADU.min()-2),float(comparison.injected_mean9_ADU.max()+2)]
    ax.plot(span,span,'--',color='gray');ax.set(xlabel='Raw base 9x9 mean (ADU)',ylabel='After injection 9x9 mean (ADU)',title='Known signal raises the whole-window mean')
    ax=axes[1,1];ax.hist(comparison.added_signal_mean9_e_per_pixel,bins=15,color='#527D96')
    ax.set(xlabel='Known injected signal in 9x9 (e- / pixel)',ylabel='Added emitters',title='Increment is known; optical background in raw data is not')
    fig.suptitle('9x9 full-window averaging retains the injected signal\nThe raw base includes measured noise and may retain undetected emitters.',fontsize=13)
    fig.savefig(out/'empirical_background9_diagnostics.png',dpi=160);plt.close(fig)


def run(config_path):
    started=time.perf_counter();c=yaml.safe_load(config_path.read_text())
    cache=Path(c['source_refinement']);signal_source=Path(c['shared_signal_simulation'])
    source=json.loads((cache/'resolved_parameters.json').read_text())['TrackMate']
    start,end=map(int,c['frames_inclusive']);frames=np.arange(start,end+1)
    assert (start,end)==(2000,2050) and c['pixel_size_nm']==source['pixel_size_nm']==117
    out=Path(c['output_root'])/('raw_based_scene300x300_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT_DIRECTORY:',out,flush=True)
    protected=[Path(source['input_tiff']),Path(source['roi_file']),Path(source['mat_file']),
               cache/'all_detections_with_refinement.csv',cache/'ROI_points_with_refinement.csv',cache/'frame_manifest.csv',
               cache/'resolved_parameters.json',cache/'nominal_camera_source.yaml',
               signal_source/'truth_spots.csv',signal_source/'scene_components.npz',signal_source/'config.yaml']
    print('Fingerprinting original TIFF and read-only caches...',flush=True)
    before=[fingerprint(p) for p in protected];dump(out/'source_fingerprints_before.json',before)
    assert before[0]['sha256']==source['expected_tiff_sha256'] and before[1]['sha256']==source['expected_roi_sha256']
    existing_hashes=json.loads((cache/'output_fingerprints.json').read_text())
    for item in before[3:6]:assert item['sha256']==existing_hashes[Path(item['path']).name]
    camera=yaml.safe_load((cache/'nominal_camera_source.yaml').read_text())['camera']
    assert camera['sensitivity_e_per_adu']==c['injection']['sensitivity_e_per_ADU']
    all_points=pd.read_csv(cache/'all_detections_with_refinement.csv',float_precision='round_trip')
    table=pd.read_csv(cache/'ROI_points_with_refinement.csv',float_precision='round_trip');original=table.copy(deep=True)
    assert len(table)==1173 and all_points.frame.between(start,end).all()
    vertices=roi_lib.load_imagej_roi(source['roi_file']);mat,_=read_mat_vertices(source['mat_file'],source['mat_record']);np.testing.assert_array_equal(vertices,mat)
    yy,xx=np.mgrid[:684,:428];roi=roi_lib.points_in_polygon(np.column_stack([xx.ravel(),yy.ravel()]),vertices).reshape(684,428)
    manifest=pd.read_csv(cache/'frame_manifest.csv');movie=[];quiet_masks=[];eligible=[];stats=[];quiet_values=[];local_records=[];temporal=[];fixed_zero=None
    with tifffile.TiffFile(source['input_tiff']) as tif:
        assert tif.series[0].axes=='TYX' and tif.series[0].shape==(2500,684,428) and tif.pages[0].dtype==np.uint16
        for f in frames:
            raw=tif.pages[int(f)].asarray();assert hashlib.sha256(raw.tobytes()).hexdigest()==manifest[manifest.frame==f].iloc[0].source_pixel_sha256
            zero=raw==0
            if fixed_zero is None:fixed_zero=zero.copy()
            else:np.testing.assert_array_equal(zero,fixed_zero)
            distance=distance_transform_edt(~seed_mask(all_points[all_points.frame==f],raw.shape))
            quiet=roi&~zero&(distance>=c['quiet_proxy']['exclusion_radius_from_all_TrackMate_rounded_centers_px'])
            ok=minimum_filter(quiet.astype(np.uint8),size=c['empirical_mosaic']['patch_size_px'],mode='constant',cval=0)>0
            coords=np.argwhere(ok);assert len(coords)>0
            z=raw[quiet];quiet_values.append(z);eligible.append(coords);movie.append(raw);quiet_masks.append(quiet)
            stat={'frame':int(f),'time_s':float(f*.02),'full_frame_zero_pixels':int(zero.sum()),'ROI_pixels':int(roi.sum()),'ROI_zero_pixels':int((roi&zero).sum()),
                  'ROI_mean_ADU':float(raw[roi].mean()),'ROI_std_ADU':float(raw[roi].std()),'quiet_pixels':len(z),
                  'quiet_mean_ADU':float(z.mean()),'quiet_std_ADU':float(z.std()),'quiet_median_ADU':float(np.median(z)),
                  'quiet_q01_ADU':float(np.quantile(z,.01)),'quiet_q99_ADU':float(np.quantile(z,.99)),'eligible_21x21_centers':len(coords)}
            for direction,one,two,pairs in [('X',raw[:,:-1],raw[:,1:],quiet[:,:-1]&quiet[:,1:]),
                                            ('Y',raw[:-1,:],raw[1:,:],quiet[:-1,:]&quiet[1:,:])]:
                stat['quiet_neighbor_correlation_'+direction]=float(np.corrcoef(one[pairs],two[pairs])[0,1])
            stats.append(stat)
            if len(movie)>1:
                common=quiet&quiet_masks[-2];delta=raw[common].astype(float)-movie[-2][common].astype(float)
                temporal.append({'frame_from':int(f-1),'frame_to':int(f),'common_quiet_pixels':int(common.sum()),
                                 'difference_mean_ADU':float(delta.mean()),'difference_std_ADU':float(delta.std()),
                                 'difference_std_div_sqrt2_ADU':float(delta.std()/np.sqrt(2)),
                                 'interpretation':'Descriptive temporal difference; not identified read noise'})
            for row in table[table.frame==f].itertuples():
                x,y=np.floor([row.x_px+.5,row.y_px+.5]).astype(int);win=(slice(y-4,y+5),slice(x-4,x+5));context=(slice(y-10,y+11),slice(x-10,x+11))
                assert raw[win].shape==(9,9)
                q=quiet[context];ref=raw[context][q];mean=float(ref.mean()) if len(ref)>=c['raw_spot_diagnostics']['minimum_reference_pixels'] else np.nan
                local_records.append({'spot_id':row.spot_id,'raw_mean9_ADU':float(raw[win].mean()),'raw_median9_ADU':float(np.median(raw[win])),
                                      'raw_std9_ADU':float(raw[win].std()),'raw_max9_ADU':int(raw[win].max()),
                                      'local_reference_n_pixels':len(ref),'local_reference_mean_ADU':mean,
                                      'mean9_minus_reference_ADU':float(raw[win].mean()-mean),
                                      'cached_MLE_background_nominal_ADU':float(row.bg/camera['sensitivity_e_per_adu']+camera['baseline_adu'])})
    movie=np.stack(movie);stats=pd.DataFrame(stats);quiet_values=np.concatenate(quiet_values)
    table=table.merge(pd.DataFrame(local_records),on='spot_id',how='left',validate='one_to_one',sort=False)
    pd.testing.assert_frame_equal(table[original.columns],original,check_exact=True)
    table.to_csv(out/'real_spot_9x9_diagnostics.csv',index=False,float_format='%.17g');stats.to_csv(out/'frame_background_statistics.csv',index=False)
    pd.DataFrame(temporal).to_csv(out/'adjacent_frame_background_differences.csv',index=False)
    rng=np.random.default_rng(c['empirical_mosaic']['seed']);shape=tuple(c['empirical_mosaic']['shape_yx']);size=c['empirical_mosaic']['patch_size_px'];half=size//2
    base=np.empty(shape,np.uint16);tile_rows=[];patches=[]
    for y in range(0,shape[0],size):
        for x in range(0,shape[1],size):
            j=int(rng.integers(len(frames)));cy,cx=eligible[j][int(rng.integers(len(eligible[j])))];patch=movie[j,cy-half:cy+half+1,cx-half:cx+half+1].copy()
            assert patch.shape==(size,size) and quiet_masks[j][cy-half:cy+half+1,cx-half:cx+half+1].all()
            h,w=min(size,shape[0]-y),min(size,shape[1]-x);base[y:y+h,x:x+w]=patch[:h,:w]
            patches.append(patch);tile_rows.append({'tile_id':len(patches)-1,'source_frame':int(frames[j]),'source_center_x_px':int(cx),'source_center_y_px':int(cy),
                                                'source_top_left_x_px':int(cx-half),'source_top_left_y_px':int(cy-half),'target_x_px':x,'target_y_px':y,
                                                'target_width_px':w,'target_height_px':h,'source_patch_sha256':hashlib.sha256(patch.tobytes()).hexdigest()})
    tiles=pd.DataFrame(tile_rows);patches=np.stack(patches);tiles.to_csv(out/'background_patch_sources.csv',index=False)
    truth=pd.read_csv(signal_source/'truth_spots.csv',float_precision='round_trip');assert len(truth)==50
    with np.load(signal_source/'scene_components.npz') as previous:
        signal=previous['signal_realized_electrons'];expected_signal=previous['signal_expected_electrons']
    assert signal.shape==shape and signal.sum()==truth.realized_total_electrons.sum()
    gain=c['injection']['sensitivity_e_per_ADU'];observed=base.astype(float)+signal/gain
    np.testing.assert_allclose((observed-base)*gain,signal,rtol=0,atol=1e-10)
    truth.to_csv(out/'shared_added_spot_truth.csv',index=False,float_format='%.17g')
    comparisons=[]
    for row in truth.itertuples():
        x,y=int(row.anchor_x_px),int(row.anchor_y_px);win=(slice(y-4,y+5),slice(x-4,x+5))
        before_mean,after_mean=float(base[win].mean()),float(observed[win].mean());increment=float(signal[win].mean())
        np.testing.assert_allclose((after_mean-before_mean)*gain,increment,rtol=0,atol=1e-10)
        comparisons.append({'spot_id':int(row.spot_id),'base_mean9_ADU':before_mean,'injected_mean9_ADU':after_mean,
                            'added_signal_mean9_e_per_pixel':increment,'added_signal_mean9_ADU':increment/gain})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'injected_spot_9x9_comparison.csv',index=False)
    np.savez_compressed(out/'scene_components.npz',base_ADU=base,observed_ADU=observed,
                        added_signal_realized_electrons=signal,added_signal_expected_electrons=expected_signal,
                        background_source_patches_ADU=patches,source_frame_ids=frames,
                        raw_example_frames=np.array([2000,2025,2050]),raw_example_crops_ADU=movie[[0,25,50],180:480,60:360])
    np.save(out/'source_fixed_zero_map.npy',fixed_zero)
    tifffile.imwrite(out/'empirical_background_uint16_ADU.tiff',base,metadata={'axes':'YX','pixel_size_nm':117,'unit':'ADU'},photometric='minisblack')
    tifffile.imwrite(out/'injected_scene_float32_ADU.tiff',observed.astype(np.float32),metadata={'axes':'YX','pixel_size_nm':117,'unit':'ADU'},photometric='minisblack')
    np.testing.assert_array_equal(tifffile.imread(out/'injected_scene_float32_ADU.tiff'),observed.astype(np.float32))
    print('Read 51 original TIFF pages; creating figures from raw pixels and real-patch injection.',flush=True)
    figures(out,c,frames,movie,vertices,table,stats,quiet_values,tiles,patches,base,observed,truth,comparison)
    summary={'source_original_frames':[start,end],'source_axes':'TYX','source_shape':[2500,684,428],'source_dtype':'uint16',
             'source_ROI_pixels':int(roi.sum()),'source_fixed_zero_pixels_each_frame':int(fixed_zero.sum()),'source_ROI_zero_pixels_each_frame':int((fixed_zero&roi).sum()),
             'pooled_quiet_pixels':len(quiet_values),'pooled_quiet_mean_ADU':float(quiet_values.mean()),'pooled_quiet_std_ADU':float(quiet_values.std()),
             'mean_quiet_neighbor_correlation_X':float(stats.quiet_neighbor_correlation_X.mean()),'mean_quiet_neighbor_correlation_Y':float(stats.quiet_neighbor_correlation_Y.mean()),
             'median_adjacent_difference_std_div_sqrt2_ADU':float(pd.DataFrame(temporal).difference_std_div_sqrt2_ADU.median()),
             'raw_point_9x9_mean_median_ADU':float(table.raw_mean9_ADU.median()),'raw_local_reference_median_ADU':float(table.local_reference_mean_ADU.median()),
             'raw_mean9_minus_reference_median_ADU':float(table.mean9_minus_reference_ADU.median()),
             'base_mosaic_mean_ADU':float(base.mean()),'base_mosaic_std_ADU':float(base.std()),'base_patch_count':len(tiles),'base_patch_size':size,
             **seam_diagnostics(base,size),
             'added_known_spots':50,'shared_added_signal_with_previous_simulation':True,'extra_noise_added':False,
             'nominal_injection_sensitivity_e_per_ADU':gain,'absolute_real_electrons_calibrated':False,
             'injected_signal_mean9_e_per_pixel':float(comparison.added_signal_mean9_e_per_pixel.mean()),
             'injected_signal_mean9_ADU':float(comparison.added_signal_mean9_ADU.mean()),
             'optical_background_known_truth':False,'low_signal_proxy_may_retain_undetected_spots':True,'mosaic_preserves_only_within_patch_correlations':True,
             'no_new_detection_fitting_or_QC_change':True}
    dump(out/'summary.json',summary)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('run_simulation.py'),Path(__file__).parent.parent/'trackmate_python_bridge/run_preview.py',Path(roi_lib.__file__),Path(draw_lib.__file__)]:shutil.copyfile(p,snapshot/p.name)
    shutil.copyfile(config_path,out/'config.yaml');shutil.copyfile(signal_source/'config.yaml',out/'shared_signal_config.yaml')
    dump(out/'software_versions.json',{**{n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','pandas','tifffile','PyYAML','h5py']},'python':sys.version,'executable':sys.executable})
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path.resolve())])+'\n')
    print('Rechecking read-only source fingerprints...',flush=True)
    after=[fingerprint(p) for p in protected];assert before==after;dump(out/'source_fingerprints_after.json',after)
    dump(out/'validation.json',{'all_checks_passed':True,'source_TIFF_pages_51_exact_cached_frame_hashes':True,'ROI_MAT_vertices_exact':True,
                              'all_real_patches_unmodified_and_pass_full_quiet_proxy':True,'shared_50_signal_truth_and_Poisson_counts_unchanged':True,
                              'ADU_signal_increment_roundtrip_verified':True,'all_original_localization_fields_QC_preserved':True,
                              'raw_mean9_diagnostics_at_original_TrackMate_centers':True,'TIFF_float32_roundtrip_exact':True,'protected_inputs_unchanged':True})
    (out/'REPORT.txt').write_text('Read original T1 C2 TIFF pages 2000-2050 directly; TYX 2500x684x428 uint16, 117nm/pixel, native C2 XY.\n'
        f'Low-signal ROI proxy pooled mean={summary["pooled_quiet_mean_ADU"]:.6g}, SD={summary["pooled_quiet_std_ADU"]:.6g} ADU.\n'
        'Proxy: inside common cell ROI, nonzero, >=7px from ALL cached rounded TrackMate centers, including failed fits. Not verified emitter-free background.\n'
        '225 uniformly sampled 21x21 eligible raw patches tile a 300x300 mosaic (outer tiles truncated). Original pixels and noise retained; tile sources saved.\n'
        'Exactly the prior simulation 50 positions, widths, total expectations and realized Poisson signals reused. Add signal_e/0.46 to raw ADU baseline; no extra background/read noise.\n'
        '0.46 e/ADU is manufacturer-nominal injection mapping, not measured real-camera calibration. Raw intensities reported in ADU.\n'
        'Real-point whole9 mean and nearby21 low-signal proxy mean compared; neither is optical background truth. Cached MLE background shown only as existing nominal field.\n'
        'The stored base is a known input to the injection test, but physical optical background/noise separation and faint undetected emitters remain unknown.\n'
        'Mosaic has tile seams and does not preserve correlations beyond21 pixels or original time structure; seam diagnostics reported.\n'
        'Original TIFF, localization values, QC and previous simulated outputs unchanged. No new detection, localization fitting, TARDIS or mitochondrial analysis.\n')
    dump(out/'output_fingerprints.json',{str(p.relative_to(out)):draw_lib.sha256(p) for p in out.rglob('*') if p.is_file()})
    dump(out/'COMPLETED.json',{'all_checks_passed':True,'output_directory':str(out),'elapsed_s':time.perf_counter()-started})
    (out/'run.log').write_text('Completed real TIFF inspection and real-patch injection.\n'+json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True);print('COMPLETE:',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config.resolve())
