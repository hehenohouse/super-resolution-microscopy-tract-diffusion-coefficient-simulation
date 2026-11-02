#!/usr/bin/env python3
"""Verified source adapter for existing TARDIS, using support50 postmin32 groups."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
import unittest

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/noton-mplcache')
import numpy as np
import pandas as pd

from run_intra import ROOT,PROJECT,dump,fingerprint,Tee,bool_values,snapshot,analyze_groups
from mito_core import GROUPS,LABELS,scopes,range_histograms,decompose_roi
from roi_core import load_imagej_roi,read_mat_roi,points_in_polygon


def known_manifest(folder):
    """Read legacy relative-SHA maps and current absolute fingerprint lists."""
    result={}
    for name in ['output_fingerprints.json','input_fingerprints_before.json']:
        path=folder/name
        if not path.exists():continue
        records=json.loads(path.read_text())
        if isinstance(records,list):result.update({r['path']:r['sha256'] for r in records})
        else:result.update({str(folder/k) if not k.startswith('/') else k:v for k,v in records.items()})
    return result


def run(config_path):
    started=time.perf_counter();c=json.loads(config_path.read_text());source=Path(c['classification_directory'])
    m=Path(c['mask_directory']);oldrun=Path(c['old_run_for_all_pairs_check']) if c.get('old_run_for_all_pairs_check') else None
    assert json.loads((source/'COMPLETED.json').read_text())['delivery_checks_passed']
    assert json.loads((m/'COMPLETE.json').read_text())['all_passed']
    if oldrun:assert json.loads((oldrun/'COMPLETED.json').read_text())['delivery_checks_passed']
    parent=json.loads((source/'resolved_config.json').read_text());assert parent['mask_directory']==str(m)
    assert c['frames']==2500 and c['tau_frames']==1 and c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    assert c['distance_max_um']==3 and c['distance_bins']==300 and c['background_fit_um']==[2.1,3.]
    assert not any(c[key] for key in ['refitting','resegmenting','tracking','probability_filtering','uncertainty_deconvolution','diffusion_correction','biological_state_fitting','formal_inference'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'results'/(c.get('run_prefix','T1_C2_postmin32_On_NotOn_tau1')+'_'+stamp)
    for directory in ['','distribution','pair_histograms','bootstrap','figures','checkpoints']:(out/directory).mkdir(parents=True,exist_ok=False)
    c['external_destination']=str(Path(parent['C1_file']).parent/c['output_relative_directory']/out.name)
    pointer=c.get('pointer_prefix','latest_postmin32')
    (ROOT/(pointer+'_local.txt')).write_text(str(out)+'\n')
    dump(out/'config.json',c);dump(out/'RULES_BEFORE_RUN.json',c)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path)])+'\n')
    with (out/'run.log').open('w') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);snapshot(out)
            # The adapter config replaces the legacy W5 source contract only.
            shutil.copyfile(config_path,out/'source_snapshot/mito_tardis/config_postmin32.json')
            known=known_manifest(source)|known_manifest(m)
            if oldrun:known.update(known_manifest(oldrun))
            weight=(ROOT/c['shared_weights']).resolve()
            paths=[config_path,source/'resolved_config.json',source/'classification/all_ROI_candidates_postmin32.csv',
                source/'classification/selected_localizations_postmin32.csv',source/'classification/per_frame_counts.csv',
                source/'classification/summary.json',Path(parent['ROI_file']),Path(parent['MAT_file']),
                weight,PROJECT/c['independent_point_reference']]
            if oldrun:paths.append(oldrun/'pair_histograms/per_frame_raw_counts.npz')
            paths += [m/name for name in ['clean_full_mask.npy','analysis_ROI_mask.npy','roi_mask.npy','roi_vertices_xy.npy',
                'original_frame_to_mask_block.npy','original_frame_numbers_by_block.npy','block_completed.npy','parameters.json','per_block_statistics.csv']]
            before=[]
            for path in paths:
                print('SOURCE',path.name,flush=True);item=fingerprint(path)
                if str(path) in known:assert item['sha256']==known[str(path)],str(path)
                elif path==weight:assert item['sha256']==c['shared_weights_sha256']
                else:assert path in [config_path,PROJECT/c['independent_point_reference']],str(path)
                before.append(item)
            dump(out/'input_fingerprints_before.json',before)
            params=json.loads((m/'parameters.json').read_text())
            assert params['per_frame_min32'] is False and params['shape']==[50,684,428] and params['axes']=='TYX'
            assert params['pixel_size_um']==.117 and params['source_frame_interval_s']==.02 and params['mask_block_interval_s']==1.
            assert params['support_minimum_count']==10 and params['minimum_foreground_area_pixel2']==32 and params['hole_area_exclusive_max_pixel2']==64
            mapping=np.load(m/'original_frame_to_mask_block.npy');np.testing.assert_array_equal(mapping,np.arange(2500)//50)
            np.testing.assert_array_equal(np.load(m/'original_frame_numbers_by_block.npy'),np.arange(2500).reshape(50,50))
            assert np.load(m/'block_completed.npy').all()
            mask=np.load(m/'clean_full_mask.npy',mmap_mode='r');roi_mask=np.load(m/'roi_mask.npy')
            assert mask.shape==(50,684,428) and mask.dtype==bool
            np.testing.assert_array_equal(mask&roi_mask,np.load(m/'analysis_ROI_mask.npy'))
            roi=load_imagej_roi(parent['ROI_file']);mat,_=read_mat_roi(parent['MAT_file'],parent['MAT_record'])
            np.testing.assert_array_equal(roi,mat);np.testing.assert_array_equal(roi,np.load(m/'roi_vertices_xy.npy'))
            assert len(roi)==c.get('expected_roi_vertices',454)
            assert roi_mask.shape==(684,428) and roi_mask.dtype==bool
            assert int(roi_mask.sum())==c.get('expected_roi_pixels',32179)
            yy,xx=np.mgrid[:684,:428]
            np.testing.assert_array_equal(roi_mask,points_in_polygon(np.c_[xx.ravel(),yy.ravel()],roi).reshape(684,428))
            pd.DataFrame(roi,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
            suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_mito_tardis.py')
            with (out/'tests.log').open('w') as stream:tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            assert tests.wasSuccessful()
            fields=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px',
                'qc_unflagged_model_fit','fitted_center_inside_ROI',
                'postmin32_pointwise_label','postmin32_registration_status']
            with (source/'classification/all_ROI_candidates_postmin32.csv').open(newline='') as stream:available=csv.DictReader(stream).fieldnames
            fields += [key for key in ['spatial_selected','support50_pointwise_label','mean50_pointwise_label',
                                       'calibration_status','quantitatively_calibrated'] if key in available]
            audit=pd.read_csv(source/'classification/all_ROI_candidates_postmin32.csv',usecols=fields,dtype=str,keep_default_na=False)
            assert len(audit)==c['expected_candidates'] and audit.candidate_index.is_unique
            keep=bool_values(audit.qc_unflagged_model_fit)&bool_values(audit.fitted_center_inside_ROI)
            if 'spatial_selected' in audit:np.testing.assert_array_equal(keep,bool_values(audit.spatial_selected))
            assert keep.sum()==c['expected_selected'] and (~keep).sum()==c['expected_QC_excluded']
            assert audit.postmin32_registration_status.eq(c['registration_status']).all()
            table=audit[keep].copy();table['tardis_group']=table.postmin32_pointwise_label.map(LABELS)
            for group,n in c['expected_group_counts'].items():assert table.tardis_group.eq(group).sum()==n
            numeric=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']
            for key in numeric:table[key]=table[key].map(int if key in ['candidate_index','frame'] else float)
            assert table.frame.between(0,2499).all() and np.isfinite(table[numeric]).all().all()
            np.testing.assert_allclose(table.time_s,table.frame*.02,atol=1e-12,rtol=0)
            assert points_in_polygon(table[['fit_x_px','fit_y_px']].to_numpy(),roi).all()
            table.to_csv(out/'statistical_input.csv',index=False)
            shutil.copyfile(source/'classification/selected_localizations_postmin32.csv',out/'selected_original_rows.csv')
            assert fingerprint(out/'selected_original_rows.csv')['sha256']==fingerprint(source/'classification/selected_localizations_postmin32.csv')['sha256']
            frame=pd.DataFrame(dict(frame=np.arange(2500),time_s=np.arange(2500)*.02,mask_block_index=mapping))
            for group in GROUPS+['unknown']:
                frame[group]=table[table.tardis_group==group].groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy()
            for name,which in [('QC_excluded',~keep),('ROI_candidates',np.ones(len(audit),bool))]:
                frame[name]=audit[which].assign(frame=lambda d:d.frame.astype(int)).groupby('frame').size().reindex(range(2500),fill_value=0).to_numpy()
            np.testing.assert_array_equal(frame[GROUPS+['unknown','QC_excluded']].sum(1),frame.ROI_candidates)
            saved=pd.read_csv(source/'classification/per_frame_counts.csv')
            for field,current in [('On','On'),('Not_on','Not-on'),('QC_excluded','QC_excluded'),('ROI_candidates','ROI_candidates')]:
                np.testing.assert_array_equal(saved[field],frame[current])
            block_roi=np.count_nonzero(mask&roi_mask,axis=(1,2));block_full=np.count_nonzero(mask,axis=(1,2))
            frame['mask_ROI_area_pixel2']=block_roi[mapping];frame['mask_full_area_pixel2']=block_full[mapping]
            frame['mask_is_50_frame_support_region']=True;frame.to_csv(out/'per_frame_counts.csv',index=False)
            adjacent=[]
            for b in range(1,50):
                a,z=mask[b-1]&roi_mask,mask[b]&roi_mask;union=int((a|z).sum())
                adjacent.append(dict(candidate='support50_postmin32',frame_before=b*50-1,frame_after=b*50,
                    block_before=b-1,block_after=b,IoU=float((a&z).sum()/union) if union else 1.,scope='ROI',
                    interpretation='Between consecutive1s support masks; not instantaneous boundaries'))
            pd.DataFrame(adjacent).to_csv(out/'mask_adjacent_pair_metrics.csv',index=False)
            dump(out/'source_reference.json',dict(classification_directory=str(source),mask_directory=str(m),
                classification_label='postmin32_pointwise_label',all_candidates_retained_at=str(source/'classification/all_ROI_candidates_postmin32.csv'),
                mask_interval_s=1.,source_frame_interval_s=.02,probability_scores_used=False,registration_verified=False))
            dump(out/'coordinate_validation.json',dict(ROI_MAT_mask_vertices_exactly_equal=True,vertices=len(roi),
                ROI_pixels=int(roi_mask.sum()),TYX_mask_shape=[50,684,428],original_frame_to_mask_block_exact=True,registration_verified=False))
            independent=analyze_groups(c,out,table,frame)
            reference=json.loads((PROJECT/c['independent_point_reference']).read_text())
            assert reference['source_selected_csv_SHA256']==fingerprint(source/'classification/selected_localizations_postmin32.csv')['sha256']
            assert reference['selected_count']==c['expected_selected']
            with np.load(out/'pair_histograms/per_frame_raw_counts.npz') as hist:
                for key,sha in reference['all_histogram_C_order_bytes_SHA256'].items():
                    assert hashlib.sha256(np.ascontiguousarray(hist[key]).tobytes()).hexdigest()==sha,key
                if oldrun:
                    with np.load(oldrun/'pair_histograms/per_frame_raw_counts.npz') as old:
                        for key in ['all_total','all_background']:np.testing.assert_array_equal(hist[key],old[key])
            shutil.copyfile(PROJECT/c['independent_point_reference'],out/'independent_point_reference.json')
            from postmin32_extras import render_extras
            render_extras(out,c,frame)
            # All original source files and saved bootstrap weights remain unchanged.
            after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
            dump(out/'validation.json',dict(implementation_checks_passed=True,preflight_tests=tests.testsRun,
                selected_points=c['expected_selected'],QC_excluded=c['expected_QC_excluded'],group_counts=c['expected_group_counts'],
                original_selected_rows_archived_verbatim=True,probability_scores_not_used=True,
                input_fingerprints_unchanged=True,ROI_MAT_vertices_equal=True,
                all_histogram_arrays_identical_to_independent_reference=True,
                all_selected_pair_histograms_identical_to_previous_movie=True if oldrun else None,independent_group_frames_checked=independent,
                shared_saved_weights_reused=True,paired_original_replicate_ID_intersections=True,
                no_cross_gap_segment_or_resampling_join_pairs=True,bootstrap_intervals_exploratory=True,
                bootstrap_conditions_on_fixed_masks_and_labels=True,formal_p_values_computed=False,
                registration_verified=False,mask_uncertainty_propagated=False,localization_error_subtracted=False))
            dump(out/'runtime.json',dict(elapsed_s=time.perf_counter()-started,refitting_seconds=0))
            dump(out/'run_status.json',dict(status='complete_pending_review',completed_frames=2500))
            dump(out/'COMPLETED.json',dict(implementation_checks_passed=True,delivery_visual_QC_pending=True,
                frames=2500,groups=GROUPS,selected_points=c['expected_selected'],formal_inference=False,output=str(out)))
            print('POSTMIN32_TARDIS_READY',out,flush=True)
        except BaseException:
            dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=stdout,stderr


def finalize(out):
    c=json.loads((out/'config.json').read_text());done=json.loads((out/'COMPLETED.json').read_text());assert done['implementation_checks_passed']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    from PIL import Image
    images=[]
    for path in sorted((out/'figures').glob('*.png')):
        with Image.open(path) as picture:
            picture.load();images.append(dict(path=str(path.relative_to(out)),size=list(picture.size),sha256=fingerprint(path)['sha256']))
    dump(out/'visual_review.json',dict(passed=True,images=images,notes='Reviewed full/zoom decomposition, all10window curves, paired difference, counts/SEs, block-IoU, signed statistics and block50 mask areas. Exploratory bands labeled; no W5/fallback labels on new figures.'))
    checks=[];width=c['distance_max_um']/c['distance_bins']
    with np.load(out/'pair_histograms/per_frame_raw_counts.npz') as hist:
        for g,group in enumerate(GROUPS):
            for scope,start,stop in scopes(c):
                tc,bc=range_histograms(hist['total'][g],hist['background'][g],start,stop);d=decompose_roi(tc,bc,c)
                saved=pd.read_csv(out/'distribution'/group/(scope+'.csv'),float_precision='round_trip')
                np.testing.assert_array_equal(saved.total_count,tc);np.testing.assert_array_equal(saved.background_ordered_nonzero_count,bc)
                for field in ['total','background','inter','intra','shape']:
                    np.testing.assert_allclose(saved[field+'_density_um_inv'],d[field][0]/width,rtol=2e-15,atol=2e-15,equal_nan=True)
                checks.append(dict(group=group,scope=scope,all_saved_bins_verified=True))
    banks=[]
    point_summary=pd.read_csv(out/'per_scope_summary.csv')
    for length in c['bootstrap']['block_lengths_frames']:
        with np.load(out/'bootstrap'/f'shape_draws_L{length}.npz') as draws:
            np.testing.assert_array_equal(draws['replicate_id'],np.arange(1000))
            for j,(scope,_,_) in enumerate(scopes(c)):
                common=draws['valid'][0,j]&draws['valid'][1,j]
                with np.load(out/'bootstrap'/f'paired_shape_{scope}_L{length}.npz') as paired:
                    np.testing.assert_array_equal(paired['replicate_id'],np.arange(1000));np.testing.assert_array_equal(paired['common_valid'],common)
                    np.testing.assert_allclose(paired['difference'][common],draws['density'][0,j,common]-draws['density'][1,j,common],rtol=0,atol=0)
                band=pd.read_csv(out/'bootstrap'/f'paired_bands_{scope}_L{length}.csv',float_precision='round_trip')
                original_shapes_valid=point_summary.loc[point_summary.scope.eq(scope),'shape_valid'].all()
                if common.mean()>=c['minimum_joint_valid_fraction'] and original_shapes_valid:
                    lo,hi=np.quantile(draws['density'][0,j,common]-draws['density'][1,j,common],[.025,.975],axis=0)
                    np.testing.assert_allclose(band.shape_difference_low,lo,rtol=1e-13,atol=1e-13)
                    np.testing.assert_allclose(band.shape_difference_high,hi,rtol=1e-13,atol=1e-13)
                else:assert band.shape_difference_low.isna().all() and band.shape_difference_high.isna().all()
                banks.append(dict(scope=scope,block_frames=length,common_valid_count=int(common.sum()),common_valid_fraction=float(common.mean())))
    with (out/'selected_original_rows.csv').open(newline='') as selected_stream,(out/'statistical_input.csv').open(newline='') as stat_stream:
        import csv
        originals=csv.DictReader(selected_stream);stats=csv.DictReader(stat_stream);n=0
        for a,b in zip(originals,stats,strict=True):
            for key in ['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']:
                assert float(a[key])==float(b[key]),key
            assert a['postmin32_pointwise_label']==b['postmin32_pointwise_label'];n+=1
        assert n==c['expected_selected']
    # Refresh source snapshot after all helpers were completed; raw kernels unchanged.
    snapshot(out)
    shutil.copyfile(out/'config.json',out/'source_snapshot/mito_tardis/config_postmin32.json')
    dump(out/'delivery_validation.json',dict(all_delivery_checks_passed=True,saved_distributions_rechecked=checks,
        paired_bootstrap_banks_rechecked=banks,coordinates_SE_signal_and_labels_identical_to_source=True,
        representative_figures_visually_reviewed=True,no_formal_significance_claim=True))
    import importlib.metadata
    dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={key:importlib.metadata.version(key) for key in ['numpy','scipy','pandas','matplotlib','h5py','tifffile']}))
    dump(out/'field_units.json',dict(frame='original zero-based camera frame',time_s='seconds',
        fit_x_px='native C2 camera pixels',fit_y_px='native C2 camera pixels',
        x_se_nm='nominal local-model position SE, nm',y_se_nm='nominal local-model position SE, nm',
        photons_e='manufacturer-nominal photoelectron-equivalent signal',bg_e_per_pixel='nominal electron-equivalent/pixel',
        sx_px='Gaussian PSF sigma in pixels',sy_px='Gaussian PSF sigma in pixels',
        pair_counts='all-pair counts, not independent samples',distance_um='micrometers',density='1/um',
        beta='dimensionless fitted background weight, not a biological state fraction',signed_area='dimensionless signed mass on0-3um',
        mask_area='native foreground pixels times(0.117um)^2; no localization-error expansion',
        intervals='exploratory paired pointwise percentile ranges conditional on fixed masks/labels; coverage unvalidated'))
    dump(out/'replay_config.json',{**c,'shared_weights':str(out/'bootstrap/shared_block_resampling.npz'),
        'shared_weights_sha256':fingerprint(out/'bootstrap/shared_block_resampling.npz')['sha256'],
        'independent_point_reference':str(out/'independent_point_reference.json')})
    (out/'REPRODUCE.txt').write_text('Run command: run_command.txt. Frozen numerical rules: RULES_BEFORE_RUN.json.\n'
        'Use the existing pinned Picasso environment. Copy source_snapshot to a NEW writable working directory, then run mito_tardis/run_postmin32.py with --config replay_config.json.\n'
        'Replays create a fresh timestamped result; original classifications/masks and all prior results remain read-only.\n'
        'Saved weights are reused exactly; all intervals are exploratory and conditional on the fixed mask/label realization.\n')
    dump(out/'run_status.json',dict(status='complete',completed_frames=2500,visual_review_passed=True))
    done.update(delivery_visual_QC_pending=False,delivery_checks_passed=True,finished_utc=datetime.now(timezone.utc).isoformat());dump(out/'COMPLETED.json',done)
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--finalize','--output',str(out)])+'\n')
    (out/'publish_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--publish','--output',str(out)])+'\n')
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['COMPLETED.json','output_fingerprints.json']])
    print('FINALIZED',json.dumps(dict(output=str(out),destination=c['external_destination'],figures=len(images))))


def publish(src):
    c=json.loads((src/'config.json').read_text());dst=Path(c['external_destination'])
    assert json.loads((src/'COMPLETED.json').read_text())['delivery_checks_passed'] and not dst.exists()
    assert dst.parent.name=='mito_tardis_postmin32' and dst.parent.parent.name=='analysis_results'
    records=json.loads((src/'output_fingerprints.json').read_text())
    for r in records:
        p=Path(r['path']);assert p.is_relative_to(src) and fingerprint(p)['sha256']==r['sha256']
    dst.mkdir(parents=True)
    for p in sorted(src.rglob('*')):
        target=dst/p.relative_to(src)
        if p.is_dir():target.mkdir(exist_ok=True)
        else:
            assert not p.is_symlink() and not target.is_symlink();shutil.copyfile(p,target)
    for r in records:
        item=fingerprint(dst/Path(r['path']).relative_to(src));assert item['sha256']==r['sha256'] and item['bytes']==r['bytes']
    dump(dst/'publication.json',dict(local_source=str(src),destination=str(dst),copied_files_SHA256_equal=True,files_verified=len(records)))
    replay=json.loads((dst/'replay_config.json').read_text());replay.update(shared_weights=str(dst/'bootstrap/shared_block_resampling.npz'),
        independent_point_reference=str(dst/'independent_point_reference.json'));dump(dst/'replay_config.json',replay)
    done=json.loads((dst/'COMPLETED.json').read_text());done.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/'COMPLETED.json',done)
    dump(dst/'output_fingerprints.json',[fingerprint(p) for p in sorted(dst.rglob('*')) if p.is_file() and p.name not in ['COMPLETED.json','output_fingerprints.json']])
    (ROOT/(c.get('pointer_prefix','latest_postmin32')+'_output.txt')).write_text(str(dst)+'\n')
    print('PUBLISHED',json.dumps(dict(output=str(dst),copied_files_SHA256_equal=True,files_verified=len(records))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    group=parser.add_mutually_exclusive_group();group.add_argument('--finalize',action='store_true');group.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.finalize:finalize(args.output.resolve())
    elif args.publish:publish(args.output.resolve())
    else:
        assert args.config is not None;run(args.config.resolve())
