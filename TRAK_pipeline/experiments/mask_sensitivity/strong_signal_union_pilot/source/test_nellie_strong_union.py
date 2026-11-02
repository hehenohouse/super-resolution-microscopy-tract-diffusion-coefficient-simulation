"""Default Nellie broad plus conservative raw-C1 intensity supplement; isolated masks-only test."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,platform,sys,time
from pathlib import Path
from datetime import datetime,timezone
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);args=ap.parse_args()
config_path=Path(args.config).resolve();cfg=json.loads(config_path.read_text());run=Path(cfg['run_directory']).resolve()
if not run.is_relative_to(Path('/Volumes/My Book')):raise ValueError('Controlled outputs must stay on My Book')
for key,folder in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','XDG_CACHE_HOME':'xdg'}.items():
    dest=run/'cache'/folder;dest.mkdir(parents=True,exist_ok=True);os.environ[key]=str(dest)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYDANTIC_DISABLE_PLUGINS='__all__',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import tempfile
tempfile.tempdir=str(run/'cache/tmp')
import numpy as np,tifffile
sys.path.insert(0,str(run/'source'))
from support50_postmin32_core import process_counts

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def uid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def desc(path):
    p=Path(path).resolve();s=p.stat();d={'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    if s.st_size<=1024*1024:d['sha256']=sha(p)
    return d
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def table(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(msg):
    line=utc()+' | '+msg
    with (run/'logs/numerical.log').open('a') as f:f.write(line+'\n')
    print(line,flush=True)
def load_broad(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(int(v) for v in z['shape']);ids=np.array(z['original_frame_ids'],copy=True)
        if shape!=(50,684,428) or str(z['bitorder'])!='little':raise ValueError('Baseline broad axes/storage differ')
        broad=np.unpackbits(z['packed'],axis=1,count=684*428,bitorder='little').reshape(shape).astype(bool)
    return broad,ids
def save_packed(path,array,ids):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,packed=np.packbits(array.reshape(len(array),-1),axis=1,bitorder='little'),shape=np.array(array.shape,np.int32),original_frame_ids=np.asarray(ids,np.int32),bitorder=np.array('little'))
def strong_signal(frame):
    valid=np.isfinite(frame)&(frame>0);v=frame[valid].astype(np.float64)
    if v.size<cfg['strong_signal_rule']['minimum_valid_pixels']:raise ValueError('Strong supplement unavailable: too few valid original C1 pixels')
    m=float(np.median(v));mad=float(np.median(np.abs(v-m)));spread=mad/0.6744897501960817
    if not np.isfinite(spread) or spread<=0:raise ValueError('Strong supplement unavailable: undefined/zero robust spread')
    threshold=m+cfg['strong_signal_rule']['k']*spread
    mask=valid&(frame>threshold)
    return mask,dict(reference_positive_pixels=int(v.size),median_intensity_ADU=m,MAD_ADU=mad,robust_intensity_scale_ADU=spread,k=cfg['strong_signal_rule']['k'],threshold_ADU=threshold)
# Formula/storage controls only; no scientific ground-truth or cross-system gate.
synthetic=np.arange(2048,dtype=np.uint16).reshape(32,64);sx,sv=strong_signal(synthetic)
assert np.array_equal(sx,(synthetic>sv['threshold_ADU'])&(synthetic>0))
source=Path(cfg['source_output']);c1=Path(cfg['source_C1']);starts=cfg['block_starts'];methods=cfg['methods']
roi=np.load(source/'roi/roi_mask.npy',allow_pickle=False);vertices=np.load(source/'roi/vertices.npy',allow_pickle=False)
job=json.loads((source/'job.json').read_text())
if roi.dtype!=bool or roi.shape!=(684,428) or int(roi.sum())!=job['ROI_pixels']:raise ValueError('ROI differs')
np.save(run/'masks/roi_mask.npy',roi);np.save(run/'masks/roi_vertices_xy.npy',vertices)
protected=[c1,Path(cfg['source_ROI']),source/'job.json',source/'COMPLETED.json',source/'classification_summary.json',source/'masks/clean_full_mask.npy',source/'masks/analysis_ROI_mask.npy',source/'masks/foreground_count.npy',source/'roi/roi_mask.npy',source/'roi/vertices.npy']
protected += [source/'masks/broad_packed'/f'b{s//50:03d}.npz' for s in starts]
protected += [Path(cfg['prior_reference_test'])/'data/preprocessing_limits.csv']
before=[desc(p) for p in protected]
dump(run/'source_records.json',dict(protected_sources=before,config_sha256=sha(config_path),source_input_policy=cfg['input_policy']))
base_full=np.load(source/'masks/clean_full_mask.npy',mmap_mode='r');base_roi=np.load(source/'masks/analysis_ROI_mask.npy',mmap_mode='r');base_counts=np.load(source/'masks/foreground_count.npy',mmap_mode='r')
products={m['id']:{k:[] for k in ['final','counts','support','cleaned','holes']} for m in methods}
threshold_rows=[];mapping=[];samples=[];zero_parts=[];summary_rows=[];pair_rows=[];bright_rows=[];stage_rows=[];stage_bright=[];detail_maxima=[]
with (Path(cfg['prior_reference_test'])/'data/preprocessing_limits.csv').open(newline='') as f:original_p99={int(r['original_frame']):float(r['effective_high_ADU']) for r in csv.DictReader(f)}
clock=time.perf_counter()
log('Starting conservative raw-C1 strong-signal union: k=6 fixed; baseline broad reused; zero Nellie/C2/localization/assignment/TARDIS calls.')
with tifffile.TiffFile(c1) as movie:
    if len(movie.pages)!=2500 or movie.pages[0].shape!=(684,428):raise ValueError('Original C1 grid/time differs')
    for ib,start in enumerate(starts):
        block=start//50;ids=np.arange(start,start+50,dtype=np.int32)
        broad_path=source/'masks/broad_packed'/f'b{block:03d}.npz';broad,broad_ids=load_broad(broad_path)
        if not np.array_equal(ids,broad_ids):raise ValueError('Baseline broad original IDs differ')
        signal=[];raw_values=[];block_thresholds=[]
        for j,f in enumerate(ids):
            raw=movie.pages[int(f)].asarray()
            if raw.dtype!=np.uint16 or raw.shape!=(684,428):raise ValueError('Original frame dtype/grid differs')
            strong,values=strong_signal(raw);signal.append(strong)
            values.update(original_frame=int(f),time_s=float(f)*.02,mask_block=block,strong_pixels_full=int(strong.sum()),strong_pixels_ROI=int((strong&roi).sum()),new_broad_pixels_full=int((strong&~broad[j]).sum()),new_broad_pixels_ROI=int((strong&~broad[j]&roi).sum()))
            threshold_rows.append(values);block_thresholds.append(values)
            mapping.append(dict(page_index=len(mapping),original_frame=int(f),time_s=float(f)*.02,mask_block=block,local_block_index=ib,original_frame_start=start,original_frame_stop_exclusive=start+50))
            raw_values.append(raw)
            if int(f) in cfg['sample_frames']:
                expected=raw>values['threshold_ADU']
                if not np.array_equal(strong,expected&(raw>0)):raise ValueError('Strong sample formula mismatch')
                samples.append(dict(original_frame=int(f),raw_pixels_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),threshold_ADU=values['threshold_ADU'],formula_match=True,original_C1_not_rewritten=True))
        raw=np.stack(raw_values);signal=np.stack(signal);union=broad|signal
        if np.any(broad&~union):raise ValueError('Per-frame union removed baseline foreground')
        zero_parts.append(np.packbits((raw==0).reshape(50,-1),axis=1,bitorder='little'))
        save_packed(run/'masks/A_default/broad_packed'/f'b{block:03d}.npz',broad,ids)
        save_packed(run/'masks/B_strong_union/strong_packed'/f'b{block:03d}.npz',signal,ids)
        save_packed(run/'masks/B_strong_union/broad_packed'/f'b{block:03d}.npz',union,ids)
        counts_a=broad.sum(axis=0,dtype=np.uint8);counts_b=union.sum(axis=0,dtype=np.uint8)
        if not np.array_equal(counts_a,base_counts[block]) or np.any(counts_b<counts_a):raise ValueError('Support count consistency/monotonicity failed')
        strong_counts=signal.sum(axis=0,dtype=np.uint8)
        (run/'masks/B_strong_union/strong_count_blocks').mkdir(exist_ok=True)
        np.save(run/'masks/B_strong_union/strong_count_blocks'/f'b{block:03d}.npy',strong_counts)
        stages_a=process_counts(counts_a,support_minimum=10,minimum_foreground=32,hole_limit=64,frame_count=50)
        stages_b=process_counts(counts_b,support_minimum=10,minimum_foreground=32,hole_limit=64,frame_count=50)
        if not np.array_equal(stages_a['final_mask'],base_full[block]) or not np.array_equal(base_roi[block],base_full[block]&roi):raise ValueError('Baseline final/ROI recipe differs')
        if np.any(stages_a['final_mask']&~stages_b['final_mask']):raise ValueError('Hybrid final unexpectedly removed baseline foreground')
        stages={'A_default':stages_a,'B_strong_union':stages_b}
        for mid,st in stages.items():
            out=products[mid]
            for key,value in [('final',st['final_mask']),('counts',counts_a if mid=='A_default' else counts_b),('support',st['support_mask']),('cleaned',st['post_min32_mask']),('holes',st['filled_holes'])]:out[key].append(value)
        bright_region=(raw[0]>original_p99[start])&roi;bright_n=int(bright_region.sum());detail_maxima.append(int(raw[0].max()))
        for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
            a=stages_a['final_mask']&gate;b=stages_b['final_mask']&gate
            n0=int(a.sum());n1=int(b.sum());inter=int((a&b).sum());union_n=int((a|b).sum());added=int((b&~a).sum());removed=int((a&~b).sum())
            assert removed==0 and n0+added-removed==n1
            for mid,mask in [('A_default',a),('B_strong_union',b)]:
                n=int(mask.sum());summary_rows.append(dict(scope=scope,block_index=block,original_frame_start=start,original_frame_stop_exclusive=start+50,method=mid,pixels=n,area_um2=n*.117**2))
            pair_rows.append(dict(scope=scope,block_index=block,original_frame_start=start,original_frame_stop_exclusive=start+50,reference_method='A_default',test_method='B_strong_union',old_pixels=n0,new_pixels=n1,area_old_um2=n0*.117**2,area_new_um2=n1*.117**2,delta_pixels=n1-n0,delta_area_um2=(n1-n0)*.117**2,delta_percent=100*(n1/n0-1) if n0 else None,intersection_pixels=inter,union_pixels=union_n,added_pixels=added,removed_pixels=removed,iou=inter/union_n if union_n else 1.,dice=2*inter/(n0+n1) if n0+n1 else 1.))
        for mid,st in stages.items():
            hit=int((bright_region&st['final_mask']).sum())
            bright_rows.append(dict(original_frame=start,block_index=block,method=mid,raw_brightness_cutoff_ADU=original_p99[start],region_definition='original full-frame P99 then ROI; same pixels across methods; not ground-truth accuracy',bright_region_pixels=bright_n,covered_bright_pixels=hit,uncovered_bright_pixels=bright_n-hit,coverage_percentage=100*hit/bright_n if bright_n else None))
            masks={'block_first_single_frame_broad':broad[0] if mid=='A_default' else (broad[0]|signal[0]),'block_support_ge10':st['support_mask'],'block_postmin32':st['post_min32_mask'],'block_final_holes_lt64':st['final_mask']}
            for stage,mask in masks.items():
                for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
                    n=int((mask&gate).sum());stage_rows.append(dict(original_frame_start=start,block_index=block,method=mid,stage=stage,scope=scope,pixels=n,area_um2=n*.117**2,temporal_scope='one original frame' if stage=='block_first_single_frame_broad' else '50-frame support block'))
                hit=int((bright_region&mask).sum());stage_bright.append(dict(original_frame=start,block_index=block,method=mid,stage=stage,bright_region_pixels=bright_n,covered_bright_pixels=hit,coverage_percentage=100*hit/bright_n if bright_n else None,temporal_scope='one original frame' if stage=='block_first_single_frame_broad' else '50-frame support block'))
        artifact_id=uid(dict(method_version=cfg['method_version'],raw_C1=before[0],baseline_broad=desc(broad_path),original_frame_ids=ids.tolist(),k=6.0,MAD_normalizer=.6744897501960817,positive_pixels_only=True,support_frames=50,support_minimum=10,foreground_min32=32,hole_exclusive_max64=64,FG_connectivity=8,BG_connectivity=4))
        dump(run/'checkpoints'/f'block_{block:03d}.json',dict(scientific_artifact_id=artifact_id,original_frame_start=start,original_frame_stop_exclusive=start+50,baseline_ROI_pixels=int((stages_a['final_mask']&roi).sum()),hybrid_ROI_pixels=int((stages_b['final_mask']&roi).sum()),removed_pixels=0,completed_utc=utc()))
        log(f'Block {block} complete: ROI pixels {int((stages_a["final_mask"]&roi).sum())} -> {int((stages_b["final_mask"]&roi).sum())}; removed0; threshold range {min(r["threshold_ADU"] for r in block_thresholds):.2f}-{max(r["threshold_ADU"] for r in block_thresholds):.2f} ADU')
for mid,data in products.items():
    dest=run/'masks'/mid
    np.save(dest/'clean_full_mask.npy',np.stack(data['final']));np.save(dest/'analysis_ROI_mask.npy',np.stack(data['final'])&roi);np.save(dest/'foreground_count.npy',np.stack(data['counts']))
    np.savez_compressed(dest/'postprocess_stages.npz',support_mask=np.stack(data['support']),post_min32_mask=np.stack(data['cleaned']),filled_holes=np.stack(data['holes']))
    dump(dest/'metadata.json',dict(method=mid,method_version=cfg['method_version'],axes='TYX',shape=[3,684,428],original_blocks=[s//50 for s in starts],frame_mapping='../../data/frame_mapping.csv',pixel_size_nm=117,recipe=cfg['recipe'],original_zero_regions='../original_zero_regions_packed.npz',threshold_rule=cfg['strong_signal_rule'] if mid=='B_strong_union' else None,registration_status=cfg['registration_status'],baseline_nellie_reused=True))
table(run/'data/frame_mapping.csv',mapping);table(run/'data/strong_thresholds.csv',threshold_rows);table(run/'data/method_summary.csv',summary_rows);table(run/'data/pairwise_mask_comparison.csv',pair_rows);table(run/'data/bright_pixel_diagnostic.csv',bright_rows);table(run/'data/stage_areas.csv',stage_rows);table(run/'data/bright_pixel_stage_coverage.csv',stage_bright)
np.savez_compressed(run/'masks/original_zero_regions_packed.npz',packed=np.concatenate(zero_parts),shape=np.array([150,684,428]),original_frame_ids=np.array([r['original_frame'] for r in mapping],np.int32),bitorder=np.array('little'))
high=float((max(detail_maxima)//100+1)*100)
dump(run/'data/display_settings.json',dict(low_ADU=350,high_ADU=high,gamma=.5,frames=starts,original_frame_maxima_ADU=detail_maxima,bright_pixels_clipped=0,description='Display only. Shared map; upper strictly above all three full-frame maxima. This display map is not used by segmentation.'))
if before!=[desc(p) for p in protected]:raise ValueError('Protected input or original results changed')
dump(run/'source_records.json',dict(protected_sources=before,config_sha256=sha(config_path),protected_sources_unchanged=True,completed_utc=utc()))
dump(run/'environment.json',dict(python=sys.version,platform=platform.platform(),analysis_versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','tifffile']},baseline_nellie_versions_as_reported=json.loads((source/'masks/metadata.json').read_text())['native_versions'],runtime_nellie_imported=False,cache_environment={k:os.environ[k] for k in ['TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME']}))
dump(run/'validation.json',dict(samples=samples,n_samples=len(samples),full_blocks=3,frames=150,original_sources_unchanged=True,baseline_reused_exact=True,baseline_broad_subset_of_union=True,count_maps_monotone=True,final_mask_contains_baseline=True,removed_final_pixels=0,ROI_intersection_exact=True,axes='TYX',registration_status=cfg['registration_status'],source_pixels_unprocessed=True,upstream_scientific_calls=dict(nellie=0,localization=0,assignment=0,TARDIS=0),scope='ten raw-frame formula checks plus computation-required consistency; no all2500 replay'))
dump(run/'results_summary.json',dict(sample_id=cfg['sample_id'],frames=150,ROI_areas_um2=[r for r in summary_rows if r['scope']=='ROI'],bright_pixel_coverage=bright_rows,pairwise_ROI=[r for r in pair_rows if r['scope']=='ROI'],threshold_rule=cfg['strong_signal_rule'],default_pipeline_changed=False,upstream_scientific_calls=dict(nellie=0,localization=0,assignment=0,TARDIS=0),interpretation='This is a conservative C1-only supplement sensitivity test, not a ground-truth accuracy comparison. Final additions can include changes induced by unchanged cleanup.'))
dump(run/'NUMERICAL_COMPLETE.json',dict(complete=True,frames=150,original_blocks=[s//50 for s in starts],baseline_reused=True,new_mask_methods=1,Nellie_calls=0,elapsed_s=time.perf_counter()-clock))
log(f'Numerical complete in {time.perf_counter()-clock:.1f}s; zero new Nellie/localization/assignment/TARDIS calls.')
print('ROI areas and deltas:',[(r['original_frame_start'],r['area_old_um2'],r['area_new_um2'],r['delta_percent']) for r in pair_rows if r['scope']=='ROI'],flush=True)
print('Bright-region coverage:',[(r['original_frame'],r['method'],r['coverage_percentage']) for r in bright_rows],flush=True)
