"""Isolated original-C1 Nellie additive-scale experiment; masks only."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,threading,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);args=ap.parse_args()
cfg_path=Path(args.config).resolve();cfg=json.loads(cfg_path.read_text());run=Path(cfg['run_directory']).resolve()
if not run.is_relative_to(Path('/Volumes/My Book')):raise ValueError('All controlled outputs must stay on My Book')
for key,folder in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','XDG_CACHE_HOME':'xdg'}.items():
    p=run/'cache'/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYDANTIC_DISABLE_PLUGINS='__all__',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import tempfile
tempfile.tempdir=str(run/'cache/tmp')
import numpy as np,tifffile
sys.path.insert(0,str(run/'source'))
from support50_postmin32_core import process_counts
def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def uid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def desc(path):
    p=Path(path).resolve();s=p.stat();r=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if s.st_size<=1024*1024:r['sha256']=sha(p)
    return r
lock=threading.Lock()
def log(msg):
    line=utc()+' | '+msg
    with lock:
        with (run/'logs/numerical.log').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
def csvwrite(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
source=Path(cfg['source_output']);c1=Path(cfg['source_C1'])
roi=np.load(source/'roi/roi_mask.npy',allow_pickle=False);vertices=np.load(source/'roi/vertices.npy',allow_pickle=False)
job=json.loads((source/'job.json').read_text())
if roi.dtype!=bool or roi.shape!=(684,428) or int(roi.sum())!=job['ROI_pixels']:raise ValueError('ROI differs')
np.save(run/'masks/roi_mask.npy',roi);np.save(run/'masks/roi_vertices_xy.npy',vertices)
protected=[c1,Path(cfg['source_ROI']),source/'job.json',source/'COMPLETED.json',source/'classification_summary.json',
 source/'masks/clean_full_mask.npy',source/'masks/analysis_ROI_mask.npy',source/'masks/foreground_count.npy',source/'roi/roi_mask.npy',source/'roi/vertices.npy']
protected += [source/'masks/broad_packed'/f'b{s//50:03d}.npz' for s in cfg['block_starts']]
protected += [Path(cfg['prior_reference_test'])/'data/preprocessing_limits.csv']
native_site=Path(cfg['runtime']['nellie_python']).parent.parent/'lib/python3.11/site-packages'
protected += [native_site/'nellie'/p for p in ['im_info/verifier.py','segmentation/filtering.py','segmentation/labelling.py']]
before=[desc(p) for p in protected]
dump(run/'source_records.json',dict(protected_sources=before,config_sha256=sha(cfg_path),source_input_policy=cfg['input_policy']))
science_id={'method_version':cfg['method_version'],'source_C1':desc(c1),'native_parameters':cfg['native_parameters'],
 'recipe':cfg['recipe'],'support_frames':50,'support_minimum':10,'foreground_minimum':32,'enclosed_holes_exclusive_maximum':64}
dump(run/'environment.json',dict(python=sys.version,platform=platform.platform(),analysis_versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','tifffile']},
 cache_environment={k:os.environ[k] for k in ['TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME']},
 source_audit_note='Source hashes are for audit, not scientific cache identity. Native TIFF dependency bootstrap reused without installs.'))
cleanup_site=Path(cfg['runtime']['cleanup_python']).parent.parent/'lib/python3.11/site-packages'
native_env=dict(os.environ);native_env.pop('PYTHONPYCACHEPREFIX',None)
methods=cfg['methods'];starts=cfg['block_starts'];blocks=[s//50 for s in starts]
old_full=np.load(source/'masks/clean_full_mask.npy',mmap_mode='r');old_roi=np.load(source/'masks/analysis_ROI_mask.npy',mmap_mode='r');old_counts=np.load(source/'masks/foreground_count.npy',mmap_mode='r')
all_data={m['id']:{'final':[],'counts':[],'support':[],'cleaned':[],'holes':[]} for m in methods}
mapping=[];sample_checks=[];zero_parts=[];threshold_rows=[];native_starts=[]
start_clock=time.perf_counter()
def unpack(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(int(x) for x in z['shape']);ids=np.array(z['original_frame_ids'],copy=True)
        if shape[1:]!=(684,428) or str(z['bitorder'])!='little':raise ValueError('Packed broad format differs')
        broad=np.unpackbits(z['packed'],axis=1,count=684*428,bitorder='little').reshape(shape).astype(bool)
    return broad,ids
def pack(path,broad,ids):
    np.savez_compressed(path,packed=np.packbits(broad.reshape(len(broad),-1),axis=1,bitorder='little'),shape=np.array(broad.shape,np.int32),original_frame_ids=np.asarray(ids,np.int32),bitorder=np.array('little'))
def process_native(method,raw,start):
    mid=method['id'];per_method=run/'masks'/mid
    for p in [per_method/'broad_packed',per_method/'native_batches']:p.mkdir(parents=True,exist_ok=True)
    pieces=[]
    for off in [0,25]:
        f0=start+off;tag=f'frames{f0:04d}_{f0+24:04d}'
        permanent=per_method/'native_batches'/tag;permanent.mkdir(exist_ok=True)
        marker=permanent/'COMPLETED.json';cache_id=uid(dict(science_id,method={k:method[k] for k in ('id','nominal_max_radius_um','sigmas_pixels')},original_frames=list(range(f0,f0+25))))
        if marker.is_file() and (permanent/'broad_packed.npz').is_file():
            if json.loads(marker.read_text())['scientific_cache_id']!=cache_id:raise ValueError('Native cache identity differs')
            broad,ids=unpack(permanent/'broad_packed.npz');log(mid+f' reused {f0}-{f0+24}')
        else:
            scratch=run/'_work'/mid/tag
            if scratch.exists():shutil.rmtree(scratch)
            scratch.mkdir(parents=True)
            tifffile.imwrite(scratch/'input.ome.tif',raw[off:off+25],ome=True,photometric='minisblack',metadata=dict(axes='TYX',PhysicalSizeX=.117,PhysicalSizeY=.117,PhysicalSizeXUnit='µm',PhysicalSizeYUnit='µm',TimeIncrement=.02,TimeIncrementUnit='s'))
            dump(scratch/'method.json',method)
            t=time.perf_counter();log(mid+f' broad starting {f0}-{f0+24}')
            command=[cfg['runtime']['nellie_python'],'-B','-u',str(run/'source/native_bootstrap.py'),str(cleanup_site),str(run/'source/scale_worker.py'),str(run/'_work'),str(scratch),str(permanent/'native_dependency_runtime.json')]
            with (permanent/'nellie.log').open('w') as logstream:
                subprocess.run(command,cwd=run,env=native_env,stdout=logstream,stderr=subprocess.STDOUT,check=True)
            broad=np.load(scratch/'broad_mask.npy',allow_pickle=False)
            if broad.shape!=(25,684,428) or broad.dtype!=bool:raise ValueError('Native broad grid differs')
            actual=json.loads((scratch/'actual_parameters.json').read_text())
            if actual['sigmas_pixels']!=method['sigmas_pixels'] or actual['native_min_area']!=4 or actual['actual_low_memory']:
                raise ValueError('Native method settings differ')
            ids=np.arange(f0,f0+25,dtype=np.int32);pack(permanent/'broad_packed.npz',broad,ids)
            for name in ['thresholds.json','actual_parameters.json','sampled_filter_response.npz']:shutil.copy2(scratch/name,permanent/name)
            dump(marker,dict(scientific_cache_id=cache_id,original_frame_start=f0,original_frame_stop_exclusive=f0+25,elapsed_s=time.perf_counter()-t,completed_utc=utc()))
            shutil.rmtree(scratch);log(mid+f' broad complete {f0}-{f0+24}; {time.perf_counter()-t:.1f}s')
        if not np.array_equal(ids,np.arange(f0,f0+25)):raise ValueError('Native frame IDs differ')
        pieces.append(broad)
    result=np.concatenate(pieces)
    pack(per_method/'broad_packed'/f'b{start//50:03d}.npz',result,np.arange(start,start+50))
    return result
log('Starting original-C1 additive-scale test: baseline reused, two appended-scale methods, three disjoint 50-frame blocks.')
with tifffile.TiffFile(c1) as movie:
    if len(movie.pages)!=2500 or movie.pages[0].shape!=(684,428):raise ValueError('Source C1 shape/time differs')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for ib,start in enumerate(starts):
            raw=np.stack([movie.pages[f].asarray() for f in range(start,start+50)])
            if raw.dtype!=np.uint16:raise ValueError('Source C1 dtype differs')
            input_path=run/'inputs'/f'C1_original_frames{start:04d}_{start+49:04d}.ome.tif'
            if not input_path.exists():
                tifffile.imwrite(input_path,raw,ome=True,photometric='minisblack',metadata=dict(axes='TYX',PhysicalSizeX=.117,PhysicalSizeY=.117,PhysicalSizeXUnit='µm',PhysicalSizeYUnit='µm',TimeIncrement=.02,TimeIncrementUnit='s'))
            zeros=np.packbits((raw==0).reshape(50,-1),axis=1,bitorder='little');zero_parts.append(zeros)
            with tifffile.TiffFile(input_path) as copied:
                for j,f in enumerate(range(start,start+50)):
                    mapping.append(dict(page_index=len(mapping),original_frame=f,time_s=f*.02,mask_block=f//50,local_block_index=ib,original_frame_start=start,original_frame_stop_exclusive=start+50))
                    if f in cfg['sample_frames']:
                        if not np.array_equal(copied.pages[j].asarray(),raw[j]):raise ValueError('Raw input copy mismatch')
                        sample_checks.append(dict(original_frame=f,unchanged_original_C1_copy=True,raw_pixel_sha256=hashlib.sha256(raw[j].tobytes()).hexdigest()))
            baseline_path=source/'masks/broad_packed'/f'b{start//50:03d}.npz'
            baseline_broad,ids=unpack(baseline_path)
            if not np.array_equal(ids,np.arange(start,start+50)):raise ValueError('Baseline original IDs differ')
            a=run/'masks/A_default/broad_packed';a.mkdir(exist_ok=True);shutil.copy2(baseline_path,a/baseline_path.name)
            futures={m['id']:pool.submit(process_native,m,raw,start) for m in methods if not m['reused']}
            broad_by_method={'A_default':baseline_broad}
            for method in methods:
                mid=method['id']
                if mid in futures:broad_by_method[mid]=futures[mid].result()
                counts=broad_by_method[mid].sum(axis=0,dtype=np.uint8)
                stages=process_counts(counts,support_minimum=10,minimum_foreground=32,hole_limit=64,frame_count=50)
                if method['reused']:
                    if not np.array_equal(counts,old_counts[start//50]) or not np.array_equal(stages['final_mask'],old_full[start//50]):
                        raise ValueError('Reused baseline saved counts/recipe mismatch')
                    if not np.array_equal(old_roi[start//50],old_full[start//50]&roi):raise ValueError('Baseline ROI mismatch')
                dst=all_data[mid]
                for k,v in [('final',stages['final_mask']),('counts',counts),('support',stages['support_mask']),('cleaned',stages['post_min32_mask']),('holes',stages['filled_holes'])]:dst[k].append(v)
                dump(run/'checkpoints'/f'{mid}_b{start//50:03d}.json',dict(method=mid,block_index=start//50,ROI_pixels=int((stages['final_mask']&roi).sum()),completed_utc=utc()))
            log(f'Block {start//50} complete: '+', '.join(m['id']+'='+str(int((all_data[m['id']]['final'][-1]&roi).sum()))+' ROI pixels' for m in methods))
for method in methods:
    mid=method['id'];d=all_data[mid];dest=run/'masks'/mid;dest.mkdir(exist_ok=True)
    np.save(dest/'clean_full_mask.npy',np.stack(d['final']));np.save(dest/'analysis_ROI_mask.npy',np.stack(d['final'])&roi);np.save(dest/'foreground_count.npy',np.stack(d['counts']))
    np.savez_compressed(dest/'postprocess_stages.npz',support_mask=np.stack(d['support']),post_min32_mask=np.stack(d['cleaned']),filled_holes=np.stack(d['holes']))
    dump(dest/'metadata.json',dict(method=method,method_version=cfg['method_version'],axes='TYX',shape=[3,684,428],original_blocks=blocks,frame_mapping='../../data/frame_mapping.csv',pixel_size_nm=117,registration=cfg['registration_status'],recipe=cfg['recipe'],original_C1_used=True,response_A_note='Baseline response cache not available; baseline masks/broad/counts reused without implicit rerun.' if method['reused'] else None))
csvwrite(run/'data/frame_mapping.csv',mapping)
np.savez_compressed(run/'masks/original_zero_regions_packed.npz',packed=np.concatenate(zero_parts),shape=np.array([150,684,428]),original_frame_ids=np.array([r['original_frame'] for r in mapping],np.int32),bitorder=np.array('little'))
if before!=[desc(p) for p in protected]:raise ValueError('Protected source changed')
dump(run/'source_records.json',dict(protected_sources=before,config_sha256=sha(cfg_path),protected_sources_unchanged=True,completed_utc=utc()))
dump(run/'validation.json',dict(samples=sample_checks,n_samples=len(sample_checks),frames=150,full_blocks=3,original_sources_unchanged=True,baseline_reused_exact=True,mask_counts_and_recipe_consistent=True,ROI_intersection_exact=True,axes='TYX',registration=cfg['registration_status'],source_pixels_unprocessed=True,localization_calls=0,assignment_calls=0,TARDIS_calls=0,scope='ten-frame raw-copy checks plus computation-required shape/mapping/count consistency; no all2500 replay'))
dump(run/'NUMERICAL_COMPLETE.json',dict(complete=True,frames=150,original_blocks=blocks,method_ids=[m['id'] for m in methods],baseline_reused=True,computed_methods=2,elapsed_s=time.perf_counter()-start_clock))
log(f'Numerical masks complete in {time.perf_counter()-start_clock:.1f}s.')
