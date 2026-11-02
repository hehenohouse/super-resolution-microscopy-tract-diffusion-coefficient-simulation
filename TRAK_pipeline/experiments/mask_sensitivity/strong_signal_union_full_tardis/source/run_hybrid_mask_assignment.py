"""Full-movie isolated raw-C1 strong-union mask and original 1SE assignment sensitivity."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.metadata,json,os,platform,sys,time,uuid
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);args=ap.parse_args()
cfgpath=Path(args.config).resolve();cfg=json.loads(cfgpath.read_text());run=Path(cfg['run_directory']).resolve()
if not run.is_relative_to(Path('/Volumes/My Book')):raise ValueError('Writes must remain on My Book')
for k,d in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','XDG_CACHE_HOME':'xdg'}.items():
 p=run/'runtime'/d;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
import tempfile
tempfile.tempdir=str(run/'runtime/tmp')
import numpy as np,tifffile
sys.path.insert(0,str(run/'source'))
from support50_postmin32_core import process_counts
import geometry as geo
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def table(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8') as s:
  w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def descriptor(p):
 p=Path(p);st=p.stat();d=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
 if st.st_size<=1024*1024:d['sha256']=sha(p)
 return d
def log(s):
 line=utc()+' | '+s
 with (run/'logs/mask_assignment.log').open('a') as f:f.write(line+'\n')
 print(line,flush=True)
def flag(x):return str(x).strip().lower() in ('true','1')
def packed(path,array,ids):
 np.savez_compressed(path,packed=np.packbits(array.reshape(len(array),-1),axis=1,bitorder='little'),shape=np.array(array.shape,np.int32),original_frame_ids=np.array(ids,np.int32),bitorder=np.array('little'))
def unpack(path):
 with np.load(path,allow_pickle=False) as z:
  shape=tuple(z['shape'].astype(int));ids=z['original_frame_ids'].copy()
  if str(z['bitorder'])!='little':raise ValueError('Unexpected packed order')
  value=np.unpackbits(z['packed'],axis=1,count=shape[1]*shape[2],bitorder='little').reshape(shape).astype(bool)
 return value,ids
def strong(raw):
 v=raw[raw>0].astype(np.float64)
 if len(v)<cfg['strong_signal_rule']['minimum_valid_pixels']:raise ValueError('Insufficient valid C1 pixels')
 m=float(np.median(v));mad=float(np.median(abs(v-m)));spread=mad/.6744897501960817
 if not np.isfinite(spread) or spread<=0:raise ValueError('Undefined robust intensity spread')
 threshold=m+float(cfg['strong_signal_rule']['k'])*spread
 return (raw>0)&(raw>threshold),dict(reference_positive_pixels=len(v),median_intensity_ADU=m,MAD_ADU=mad,robust_intensity_scale_ADU=spread,threshold_ADU=threshold,k=cfg['strong_signal_rule']['k'])
def independent(mask,zero,x,y,a,b):
 if not np.isfinite([x,y,a,b]).all() or min(a,b)<=0:return 'unknown'
 h,w=mask.shape
 outside=x-a<-.5-1e-12 or x+a>w-.5+1e-12 or y-b<-.5-1e-12 or y+b>h-.5+1e-12
 if outside:return 'unknown'
 hits=[]
 for z in (mask,zero):
  py,px=np.nonzero(z)
  dx=np.maximum(abs(px.astype(float)-x)-.5,0);dy=np.maximum(abs(py.astype(float)-y)-.5,0)
  hits.append(bool(np.any((dx/a)**2+(dy/b)**2<=1+1e-12)))
 return 'unknown' if hits[1] else ('mito-compatible' if hits[0] else 'non-mito-compatible')
source=Path(cfg['source_output']);prior=Path(cfg['prior_pilot'])
protected=[Path(cfg['source_C1']),Path(cfg['source_ROI']),source/'data/localizations.csv.gz',source/'classification_summary.json',source/'COMPLETED.json',source/'masks/clean_full_mask.npy',source/'masks/foreground_count.npy',source/'roi/roi_mask.npy',source/'roi/vertices.npy']
protected += [source/'masks/broad_packed'/f'b{b:03d}.npz' for b in range(50)]
protected += [source/'data/tardis/metadata.json',source/'data/tardis/pair_histograms/per_frame_raw_counts.npz']
before=[descriptor(p) for p in protected];signature=hashlib.sha256(json.dumps(dict(cfg=cfg,inputs=before),sort_keys=True,separators=(',',':')).encode()).hexdigest()
if (run/'MASK_ASSIGNMENT_COMPLETE.json').is_file():
 if read(run/'MASK_ASSIGNMENT_COMPLETE.json')['signature']!=signature:raise ValueError('Completed inputs changed')
 print('MASK_ASSIGNMENT_ALREADY_COMPLETE',flush=True);sys.exit(0)
dump(run/'source_records.json',dict(signature=signature,protected_inputs=before,source_master_read_only=True,cache_policy='Explicit method + scientific parameters + upstream identity; source hashes audit only'))
roi=np.load(source/'roi/roi_mask.npy');vertices=np.load(source/'roi/vertices.npy')
full0=np.load(source/'masks/clean_full_mask.npy',mmap_mode='r');count0=np.load(source/'masks/foreground_count.npy',mmap_mode='r')
if roi.shape!=(684,428) or roi.dtype!=bool or full0.shape!=(50,684,428):raise ValueError('Mask/ROI grid mismatch')
for fn,a in [('roi_mask.npy',roi),('roi_vertices_xy.npy',vertices),('original_frame_to_mask_block.npy',np.arange(2500,dtype=np.int32)//50)]:np.save(run/'masks'/fn,a)
rows=defaultdict(list);seen=set();oldtotals=Counter()
input_keys=['source_detection_uid','frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm','qc_unflagged_model_fit','fitted_center_inside_ROI','uncertainty_status','pointwise_label','spatial_selected']
with gzip.open(source/'data/localizations.csv.gz','rt',newline='',encoding='utf-8') as f:
 reader=csv.DictReader(f)
 if not set(input_keys).issubset(reader.fieldnames):raise ValueError('Missing required UID/fit/error semantics')
 for r in reader:
  uid=r['source_detection_uid']
  if not uid or uid in seen:raise ValueError('Duplicate or missing UID')
  seen.add(uid);frame=int(r['frame'])
  if frame<0 or frame>=2500 or float(r['time_s'])!=frame*.02:raise ValueError('Original frame/time inconsistency')
  keep=flag(r['qc_unflagged_model_fit']) and flag(r['fitted_center_inside_ROI'])
  if keep!=flag(r['spatial_selected']):raise ValueError('Saved selection differs from original QC and ROI')
  if keep and 'full_observed_Hessian' not in r['uncertainty_status']:raise ValueError('Unclear standard error semantics')
  oldtotals[r['pointwise_label']]+=1
  rows[frame].append({k:r[k] for k in input_keys})
expected=read(source/'classification_summary.json')
if dict(On=oldtotals['mito-compatible'],Not_on=oldtotals['non-mito-compatible'],unknown=oldtotals['unknown'],QC_excluded=oldtotals['excluded_qc'],ROI_candidates=len(seen))!={k:expected[k] for k in ['On','Not_on','unknown','QC_excluded','ROI_candidates']}:raise ValueError('Baseline membership inconsistency')
for d in ['broad_packed','zero_packed','blocks']:(run/'masks'/d).mkdir(exist_ok=True)
label_fields=['source_detection_uid','frame','spatial_selected','pointwise_label','pointwise_unknown_reason','mask_block_index','pointwise_axis_x_px','pointwise_axis_y_px','pointwise_intersects_known_mask','pointwise_intersects_C1_zero_unknown','pointwise_min_q','default_label']
allcounts=[];allstats=[];allthreshold=[];allchecks=[];transitions=Counter();unknownreasons=Counter();blockfinal=[];blockcounts=[]
startclock=time.perf_counter()
log('Starting all2500 frames. Reuse default Nellie broad and localization master; original1SE assignment, closed pixel squares, unknown priority unchanged.')
with tifffile.TiffFile(cfg['source_C1']) as movie:
 if len(movie.pages)!=2500 or movie.pages[0].shape!=(684,428):raise ValueError('Raw C1 size/time mismatch')
 for block,start in enumerate(range(0,2500,50)):
  cp=run/'checkpoints'/f'block_{block:03d}.json';part=run/'data'/f'assignment_part_{block:03d}.csv.gz'
  if cp.exists():
   saved=read(cp)
   if saved['signature']!=signature:raise ValueError('Checkpoint method/source mismatch')
   final=np.load(run/'masks/blocks'/f'final_{block:03d}.npy');counts=np.load(run/'masks/blocks'/f'counts_{block:03d}.npy')
   if not part.is_file():raise ValueError('Incomplete assignment checkpoint')
   log(f'Reused block{block}')
  else:
   ids=np.arange(start,start+50,dtype=np.int32)
   broad,bids=unpack(source/'masks/broad_packed'/f'b{block:03d}.npz')
   if broad.shape!=(50,684,428) or not np.array_equal(ids,bids):raise ValueError('Broad grid/original-frame mismatch')
   raw=np.stack([movie.pages[int(f)].asarray() for f in ids])
   if raw.dtype!=np.uint16:raise ValueError('Raw C1 type changed')
   signal=[];thresholds=[]
   for j,f in enumerate(ids):
    s,t=strong(raw[j]);signal.append(s);t.update(original_frame=int(f),time_s=float(f)*.02,mask_block=block,strong_pixels_full=int(s.sum()),strong_pixels_ROI=int((s&roi).sum()));thresholds.append(t)
   signal=np.stack(signal);union=broad|signal;counts=union.sum(0,dtype=np.uint8)
   if not np.array_equal(broad.sum(0,dtype=np.uint8),count0[block]) or np.any(counts<count0[block]):raise ValueError('Support cache/monotonicity mismatch')
   stages=process_counts(counts,support_minimum=10,minimum_foreground=32,hole_limit=64,frame_count=50)
   final=stages['final_mask']
   if np.any(full0[block]&~final):raise ValueError('Hybrid removed original foreground')
   if start in [0,1200,2450]:
    pilotindex=[0,1200,2450].index(start)
    pilot=np.load(prior/'masks/B_strong_union/clean_full_mask.npy',mmap_mode='r')
    if not np.array_equal(pilot[pilotindex],final):raise ValueError('Previously approved pilot method changed')
    pilot._mmap.close()
   packed(run/'masks/broad_packed'/f'b{block:03d}.npz',union,ids)
   packed(run/'masks/zero_packed'/f'b{block:03d}.npz',raw==0,ids)
   np.save(run/'masks/blocks'/f'final_{block:03d}.npy',final);np.save(run/'masks/blocks'/f'counts_{block:03d}.npy',counts)
   checks=[];fcounts=[];trans=Counter();reasons=Counter();temps=part.with_name(part.name+'.partial')
   with gzip.open(temps,'wt',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=label_fields);writer.writeheader()
    for j,frame in enumerate(ids):
     tally=Counter();old=Counter();zero=raw[j]==0
     for r in rows[int(frame)]:
      keep=flag(r['spatial_selected']);uid=r['source_detection_uid'];old[r['pointwise_label']]+=1
      extra=dict(source_detection_uid=uid,frame=int(frame),spatial_selected=keep,pointwise_label='excluded_qc',pointwise_unknown_reason='',mask_block_index=block,pointwise_axis_x_px='',pointwise_axis_y_px='',pointwise_intersects_known_mask='',pointwise_intersects_C1_zero_unknown='',pointwise_min_q='',default_label=r['pointwise_label'])
      if keep:
       x,y,a,b=[float(r[k]) for k in ['fit_x_px','fit_y_px','x_se_nm','y_se_nm']];a/=117.;b/=117.
       answer=geo.classify(final,zero,x,y,a,b,1e-12)
       extra.update(pointwise_label=answer['label'],pointwise_unknown_reason=answer['reason'],pointwise_axis_x_px=a,pointwise_axis_y_px=b,pointwise_intersects_known_mask=answer['intersects_known_mask'],pointwise_intersects_C1_zero_unknown=answer['intersects_zero_unknown'],pointwise_min_q='' if answer['min_q'] is None else answer['min_q'])
       if int(frame) in cfg['sample_frames']:
        if not geo.point_inside_mask(roi,x,y,1e-12):raise ValueError('Selected center outside ROI')
        if independent(final,zero,x,y,a,b)!=extra['pointwise_label']:raise ValueError('Independent hybrid geometry mismatch')
        if independent(full0[block],zero,x,y,a,b)!=r['pointwise_label']:raise ValueError('Baseline geometry rules differ')
        checks.append(dict(source_detection_uid=uid,frame=int(frame),hybrid_independent_geometry=True,default_independent_geometry=True))
       if (r['pointwise_label']=='unknown')!=(extra['pointwise_label']=='unknown'):raise ValueError('Unknown membership changed unexpectedly')
       if r['pointwise_label']=='mito-compatible' and extra['pointwise_label']!='mito-compatible':raise ValueError('Original On not retained in hybrid')
       trans[(r['pointwise_label'],extra['pointwise_label'])]+=1
       if extra['pointwise_label']=='unknown':reasons[extra['pointwise_unknown_reason']]+=1
      writer.writerow(extra);tally[extra['pointwise_label']]+=1
     count=dict(frame=int(frame),time_s=float(frame)*.02,block_index=block,ROI_candidates=len(rows[int(frame)]),selected=len(rows[int(frame)])-tally['excluded_qc'],QC_excluded=tally['excluded_qc'],On=tally['mito-compatible'],Not_on=tally['non-mito-compatible'],unknown=tally['unknown'],default_On=old['mito-compatible'],default_Not_on=old['non-mito-compatible'],default_unknown=old['unknown'],mask_full_area_px2=int(final.sum()),mask_ROI_area_px2=int((final&roi).sum()))
     if count['On']+count['Not_on']+count['unknown']!=count['selected'] or count['selected']+count['QC_excluded']!=count['ROI_candidates']:raise ValueError('Count conservation failed')
     fcounts.append(count)
   os.replace(temps,part)
   stats=[]
   for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
    old=full0[block]&gate;new=final&gate;nold=int(old.sum());nnew=int(new.sum())
    stats.append(dict(scope=scope,block_index=block,original_frame_start=start,original_frame_stop_exclusive=start+50,default_pixels=nold,hybrid_pixels=nnew,default_area_um2=nold*.117**2,hybrid_area_um2=nnew*.117**2,added_pixels=int((new&~old).sum()),removed_pixels=int((old&~new).sum()),change_percent=100*(nnew/nold-1) if nold else None))
   saved=dict(signature=signature,counts=fcounts,statistics=stats,thresholds=thresholds,checks=checks,transitions=[dict(default_label=k[0],hybrid_label=k[1],count=v) for k,v in trans.items()],unknown_reasons=dict(reasons),completed_utc=utc())
   dump(cp,saved)
  allcounts.extend(saved['counts']);allstats.extend(saved['statistics']);allthreshold.extend(saved['thresholds']);allchecks.extend(saved['checks']);unknownreasons.update(saved['unknown_reasons'])
  for x in saved['transitions']:transitions[(x['default_label'],x['hybrid_label'])]+=x['count']
  blockfinal.append(final);blockcounts.append(counts)
  if (block+1)%5==0:log(f'MASK+ASSIGNMENT {start+50}/2500 frames | elapsed {time.perf_counter()-startclock:.1f}s')
labelpath=run/'data/assignment_labels.csv.gz';temp=labelpath.with_name(labelpath.name+'.partial')
with gzip.open(temp,'wt',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=label_fields);w.writeheader()
 for b in range(50):
  with gzip.open(run/'data'/f'assignment_part_{b:03d}.csv.gz','rt',encoding='utf-8',newline='') as src:w.writerows(csv.DictReader(src))
os.replace(temp,labelpath)
final=np.stack(blockfinal);np.save(run/'masks/clean_full_mask.npy',final);np.save(run/'masks/analysis_ROI_mask.npy',final&roi);np.save(run/'masks/foreground_count.npy',np.stack(blockcounts))
table(run/'data/per_frame_counts.csv',allcounts);table(run/'data/mask_area_comparison.csv',allstats);table(run/'data/strong_thresholds.csv',allthreshold)
tr=[dict(default_label=k[0],hybrid_label=k[1],count=v) for k,v in transitions.items()];table(run/'data/classification_transitions.csv',tr)
totals={k:sum(r[k] for r in allcounts) for k in ['On','Not_on','unknown','selected','QC_excluded','ROI_candidates']}
if totals['selected']!=expected['selected'] or totals['QC_excluded']!=expected['QC_excluded'] or totals['ROI_candidates']!=len(seen):raise ValueError('Master membership changed')
summary=dict(sample_id=cfg['sample_id'],**totals,On_percentage=100*totals['On']/(totals['On']+totals['Not_on']),default_counts={k:expected[k] for k in ['On','Not_on','unknown','selected','QC_excluded','ROI_candidates']},default_On_percentage=100*expected['On']/(expected['On']+expected['Not_on']),transitions=tr,unknown_reasons=dict(unknownreasons),method_version=cfg['method_version'],original_QC_and_fit_fields_unchanged=True,assignment='original1SE diagonal ellipse closed-pixel intersection, unknown priority, tolerance1e-12',original_source_master=str(source/'data/localizations.csv.gz'),labels_key='source_detection_uid',independent_checked_frames=cfg['sample_frames'],independent_checked_points=len(allchecks),ROI_first_mask_area_um2=int((final[0]&roi).sum())*.117**2,ROI_mean_mask_area_um2=float((final&roi).sum((1,2)).mean())*.117**2,default_ROI_first_mask_area_um2=int((full0[0]&roi).sum())*.117**2,default_ROI_mean_mask_area_um2=float((full0&roi).sum((1,2)).mean())*.117**2)
dump(run/'classification_summary.json',summary)
dump(run/'validation_mask_assignment.json',dict(ten_frame_geometry_checks=allchecks,baseline_On_retained=True,unknown_membership_unchanged=True,pilot_three_blocks_match=True,ROI_intersection_exact=True,originals_unchanged=before==[descriptor(p) for p in protected]))
dump(run/'masks/metadata.json',dict(sample_id=cfg['sample_id'],recipe=cfg['recipe'],method_version=cfg['method_version'],axes='TYX',shape=[50,684,428],pixel_size_nm=117,original_frame_interval_s=.02,pixel_centers='Integer x,y; closed squares center+/-.5',crop_offset_xy_px=[0,0],frame_mapping='original_frame_to_mask_block.npy',block_frame_ids=[list(range(s,s+50)) for s in range(0,2500,50)],frame_to_block={str(f):f//50 for f in range(2500)},registration='identity/unverified',C1_zero_unknown_source='zero_packed/b*.npz',block_valid=[True]*50))
dump(run/'environment_mask_assignment.json',dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','tifffile']},source_fingerprints={f.name:sha(f) for f in (run/'source').glob('*.py')},new_Nellie_calls=0,new_localization_calls=0,cache_root=str(run/'runtime')))
if before!=[descriptor(p) for p in protected]:raise ValueError('Protected original or baseline source changed')
full0._mmap.close();count0._mmap.close()
dump(run/'MASK_ASSIGNMENT_COMPLETE.json',dict(complete=True,signature=signature,frames=2500,blocks=50,selected=totals['selected'],On=totals['On'],Not_on=totals['Not_on'],unknown=totals['unknown'],QC_excluded=totals['QC_excluded'],elapsed_seconds=time.perf_counter()-startclock,completed_utc=utc(),new_Nellie_calls=0,new_localization_calls=0,source_master_unchanged=True))
log('MASK_ASSIGNMENT_COMPLETE '+json.dumps(summary))
