"""Pandas-free I/O for unchanged frozen TARDIS histogram algebra; one isolated cell."""
import argparse,ast,csv,gzip,hashlib,json,os,sys,time,uuid
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
run=args.run.resolve();cfg=json.loads((run/'config.json').read_text())
if not run.is_relative_to(Path('/Volumes/My Book')):raise ValueError('All outputs require My Book')
for k,d in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
 p=run/'runtime'/d;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import tempfile
tempfile.tempdir=str(run/'runtime/tmp')
print('COUNT_IMPORT NumPy',flush=True)
import numpy as np
print('COUNT_IMPORT direct SciPy native Euclidean distance kernel',flush=True)
import importlib.util
native_path=next((Path(sys.prefix)/'lib').glob('python*/site-packages/scipy/spatial/_distance_pybind*.so'))
spec=importlib.util.spec_from_file_location('_distance_pybind',native_path)
_native=importlib.util.module_from_spec(spec);spec.loader.exec_module(_native)
cdist=_native.cdist_euclidean
print('COUNT_IMPORT_COMPLETE',flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def dump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def flag(s):return str(s).lower() in ('true','1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):
 p=Path(p);st=p.stat();v=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
 if st.st_size<1024*1024:v['sha256']=sha(p)
 return v
if not (run/'MASK_ASSIGNMENT_COMPLETE.json').is_file():raise ValueError('Classification must complete first')
if (run/'COUNTS_COMPLETE.json').is_file():print('COUNTS_ALREADY_COMPLETE',flush=True);sys.exit(0)
source=Path(cfg['source_output']);out=run/'data/tardis/pair_histograms';out.mkdir(parents=True,exist_ok=True)
protected=[source/'data/localizations.csv.gz',source/'data/tardis/pair_histograms/per_frame_raw_counts.npz',run/'data/assignment_labels.csv.gz']
before=[desc(p) for p in protected]
vendor=run/'source/frozen_tardis/src/trak_pipeline/compat/vendor'
module=vendor/'mito_tardis/mito_core.py'
tree=ast.parse(module.read_text());nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='histogram']
if len(nodes)!=1:raise ValueError('Frozen histogram function unavailable')
ns=dict(np=np,cdist=cdist);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(module),'exec'),ns);histogram=ns['histogram']
labels={}
with gzip.open(run/'data/assignment_labels.csv.gz','rt',encoding='utf-8',newline='') as f:
 for row in csv.DictReader(f):
  uid=row['source_detection_uid']
  if not uid or uid in labels:raise ValueError('Duplicate label UID')
  labels[uid]=row
GROUPS=['On','Not-on'];mapping={'mito-compatible':'On','non-mito-compatible':'Not-on','unknown':'unknown'}
coords={g:[[] for _ in range(2500)] for g in GROUPS};selected=[];seen=set();exclusions=0;unknown=0
started=time.perf_counter()
with gzip.open(source/'data/localizations.csv.gz','rt',encoding='utf-8',newline='') as f:
 for row in csv.DictReader(f):
  uid=row['source_detection_uid']
  if uid in seen or uid not in labels:raise ValueError('Master UID correspondence failed')
  seen.add(uid);label=labels[uid];frame=int(row['frame'])
  if frame!=int(label['frame']) or not 0<=frame<2500:raise ValueError('Original frame correspondence failed')
  keep=flag(row['qc_unflagged_model_fit']) and flag(row['fitted_center_inside_ROI'])
  if keep!=flag(label['spatial_selected']) or keep!=flag(row['spatial_selected']):raise ValueError('Original QC/ROI selection changed')
  if not keep:
   if label['pointwise_label']!='excluded_qc':raise ValueError('QC exclusion restored')
   exclusions+=1;continue
  x,y=float(row['fit_x_px']),float(row['fit_y_px'])
  if not np.isfinite([x,y]).all() or min(float(row['x_se_nm']),float(row['y_se_nm']))<=0:raise ValueError('Selected coordinate/error contract differs')
  group=mapping[label['pointwise_label']]
  selected.append(dict(source_detection_uid=uid,frame=frame,fit_x_px=x,fit_y_px=y,tardis_group=group))
  if group=='unknown':unknown+=1
  else:coords[group][frame].append([x,y])
if set(labels)!=seen:raise ValueError('Master/label UID sets differ')
summary=read(run/'classification_summary.json')
if len(selected)!=summary['selected'] or exclusions!=summary['QC_excluded'] or unknown!=summary['unknown']:raise ValueError('Selection conservation differs')
for group in GROUPS:
 for f in range(2500):coords[group][f]=np.asarray(coords[group][f],dtype=float).reshape(-1,2)*cfg['pixel_size_nm']/1000
edges=np.linspace(0,3,301);hist={k:np.zeros((2,2500,300),np.int64) for k in ['total','background','cross']}
hist.update({k:np.zeros((2500,300),np.int64) for k in ['same_cross_background','all_total','all_background']})
zero=np.zeros((2,2500),np.int64)
for f in range(2500):
 for g,group in enumerate(GROUPS):
  hist['background'][g,f],zero[g,f]=histogram(coords[group][f],coords[group][f],edges,True)
 hist['same_cross_background'][f]=2*histogram(coords['On'][f],coords['Not-on'][f],edges,True)[0]
 both=np.concatenate([coords[g][f] for g in GROUPS]);hist['all_background'][f]=histogram(both,both,edges,True)[0]
 if f<2499:
  for g,group in enumerate(GROUPS):hist['total'][g,f]=histogram(coords[group][f],coords[group][f+1],edges)[0]
  hist['cross'][0,f]=histogram(coords['On'][f],coords['Not-on'][f+1],edges)[0]
  hist['cross'][1,f]=histogram(coords['Not-on'][f],coords['On'][f+1],edges)[0]
  hist['all_total'][f]=hist['total'][:,f].sum(0)+hist['cross'][:,f].sum(0)
 if (f+1)%250==0:print(f'PAIR_COUNTS {f+1}/2500',flush=True)
np.testing.assert_array_equal(hist['all_total'],hist['total'].sum(0)+hist['cross'].sum(0))
np.testing.assert_array_equal(hist['all_background'],hist['background'].sum(0)+hist['same_cross_background'])
with np.load(source/'data/tardis/pair_histograms/per_frame_raw_counts.npz',allow_pickle=False) as old:
 for k in ['all_total','all_background']:np.testing.assert_array_equal(hist[k],old[k])
def independent(p,q,remove_zero=False):
 if not len(p) or not len(q):return np.zeros(300,np.int64),0
 d=np.sqrt(np.sum((p[:,None,:]-q[None,:,:])**2,axis=2)).ravel();removed=int(np.count_nonzero(d==0)) if remove_zero else 0
 if remove_zero:d=d[d>0]
 return np.histogram(d,bins=edges)[0],removed
checked=0
for f in cfg['sample_frames']:
 for g,group in enumerate(GROUPS):
  h,z=independent(coords[group][f],coords[group][f],True)
  np.testing.assert_array_equal(h,hist['background'][g,f]);assert z==zero[g,f]
  if f<2499:np.testing.assert_array_equal(independent(coords[group][f],coords[group][f+1])[0],hist['total'][g,f])
  checked+=1
hist.update(background_zero_removed=zero,edges_um=edges,original_frame=np.arange(2500),source_frame=np.arange(2500),target_frame=np.arange(2500)+1,frame_valid=np.ones(2500,bool),endpoint_valid=np.r_[np.ones(2499,bool),False],group_order=np.array(GROUPS))
np.savez_compressed(out/'per_frame_raw_counts.npz',**hist)
with gzip.open(out/'selected_uid_coordinates_labels.csv.gz','wt',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
if before!=[desc(p) for p in protected]:raise ValueError('Read-only input changed')
actual_rules=dict(cfg['tardis']);actual_rules.update({'classification_label_field': 'pointwise_label', 'assignment_label_artifact': 'data/assignment_labels.csv.gz', 'assignment_uid_key': 'source_detection_uid', 'group_rule': 'New UID labels; original1SE ellipse, original QC/ROI/unknown policy; no weighting', 'mask_label': 'Default Nellie OR raw-C1 median+6MADscale; support50>=10/postmin32/holes<64'})
metadata=dict(config_rules=actual_rules,groups=GROUPS,raw_axes=['group','original_source_frame','distance_bin'],bootstrap_sufficient_units='Per-original-source-frame Total, per-frame ordered B, zero removals and source/target validity',cross_frame_exact_zero_kept=True,same_frame_all_exact_zero_removed=True,selected_count=len(selected),QC_excluded=exclusions,unknown=unknown,group_counts=dict(Counter(r['tardis_group'] for r in selected)),original_candidate_count=len(seen),independent_histogram_group_frames_checked=checked,all_selected_counts_exactly_baseline=True,frozen_histogram_source=str(module),frozen_histogram_SHA256=sha(module),pandas_imported=False,numerical_function_body_changed=False,distance_backend='Same SciPy _distance_pybind.cdist_euclidean kernel loaded directly; no optional spatial Python imports',native_distance_module=str(native_path),timing_seconds=time.perf_counter()-started,completed_utc=utc())
dump(out/'metadata.json',metadata);dump(run/'COUNTS_COMPLETE.json',dict(complete=True,**metadata))
print('COUNTS_COMPLETE '+json.dumps(metadata['group_counts']),flush=True)
