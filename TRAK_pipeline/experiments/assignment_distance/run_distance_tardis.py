#!/usr/bin/env python3
"""T5-only TARDIS adapter for saved fixed-distance point assignments.

Frozen numerical cores are reused without editing. The classified master is
read once; source coordinates, errors, QC, labels, and masks remain unchanged.
Only the in-memory TARDIS grouping column selects a new saved label field.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,hashlib,importlib.metadata,importlib.util,json,os,platform,shutil,sys,time,traceback,types

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parents[1]
DATA_ROOT=CONTROL.parent
DEFAULT_CELL=DATA_ROOT/'WT/TRAK2_WT_BR1/101625_WT_TRAK2_T5f'
RUN_ID='20261026T194830Z'
METHODS=[dict(key='close_1px',label='1-pixel proximity (117 nm)',field='close_1px_label',radius_px=1,radius_nm=117),dict(key='close_2px',label='2-pixel proximity (234 nm)',field='close_2px_label',radius_px=2,radius_nm=234)]
FRAMES=[0,49,200,625,1200,1250,1875,2000,2450,2499]


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guard(path):
 p=Path(path).resolve()
 if not DATA_ROOT.is_dir() or not p.is_relative_to(DATA_ROOT.resolve()):raise OSError(19,'Output is outside the mounted data root: '+str(path))
 return p

def dump(path,value):
 path=guard(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)

def descriptor(path):
 p=Path(path);st=p.stat();h=hashlib.sha256()
 with p.open('rb') as stream:
  if st.st_size<=2*1024*1024:h.update(stream.read());kind='full_small_file'
  else:h.update(stream.read(65536));stream.seek(max(0,st.st_size-65536));h.update(stream.read(65536));kind='first_last_64KiB'
 return dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h.hexdigest(),fingerprint_kind=kind)


def setup_config(cell,release):
 cell=guard(cell);source=cell/'analysis_results/batch_TRAK2_20261021T161234Z';classified=cell/'analysis_results/distance_sensitivity_20261026T192504Z'
 if not (classified/'CLASSIFIED.json').is_file():raise ValueError('Distance classifications are not complete')
 output=cell/('analysis_results/distance_tardis_'+RUN_ID);figures=cell/('Analysis_Figures_distance_TARDIS_'+RUN_ID)
 output.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
 methods=[dict(m,analysis_directory=str(output/m['key']/'data/tardis'),job_path=str(output/m['key']/'job.json'),figures_directory=str(figures/m['key'])) for m in METHODS]
 config=dict(schema_version=1,run_id=RUN_ID,sample_id='WT_TRAK2_BR1_T5',sample_name='WT TRAK2 BR1 cell T5',cell=str(cell),source_output=str(source),classification_output=str(classified),source_release=str(release),output=str(output),figures=str(figures),methods=methods,old_tardis_directory=str(source/'data/tardis'),sample_frames=FRAMES,frames=2500,frame_interval_s=.02,pixel_size_nm=117,
  numerical_rules='Frozen analysis_stage.py / mito_core.py; full and ten250-frame scopes; original shared1000 bootstrap draws L10/L25/L50',
  classification_rule='Euclidean distance to unchanged full-mask closed foreground pixel squares; saved labels from fixed-radius sensitivity run',
  nominal_localization_SE_unchanged=True,geometric_radius_is_not_localization_SE=True,new_cells_started=0,new_detection_fits_or_segmentation=0,
  no_cross_cell_summary_registration=True,registration_status='exploratory_identity_registration_unverified')
 dump(output/'config.json',config);return config


class NoFittingPermittedError(RuntimeError):
 pass


def install_lean_pilot_imports(release):
 """Avoid importing unused fitting/plotting dependencies in this private process.

 Pure utility function ASTs are copied unchanged from the frozen files. Every
 fitting/detection stub fails immediately if an unexpected caller reaches it.
 The actual full_core, roi_core and mito_core numerical modules load normally.
 """
 proof=[]
 def forbidden(*args,**kwargs):raise NoFittingPermittedError('Fitting/detection is forbidden in the cached TARDIS-only adapter')
 for name,filename,wanted,blocked in [('pilot_core','pilot_core.py',{'dump_json','sha256_file'},['fit_candidates','identify_candidates','model_images']),('pilot_qc','pilot_qc.py',{'key_name'},[])]:
  path=release/'vendor/picasso_pilot'/filename;tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
  if {n.name for n in nodes}!=wanted:raise RuntimeError('Frozen pure-utility AST missing: '+name)
  module=types.ModuleType(name);module.__file__=str(path);module.__dict__.update(Path=Path,json=json,hashlib=hashlib)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
  for function_name in blocked:setattr(module,function_name,forbidden)
  sys.modules[name]=module
  proof.append(dict(module=name,frozen_source_SHA256=sha(path),unchanged_pure_functions=sorted(wanted),forbidden_unused_helpers=blocked))
 return dict(mode='strict_private_import_stubs_for_unused_pilot_dependencies',modules=proof,numerical_core_source_files_modified=False)


class Tee:
 def __init__(self,original,stream):self.original=original;self.stream=stream
 def write(self,text):self.original.write(text);self.stream.write(text);self.stream.flush();return len(text)
 def flush(self):self.original.flush();self.stream.flush()


def run(config):
 output=Path(config['output']);source=Path(config['source_output']);classified=Path(config['classification_output']);release=Path(config['source_release'])
 sys.path.insert(0,str(release))
 spec=importlib.util.spec_from_file_location('distance_frozen_analysis',release/'analysis_stage.py');a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
 original_job=read(source/'job.json');base_meta=read(source/'data/tardis/metadata.json');fingerprint=base_meta['scientific_fingerprint']
 common_job=dict(original_job,runtime_dir=str(output/'runtime'))
 print('FROZEN_SCIENCE_IMPORT_START',utc(),flush=True)
 a.configure(common_job);lean_import=install_lean_pilot_imports(release)
 print('LEAN_UNUSED_PILOT_IMPORTS_INSTALLED',flush=True)
 a.load_science();np,pd=a.np,a.pd
 print('FROZEN_SCIENCE_IMPORT_COMPLETE',utc(),flush=True)
 master=classified/'data/pointwise_comparison.csv.gz'
 watched=[master,classified/'config.json',classified/'summary.json',source/'data/localizations.csv.gz',source/'masks/clean_full_mask.npy',source/'masks/analysis_ROI_mask.npy',source/'roi/vertices.npy',source/'COMPLETED.json',source/'data/tardis/pair_histograms/per_frame_raw_counts.npz',release/'analysis_stage.py',release/'vendor/mito_tardis/mito_core.py',release/'bootstrap_weights.npz']
 before=[descriptor(p) for p in watched];dump(output/'source_fingerprints_before.json',before)
 versions=dict(created_utc=utc(),python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy']},source_release=str(release),lean_import_adapter=lean_import,frozen_analysis_sha256=sha(release/'analysis_stage.py'),frozen_core_sha256=sha(release/'vendor/mito_tardis/mito_core.py'),bootstrap_weights_SHA256=sha(release/'bootstrap_weights.npz'))
 dump(output/'versions.json',versions)
 (output/'source').mkdir(exist_ok=True);shutil.copy2(__file__,output/'source/run_distance_tardis.py')
 print('READ_CLASSIFIED_MASTER_ONCE',str(master),flush=True)
 real_read_csv=pd.read_csv;audit=real_read_csv(master,dtype=str,keep_default_na=False)
 require=lambda v,message: a.require(bool(v),message)
 require(len(audit)==94204,'Unexpected T5 ROI candidate count')
 original_selected=a.selected_table;original_compat=a.write_compatibility;original_dump=a.dump;original_hist=a.mc.calculate_histograms
 master_target=master.resolve()
 def cached_read_csv(path,*args,**kwargs):
  if isinstance(path,(str,Path)) and Path(path).resolve()==master_target:return audit.copy(deep=False)
  return real_read_csv(path,*args,**kwargs)
 pd.read_csv=cached_read_csv
 results=[];start=time.perf_counter()
 try:
  for method in config['methods']:
   key=method['key'];method_output=output/key;out=method_output/'data/tardis';proofpath=method_output/'DISTANCE_TARDIS_COMPLETE.json'
   if proofpath.is_file():
    proof=read(proofpath)
    require(proof['source_classification_signature']==read(classified/'config.json')['classification_signature'],'Existing method proof has a different source classification')
    results.append(proof);print('DISTANCE_TARDIS_REUSED',key,flush=True);continue
   method_output.mkdir(parents=True,exist_ok=True);(method_output/'data').mkdir(exist_ok=True);(method_output/'state').mkdir(exist_ok=True)
   canonical=method_output/'data/localizations.csv.gz'
   if not canonical.exists():canonical.symlink_to(master)
   require(canonical.resolve()==master.resolve(),'Private canonical symlink targets a different classified master')
   masks=method_output/'masks'
   if not masks.exists():masks.symlink_to(source/'masks',target_is_directory=True)
   require(masks.resolve()==(source/'masks').resolve(),'Private mask symlink targets a different source')
   assignment=dict(mode='fixed_euclidean_distance_to_closed_mask_pixel_squares',radius_px=method['radius_px'],radius_nm=method['radius_nm'],classification_label_field=method['field'],source_classification=str(master),source_classification_signature=read(classified/'config.json')['classification_signature'],original_QC_and_nominal_SE_unchanged=True,registration_status=config['registration_status'],geometric_radius_is_not_localization_SE=True)
   assignment['assignment_signature']=hashlib.sha256(json.dumps(assignment,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
   overrides=dict(classification_label_field=method['field'],group_rule='Both endpoints use saved '+method['field']+' (fixed '+str(method['radius_px'])+'-pixel Euclidean proximity to unchanged full mask); no additional filter or weighting',mask_label='support50>=10/post-min32 | assignment: '+method['label'],assignment_rule=assignment)
   tardis=dict(original_job.get('tardis',{}));tardis.update(overrides)
   job=dict(original_job,output=str(method_output),figures=method['figures_directory'],work=str(method_output/'_work'),runtime_dir=str(output/'runtime'),release=str(release),run_id=RUN_ID+'_'+key,sample_name=config['sample_name']+' | '+method['label'],sample_frames=FRAMES,scientific_fingerprint=fingerprint,tardis=tardis,assignment_rule=assignment,assignment_signature=assignment['assignment_signature'])
   dump(method_output/'job.json',job)
   reference=read(release/'tardis_reference.json');reference.update(tardis)
   immutable=['frames','segment_frames','tau_frames','pixel_size_nm','frame_interval_s','distance_max_um','distance_bins','background_fit_um','minimum_background_tail_ordered_pairs','minimum_total_tail_pairs','signed_area_min_abs','main_area_min_positive','main_region_um','tail_region_um','minimum_joint_valid_fraction','total_rule','background_rule','normalization_rule','beta_rule','shape_rule','bootstrap']
   for rule in immutable:require(reference[rule]==base_meta['config_rules'][rule],'Original numerical rule changed: '+rule)
   dump(method_output/'numerical_rule_comparison.json',dict(original_tardis_configuration_preserved=True,exactly_equal_rules=immutable,assignment_rule_only_changed=True,assignment_signature=assignment['assignment_signature']))
   def selected_adapter(frame,c):
    table,keep=original_selected(frame,c)
    labels=frame.loc[keep,method['field']].map({**a.mc.LABELS,'On':'On','Not-on':'Not-on'})
    require(labels.notna().all(),'Unknown saved distance label')
    original_numbers=table[['candidate_index','frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm']].to_numpy(float).copy()
    table['tardis_group']=labels
    np.testing.assert_array_equal(original_numbers,table[['candidate_index','frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm']].to_numpy(float))
    return table,keep
   def compatible_adapter(destination,c,active_job):
    original_compat(destination,c,active_job);path=Path(destination)/'compatibility.json';v=read(path)
    v['pipeline_method_definition']['classification']=assignment
    v['pipeline_method_SHA256']=hashlib.sha256(json.dumps(a.jsonable(v['pipeline_method_definition']),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    v['distance_assignment_explicit']=True;original_dump(path,v)
   def dump_adapter(path,value):
    path=Path(path)
    if path.name in ('metadata.json','COMPLETE.json') and path.is_relative_to(out):value=dict(value,assignment_rule=assignment,classification_label_field=method['field'],original_nominal_SE_unchanged=True)
    return original_dump(path,value)
   def histogram_adapter(table,c,progress=None):
    def callback(n,*args):
     if n%250==0:dump(method_output/'state/progress.json',dict(updated_utc=utc(),method=key,stage='pair_histograms',processed_frames=int(n),total_frames=2500))
     if progress:progress(n,*args)
    return original_hist(table,c,callback)
   a.selected_table=selected_adapter;a.write_compatibility=compatible_adapter;a.dump=dump_adapter;a.mc.calculate_histograms=histogram_adapter
   dump(method_output/'state/progress.json',dict(updated_utc=utc(),method=key,stage='starting'))
   def bootstrap_progress(length,groups,controls):dump(method_output/'state/progress.json',dict(updated_utc=utc(),method=key,stage='bootstrap',computed_block_length_frames=int(length),replicates=1000))
   print('DISTANCE_TARDIS_START',key,flush=True);method_start=time.perf_counter();a.run(job,regression=bootstrap_progress)
   diagnostic=[]
   with np.load(source/'data/tardis/pair_histograms/per_frame_raw_counts.npz',allow_pickle=False) as old,np.load(out/'pair_histograms/per_frame_raw_counts.npz',allow_pickle=False) as new:
    for field in ['all_total','all_background']:
     np.testing.assert_array_equal(new[field],old[field]);diagnostic.append(dict(field=field,exactly_equal=True,array_SHA256=a.array_sha(new[field])))
   metadata=read(out/'metadata.json');expected=next(m for m in read(classified/'summary.json')['methods'] if m['key']==key)
   require(metadata['group_counts'].get('On',0)==expected['On'] and metadata['group_counts'].get('Not-on',0)==expected['Not_on'],'TARDIS membership count mismatch')
   require(metadata['selected_count']==91860 and metadata['unknown_count']==0,'T5 selected/unknown count changed')
   proof=dict(method=key,label=method['label'],classification_label_field=method['field'],source_classification_signature=assignment['source_classification_signature'],analysis_directory=str(out),selected_count=metadata['selected_count'],group_counts=metadata['group_counts'],all_pairs_diagnostic=diagnostic,numerical_complete=True,elapsed_seconds=time.perf_counter()-method_start,completed_utc=utc(),nominal_localization_SE_unchanged=True)
   dump(proofpath,proof);results.append(proof);dump(output/'progress.json',dict(updated_utc=utc(),completed_methods=[v['method'] for v in results],total_methods=len(config['methods'])))
   print('DISTANCE_TARDIS_METHOD_COMPLETE',key,proof['elapsed_seconds'],flush=True)
 finally:
  pd.read_csv=real_read_csv;a.selected_table=original_selected;a.write_compatibility=original_compat;a.dump=original_dump;a.mc.calculate_histograms=original_hist
 after=[descriptor(p) for p in watched];require(before==after,'Protected input fingerprint changed during TARDIS')
 dump(output/'source_fingerprints_after.json',after)
 proof=dict(numerical_complete=True,methods=results,old_TARDIS_results_unchanged=True,source_inputs_unchanged=True,all_selected_pair_diagnostic_exactly_reproduced=True,full_and_ten_segments=True,bootstrap_replicates=1000,block_lengths_frames=[10,25,50],elapsed_seconds=time.perf_counter()-start,completed_utc=utc(),no_cross_cell_summary_registration=True,new_detection_fits_or_segmentation=0)
 dump(output/'NUMERICAL_COMPLETE.json',proof);print('DISTANCE_TARDIS_NUMERICAL_COMPLETE',flush=True);return proof


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cell',type=Path,default=DEFAULT_CELL);parser.add_argument('--release',type=Path);parser.add_argument('--config',type=Path);args=parser.parse_args()
 if args.config:config=read(args.config)
 else:
  release=args.release or Path((CONTROL/'CURRENT_RELEASE.txt').read_text().strip());config=setup_config(args.cell,release)
 output=Path(config['output']);(output/'logs').mkdir(exist_ok=True)
 if (output/'FAILED.json').is_file():os.replace(output/'FAILED.json',output/'logs'/('previous_startup_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'))
 dump(output/'command.json',dict(argv=sys.argv,python=sys.executable,started_utc=utc()))
 with (output/'logs/numerical_run.log').open('a',encoding='utf-8') as log:
  oldout,olderr=sys.stdout,sys.stderr;sys.stdout=Tee(oldout,log);sys.stderr=Tee(olderr,log)
  try:run(config)
  except BaseException as error:
   dump(output/'FAILED.json',dict(failed_utc=utc(),error=str(error),traceback=traceback.format_exc()));raise
  finally:sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__':main()
