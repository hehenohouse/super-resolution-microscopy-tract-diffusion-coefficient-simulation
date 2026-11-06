"""Native TrackMate detection and frozen integrated-Gaussian fits, independent of masks."""
from pathlib import Path
from collections import Counter
import argparse,concurrent.futures,csv,gc,gzip,hashlib,json,multiprocessing,os,sys,time
from ..types import StageResult
from ..io import atomic_json,read_json,fingerprint,replace
from ..kernels import paths,COMPAT

NATIVE_FIELDS=['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response','radius_setting_um']
def _csv(table,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    try:table.to_csv(temp,index=False,compression={'method':'gzip','mtime':0}if str(path).endswith('.gz')else None);replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def execute_roi_geometry(ctx):
    import numpy as np
    paths('TARDIS/path_irrelevant_python/tardis_tau1')
    from locbio import load_imagej_roi,points_in_polygon
    vertices=np.asarray(load_imagej_roi(ctx.sample.ROI_file),dtype=float)
    h,w=ctx.config['acquisition']['shape'][1:]
    if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all():raise ValueError('Invalid ROI vertices')
    if not((vertices[:,0]>=0)&(vertices[:,0]<w)&(vertices[:,1]>=0)&(vertices[:,1]<h)).all():raise ValueError('ROI vertices outside original image')
    y,x=np.mgrid[:h,:w];mask=points_in_polygon(np.c_[x.ravel(),y.ravel()],vertices).reshape(h,w)
    ctx.output_dir.mkdir(parents=True,exist_ok=True)
    np.save(ctx.output_dir/'roi_mask.npy',mask);np.save(ctx.output_dir/'vertices.npy',vertices)
    crop=[max(0,int(np.floor(vertices[:,0].min()))-10),max(0,int(np.floor(vertices[:,1].min()))-10),min(w,int(np.ceil(vertices[:,0].max()))+11),min(h,int(np.ceil(vertices[:,1].max()))+11)]
    return StageResult(dict(mask=ctx.output_dir/'roi_mask.npy',vertices=ctx.output_dir/'vertices.npy',roi_file=ctx.sample.ROI_file),dict(vertices=len(vertices),pixels=int(mask.sum()),crop_xyxy_px=crop,coordinate_system='original_image_integer_pixel_centers'))

def validate_detection_settings(settings):
    if settings.get('subpixel_localization',True)is not True:raise ValueError('TrackMate v1 requires subpixel_localization=True; a different implementation/version is required')
    if settings.get('trackmate_version','7.14.0')!='7.14.0':raise ValueError('This detector method requires TrackMate 7.14.0')
    if float(settings.get('radius_um',.28))<=0:raise ValueError('Detection radius must be positive')

# 625 frames with a 6 GiB heap keeps the frames-per-GiB ratio of the 250-frame/2 GiB
# detector_memory_fix (about 80 MB of retained heap per 25-frame chunk) while restarting
# the JVM 3 times per 2500-frame cell instead of 9.
NATIVE_JVM_FRAME_LIMIT = 625
NATIVE_JVM_HEAP_GB = 6


def detection_jvm_ranges(frames, chunk_frames, limit=NATIVE_JVM_FRAME_LIMIT):
    """Fixed chunk-aligned resource windows, independent of saved chunk IDs."""
    frames,chunk_frames,limit=map(int,(frames,chunk_frames,limit))
    if min(frames,chunk_frames,limit)<=0 or chunk_frames>limit:
        raise ValueError('Detection chunks must be positive and fit inside the bounded JVM frame limit')
    step=(limit//chunk_frames)*chunk_frames
    return [(start,min(start+step,frames)) for start in range(0,frames,step)]


def checked_detection_chunk(out,start,stop,source_fingerprint,settings):
    """Read a committed native table verbatim; never reseed/relabel old spots."""
    tag=f'frames{start:04d}_{stop-1:04d}';csv_path=Path(out)/(tag+'.csv');marker=Path(out)/(tag+'.json')
    if not marker.is_file():return None
    old=read_json(marker)
    if (old.get('source_key')!=tag or old.get('start')!=start or old.get('stop_exclusive')!=stop
            or old.get('source_fingerprint')!=source_fingerprint or old.get('settings')!=settings
            or not csv_path.is_file() or old.get('table_fingerprint')!=fingerprint(csv_path)):
        raise ValueError('Detection checkpoint incompatible: '+tag)
    if len(old.get('counts',[]))!=stop-start:raise ValueError('Detection checkpoint frame map differs: '+tag)
    return dict(old,table=str(csv_path))


def _java_heap_exhausted(error):
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        text=type(error).__name__+': '+str(error)
        if any(term in text for term in ('OutOfMemoryError','Java heap space','GC overhead limit exceeded')):return True
        error=getattr(error,'__cause__',None) or getattr(error,'__context__',None)
    return False


def _raise_native_failure(error,start,stop):
    if _java_heap_exhausted(error):
        from ..runtime import platform
        raise platform.BatchInfrastructureError(
            f'Native TrackMate JVM heap exhausted in original frames {start}-{stop-1}. '
            'Committed 25-frame detection checkpoints are retained; stop the batch instead of retrying every cell. '
            'TRAK_DETECTION_THREADS=2 reduces concurrent native memory if this recurs.') from error
    raise error


def _heap_snapshot(java_runtime):
    try:return dict(used_bytes=int(java_runtime.totalMemory()-java_runtime.freeMemory()),
                    committed_bytes=int(java_runtime.totalMemory()),maximum_bytes=int(java_runtime.maxMemory()))
    except Exception:return dict(unavailable=True)


def release_detection_image(imp):
    """Headless ImagePlus.close alone does not release stack pixels."""
    if imp is None:return
    try:imp.changes=False
    except Exception:pass
    for operation in ('close','flush'):
        try:getattr(imp,operation)()
        except Exception as cleanup_error:print('IMAGE_RELEASE_WARNING '+str(cleanup_error),file=sys.stderr,flush=True)


def run_detection_batches(ctx):
    validate_detection_settings(ctx.parameters)
    ctx.output_dir.mkdir(parents=True,exist_ok=True)
    request=dict(data_root=ctx.config['data_root'],control_root=ctx.config['control_root'],runtime_dir=str(Path(ctx.config['control_root'])/'modular/runtime'),runtime=ctx.runtime,acquisition=ctx.config['acquisition'],settings=ctx.parameters,source=str(ctx.sample.C2_file),output=str(ctx.output_dir),sample_id=ctx.sample_id,source_id=ctx.output_dir.name,sample_frames=ctx.config['sample_frames'],chunk_frames=int(ctx.config.get('execution',{}).get('chunk_frames',25)),native_jvm_frame_limit=NATIVE_JVM_FRAME_LIMIT,detection_threads=detection_thread_count(ctx.config))
    job=ctx.output_dir/'native_request.json';atomic_json(job,request)
    from ..runtime import invoke
    script=ctx.package_root/'tools/native_detection_bootstrap.py'
    n=ctx.config['acquisition']['frames'];chunk=request['chunk_frames'];source=fingerprint(ctx.sample.C2_file)
    for start,stop in detection_jvm_ranges(n,chunk):
        existing=[checked_detection_chunk(ctx.output_dir,f,min(f+chunk,n),source,ctx.parameters) for f in range(start,stop,chunk)]
        if all(value is not None for value in existing):
            print(f'DETECTION_JVM_RANGE {start}-{stop-1} | all committed chunks reused; JVM not started',flush=True)
            continue
        print(f'DETECTION_JVM_RANGE {start}-{stop-1} | bounded native process; maximum {NATIVE_JVM_FRAME_LIMIT} frames',flush=True)
        invoke(ctx.config,[ctx.runtime['detector_python'],'-B','-s','-u',str(script),'--job',str(job),'--start',str(start),'--stop',str(stop)],ctx.output_dir/f'native_{start:04d}_{stop-1:04d}.log')
        for f in range(start,stop,chunk):
            if checked_detection_chunk(ctx.output_dir,f,min(f+chunk,n),source,ctx.parameters) is None:
                raise RuntimeError('Bounded detector exited without a committed chunk at original frame '+str(f))
    chunks=[checked_detection_chunk(ctx.output_dir,f,min(f+chunk,n),source,ctx.parameters) for f in range(0,n,chunk)]
    if any(value is None for value in chunks):raise RuntimeError('Native detection index is incomplete')
    atomic_json(ctx.output_dir/'native_chunks.json',chunks)
    return request,chunks


def native_detection_uid(sample_id,source_id,source_key,spot_id):
    return f'{sample_id}:{source_id}:{source_key}:{int(spot_id)}'


def execute_detection(ctx):
    import pandas as pd
    request,chunks=run_detection_batches(ctx)
    n=ctx.config['acquisition']['frames']
    tables=[]
    for item in chunks:
        table=pd.read_csv(item['table'],float_precision='round_trip')
        table['source_detection_uid']=[native_detection_uid(ctx.sample_id,request["source_id"],item["source_key"],v)for v in table.spot_id]
        table['source_detection_cache']=item['table'];tables.append(table)
    all_rows=pd.concat(tables,ignore_index=True).sort_values(['frame','x_px','y_px'],kind='stable').reset_index(drop=True)
    all_rows.insert(0,'candidate_index',range(len(all_rows)))
    if not all_rows.source_detection_uid.is_unique:raise ValueError('Native detection source identity is not unique')
    for key in['molecule','genotype','BR','cell','acquisition_date']:
        all_rows[key]=ctx.sample.metadata.get(key,'')
    all_rows['sample_id']=ctx.sample_id
    out=ctx.output_dir/'all_detections.csv.gz';_csv(all_rows,out)
    counts=all_rows.groupby('frame').size();count_path=ctx.output_dir/'per_frame_counts.csv'
    pd.DataFrame([dict(frame=f,time_s=f*ctx.config['acquisition']['frame_interval_s'],FOV_candidates=int(counts.get(f,0)))for f in range(n)]).to_csv(count_path,index=False)
    return StageResult(dict(table=out,per_frame_counts=count_path),dict(FOV_candidates=len(all_rows),frames=n,UID_rule='immutable_detection_source_and_native_spot_id',coordinates='original_C2_pixel_centers',method='TrackMate LoG',native_chunks=chunks,native_jvm_frame_limit=NATIVE_JVM_FRAME_LIMIT,committed_native_IDs_preserved=True))


def native_main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);args=ap.parse_args(argv);job=read_json(args.job)
    acq=job['acquisition'];chunk=int(job['chunk_frames']);start_frame=args.start;stop_frame=args.stop if args.stop is not None else min(start_frame+NATIVE_JVM_FRAME_LIMIT,acq['frames'])
    if (not 0<=start_frame<stop_frame<=acq['frames'] or stop_frame-start_frame>NATIVE_JVM_FRAME_LIMIT
            or chunk<1 or chunk>NATIVE_JVM_FRAME_LIMIT or start_frame%chunk or (stop_frame!=acq['frames'] and stop_frame%chunk)):
        raise ValueError(f'Native TrackMate request must be a chunk-aligned range of at most{NATIVE_JVM_FRAME_LIMIT} original frames')
    os.environ['TRAK_DATA_ROOT']=job['data_root'];os.environ['TRAK_CONTROL_ROOT']=job['control_root']
    from ..runtime import configure
    configure(dict(data_root=job['data_root'],control_root=job['control_root'],runtime=job['runtime']))
    paths('trackmate_python_bridge')
    from detector_worker import configure_java_environment,initialize_imagej
    configure_java_environment(job['runtime'])
    import numpy as np,tifffile,scyjava,imagej
    import jgo.env._builder as builder
    from jgo.env._linking import link_file,LinkStrategy
    builder.link_file=lambda source,target,strategy:link_file(source,target,LinkStrategy.COPY)
    runtime=Path(job['runtime_dir']);owned=Path(job['output'])/'java_runtime';owned.mkdir(exist_ok=True)
    for part in['home','tmp','prefs']:(owned/part).mkdir(exist_ok=True)
    scyjava.config.set_cache_dir(runtime/'jgo');scyjava.config.set_m2_repo(runtime/'m2');scyjava.config.set_java_constraints(fetch='never')
    for option in [f'-Xmx{NATIVE_JVM_HEAP_GB}g','-XX:-UsePerfData','-Dscijava.log.level=warn','-Dimagej.updater.disableAutocheck=true','-Djava.util.prefs.PreferencesFactory=DriveMemoryPreferencesFactory','-Djava.io.tmpdir='+str(owned/'tmp'),'-Duser.home='+str(owned/'home'),'-Djava.util.prefs.userRoot='+str(owned/'prefs'),'-Djava.util.prefs.systemRoot='+str(owned/'prefs'),'-Dmaven.repo.local='+str(runtime/'m2')]:scyjava.config.add_option(option)
    scyjava.config.add_classpath(str(COMPAT/'java'))
    from run_detection import make_imageplus,detect
    threads=int(job.get('detection_threads',2))
    print(f'NATIVE_JVM_RANGE {start_frame}-{stop_frame-1} | heap limit{NATIVE_JVM_HEAP_GB}GiB | chunk{chunk} | TrackMate threads {threads}',flush=True)
    ij,System,factory,arguments=initialize_imagej(imagej,scyjava,job['runtime'])
    java_runtime=scyjava.jimport('java.lang.Runtime').getRuntime()
    source_fingerprint=fingerprint(job['source']);settings=job['settings'];validate_detection_settings(settings);out=Path(job['output']);rows=[]
    try:
        with tifffile.TiffFile(job['source'])as tf:
            if list(tf.series[0].shape)!=acq['shape']or tf.series[0].axes!=acq['axes']:raise ValueError('C2 acquisition contract mismatch')
            for start in range(start_frame,stop_frame,chunk):
                stop=min(start+chunk,stop_frame);tag=f'frames{start:04d}_{stop-1:04d}';csv_path=out/(tag+'.csv');marker=out/(tag+'.json')
                old=checked_detection_chunk(out,start,stop,source_fingerprint,settings)
                if old is not None:
                    rows.append(old);print('DETECTION_CHUNK '+tag+' | committed native IDs reused',flush=True);continue
                began=time.monotonic();movie=None;imp=None;detected=None
                try:
                    movie=np.stack([tf.pages[f].asarray()for f in range(start,stop)])
                    if movie.dtype!=np.uint16 or list(movie.shape[1:])!=acq['shape'][1:]:raise ValueError('C2 raw page contract mismatch')
                    imp=make_imageplus(ij,movie,acq['pixel_size_nm']/1000,acq['frame_interval_s'])
                    detected,counts,version=detect(imp,settings['radius_um'],settings['threshold'],settings['median_filter'],start,acq['frame_interval_s'],num_threads=threads)
                    if version!=settings['trackmate_version']or sum(counts)!=len(detected):raise ValueError('TrackMate output contract mismatch')
                    temp=csv_path.with_suffix('.csv.tmp')
                    with temp.open('w',newline='',encoding='utf-8')as stream:
                        writer=csv.DictWriter(stream,fieldnames=NATIVE_FIELDS);writer.writeheader();writer.writerows(detected)
                    replace(temp,csv_path)
                    record=dict(source_key=tag,table=str(csv_path),start=start,stop_exclusive=stop,counts=list(map(int,counts)),source_fingerprint=source_fingerprint,settings=settings,table_fingerprint=fingerprint(csv_path),sample_pixels={str(f):hashlib.sha256(movie[f-start].tobytes()).hexdigest()for f in job['sample_frames']if start<=f<stop},native_jvm_range=[start_frame,stop_frame],native_jvm_frame_limit=NATIVE_JVM_FRAME_LIMIT,java_heap_before_release=_heap_snapshot(java_runtime))
                    atomic_json(marker,record);rows.append(record)
                except BaseException as error:
                    _raise_native_failure(error,start,stop)
                finally:
                    release_detection_image(imp)
                    imp=None;movie=None;detected=None
                    try:gc.collect();System.gc()
                    except Exception as cleanup_error:print('JVM_GC_WARNING '+str(cleanup_error),file=sys.stderr,flush=True)
                print(json.dumps(dict(stage='native_detection_chunk',tag=tag,rows=sum(counts),elapsed_s=time.monotonic()-began,heap_after_release=_heap_snapshot(java_runtime))),flush=True)
        if fingerprint(job['source'])!=source_fingerprint:raise ValueError('C2 source changed during detection')
        atomic_json(out/f'native_batch_{start_frame:04d}_{stop_frame-1:04d}.json',rows)
    finally:
        try:ij.dispose()
        except Exception as cleanup_error:print('JVM_DISPOSE_WARNING '+str(cleanup_error),file=sys.stderr,flush=True)


FIT_WORKERS_AUTO_CAP=10
DETECTION_THREADS_AUTO_CAP=12  # detection runs before fit; only single-thread Nellie overlaps
FIT_CHUNKS_IN_FLIGHT=2


def execution_count(config,key,env,default,cap,cpu_count=None,environ=None):
    """Execution-only CPU count; never part of a scientific cache identity.

    The environment variable overrides the config. "auto" leaves two logical
    CPUs for the parent process, Nellie and the desktop, capped at `cap`.
    """
    environ=os.environ if environ is None else environ
    value=environ.get(env) or config.get('execution',{}).get(key,default)
    if str(value).strip().lower()=='auto':
        value=max(1,min(cap,(cpu_count or os.cpu_count() or 4)-2))
    value=int(value)
    if value<1:raise ValueError(key+' must be a positive integer or "auto"')
    return value


def fit_worker_count(config,cpu_count=None,environ=None):
    return execution_count(config,'fit_workers','TRAK_FIT_WORKERS',4,FIT_WORKERS_AUTO_CAP,cpu_count,environ)


def detection_thread_count(config,cpu_count=None,environ=None):
    return execution_count(config,'detection_threads','TRAK_DETECTION_THREADS',2,DETECTION_THREADS_AUTO_CAP,cpu_count,environ)


def stop_process_pool(pool,grace_s=5):
    """Cancel queued frames and stop running fit workers immediately.

    Workers only return results; nothing they compute is committed after an error,
    so terminating them releases CPU and file handles without losing checkpoints.
    The executor's process table is read before shutdown() clears it.
    """
    processes=list((getattr(pool,'_processes',None) or {}).values())
    pool.shutdown(wait=False,cancel_futures=True)
    for process in processes:
        try:
            if process.is_alive():process.terminate()
        except Exception:pass
    for process in processes:
        try:
            process.join(grace_s)
            if process.is_alive():process.kill();process.join(grace_s)
        except Exception:pass


_FIT_SHARED={}


def _fit_worker_init(roi,vertices,c):
    paths('real_annular_background')
    _FIT_SHARED.update(roi=roi,vertices=vertices,c=c)


def fit_frame(f,all_frame,raw,roi,vertices,c):
    """One original frame: the frozen annular background and per-candidate fits.

    Identical per-candidate calls to the v1 frame loop; only process placement differs.
    """
    import numpy as np,pandas as pd
    paths('real_annular_background')
    import run_real,run_full
    targets=all_frame[all_frame.roi_inside==1].copy()
    if len(targets):
        background,masks,movie=run_real.background_geometry(targets,all_frame,raw,roi,c,c['camera'])
        items=[(int(r.candidate_index),int(r.spot_id),f,float(r.x_px),float(r.y_px))for r in targets.itertuples()]
        values=[run_real.fit_one(item,raw,movie,background.iloc[j],masks[j],c)for j,item in enumerate(items)]
        fits=run_full.decorate(pd.DataFrame(values),vertices,c)
        extras=[k for k in targets if k not in fits]
        fits=fits.merge(targets[['candidate_index']+extras],on='candidate_index',how='left',validate='one_to_one',sort=False)
        for key in background:
            if key!='candidate_index':fits[key if key not in fits else'annular_'+key]=background[key].to_numpy()
        fits['reused_from_pilot']=False;fits['pilot_candidate_index']=np.nan
        if fits.source_detection_uid.tolist()!=targets.source_detection_uid.tolist():raise ValueError('Fit changed detection identity')
    else:
        from localization_stage import empty_fits
        fits=empty_fits(pd)
    count=dict(frame=f,time_s=f*c['frame_interval_s'],FOV_candidates=len(all_frame),ROI_candidates=len(fits),selected=int(fits.qc_unflagged_model_fit.sum()),QC_excluded=int((~fits.qc_unflagged_model_fit).sum()))
    return fits,count


def _fit_frame_task(f,all_frame,raw):
    return fit_frame(f,all_frame,raw,_FIT_SHARED['roi'],_FIT_SHARED['vertices'],_FIT_SHARED['c'])


def execute_fit(ctx):
    import collections,numpy as np,pandas as pd,tifffile
    paths('real_annular_background')
    import run_real
    c=read_json(COMPAT/'fit_reference.json');c.update(ctx.parameters)
    c.update(pixel_size_nm=ctx.config['acquisition']['pixel_size_nm'],frame_interval_s=ctx.config['acquisition']['frame_interval_s'])
    c['workers']=fit_worker_count(ctx.config)
    table=pd.read_csv(ctx.file('detection','table'),float_precision='round_trip')
    if not table.source_detection_uid.is_unique:raise ValueError('Detection UID duplicate')
    vertices=np.load(ctx.file('roi','vertices'),allow_pickle=False);roi=np.load(ctx.file('roi','mask'),allow_pickle=False)
    inside=run_real.roi_lib.points_in_polygon(table[['x_px','y_px']].to_numpy(float),vertices)
    table['roi_inside']=inside.astype('int8');groups={int(f):v for f,v in table.groupby('frame',sort=False)}
    ctx.output_dir.mkdir(parents=True,exist_ok=True);root=ctx.output_dir/'chunks';root.mkdir(exist_ok=True)
    parts=[];counts=[];n=ctx.config['acquisition']['frames'];chunk=int(ctx.config.get('execution',{}).get('chunk_frames',25));source=fingerprint(ctx.sample.C2_file)
    print(json.dumps(dict(stage='fit_scheduler',workers=c['workers'],task='one_frame',chunks_in_flight=FIT_CHUNKS_IN_FLIGHT,cpu_count=os.cpu_count())),flush=True)
    # Frames are dispatched whole and up to FIT_CHUNKS_IN_FLIGHT checkpoint chunks ahead, so
    # workers never idle on a per-frame barrier. Chunks still commit strictly in frame order.
    with tifffile.TiffFile(ctx.sample.C2_file)as tf:
        pool=concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=_fit_worker_init,initargs=(roi,vertices,c))
        def submit(start):
            stop=min(n,start+chunk);tag=f'frames{start:04d}_{stop-1:04d}';p=root/(tag+'.csv.gz');marker=root/(tag+'.json')
            if marker.is_file():return dict(stop=stop,p=p,marker=marker,saved=True)
            futures=[pool.submit(_fit_frame_task,f,groups.get(f,table.iloc[:0]),tf.pages[f].asarray())for f in range(start,stop)]
            return dict(stop=stop,p=p,marker=marker,saved=False,futures=futures)
        def finish(entry):
            stop,p,marker=entry['stop'],entry['p'],entry['marker']
            if entry['saved']:
                saved=read_json(marker)
                if saved['source']!=source or saved['parameters']!=ctx.parameters or saved['table_fingerprint']!=fingerprint(p):raise ValueError('Fit checkpoint incompatible')
                parts.append(pd.read_csv(p,float_precision='round_trip'));counts.extend(saved['counts'])
            else:
                results=[future.result()for future in entry['futures']]
                chunk_parts=[fits for fits,_ in results];chunk_counts=[count for _,count in results]
                data=pd.concat(chunk_parts,ignore_index=True);_csv(data,p)
                atomic_json(marker,dict(source=source,parameters=ctx.parameters,table_fingerprint=fingerprint(p),counts=chunk_counts))
                parts.append(data);counts.extend(chunk_counts)
            if stop%250==0 or stop==n:ctx.progress('fit checkpoint',completed_frames=stop,total_frames=n)
        pending=collections.deque()
        try:
            for start in range(0,n,chunk):
                pending.append(submit(start))
                while len(pending)>FIT_CHUNKS_IN_FLIGHT:finish(pending.popleft())
            while pending:finish(pending.popleft())
        except BaseException:
            stop_process_pool(pool)
            raise
        pool.shutdown(wait=True)
    result=pd.concat(parts,ignore_index=True)
    if not result.source_detection_uid.is_unique:raise ValueError('Duplicate fit UID')
    expected=table.loc[inside,'source_detection_uid'].tolist()
    if result.source_detection_uid.tolist()!=expected:raise ValueError('ROI fitting changed source correspondence')
    if fingerprint(ctx.sample.C2_file)!=source:raise ValueError('C2 source changed during fit')
    path=ctx.output_dir/'localizations.csv.gz';_csv(result,path);count_path=ctx.output_dir/'per_frame_counts.csv';pd.DataFrame(counts).to_csv(count_path,index=False)
    metadata=dict(rows=len(result),selected=int(result.qc_unflagged_model_fit.sum()),QC_excluded=int((~result.qc_unflagged_model_fit).sum()),uncertainty='per_axis_full_observed_Hessian_with_nuisances',uncertainty_units='nm',calibration_status='manufacturer_nominal',source_field='source_detection_uid',fit_parameters=c)
    atomic_json(ctx.output_dir/'metadata.json',metadata)
    return StageResult(dict(table=path,per_frame_counts=count_path,metadata=ctx.output_dir/'metadata.json'),metadata)
