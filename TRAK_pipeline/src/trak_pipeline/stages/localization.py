"""Native TrackMate detection and frozen integrated-Gaussian fits, independent of masks."""
from pathlib import Path
from collections import Counter
import argparse,concurrent.futures,csv,gzip,hashlib,json,multiprocessing,os,sys,time
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

def execute_detection(ctx):
    validate_detection_settings(ctx.parameters)
    import pandas as pd
    ctx.output_dir.mkdir(parents=True,exist_ok=True)
    request=dict(data_root=ctx.config['data_root'],control_root=ctx.config['control_root'],runtime_dir=str(Path(ctx.config['control_root'])/'modular/runtime'),runtime=ctx.runtime,acquisition=ctx.config['acquisition'],settings=ctx.parameters,source=str(ctx.sample.C2_file),output=str(ctx.output_dir),sample_id=ctx.sample_id,source_id=ctx.output_dir.name,sample_frames=ctx.config['sample_frames'],chunk_frames=int(ctx.config.get('execution',{}).get('chunk_frames',25)))
    job=ctx.output_dir/'native_request.json';atomic_json(job,request)
    from ..runtime import invoke
    script=ctx.package_root/'tools/native_detection_bootstrap.py'
    invoke(ctx.config,[ctx.runtime['detector_python'],'-B','-s','-u',str(script),'--job',str(job)],ctx.output_dir/'native.log')
    chunks=read_json(ctx.output_dir/'native_chunks.json');tables=[]
    for item in chunks:
        table=pd.read_csv(item['table'],float_precision='round_trip')
        table['source_detection_uid']=[f'{ctx.sample_id}:{request["source_id"]}:{item["source_key"]}:{int(v)}'for v in table.spot_id]
        table['source_detection_cache']=item['table'];tables.append(table)
    all_rows=pd.concat(tables,ignore_index=True).sort_values(['frame','x_px','y_px'],kind='stable').reset_index(drop=True)
    all_rows.insert(0,'candidate_index',range(len(all_rows)))
    if not all_rows.source_detection_uid.is_unique:raise ValueError('Native detection source identity is not unique')
    for key in['molecule','genotype','BR','cell','acquisition_date']:
        all_rows[key]=ctx.sample.metadata.get(key,'')
    all_rows['sample_id']=ctx.sample_id
    out=ctx.output_dir/'all_detections.csv.gz';_csv(all_rows,out)
    n=ctx.config['acquisition']['frames'];counts=all_rows.groupby('frame').size()
    count_path=ctx.output_dir/'per_frame_counts.csv'
    pd.DataFrame([dict(frame=f,time_s=f*ctx.config['acquisition']['frame_interval_s'],FOV_candidates=int(counts.get(f,0)))for f in range(n)]).to_csv(count_path,index=False)
    return StageResult(dict(table=out,per_frame_counts=count_path),dict(FOV_candidates=len(all_rows),frames=n,UID_rule='immutable_detection_source_and_native_spot_id',coordinates='original_C2_pixel_centers',method='TrackMate LoG',native_chunks=chunks))

def native_main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);args=ap.parse_args(argv);job=read_json(args.job)
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
    for option in ['-Xmx2g','-XX:-UsePerfData','-Dscijava.log.level=warn','-Dimagej.updater.disableAutocheck=true','-Djava.util.prefs.PreferencesFactory=DriveMemoryPreferencesFactory','-Djava.io.tmpdir='+str(owned/'tmp'),'-Duser.home='+str(owned/'home'),'-Djava.util.prefs.userRoot='+str(owned/'prefs'),'-Djava.util.prefs.systemRoot='+str(owned/'prefs'),'-Dmaven.repo.local='+str(runtime/'m2')]:scyjava.config.add_option(option)
    scyjava.config.add_classpath(str(COMPAT/'java'))
    from run_detection import make_imageplus,detect
    ij,System,factory,arguments=initialize_imagej(imagej,scyjava,job['runtime'])
    source_fingerprint=fingerprint(job['source']);settings=job['settings'];validate_detection_settings(settings);acq=job['acquisition'];out=Path(job['output']);rows=[]
    try:
        with tifffile.TiffFile(job['source'])as tf:
            if list(tf.series[0].shape)!=acq['shape']or tf.series[0].axes!=acq['axes']:raise ValueError('C2 acquisition contract mismatch')
            for start in range(0,acq['frames'],job['chunk_frames']):
                stop=min(start+job['chunk_frames'],acq['frames']);tag=f'frames{start:04d}_{stop-1:04d}';csv_path=out/(tag+'.csv');marker=out/(tag+'.json')
                if marker.is_file():
                    old=read_json(marker)
                    if old['source_fingerprint']!=source_fingerprint or old['settings']!=settings or old['table_fingerprint']!=fingerprint(csv_path):raise ValueError('Detection checkpoint incompatible: '+tag)
                    rows.append(dict(old,table=str(csv_path)));continue
                movie=np.stack([tf.pages[f].asarray()for f in range(start,stop)])
                if movie.dtype!=np.uint16 or list(movie.shape[1:])!=acq['shape'][1:]:raise ValueError('C2 raw page contract mismatch')
                imp=make_imageplus(ij,movie,acq['pixel_size_nm']/1000,acq['frame_interval_s'])
                try:detected,counts,version=detect(imp,settings['radius_um'],settings['threshold'],settings['median_filter'],start,acq['frame_interval_s'])
                finally:imp.close()
                if version!=settings['trackmate_version']or sum(counts)!=len(detected):raise ValueError('TrackMate output contract mismatch')
                temp=csv_path.with_suffix('.csv.tmp')
                with temp.open('w',newline='',encoding='utf-8')as stream:
                    writer=csv.DictWriter(stream,fieldnames=NATIVE_FIELDS);writer.writeheader();writer.writerows(detected)
                replace(temp,csv_path)
                record=dict(source_key=tag,table=str(csv_path),start=start,stop_exclusive=stop,counts=list(map(int,counts)),source_fingerprint=source_fingerprint,settings=settings,table_fingerprint=fingerprint(csv_path),sample_pixels={str(f):hashlib.sha256(movie[f-start].tobytes()).hexdigest()for f in job['sample_frames']if start<=f<stop})
                atomic_json(marker,record);rows.append(record)
                if stop%250==0 or stop==acq['frames']:print('DETECTION_FRAME',stop,flush=True)
        if fingerprint(job['source'])!=source_fingerprint:raise ValueError('C2 source changed during detection')
        atomic_json(out/'native_chunks.json',rows)
    finally:ij.dispose()


def execute_fit(ctx):
    import numpy as np,pandas as pd,tifffile
    paths('real_annular_background')
    import run_real,run_full
    from parallel_fit import fit_batch
    c=read_json(COMPAT/'fit_reference.json');c.update(ctx.parameters)
    c.update(pixel_size_nm=ctx.config['acquisition']['pixel_size_nm'],frame_interval_s=ctx.config['acquisition']['frame_interval_s'])
    c['workers']=int(ctx.config.get('execution',{}).get('fit_workers',4))
    table=pd.read_csv(ctx.file('detection','table'),float_precision='round_trip')
    if not table.source_detection_uid.is_unique:raise ValueError('Detection UID duplicate')
    vertices=np.load(ctx.file('roi','vertices'),allow_pickle=False);roi=np.load(ctx.file('roi','mask'),allow_pickle=False)
    inside=run_real.roi_lib.points_in_polygon(table[['x_px','y_px']].to_numpy(float),vertices)
    table['roi_inside']=inside.astype('int8');groups={int(f):v for f,v in table.groupby('frame',sort=False)}
    ctx.output_dir.mkdir(parents=True,exist_ok=True);root=ctx.output_dir/'chunks';root.mkdir(exist_ok=True)
    parts=[];counts=[];n=ctx.config['acquisition']['frames'];chunk=int(ctx.config.get('execution',{}).get('chunk_frames',25));source=fingerprint(ctx.sample.C2_file)
    with tifffile.TiffFile(ctx.sample.C2_file)as tf,concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn'))as pool:
        for start in range(0,n,chunk):
            stop=min(n,start+chunk);tag=f'frames{start:04d}_{stop-1:04d}';p=root/(tag+'.csv.gz');marker=root/(tag+'.json')
            if marker.is_file():
                saved=read_json(marker)
                if saved['source']!=source or saved['parameters']!=ctx.parameters or saved['table_fingerprint']!=fingerprint(p):raise ValueError('Fit checkpoint incompatible')
                parts.append(pd.read_csv(p,float_precision='round_trip'));counts.extend(saved['counts']);continue
            chunk_parts=[];chunk_counts=[]
            for f in range(start,stop):
                all_frame=groups.get(f,table.iloc[:0]);targets=all_frame[all_frame.roi_inside==1].copy();raw=tf.pages[f].asarray()
                if len(targets):
                    background,masks,movie=run_real.background_geometry(targets,all_frame,raw,roi,c,c['camera'])
                    items=[(int(r.candidate_index),int(r.spot_id),f,float(r.x_px),float(r.y_px))for r in targets.itertuples()]
                    step=max(1,int(np.ceil(len(items)/c['workers'])))
                    futures=[pool.submit(fit_batch,(items[a:a+step],raw,movie,background.iloc[a:a+step].reset_index(drop=True),masks[a:a+step],c))for a in range(0,len(items),step)]
                    values=[v for future in futures for v in future.result()];fits=run_full.decorate(pd.DataFrame(values),vertices,c)
                    extras=[k for k in targets if k not in fits]
                    fits=fits.merge(targets[['candidate_index']+extras],on='candidate_index',how='left',validate='one_to_one',sort=False)
                    for key in background:
                        if key!='candidate_index':fits[key if key not in fits else'annular_'+key]=background[key].to_numpy()
                    fits['reused_from_pilot']=False;fits['pilot_candidate_index']=np.nan
                    if fits.source_detection_uid.tolist()!=targets.source_detection_uid.tolist():raise ValueError('Fit changed detection identity')
                else:
                    from localization_stage import empty_fits
                    fits=empty_fits(pd)
                chunk_parts.append(fits)
                chunk_counts.append(dict(frame=f,time_s=f*c['frame_interval_s'],FOV_candidates=len(all_frame),ROI_candidates=len(fits),selected=int(fits.qc_unflagged_model_fit.sum()),QC_excluded=int((~fits.qc_unflagged_model_fit).sum())))
            data=pd.concat(chunk_parts,ignore_index=True);_csv(data,p)
            atomic_json(marker,dict(source=source,parameters=ctx.parameters,table_fingerprint=fingerprint(p),counts=chunk_counts))
            parts.append(data);counts.extend(chunk_counts)
            if stop%250==0 or stop==n:ctx.progress('fit checkpoint',completed_frames=stop,total_frames=n)
    result=pd.concat(parts,ignore_index=True)
    if not result.source_detection_uid.is_unique:raise ValueError('Duplicate fit UID')
    expected=table.loc[inside,'source_detection_uid'].tolist()
    if result.source_detection_uid.tolist()!=expected:raise ValueError('ROI fitting changed source correspondence')
    if fingerprint(ctx.sample.C2_file)!=source:raise ValueError('C2 source changed during fit')
    path=ctx.output_dir/'localizations.csv.gz';_csv(result,path);count_path=ctx.output_dir/'per_frame_counts.csv';pd.DataFrame(counts).to_csv(count_path,index=False)
    metadata=dict(rows=len(result),selected=int(result.qc_unflagged_model_fit.sum()),QC_excluded=int((~result.qc_unflagged_model_fit).sum()),uncertainty='per_axis_full_observed_Hessian_with_nuisances',uncertainty_units='nm',calibration_status='manufacturer_nominal',source_field='source_detection_uid',fit_parameters=c)
    atomic_json(ctx.output_dir/'metadata.json',metadata)
    return StageResult(dict(table=path,per_frame_counts=count_path,metadata=ctx.output_dir/'metadata.json'),metadata)
