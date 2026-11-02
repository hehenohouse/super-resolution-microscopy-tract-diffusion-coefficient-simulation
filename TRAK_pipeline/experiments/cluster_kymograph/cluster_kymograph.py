#!/usr/bin/env python3
"""Selected-cluster C2 intensity kymographs with frozen localization overlays."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,gzip,hashlib,json,os,shlex,shutil,sys,time,traceback


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
def sample_descriptor(path):
    p=Path(path);st=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:h.update(f.read(65536));f.seek(max(0,st.st_size-65536));h.update(f.read(65536))
    return dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,first_last_64KiB_sha256=h.hexdigest())
def table_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)

def run(config_path):
    config_path=Path(config_path).resolve();cfg=json.loads(config_path.read_text());out=Path(cfg['output_directory']).resolve();cell=Path(cfg['cell_directory']).resolve()
    drive=Path('/Volumes/My Book');root=drive/'Xinran_FullsetTRAK2'
    if not os.path.ismount(drive) or not cell.is_relative_to(root) or not out.is_relative_to(cell):raise RuntimeError('My Book/cell output unavailable; no local fallback')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        complete=json.loads((out/'COMPLETE.json').read_text())
        if complete.get('complete') and complete.get('config_sha256')==sha(config_path) and all(Path(r['path']).is_file() and sha(r['path'])==r['sha256'] for r in complete['artifacts']):
            print('Existing completed output reused; no changes made.',flush=True);return
        raise RuntimeError('Existing output differs; use a new timestamp directory')
    cache=root/'_batch_pipeline/runtime/cluster_kymograph'
    for key,folder in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
        p=cache/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import numpy as np
    import tifffile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    started=time.monotonic()
    def log(message):
        line=utc()+' | '+message;print(line,flush=True)
        with (out/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    input_points=Path(cfg['localization_subset']);selection_path=Path(cfg['selection_record']);master=Path(cfg['master_localizations']);raw_path=Path(cfg['C2_file'])
    before={str(p):sample_descriptor(p) for p in [raw_path,master]};small_before={str(p):sha(p) for p in [input_points,selection_path,config_path,Path(__file__)]}
    selection=json.loads(selection_path.read_text());fields,points=table_rows(input_points)
    if len(points)!=96 or selection['selected_localizations']!=96:raise ValueError('Expected the fixed 96 localization records')
    frames=np.array([int(float(r['frame'])) for r in points],dtype=np.int64)
    if not np.array_equal(frames,np.array(list(range(95))+[196])):raise ValueError('Selected frame identity/order changed')
    center=np.array(cfg['center_xy_px']);bounds=np.array(cfg['localization_window_xyxy_px'])
    if not np.array_equal(center,np.array([134.125,226.875])) or not np.array_equal(bounds,np.array([132.125,224.875,136.125,228.875])):raise ValueError('Confirmed spatial window changed')
    for row in points:
        if row['spatial_selected'].lower() not in ('true','1') or row['postmin32_pointwise_label']!='non-mito-compatible':raise ValueError('Frozen localization QC/group changed')
        f=int(float(row['frame']))
        if not np.isclose(float(row['time_s']),f*.02,rtol=0,atol=1e-12):raise ValueError('Localization frame/time mismatch')
    selected=[]
    with gzip.open(master,'rt',newline='',encoding='utf-8') as source:
        reader=csv.DictReader(source);master_fields=reader.fieldnames
        for row in reader:
            try:x=float(row['fit_x_px']);y=float(row['fit_y_px'])
            except ValueError:continue
            if bounds[0]<=x<=bounds[2] and bounds[1]<=y<=bounds[3] and row['spatial_selected'].lower() in ('true','1'):selected.append(row)
    if fields!=master_fields or points!=selected:raise ValueError('Original master does not exactly reproduce all 96 rows/fields')
    del selected
    shutil.copyfile(input_points,out/'localizations_96.csv')
    log('Confirmed 96 unchanged Not-on records: frames 0-94 and 196; no tracking')
    n=cfg['frames'];dt=cfg['frame_interval_s'];scale=cfg['pixel_size_nm']
    if n!=2500 or dt!=.02 or scale!=117:raise ValueError('Original time/pixel scale changed')
    xb=cfg['X_profile'];yb=cfg['Y_profile'];xcols=np.arange(*xb['x_slice']);yrows=np.arange(*yb['y_slice'])
    xprofile=np.empty((n,len(xcols)),dtype=np.float64);yprofile=np.empty((n,len(yrows)),dtype=np.float64)
    checks=cfg['check_frames'];validation=[];raw_reads=0
    with tifffile.TiffFile(raw_path) as tf:
        if tf.series[0].axes!='TYX' or tuple(tf.series[0].shape)!=(n,684,428) or len(tf.pages)!=n:raise ValueError('C2 TYX shape/frame mapping differs')
        if np.dtype(tf.pages[0].dtype).kind!='u' or np.dtype(tf.pages[0].dtype).itemsize!=2:raise ValueError('Expected original uint16 C2 pixels')
        ij=tf.imagej_metadata or {}
        if 'finterval' in ij and not np.isclose(float(ij['finterval']),dt,rtol=0,atol=1e-12):raise ValueError('TIFF frame interval differs')
        pixel_dtype=np.dtype(tf.byteorder+'u2');width=428;row_stride=width*2
        x0=min(xb['x_slice'][0],yb['x_slice'][0]);x1=max(xb['x_slice'][1],yb['x_slice'][1]);y0=min(xb['y_slice'][0],yb['y_slice'][0]);y1=max(xb['y_slice'][1],yb['y_slice'][1])
        span=(y1-y0-1)*row_stride+(x1-x0)*2
        saved_checks=[]
        with raw_path.open('rb') as source:
            for f in range(n):
                page=tf.pages[f]
                if int(page.compression)!=1 or page.is_tiled or len(page.dataoffsets)!=1 or page.shape!=(684,428) or page.databytecounts[0]!=684*428*2:raise ValueError('Unsupported raw TIFF strip layout at frame '+str(f))
                source.seek(page.dataoffsets[0]+y0*row_stride+x0*2);buf=source.read(span);raw_reads+=len(buf)
                if len(buf)!=span:raise OSError('Short raw C2 window read at frame '+str(f))
                patch=np.ndarray((y1-y0,x1-x0),dtype=pixel_dtype,buffer=buf,strides=(row_stride,2))
                xpatch=patch[xb['y_slice'][0]-y0:xb['y_slice'][1]-y0,xb['x_slice'][0]-x0:xb['x_slice'][1]-x0]
                ypatch=patch[yb['y_slice'][0]-y0:yb['y_slice'][1]-y0,yb['x_slice'][0]-x0:yb['x_slice'][1]-x0]
                xprofile[f]=xpatch.mean(axis=0,dtype=np.float64);yprofile[f]=ypatch.mean(axis=1,dtype=np.float64)
                if f in checks:
                    reference=page.asarray();reference_patch=reference[y0:y1,x0:x1]
                    np.testing.assert_array_equal(patch,reference_patch)
                    np.testing.assert_array_equal(xprofile[f],reference[xb['y_slice'][0]:xb['y_slice'][1],xb['x_slice'][0]:xb['x_slice'][1]].mean(axis=0,dtype=np.float64))
                    np.testing.assert_array_equal(yprofile[f],reference[yb['y_slice'][0]:yb['y_slice'][1],yb['x_slice'][0]:yb['x_slice'][1]].mean(axis=1,dtype=np.float64))
                    saved_checks.append(patch.astype(np.uint16));validation.append(dict(frame=f,time_s=f*dt,direct_patch_exact=True,X_mean_exact=True,Y_mean_exact=True))
                if (f+1)%250==0:
                    log('Raw C2 profiles extracted '+str(f+1)+'/2500 original frames')
                    dump(out/'progress.json',dict(status='extracting',frames_read=f+1,total_frames=n,raw_window_bytes_read=raw_reads))
        layout=dict(TIFF_byteorder=tf.byteorder,pixel_dtype=str(pixel_dtype),raw_window_xyxy=[x0,y0,x1,y1],read_span_bytes_per_frame=span,raw_window_bytes_read=raw_reads,layout='Uncompressed one-strip original uint16 frames; original TIFF offsets used')
    times=np.arange(n,dtype=np.float64)*dt
    np.savez_compressed(out/'profiles_all_frames.npz',frame=np.arange(n),time_s=times,X_mean_ADU=xprofile,Y_mean_ADU=yprofile,X_pixel_centers=xcols,Y_pixel_centers=yrows,center_xy_px=center,frame_interval_s=np.array(dt),pixel_size_nm=np.array(scale),check_frames=np.array(checks),check_C2_windows=np.stack(saved_checks))
    fitx=np.array([float(r['fit_x_px']) for r in points]);fity=np.array([float(r['fit_y_px']) for r in points]);ptimes=frames*dt
    figure,axes=plt.subplots(2,2,figsize=(13.8,13.1),dpi=220,sharey='row')
    purple='#C58BFF';vmin,vmax=cfg['display_ADU'];image=None
    for col,(axis_name,profile,positions,fitted,c) in enumerate([('X',xprofile,xcols,fitx,center[0]),('Y',yprofile,yrows,fity,center[1])]):
        spatial_edges=((positions[0]-.5-c)*scale,(positions[-1]+.5-c)*scale)
        for row,tmax in enumerate([50.,5.]):
            ax=axes[row,col];limit=n if row==0 else 250;data=profile[:limit]
            image=ax.imshow(data,cmap='gray',vmin=vmin,vmax=vmax,origin='upper',aspect='auto',interpolation='nearest',resample=False,extent=[spatial_edges[0],spatial_edges[1],(limit-.5)*dt,-.5*dt])
            use=ptimes<=tmax
            ax.scatter((fitted[use]-c)*scale,ptimes[use],s=7.0,c=purple,marker='o',linewidths=0,zorder=3)
            for edge in [-234.,234.]:ax.axvline(edge,color='#F1CE56',ls=(0,(4,4)),lw=.85,alpha=.8,zorder=2)
            ax.axvline(0,color='white',lw=.5,alpha=.18,zorder=2)
            ax.set_ylim(tmax,0);ax.set_xlim(*spatial_edges)
            ax.set_xlabel(axis_name+' relative to selected center (nm)',fontsize=11)
            ax.set_title(axis_name+'–time | '+('full 0–50 s' if row==0 else 'zoom 0–5 s'),fontsize=13,pad=10)
            if col==0:ax.set_ylabel('Original acquisition time (s)',fontsize=11)
            ax.set_xticks([-600,-300,0,300,600]);ax.tick_params(labelsize=10)
            ax.text(.98,.98,'96 Not-on records',transform=ax.transAxes,ha='right',va='top',fontsize=9,color=purple,bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=3))
    figure.suptitle('WT TRAK2 BR1 T5 | selected-cluster C2 kymographs',fontsize=18,y=.975)
    figure.text(.5,.943,'Center (X, Y) = (134.125, 226.875) px | 117 nm/px | 20 ms/frame | raw C2 + refined localizations',ha='center',fontsize=10.5,color='#405263')
    figure.subplots_adjust(left=.085,right=.885,bottom=.115,top=.90,hspace=.28,wspace=.16)
    cax=figure.add_axes([.915,.24,.015,.51]);cb=figure.colorbar(image,cax=cax);cb.set_label('C2 spatial-strip mean (raw ADU)',fontsize=10)
    figure.text(.5,.073,'Gray: one original frame per time row; four-pixel spatial means only. No temporal averaging, smoothing or trajectory links.',ha='center',fontsize=9.3)
    figure.text(.5,.050,'Purple: 96 unchanged QC-passed Not-on observations within the fixed 4×4 px window (dashed bounds); frames 0–94 and 196.',ha='center',fontsize=9.3)
    figure.text(.5,.027,'Window membership is not molecular identity. Gaps do not establish bleaching, dissociation or departure; C1/C2 registration remains unverified.',ha='center',fontsize=8.8,color='#52616D')
    name=cfg['figure_name'];target=out/name;temp=out/('.'+name+'.tmp');figure.savefig(temp,format='png',dpi=220,facecolor='white');plt.close(figure);os.replace(temp,target)
    gallery=Path(cfg['figure_directory']).resolve()
    if not gallery.is_relative_to(cell) or gallery.name!='cluster_kymograph':raise ValueError('Figure-copy directory must be a cell-owned child folder')
    gallery.mkdir(parents=True,exist_ok=True);copy=gallery/name;shutil.copyfile(target,copy)
    if sha(target)!=sha(copy):raise ValueError('Delivered PNG copy differs')
    if any(sample_descriptor(Path(p))!=record for p,record in before.items()) or any(sha(Path(p))!=digest for p,digest in small_before.items()):raise ValueError('Protected source/config/script changed')
    dictionary=dict(profile_shape='(2500 original frames, 12 native camera-pixel centers)',X_mean_ADU='Arithmetic mean over C2 y=[225,229), for each original x=[128,140)',Y_mean_ADU='Arithmetic mean over C2 x=[133,137), for each original y=[221,233)',time_s='Original frame * 0.02; localization and intensity row centers use this same time',pixel_convention='Camera coordinates denote pixel centers; imshow uses closed pixel-square edges at center +/-0.5',localizations='Full original string fields retained in localizations_96.csv, exactly matched to the master CSV',display='Fixed406-554ADU; no offset subtraction or electron conversion; original 16-bit TIFF unchanged',selection='Only the fixed 4x4 native-pixel inspection window; not a formal cluster boundary or tracking identity')
    dump(out/'data_dictionary.json',dictionary)
    report=dict(all_passed=True,checked_frames=checks,ten_frame_checks=validation,selected_count=96,unique_frames=96,exact_original_rows_and_fields=True,frames=list(range(95))+[196],all_frames_in_intensity_profiles=n,no_temporal_averaging=True,no_smoothing=True,no_trajectory_links=True,source_inputs_unchanged=True,PNG_copies_identical=True)
    dump(out/'validation.json',report)
    provenance=dict(created_utc=utc(),sample_id=cfg['sample_id'],inputs=before,small_input_SHA256=small_before,script_SHA256=sha(__file__),config_SHA256=sha(config_path),layout=layout,versions=dict(Python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,tifffile=tifffile.__version__),command=[sys.executable,'-B','-s','-u',str(Path(__file__).resolve()),'--config',str(config_path)],elapsed_seconds=time.monotonic()-started,scientific_stages_rerun=[],tracking_performed=False,TARDIS_performed=False)
    dump(out/'provenance.json',provenance);(out/'command.txt').write_text(shlex.join(provenance['command'])+'\n')
    (out/'README.txt').write_text('Selected-cluster C2 kymographs\n\nOpen '+name+'.\nFull 50-second views are on top; first 5 seconds below. Horizontal axes are X/Y displacement from the selected center, in nm; time increases downwards.\nRaw C2 spatial-strip means use all 2500 original frames, including frames without a selected localization. Purple marks are the unchanged 96 records, not trajectories, error bars or distinct molecule counts.\nFrame0 is time0; frame196 is3.92s. The original localization window is continuous [132.125,136.125] x [224.875,228.875] native pixels.\nThe wider intensity strips provide context; only the fixed-window records are overlaid. No localization fitting, segmentation, reclassification, tracking or TARDIS was run.\nThe NPZ retains complete float64 intensity profiles, original frame/time and native pixel coordinates. All output/data/cache files are on My Book.\n')
    dump(out/'progress.json',dict(status='complete',frames_read=n,total_frames=n,raw_window_bytes_read=raw_reads))
    artifacts=[out/name,copy,out/'profiles_all_frames.npz',out/'localizations_96.csv',out/'validation.json',out/'provenance.json',out/'data_dictionary.json',out/'README.txt',out/'command.txt']
    dump(out/'COMPLETE.json',dict(complete=True,all_passed=True,completed_utc=utc(),sample_id=cfg['sample_id'],config_sha256=sha(config_path),figure=str(target),figure_copy=str(copy),selected_records=96,frames=n,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts],scientific_stages_rerun=[],tracking_performed=False,TARDIS_performed=False))
    log('COMPLETE: 2x2 kymograph saved with full 2500-frame profiles; ten sampled frame checks passed')
    print('FIGURE '+str(target),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    try:run(args.config)
    except Exception:
        cfg=json.loads(args.config.read_text());out=Path(cfg['output_directory'])
        if out.is_dir() and out.is_relative_to(Path('/Volumes/My Book/Xinran_FullsetTRAK2')):dump(out/'FAILED.json',dict(failed_utc=utc(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
