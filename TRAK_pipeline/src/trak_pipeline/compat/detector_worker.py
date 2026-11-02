#!/usr/bin/env python3
"""Bounded native TrackMate JVM. All mutable runtime paths belong to My Book."""
import argparse,csv,gc,hashlib,json,os,sys,time
from pathlib import Path
from batch_common import load_job,guard_path,dump,sha,descriptor,vendor_paths,utc,BatchInfrastructureError,atomic_replace

FIELDS=['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response','radius_setting_um']

def configure_java_environment(paths):
    """Make the configured Java discoverable before importing ScyJava/JGo."""
    home=Path(paths['java_home']).resolve()
    java_bin=home/'bin'
    executable=java_bin/('java.exe' if os.name=='nt' else 'java')
    if not executable.is_file():
        raise BatchInfrastructureError('Configured Java executable is missing: '+str(executable))
    os.environ['JAVA_HOME']=str(home)
    current=os.environ.get('PATH','')
    if current.split(os.pathsep,1)[0]!=str(java_bin):
        os.environ['PATH']=str(java_bin)+(os.pathsep+current if current else '')
    return executable

def initialize_imagej(imagej,scyjava,paths):
    """Treat shared JVM/preferences initialization failures as infrastructure."""
    ij=None
    try:
        ij=imagej.init(paths['fiji'],mode='headless')
        System=scyjava.jimport('java.lang.System')
        factory=str(System.getProperty('java.util.prefs.PreferencesFactory'))
        assert factory=='DriveMemoryPreferencesFactory'
        ManagementFactory=scyjava.jimport('java.lang.management.ManagementFactory')
        arguments=[str(v) for v in ManagementFactory.getRuntimeMXBean().getInputArguments()]
        assert '-XX:-UsePerfData' in arguments,'HotSpot perf-data storage must be disabled'
        Preferences=scyjava.jimport('java.util.prefs.Preferences')
        node=Preferences.userRoot().node('batch-runtime-selftest')
        node.put('roundtrip','external-only')
        assert str(node.get('roundtrip',''))=='external-only'
        node.removeNode()
        return ij,System,factory,arguments
    except Exception as error:
        if ij is not None:
            try:ij.dispose()
            except Exception:pass
        raise BatchInfrastructureError('Shared ImageJ/JVM/preferences initialization failed: '+str(error)) from error

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,type=Path)
    ap.add_argument('--request',required=True,type=Path);args=ap.parse_args()
    job=load_job(args.job);request=json.loads(args.request.read_text(encoding='utf-8'));runtime=Path(job['runtime_dir'])
    settings=job['fit'];paths=job['runtime'];configure_java_environment(paths)
    owned=guard_path(Path(job['work'])/'java_runtime');owned.mkdir(parents=True,exist_ok=True)
    for folder in ['java_home','java_tmp','java_prefs']:(owned/folder).mkdir(parents=True,exist_ok=True)
    # Set scyjava explicitly: its import-time defaults otherwise refer to local HOME.
    import numpy as np,tifffile,scyjava,imagej
    # jgo 3.1 AUTO only handles EXDEV, while this external filesystem reports
    # ENOTSUP for hard links. Select its documented COPY staging operation
    # process-locally, without editing the installed dependency or any jars.
    import jgo.env._builder as jgo_builder
    from jgo.env._linking import link_file,LinkStrategy
    def drive_copy(source,target,strategy):
        return link_file(source,target,LinkStrategy.COPY)
    jgo_builder.link_file=drive_copy
    scyjava.config.set_cache_dir(runtime/'jgo');scyjava.config.set_m2_repo(runtime/'m2')
    scyjava.config.set_java_constraints(fetch='never')
    for option in ['-Xmx2g','-XX:-UsePerfData','-Dscijava.log.level=warn','-Dimagej.updater.disableAutocheck=true',
        '-Djava.util.prefs.PreferencesFactory=DriveMemoryPreferencesFactory',
        '-Djava.io.tmpdir='+str(owned/'java_tmp'),'-Duser.home='+str(owned/'java_home'),
        '-Djava.util.prefs.userRoot='+str(owned/'java_prefs'),
        '-Djava.util.prefs.systemRoot='+str(owned/'java_prefs'),
        '-Dmaven.repo.local='+str(runtime/'m2')]:scyjava.config.add_option(option)
    scyjava.config.add_classpath(str(Path(job['release'])/'java'))
    vendor_paths('trackmate_python_bridge')
    from run_detection import make_imageplus,detect,synthetic_movie
    started=time.perf_counter();ij,System,factory,jvm_arguments=initialize_imagej(imagej,scyjava,paths)
    samples=set(map(int,job['sample_frames']));reports=[];smoke=None
    if request.get('synthetic_smoke'):
        assert job.get('test_mode') is True
        movie,truth=synthetic_movie();imp=make_imageplus(ij,movie,.117,.02)
        try:rows,counts,version=detect(imp,.28,6.5,True,0,.02)
        finally:imp.close()
        assert counts==[2,0,2] and version=='7.14.0'
        for f in [0,2]:
            observed=sorted([(x['x_px'],x['y_px']) for x in rows if x['frame']==f])
            np.testing.assert_allclose(observed,sorted(truth[f]),atol=.12,rtol=0)
        smoke=dict(non_square_shape=list(movie.shape),counts=counts,blank_frame_preserved=True,
                   coordinate_direction_verified=True,actual_native_smoke=True)

    try:
        with tifffile.TiffFile(job['C2_file']) as tf:
            assert len(tf.pages)==2500 and tf.series[0].shape==(2500,684,428)
            for start,stop in request['chunks']:
                assert 0<=start<stop<=2500 and stop-start<=25
                frame_start=time.perf_counter();tag=f'frames{start:04d}_{stop-1:04d}'
                directory=guard_path(Path(job['work'])/'localization_detection');directory.mkdir(parents=True,exist_ok=True)
                movie=np.stack([tf.pages[f].asarray() for f in range(start,stop)])
                assert movie.shape==(stop-start,684,428) and movie.dtype==np.uint16
                hashes=[dict(frame=f,pixel_sha256=hashlib.sha256(movie[f-start].tobytes(order='C')).hexdigest())
                        for f in range(start,stop) if f in samples]
                imp=make_imageplus(ij,movie,.117,.02)
                try:rows,counts,version=detect(imp,.28,6.5,True,start,.02)
                finally:imp.close()
                assert version=='7.14.0' and sum(counts)==len(rows) and len(counts)==stop-start
                p=directory/(tag+'.csv');tmp=p.with_suffix('.csv.tmp')
                with tmp.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
                atomic_replace(tmp,p)
                report=dict(complete=True,scientific_fingerprint=job['scientific_fingerprint'],
                    C2_source_descriptor=request['C2_source_descriptor'],start=start,stop_exclusive=stop,
                    native_fields=FIELDS,rows=len(rows),counts=counts,trackmate_version=version,
                    radius_um=.28,threshold=6.5,median_filter=True,subpixel=True,tracking=False,
                    sample_page_hashes=hashes,pixel_roundtrip_exact=True,
                    csv_sha256=sha(p),preferences_factory=factory,java_input_arguments=jvm_arguments,perf_data_disabled=True,
                    java_user_home=str(System.getProperty('user.home')),
                    java_temp=str(System.getProperty('java.io.tmpdir')),
                    java_version=str(System.getProperty('java.version')),jgo_jar_staging='COPY (external filesystem supports no hardlinks)',
                    elapsed_s=time.perf_counter()-frame_start,completed_utc=utc())
                dump(directory/(tag+'.json'),report);reports.append(report)
                print(json.dumps(dict(stage='native_detection_chunk',tag=tag,rows=len(rows),elapsed_s=report['elapsed_s'])),flush=True)
                del movie,rows;gc.collect();System.gc()
        dump(request['result'],dict(complete=True,chunks=len(reports),frames=sum(x['stop_exclusive']-x['start'] for x in reports),
             bounded_JVM_frame_limit=250,synthetic_smoke=smoke,
             runtime_isolation=dict(perf_data_disabled=True,java_input_arguments=jvm_arguments,
                preferences_factory=factory,user_home=str(System.getProperty('user.home')),
                java_temp=str(System.getProperty('java.io.tmpdir'))),
             elapsed_s=time.perf_counter()-started))
    finally:ij.dispose()

if __name__=='__main__':main()
