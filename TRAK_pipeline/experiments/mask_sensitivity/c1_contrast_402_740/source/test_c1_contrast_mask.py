#!/usr/bin/env python3
"""Isolated fixed C1 pixel-contrast sensitivity experiment; no batch mutation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    stat = path.stat()
    result = dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size <= 1024 * 1024:
        result['sha256'] = sha(path)
    return result


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configure(run):
    envpaths = {
        'TMPDIR': run / 'cache/tmp', 'TMP': run / 'cache/tmp',
        'TEMP': run / 'cache/tmp', 'MPLCONFIGDIR': run / 'cache/matplotlib',
        'NUMBA_CACHE_DIR': run / 'cache/numba', 'XDG_CACHE_HOME': run / 'cache/xdg',
        'PYTHONPYCACHEPREFIX': run / 'cache/python', 'SCIJAVA_CACHE_DIR': run / 'cache/scijava',
    }
    for key, path in envpaths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import tempfile
    tempfile.tempdir = str(run / 'cache/tmp')


def main(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    run = Path(cfg['run_directory']).resolve()
    if '/Volumes/My Book/' not in str(run) or not run.is_dir():
        raise ValueError('This experiment must write only into its existing My Book test directory')
    configure(run)
    import numpy as np
    import tifffile
    sys.path.insert(0, str(run / 'source'))
    from support50_postmin32_core import process_counts
    begun = time.perf_counter()
    logpath = run / 'logs/numerical.log'
    def log(message):
        line = utc() + ' | ' + message
        print(line, flush=True)
        with logpath.open('a') as stream:
            stream.write(line + '\n')
    for folder in ['inputs','data','masks/broad_packed','masks/native_batches','checkpoints','_work','validation']:
        (run / folder).mkdir(parents=True, exist_ok=True)
    binding = hashlib.sha256(config_path.read_bytes()).hexdigest()
    src = Path(cfg['source_output'])
    source_job = json.loads((src / 'job.json').read_text())
    pipeline = Path(cfg['source_release']).parent.parent
    protected_paths = [Path(cfg[k]) for k in ['source_C1','source_C2','source_ROI']]
    protected_paths += [src/'job.json',src/'COMPLETED.json',src/'classification_summary.json',
                        src/'masks/clean_full_mask.npy',src/'masks/analysis_ROI_mask.npy',
                        src/'masks/foreground_count.npy',src/'masks/metadata.json',
                        src/'roi/roi_mask.npy',src/'roi/vertices.npy',
                        pipeline/'CURRENT_RELEASE.txt',pipeline/'batch_config.json',
                        pipeline/'windows/batch_config_windows.json']
    for candidate in [pipeline/'windows/CURRENT_RELEASE.txt',pipeline/'windows/package_manifest.json',
                      pipeline/'WINDOWS_PACKAGE_READY.json',pipeline/'windows/WINDOWS_VALIDATED.json']:
        if candidate.exists():
            protected_paths.append(candidate)
    records_path = run/'source_records.json'
    if records_path.exists():
        records = json.loads(records_path.read_text())
        if records['config_sha256'] != binding:
            raise ValueError('Configuration changed; refusing mixed checkpoint reuse')
        for previous in records['protected_sources']:
            if descriptor(previous['path']) != previous:
                raise ValueError('Protected input changed before resume: '+previous['path'])
    else:
        records = dict(schema_version=1,created_utc=utc(),config_sha256=binding,
                       protected_sources=[descriptor(p) for p in protected_paths],
                       protected_descriptor_fields=['path','size_bytes','mtime_ns','sha256_for_files_at_most_1MiB'],
                       source_C1=cfg['source_C1'],copied_C1=None,
                       registration_status=cfg['registration_status'])
        dump(records_path, records)
    reference = json.loads((run/'source/native_parameters.json').read_text())
    module_checks = []
    for file, expected in reference['source_modules_sha256'].items():
        actual = sha(file)
        module_checks.append(dict(path=file,expected_sha256=expected,actual_sha256=actual,match=actual==expected))
        if actual != expected:
            raise ValueError('Installed Nellie source differs from frozen reference: '+file)
    for key, expected in dict(version='1.0.4',device='cpu',filter_min_radius_um=.25,
                              filter_max_radius_um=1.,remove_edges=False,otsu_thresh_intensity=False).items():
        if reference[key] != expected:
            raise ValueError('Frozen native setting changed: '+key)
    query = ('import json,importlib.metadata as m;'
             'print(json.dumps({k:m.version(k) for k in ["nellie","numpy","scipy","tifffile"]}))')
    versions = json.loads(subprocess.check_output([cfg['runtime']['nellie_python'],'-B','-c',query],
                          cwd=run,env=dict(os.environ),text=True))
    if versions['nellie'] != '1.0.4':
        raise ValueError('Nellie version must remain 1.0.4')
    dump(run/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
         cleanup_versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','tifffile']},
         native_versions=versions,nellie_source_checks=module_checks,
         source_fingerprints=[descriptor(p) for p in sorted((run/'source').glob('*')) if p.is_file()],
         cache_environment={k:v for k,v in os.environ.items() if k in ['TMPDIR','TMP','TEMP','MPLCONFIGDIR',
                           'NUMBA_CACHE_DIR','XDG_CACHE_HOME','PYTHONPYCACHEPREFIX','SCIJAVA_CACHE_DIR']},
         command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path)],
         native_bytecode_policy='Read existing installed bytecode; -B and PYTHONDONTWRITEBYTECODE prevent all bytecode writes; remove empty PYTHONPYCACHEPREFIX only for native worker'))
    # -B prevents writes; allow the native worker to READ existing installed bytecode.
    native_env=dict(os.environ)
    native_env.pop('PYTHONPYCACHEPREFIX',None)
    native_env['PYTHONDONTWRITEBYTECODE']='1'
    # Read-only wheel RECORD comparison proves the package content is identical.
    cleanup_site=Path(cfg['runtime']['cleanup_python']).parent.parent/'lib/python3.11/site-packages'
    nellie_site=Path(cfg['runtime']['nellie_python']).parent.parent/'lib/python3.11/site-packages'
    dependency_proof=[]
    for name,version in [('numpy','2.4.6'),('scipy','1.17.1'),('tifffile','2026.3.3')]:
        entries=[]
        for site in [cleanup_site,nellie_site]:
            with (site/f'{name}-{version}.dist-info/RECORD').open(newline='') as stream:
                entries.append({a:(b,c) for a,b,c in csv.reader(stream)
                                if b and (a.startswith(name+'/') or a.startswith(name+'.'))})
        if entries[0]!=entries[1]:
            raise ValueError('Hydrated native dependency wheel content differs: '+name)
        dependency_proof.append(dict(package=name,version=version,hashed_members=len(entries[0]),
            wheel_package_RECORD_hashes_identical=True,cleanup_location=str(cleanup_site/name),
            original_native_location=str(nellie_site/name)))
    dump(run/'native_dependency_binding.json',dict(packages=dependency_proof,
        policy='Process-local preimport of identical existing hydrated packages; no installation or kernel change',
        generated_bytecode_excluded=True,bytecode_writes_disabled=True,
        optional_TIFF_unused_imports=['skimage.filters','skimage.morphology','skimage.measure','nd2','dask (optional, unavailable)'],
        optional_import_rule='Fail closed on any unexpected attribute access; no used numerical implementation replaced'))
    native_bootstrap = """
import sys,json,runpy,types
from pathlib import Path
site,worker,work,batch,record=sys.argv[1:]
sys.path.insert(0,site)
import numpy,scipy,scipy.ndimage,tifffile
sys.path.pop(0)
class UnusedTIFFDependency(types.ModuleType):
    def __getattr__(self,name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError('Unexpected access to TIFF-unused optional dependency '+self.__name__+'.'+name)
# Nellie __init__ eagerly imports these modules, but its frozen TIFF Filter/Label
# path does not call them. Fail immediately if an unexpected use is attempted.
skimage_proxy=types.ModuleType('skimage')
for optional in ['filters','morphology','measure']:
    setattr(skimage_proxy,optional,UnusedTIFFDependency('skimage.'+optional))
sys.modules['skimage']=skimage_proxy
sys.modules['nd2']=UnusedTIFFDependency('nd2')
# Standard absence behavior for unused optional Dask integration in Pint.
sys.modules['dask']=None
# Import the unmodified Nellie root, but do not eagerly load unused Network/Markers.
import nellie
from importlib.machinery import ModuleSpec
segmentation=types.ModuleType('nellie.segmentation')
segmentation.__path__=[str(Path(nellie.__file__).parent/'segmentation')]
segmentation.__package__='nellie.segmentation'
segmentation.__spec__=ModuleSpec('nellie.segmentation',loader=None,is_package=True)
sys.modules['nellie.segmentation']=segmentation
nellie.segmentation=segmentation
modules=[numpy,scipy,scipy.ndimage,tifffile]
runtime={m.__name__:{'file':m.__file__,'version':getattr(m,'__version__',None)} for m in modules}
runtime['direct_segmentation_namespace_path']=segmentation.__path__
runtime['unused_segmentation_initializer_classes_not_imported']=['Markers','Network']
runtime['TIFF_unused_optional_import_policy']=['skimage.filters','skimage.morphology','skimage.measure','nd2','dask (optional, unavailable)']
Path(record).write_text(json.dumps(runtime,indent=2))
print('Using identical hydrated numerical dependencies; TIFF-unused optional imports fail closed; frozen Nellie worker unchanged',flush=True)
sys.argv=[worker,work,batch]
runpy.run_path(worker,run_name='__main__')
"""
    # A single byte-for-byte copy pass also computes the complete source hash.
    copy_path = run/'inputs/C1_original.tiff'
    copied_marker = run/'checkpoints/original_copy.json'
    if copy_path.exists() and copied_marker.exists():
        copied = json.loads(copied_marker.read_text())
        if copied['config_sha256'] != binding or copy_path.stat().st_size != copied['bytes']:
            raise ValueError('Original copy checkpoint changed')
        log('Reusing committed full original C1 copy')
    else:
        log('Copying the complete original C1 TIFF onto the isolated test directory')
        before = descriptor(cfg['source_C1'])
        partial = copy_path.with_name(copy_path.name+'.partial')
        digest = hashlib.sha256()
        size = 0
        with Path(cfg['source_C1']).open('rb') as source, partial.open('wb') as destination:
            while True:
                chunk = source.read(8*1024*1024)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if before != descriptor(cfg['source_C1']) or size != before['size_bytes']:
            raise ValueError('Original C1 changed during the copy')
        os.replace(partial,copy_path)
        copied = dict(path=str(copy_path),source_path=cfg['source_C1'],bytes=size,
                      sha256=digest.hexdigest(),config_sha256=binding,copy_method='single_stream_byte_for_byte',
                      completed_utc=utc())
        dump(copied_marker,copied)
        records['copied_C1'] = copied
        dump(records_path,records)
        log('Full original C1 copy committed; bytes='+str(size))
    shape = tuple(cfg['shape_TYX'])
    copy_checks = []
    with tifffile.TiffFile(cfg['source_C1']) as original, tifffile.TiffFile(copy_path) as copied_movie:
        if tuple(copied_movie.series[0].shape) != shape or len(copied_movie.pages) != shape[0]:
            raise ValueError('Original copy header mismatch')
        for frame in cfg['sample_frames']:
            a = original.pages[frame].asarray()
            b = copied_movie.pages[frame].asarray()
            if not np.array_equal(a,b):
                raise ValueError('Copied C1 sample page mismatch: '+str(frame))
            copy_checks.append(dict(original_frame=frame,exact_equal=True,pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    dump(run/'validation/original_copy_ten_frame_checks.json',dict(sample_count=len(copy_checks),checks=copy_checks))
    roi = np.load(src/'roi/roi_mask.npy',allow_pickle=False)
    vertices = np.load(src/'roi/vertices.npy',allow_pickle=False)
    if roi.shape != shape[1:] or roi.dtype != bool or int(roi.sum()) != source_job['ROI_pixels']:
        raise ValueError('Original ROI shape/count mismatch')
    np.save(run/'masks/roi_mask.npy',roi)
    np.save(run/'masks/roi_vertices_xy.npy',vertices)
    baseline_full = np.load(src/'masks/clean_full_mask.npy',mmap_mode='r',allow_pickle=False)
    baseline_roi = np.load(src/'masks/analysis_ROI_mask.npy',mmap_mode='r',allow_pickle=False)
    all_final=[];all_counts=[];old_final=[];old_roi=[];zero_packed=[];mapping=[];transform_rows=[];threshold_rows=[];comparison=[]
    block_summaries=[]
    with tifffile.TiffFile(copy_path) as movie:
        for local_block,start in enumerate(cfg['block_starts']):
            block_index=start//50
            ids=np.arange(start,start+50,dtype=np.int32)
            raw=np.stack([movie.pages[int(f)].asarray() for f in ids])
            if raw.shape!=(50,)+shape[1:] or raw.dtype!=np.uint16:
                raise ValueError('Original block shape/type mismatch')
            transformed=np.rint(65535*np.clip((raw.astype(np.float64)-cfg['contrast_low_adu'])/
                         (cfg['contrast_high_adu']-cfg['contrast_low_adu']),0,1)).astype(np.uint16)
            if np.any(transformed[raw==0]!=0):
                raise ValueError('Original zeros were not preserved')
            zero_packed.append(np.packbits((raw==0).reshape(50,-1),axis=1,bitorder='little'))
            input_path=run/'inputs'/f'C1_contrast_frames{start:04d}_{start+49:04d}.ome.tif'
            input_marker=run/'checkpoints'/f'transform_b{block_index:03d}.json'
            if not(input_path.exists() and input_marker.exists()):
                temporary=input_path.with_name(input_path.name+'.partial')
                tifffile.imwrite(temporary,transformed,ome=True,photometric='minisblack',
                    metadata=dict(axes='TYX',PhysicalSizeX=.117,PhysicalSizeY=.117,
                    PhysicalSizeXUnit='µm',PhysicalSizeYUnit='µm',TimeIncrement=.02,TimeIncrementUnit='s'))
                os.replace(temporary,input_path)
                dump(input_marker,dict(config_sha256=binding,original_frames=ids.tolist(),dtype='uint16',shape=list(transformed.shape)))
            elif json.loads(input_marker.read_text())['config_sha256']!=binding:
                raise ValueError('Contrast input binding changed')
            chunk_masks=[]
            for offset in [0,25]:
                native_start=start+offset
                tag=f'frames{native_start:04d}_{native_start+24:04d}'
                permanent=run/'masks/native_batches'/tag
                permanent.mkdir(parents=True,exist_ok=True)
                marker=permanent/'COMPLETED.json'
                packed_path=permanent/'broad_packed.npz'
                if marker.exists() and packed_path.exists():
                    saved=json.loads(marker.read_text())
                    if saved['config_sha256']!=binding:
                        raise ValueError('Native chunk binding changed')
                    with np.load(packed_path,allow_pickle=False) as packed:
                        native=np.unpackbits(packed['packed'],axis=1,count=shape[1]*shape[2],bitorder='little').reshape(25,*shape[1:]).astype(bool)
                        if packed['original_frame_ids'].tolist()!=list(range(native_start,native_start+25)):
                            raise ValueError('Native chunk frame mapping changed')
                    chunk_thresholds=json.loads((permanent/'thresholds.json').read_text())
                    log(f'Reusing native chunk {native_start}-{native_start+24}')
                else:
                    scratch=run/'_work'/tag
                    if scratch.exists():
                        owner=scratch/'.owned_contrast_test'
                        if not owner.exists():
                            raise ValueError('Refusing to remove unowned native scratch')
                        shutil.rmtree(scratch)
                    scratch.mkdir()
                    (scratch/'.owned_contrast_test').write_text(binding+'\n')
                    tifffile.imwrite(scratch/'input.ome.tif',transformed[offset:offset+25],ome=True,
                        metadata=dict(axes='TYX',PhysicalSizeX=.117,PhysicalSizeY=.117,
                        PhysicalSizeXUnit='µm',PhysicalSizeYUnit='µm',TimeIncrement=.02,TimeIncrementUnit='s'))
                    log(f'Native Nellie broad starting: original frames {native_start}-{native_start+24}')
                    t0=time.perf_counter()
                    with (permanent/'nellie.log').open('w') as stream:
                        subprocess.run([cfg['runtime']['nellie_python'],'-B','-u','-c',native_bootstrap,str(cleanup_site),
                                        str(run/'source/nellie_worker.py'),str(run/'_work'),str(scratch),
                                        str(permanent/'native_dependency_runtime.json')],cwd=run,env=native_env,
                                        stdout=stream,stderr=subprocess.STDOUT,check=True)
                    native=np.load(scratch/'broad_mask.npy',allow_pickle=False)
                    if native.shape!=(25,)+shape[1:] or native.dtype!=bool:
                        raise ValueError('Native broad mask shape/type changed')
                    actual=json.loads((scratch/'actual_parameters.json').read_text())
                    if actual['native_min_area']!=4 or not np.allclose(actual['sigmas_pixels'],reference['actual_sigmas_pixels'],rtol=0,atol=1e-14):
                        raise ValueError('Native broad parameters changed')
                    chunk_thresholds=json.loads((scratch/'thresholds.json').read_text())
                    if len(chunk_thresholds)!=25:
                        raise ValueError('Native threshold count mismatch')
                    for j,row in enumerate(chunk_thresholds):
                        row['original_frame']=native_start+j
                        row['block_index']=block_index
                    np.savez_compressed(packed_path,packed=np.packbits(native.reshape(25,-1),axis=1,bitorder='little'),
                        shape=np.array(native.shape,dtype=np.int32),original_frame_ids=np.arange(native_start,native_start+25,dtype=np.int32),bitorder=np.array('little'))
                    dump(permanent/'thresholds.json',chunk_thresholds)
                    dump(permanent/'actual_parameters.json',actual)
                    dump(marker,dict(config_sha256=binding,original_frame_start=native_start,
                        original_frame_stop_exclusive=native_start+25,completed_utc=utc(),elapsed_s=time.perf_counter()-t0))
                    shutil.rmtree(scratch)
                    log(f'Native Nellie broad committed: {native_start}-{native_start+24}; {time.perf_counter()-t0:.1f} s')
                chunk_masks.append(native)
                threshold_rows.extend(chunk_thresholds)
            broad=np.concatenate(chunk_masks,axis=0)
            np.savez_compressed(run/'masks/broad_packed'/f'b{block_index:03d}.npz',
                packed=np.packbits(broad.reshape(50,-1),axis=1,bitorder='little'),shape=np.array(broad.shape,dtype=np.int32),
                original_frame_ids=ids,bitorder=np.array('little'))
            counts=broad.sum(axis=0,dtype=np.uint8)
            result=process_counts(counts,support_minimum=cfg['support_minimum'],
                minimum_foreground=cfg['foreground_min_pixels'],hole_limit=cfg['hole_exclusive_max_pixels'],frame_count=50)
            final=result['final_mask']
            old=np.asarray(baseline_full[block_index])
            oldr=np.asarray(baseline_roi[block_index])
            if not np.array_equal(oldr,old & roi):
                raise ValueError('Baseline ROI intersection mismatch')
            all_final.append(final);all_counts.append(counts);old_final.append(old.copy());old_roi.append(oldr.copy())
            for scope,gate in [('ROI',roi),('full',np.ones(roi.shape,dtype=bool))]:
                before=old & gate;after=final & gate
                n0=int(before.sum());n1=int(after.sum());intersection=int((before&after).sum());union=int((before|after).sum())
                added=int((after&~before).sum());removed=int((before&~after).sum());area=(cfg['pixel_size_nm']/1000)**2
                comparison.append(dict(scope=scope,block_index=block_index,original_frame_start=start,
                    original_frame_stop_exclusive=start+50,old_pixels=n0,new_pixels=n1,area_old_um2=n0*area,
                    area_new_um2=n1*area,delta_pixels=n1-n0,delta_area_um2=(n1-n0)*area,
                    intersection_pixels=intersection,union_pixels=union,added_pixels=added,removed_pixels=removed,
                    iou=intersection/union if union else 1.,dice=2*intersection/(n0+n1) if n0+n1 else 1.))
            for j,frame in enumerate(ids):
                f=int(frame)
                mapping.append(dict(page_index=len(mapping),original_frame=f,time_s=f*cfg['frame_interval_s'],
                    mask_block=block_index,local_block_index=local_block,original_frame_start=start,original_frame_stop_exclusive=start+50))
                row=dict(original_frame=f,block_index=block_index)
                for scope,values in [('full',raw[j]),('ROI',raw[j][roi])]:
                    row.update({f'{scope}_pixels':int(values.size),f'{scope}_original_zero_pixels':int(np.count_nonzero(values==0)),
                        f'{scope}_low_clipped_pixels':int(np.count_nonzero(values<=cfg['contrast_low_adu'])),
                        f'{scope}_high_clipped_pixels':int(np.count_nonzero(values>=cfg['contrast_high_adu'])),
                        f'{scope}_new_zero_nonzero_input_pixels':int(np.count_nonzero((values>0)&(values<=cfg['contrast_low_adu'])))})
                transform_rows.append(row)
            summary=dict(block_index=block_index,local_block_index=local_block,original_frames=[start,start+49],
                full_mask_pixels=int(final.sum()),ROI_mask_pixels=int((final&roi).sum()),completed_utc=utc())
            block_summaries.append(summary)
            dump(run/'checkpoints'/f'b{block_index:03d}.json',dict(config_sha256=binding,**summary))
            log(f'Block {block_index} committed (original {start}-{start+49}); ROI pixels old={int(oldr.sum())}, new={int((final&roi).sum())}')
    np.save(run/'masks/foreground_count.npy',np.stack(all_counts))
    np.save(run/'masks/clean_full_mask.npy',np.stack(all_final))
    np.save(run/'masks/analysis_ROI_mask.npy',np.stack(all_final)&roi)
    np.save(run/'masks/baseline_clean_full_mask.npy',np.stack(old_final))
    np.save(run/'masks/baseline_analysis_ROI_mask.npy',np.stack(old_roi))
    all_ids=np.concatenate([np.arange(s,s+50,dtype=np.int32) for s in cfg['block_starts']])
    np.savez_compressed(run/'masks/original_zero_regions_packed.npz',packed=np.concatenate(zero_packed),
        shape=np.array((150,)+shape[1:],dtype=np.int32),original_frame_ids=all_ids,bitorder=np.array('little'))
    write_csv(run/'data/mask_comparison.csv',comparison)
    write_csv(run/'data/frame_mapping.csv',mapping)
    write_csv(run/'data/transform_statistics.csv',transform_rows)
    write_csv(run/'data/native_thresholds.csv',threshold_rows)
    after=[descriptor(row['path']) for row in records['protected_sources']]
    if after!=records['protected_sources']:
        raise ValueError('Protected source descriptors changed during the test')
    records['protected_sources_after']=after
    records['protected_sources_unchanged']=True
    records['completed_utc']=utc()
    dump(records_path,records)
    dump(run/'masks/metadata.json',dict(shape=[3,*shape[1:]],axes='TYX',original_blocks=[s//50 for s in cfg['block_starts']],
        original_frame_ids=all_ids.tolist(),pixel_size_nm=cfg['pixel_size_nm'],recipe=cfg['recipe'],
        raw_zero_unknown_reference='original_zero_regions_packed.npz; never use contrast-created zeros',
        native_versions=versions,formula=cfg['formula']))
    dump(run/'NUMERICAL_COMPLETE.json',dict(status='NUMERICAL_COMPLETE',complete=True,completed_utc=utc(),
        elapsed_s=time.perf_counter()-begun,config_sha256=binding,frames=150,native_chunks=6,blocks=block_summaries,
        protected_sources_unchanged=True,source_copy_sha256=copied['sha256'],original_copy_sample_checks=10,
        numerical_outputs=['masks/clean_full_mask.npy','masks/analysis_ROI_mask.npy','masks/foreground_count.npy',
                           'data/mask_comparison.csv','data/frame_mapping.csv','data/transform_statistics.csv']))
    log(f'Numerical experiment complete: 150 frames, 3 support blocks; {time.perf_counter()-begun:.1f} s')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--numerical-only',action='store_true',
        help='Run/resume only contrast transformation and native mask stages')
    args=parser.parse_args()
    main(args.config)
    if not args.numerical_only:
        config_path=Path(args.config).resolve()
        configuration=json.loads(config_path.read_text())
        test_run=Path(configuration['run_directory'])
        cleanup_python=configuration['runtime']['cleanup_python']
        renderer_python=cleanup_python
        renderer_record=test_run/'render_runtime.json'
        if renderer_record.exists():
            renderer_python=json.loads(renderer_record.read_text()).get('render_python',cleanup_python)
        render_command=[renderer_python,'-B','-u',str(test_run/'source/render_contrast_test.py'),
                        '--config',str(config_path)]
        validation_command=[cleanup_python,'-B','-u',str(test_run/'source/validate_contrast_test.py'),
                            '--config',str(config_path),'--include-render']
        dump(test_run/'command_reproduce_all.json',dict(numerical_entry=str(Path(__file__).resolve()),
             config=str(config_path),render=render_command,validation=validation_command,
             completion_policy='Global COMPLETE is written only after final visual review'))
        subprocess.run(render_command,cwd=test_run,env=dict(os.environ),check=True)
        subprocess.run(validation_command,cwd=test_run,env=dict(os.environ),check=True)
