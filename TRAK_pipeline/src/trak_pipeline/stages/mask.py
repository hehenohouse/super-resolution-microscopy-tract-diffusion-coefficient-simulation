"""Independent full-field Nellie, temporal counts, threshold, cleanup and ROI stages.

No downstream threshold or ROI is imported by the broad/count stages. Original
frame IDs are explicit and an incomplete temporal window remains unavailable.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def _result(files, metadata):
    from trak_pipeline.types import StageResult
    return StageResult(files, metadata)


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _array(path, value):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as stream: np.save(stream, value, allow_pickle=False)
        os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)


def _close(array):
    if hasattr(array, 'flush'): array.flush()
    mm = getattr(array, '_mmap', None)
    if mm is not None: mm.close()


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def _input(ctx, *names):
    for name in names:
        if name in ctx.inputs: return ctx.inputs[name]
    raise ValueError('Missing input artifact: ' + '/'.join(names))


def _file(artifact, *names):
    for name in names:
        if name in artifact.files: return Path(artifact.files[name])
    raise ValueError(artifact.stage + ' artifact lacks ' + '/'.join(names))


def _acquisition(ctx):
    acquisition = dict(ctx.config.get('acquisition', {}))
    acquisition.update({key:value for key,value in ctx.parameters.items()
                        if key in ('shape','shape_yx','n_frames','frames','pixel_size_nm','frame_interval_s','original_frame_ids')})
    shape = acquisition.get('shape_yx', acquisition.get('shape'))
    if shape is None: raise ValueError('Explicit acquisition shape is required')
    shape = tuple(int(x) for x in shape)
    if len(shape) == 3:
        n = shape[0]; shape = shape[1:]
    else: n = acquisition.get('n_frames', acquisition.get('frames', acquisition.get('frame_count')))
    frames = acquisition.get('original_frame_ids')
    if frames is None:
        if n is None: raise ValueError('Explicit original frame count is required')
        frames = list(range(int(n)))
    frames = [int(x) for x in frames]
    if len(shape)!=2 or min(shape)<=0 or not frames or len(set(frames))!=len(frames) or any(f<0 for f in frames):
        raise ValueError('Invalid acquisition grid/frame IDs')
    scale = float(acquisition['pixel_size_nm']); dt = float(acquisition['frame_interval_s'])
    if not math.isfinite(scale) or not math.isfinite(dt) or min(scale,dt)<=0:
        raise ValueError('Explicit positive pixel/time units required')
    return shape, frames, scale, dt


def _contract(ctx, shape, frames, scale, dt, blocks, participants, valid, reasons):
    frame_to_block = {}
    for b, group in enumerate(blocks):
        for f in group:
            if str(f) in frame_to_block: raise ValueError('Overlapping support windows are not supported by one-mask-per-frame contract')
            frame_to_block[str(f)] = b
    return dict(axes='TYX', shape=[len(blocks),*shape], original_frame_ids=frames,
        pixel_center_convention='integer_centers_xy;closed_unit_squares', pixel_size_nm=scale,
        frame_interval_s=dt,crop_offset_xy_px=[0,0], coordinate_transform='identity',
        registration=dict(transform='identity',status='unverified'),
        frame_to_block=frame_to_block, block_frame_ids=participants, expected_block_frame_ids=blocks,
        temporal_window_frame_ids=blocks, participating_frame_ids=participants, block_valid=[bool(v) for v in valid],
        block_reasons=list(reasons), time_semantics='original frame f uses explicitly mapped temporal support block',
        missing_frames_are_not_empty_masks=True)


def _load_contract(artifact):
    if 'contract' in artifact.files: return _read(artifact.files['contract'])
    if isinstance(artifact.metadata.get('contract'), dict): return dict(artifact.metadata['contract'])
    raise ValueError(artifact.stage + ' lacks explicit spatial/time contract')


def _validate_contract(contract, shape=None, scale=None, dt=None):
    try:
        from trak_pipeline.contracts import validate_mask_contract
    except ImportError:
        # The public contract validator is installed by the foundation module.
        raise ValueError('Mask contract validator is unavailable')
    return validate_mask_contract(contract, expected_shape=shape,
        expected_pixel_size_nm=scale, expected_frame_interval_s=dt)


def pack_broad(masks):
    import numpy as np
    masks=np.asarray(masks)
    if masks.dtype!=bool or masks.ndim!=3 or min(masks.shape)<1: raise ValueError('Expected nonempty boolean TYX broad masks')
    return np.packbits(masks.reshape(len(masks),-1), axis=1, bitorder='little')


def unpack_broad(packed, shape):
    import numpy as np
    shape=tuple(int(x) for x in shape); packed=np.asarray(packed)
    if len(shape)!=3 or min(shape)<1 or packed.dtype!=np.uint8: raise ValueError('Invalid packed broad mask')
    n=shape[1]*shape[2]
    if packed.shape!=(shape[0], (n+7)//8): raise ValueError('Packed broad shape mismatch')
    if n%8 and np.any(packed[:,-1]>>(n%8)): raise ValueError('Packed broad padding is nonzero')
    return np.unpackbits(packed, axis=1, count=n, bitorder='little').reshape(shape).astype(bool)


def save_packed(path, masks, frame_ids):
    import numpy as np
    path=Path(path); frames=np.asarray(frame_ids, dtype=np.int32)
    if frames.shape!=(len(masks),) or len(set(frames.tolist()))!=len(frames): raise ValueError('Invalid broad frame IDs')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,packed=pack_broad(masks),shape=np.asarray(masks.shape,dtype=np.int32),
                original_frame_ids=frames,bitorder=np.asarray('little'))
        os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)


def load_packed(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as bank:
        if str(bank['bitorder'].item())!='little': raise ValueError('Unexpected broad bit order')
        masks=unpack_broad(bank['packed'],bank['shape']); frames=bank['original_frame_ids'].astype(int).tolist()
    if len(frames)!=len(masks) or len(frames)!=len(set(frames)): raise ValueError('Invalid packed broad frame IDs')
    return masks,frames


def _broad_records(artifact):
    if 'index' in artifact.files:
        records=_read(artifact.files['index'])['chunks']
        parent=Path(artifact.files['index']).parent
        return [dict(record,path=str((parent/record['path']).resolve()) if not Path(record['path']).is_absolute() else record['path']) for record in records]
    folder=_file(artifact,'packed_dir','broad_packed','packed')
    if not folder.is_dir(): raise ValueError('Broad packed directory is missing')
    return [dict(path=str(path)) for path in sorted(folder.glob('*.npz')) if not path.name.startswith('._')]


def _iter_broad_masks(artifact, chunk_frames):
    """Stream old boolean NPY stacks or packed chunks without whole-movie loading."""
    import numpy as np
    if 'array' in artifact.files:
        values=np.load(artifact.files['array'],mmap_mode='r',allow_pickle=False)
        frames=np.load(_file(artifact,'frame_ids','original_frame_ids'),allow_pickle=False)
        try:
            if values.ndim!=3 or values.dtype!=bool or frames.shape!=(len(values),) or frames.dtype.kind not in 'ui':
                raise ValueError('Legacy broad array/frame mapping contract mismatch')
            ids=[int(f) for f in frames]
            if len(set(ids))!=len(ids):raise ValueError('Duplicate legacy broad original frame')
            for start in range(0,len(values),chunk_frames):
                yield values[start:start+chunk_frames],ids[start:start+chunk_frames]
        finally:_close(values)
    else:
        for record in _broad_records(artifact):yield load_packed(record['path'])


def _native_worker(request_path):
    """Exact frozen Nellie call sequence, with explicit grid, units and radii."""
    import importlib.metadata
    import numpy as np
    import tifffile
    request=_read(request_path); work=Path(request['work']); work.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(work/'cache/matplotlib')
    from nellie.im_info.verifier import FileInfo,ImInfo
    from nellie.segmentation.filtering import Filter
    from nellie.segmentation.labelling import Label
    if importlib.metadata.version('nellie')!=request['nellie_version']: raise ValueError('Nellie version differs from explicit method')
    fi=FileInfo(request['input'],output_dir=str(work/'native'))
    fi.find_metadata();fi.load_metadata()
    if fi.axes!='TYX': raise ValueError('Native input must have TYX axes')
    for key,value in [('X',request['pixel_size_nm']/1000),('Y',request['pixel_size_nm']/1000),('T',request['frame_interval_s'])]: fi.change_dim_res(key,value)
    im=ImInfo(fi); n=len(request['frame_ids'])
    if tuple(im.shape)!=(n,*request['shape_yx']) or not im.no_z: raise ValueError('Native input grid differs from request')
    lo,hi=request['filter_radii_um']
    filt=Filter(im,num_t=n,min_radius_um=lo,max_radius_um=hi,remove_edges=False,device='cpu');filt.run()
    lab=Label(im,num_t=n,min_radius_um=request['pixel_size_nm']/1000,otsu_thresh_intensity=False,device='cpu')
    thresholds=[]; original=lab._compute_frame_thresholds
    def record(a,b):
        intensity,frangi=original(a,b)
        thresholds.append(dict(local_frame=len(thresholds),original_frame=request['frame_ids'][len(thresholds)],
            intensity_threshold_ADU=None if intensity is None else float(intensity),frangi_threshold=None if frangi is None else float(frangi)))
        return intensity,frangi
    lab._compute_frame_thresholds=record;lab.run()
    labels=tifffile.imread(im.pipeline_paths['im_instance_label']); masks=labels>0
    if masks.shape!=(n,*request['shape_yx']) or len(thresholds)!=n: raise ValueError('Nellie result mapping mismatch')
    _array(request['output'],masks)
    _json(work/'thresholds.json',thresholds)
    _json(work/'actual_parameters.json',dict(nellie_version=importlib.metadata.version('nellie'),
        sigmas_pixels=[float(v) for v in filt.sigmas],native_min_area=int(lab.min_area_pixels),
        filter_radii_um=request['filter_radii_um'],pixel_size_nm=request['pixel_size_nm'],frame_interval_s=request['frame_interval_s']))


def execute_broad(ctx):
    import numpy as np
    import tifffile
    shape,frames,scale,dt=_acquisition(ctx); p=ctx.parameters
    chunk=int(ctx.config.get('execution',{}).get('chunk_frames',p.get('chunk_frames',25)))
    if chunk<=0: raise ValueError('chunk_frames must be positive')
    radii=[float(x) for x in p['filter_radii_um']]
    if len(radii)!=2 or not 0<radii[0]<=radii[1]: raise ValueError('Invalid Nellie filtering radii')
    c1=_input(ctx,'C1','c1'); source=_file(c1,'image','tiff')
    executable=ctx.runtime.get('nellie_python',ctx.config.get('runtime',{}).get('nellie_python'))
    if not executable: raise ValueError('Explicit existing Nellie Python executable is required')
    out=ctx.output_dir; out.mkdir(parents=True,exist_ok=True)
    packed=out/'broad_packed';packed.mkdir(exist_ok=True)
    scratch=out/'native_work';scratch.mkdir(exist_ok=True)
    records=[];available=[]; sampled=[]
    sample_frames=set(ctx.config.get('sample_frames',ctx.config.get('checking',{}).get('frames',[])))
    declared=c1.metadata.get('original_frame_ids',frames)
    if len(set(declared))!=len(declared): raise ValueError('Duplicate C1 original frame IDs')
    with tifffile.TiffFile(source) as movie:
        if len(movie.pages)!=len(declared): raise ValueError('C1 page count and declared original frames differ')
        page_for={int(f):i for i,f in enumerate(declared)}
        available=[f for f in frames if f in page_for]
        for start in range(0,len(available),chunk):
            ids=available[start:start+chunk];tag='chunk_%06d'%start; dest=packed/(tag+'.npz'); complete=packed/(tag+'.json')
            if dest.exists() and complete.exists():
                record=_read(complete)
                if record.get('frame_ids')!=ids or record.get('parameters')!=p: raise ValueError('Native chunk checkpoint belongs to different input/parameters')
                # Sampled pages only are independently read back; presence/shape are lightweight checks.
                if sample_frames.intersection(ids):
                    saved,saved_ids=load_packed(dest)
                    if saved_ids!=ids or saved.shape!=(len(ids),*shape):raise ValueError('Native checkpoint mapping/shape mismatch')
                records.append(dict(path=str(dest.relative_to(out)),frame_ids=ids));continue
            work=scratch/tag
            if work.exists(): shutil.rmtree(work)
            work.mkdir()
            raw=np.stack([movie.pages[page_for[f]].asarray() for f in ids])
            if raw.shape!=(len(ids),*shape) or raw.dtype!=np.uint16: raise ValueError('Original C1 grid/type mismatch')
            input_tif=work/'input.ome.tif'
            tifffile.imwrite(input_tif,raw,ome=True,metadata=dict(axes='TYX',PhysicalSizeX=scale/1000,PhysicalSizeY=scale/1000,
                PhysicalSizeXUnit='µm',PhysicalSizeYUnit='µm',TimeIncrement=dt,TimeIncrementUnit='s'))
            if sample_frames.intersection(ids):
                with tifffile.TiffFile(input_tif) as own:
                    for i,f in enumerate(ids):
                        if f in sample_frames:
                            np.testing.assert_array_equal(own.pages[i].asarray(),raw[i]);sampled.append(f)
            request=dict(work=str(work),input=str(input_tif),output=str(work/'broad_mask.npy'),frame_ids=ids,
                shape_yx=list(shape),pixel_size_nm=scale,frame_interval_s=dt,filter_radii_um=radii,nellie_version=p['nellie_version'])
            req=work/'request.json';_json(req,request)
            with (out/(tag+'_nellie.log')).open('w',encoding='utf-8') as stream:
                subprocess.run([str(executable),'-B','-u',str(Path(__file__).resolve()),'--native-worker',str(req)],
                    cwd=work,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
            masks=np.load(work/'broad_mask.npy',allow_pickle=False)
            if masks.shape!=raw.shape or masks.dtype!=bool:raise ValueError('Native broad result differs from original grid')
            save_packed(dest,masks,ids)
            for name in ('thresholds.json','actual_parameters.json'):shutil.copy2(work/name,packed/(tag+'_'+name))
            if sample_frames.intersection(ids):
                saved,saved_ids=load_packed(dest)
                for i,f in enumerate(ids):
                    if f in sample_frames:np.testing.assert_array_equal(saved[i],masks[i])
            _json(complete,dict(frame_ids=ids,parameters=p,shape=list(masks.shape),complete=True))
            records.append(dict(path=str(dest.relative_to(out)),frame_ids=ids))
            del masks,raw;shutil.rmtree(work)
            if (start+len(ids))%250==0 or start+len(ids)==len(available):
                _json(out/'checkpoint.json',dict(completed_frames=start+len(ids),expected_frames=len(frames),available_frames=len(available)))
                ctx.progress('Nellie broad checkpoint',completed_frames=start+len(ids),expected_frames=len(frames))
    index=out/'broad_index.json';_json(index,dict(chunks=records,original_frame_ids=frames,available_frame_ids=available,
        shape_yx=list(shape),pixel_size_nm=scale,frame_interval_s=dt,axes='TYX',roi_independent=True,
        native_post_min32=False,sampled_original_frames=sorted(set(sampled)),sampled_check_only=True))
    return _result(dict(index=index,packed_dir=packed),dict(original_frame_ids=frames,available_frame_ids=available,
        shape_yx=list(shape),pixel_size_nm=scale,frame_interval_s=dt,sampled_check_only=True))


def _support_windows(frames,window,stride,participation=None):
    if window<=0 or stride<=0:raise ValueError('Temporal window and stride must be positive')
    if stride!=window:raise ValueError('Current one-mask-per-frame method requires stride_frames == window_frames; overlapping/gapped schemes need an explicit mapping method')
    if frames!=list(range(min(frames),max(frames)+1)):raise ValueError('Acquisition frame IDs must declare missing frames explicitly, not omit them')
    blocks=[list(range(f,min(f+window,max(frames)+1))) for f in range(min(frames),max(frames)+1,stride)]
    wanted=set(frames if participation is None else participation)
    if not wanted.issubset(set(frames)):raise ValueError('Support participation references undeclared original frame')
    return blocks,wanted


def execute_support_counts(ctx):
    import numpy as np
    shape,frames,scale,dt=_acquisition(ctx);p=ctx.parameters
    if p.get('count_rule')!='sum_bool_per_pixel':raise ValueError('Unsupported support count method')
    if p.get('partial_window_policy')!='unknown':raise ValueError('Incomplete windows must be unknown')
    window=int(p['window_frames']);stride=int(p['stride_frames'])
    participation=p.get('frame_ids',frames)
    blocks,wanted=_support_windows(frames,window,stride,participation)
    mapping={f:b for b,group in enumerate(blocks) for f in group}
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True)
    path=out/'foreground_count.npy'
    dtype=np.min_scalar_type(window)
    counts=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=(len(blocks),*shape));counts[:]=0
    participants=[[] for _ in blocks];seen=set()
    try:
        broad=_input(ctx,'broad','mask_broad')
        for values,ids in _iter_broad_masks(broad,int(ctx.config.get('execution',{}).get('chunk_frames',25))):
            if values.shape[1:]!=shape:raise ValueError('Broad spatial grid differs from acquisition')
            for value,f in zip(values,ids):
                if f in seen:raise ValueError('Duplicate broad original frame '+str(f))
                seen.add(f)
                if f not in mapping:raise ValueError('Broad frame lies outside declared acquisition')
                if f in wanted:counts[mapping[f]]+=value;participants[mapping[f]].append(f)
            if len(seen)%250==0:ctx.progress('Support count checkpoint',frames_seen=len(seen))
        for group in participants:group.sort()
        # Intentional participant selection is part of the cache identity. All selected
        # members must exist, and the physical acquisition window must be complete.
        valid=[];reasons=[]
        for group,actual in zip(blocks,participants):
            missing=set(group)-seen
            reason='incomplete_support_block' if missing or len(group)!=window else ''
            if not reason and not actual:reason='no_participating_frames'
            valid.append(not reason);reasons.append(reason)
        counts.flush()
    finally:_close(counts)
    contract=_contract(ctx,shape,frames,scale,dt,blocks,participants,valid,reasons)
    _validate_contract(contract,scale=scale,dt=dt)
    cp=out/'contract.json';_json(cp,contract)
    mapping_file=out/'frame_mapping.json';_json(mapping_file,dict(frame_to_block=contract['frame_to_block'],
        block_frame_ids=blocks,participating_frame_ids=participants,block_valid=valid,block_reasons=reasons))
    return _result(dict(counts=path,contract=cp,mapping=mapping_file),dict(contract=contract,
        count_rule=p['count_rule'],window_frames=window,stride_frames=stride))


def execute_threshold(ctx):
    import numpy as np
    source=_input(ctx,'support_counts','mask_support');contract=_load_contract(source)
    minimum=int(ctx.parameters['minimum_support'])
    if minimum<1:raise ValueError('minimum_support must be positive')
    counts=np.load(_file(source,'counts','foreground_count'),mmap_mode='r',allow_pickle=False)
    _validate_contract(contract,shape=counts.shape)
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True);dest=out/'support_mask.npy'
    result=np.lib.format.open_memmap(dest,mode='w+',dtype=bool,shape=counts.shape)
    try:
        if counts.dtype.kind not in 'ui':raise ValueError('Support counts must be unsigned integers')
        if minimum>int(source.parameters.get('window_frames',max(len(g) for g in contract['block_frame_ids']))):raise ValueError('Support threshold exceeds temporal window')
        for b in range(len(counts)):
            result[b]=counts[b]>=minimum if contract['block_valid'][b] else False
        result.flush()
    finally:_close(result);_close(counts)
    cp=out/'contract.json';_json(cp,contract)
    return _result(dict(threshold_mask=dest,contract=cp),dict(contract=contract,minimum_support=minimum))


def _cleanup_core(package_root):
    path=Path(package_root)/'src/trak_pipeline/compat/vendor/spatial_mask_preview/support50_postmin32_core.py'
    spec=importlib.util.spec_from_file_location('_trak_frozen_support_cleanup',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cleanup_mask(mask,minimum,hole_limit,package_root):
    core=_cleanup_core(package_root)
    cleaned,*_=core.clean_frame(mask,int(minimum))
    final,added,*_=core.fill_enclosed(cleaned,int(hole_limit))
    return final,dict(post_cleanup_foreground_pixels=int(cleaned.sum()),filled_pixels=int(added.sum()))


def execute_cleanup(ctx):
    import numpy as np
    p=ctx.parameters
    if int(p['foreground_connectivity'])!=8 or int(p['background_connectivity'])!=4:
        raise ValueError('Current cleanup method requires foreground8/background4 connectivity')
    source=_input(ctx,'mask_threshold','threshold');contract=_load_contract(source)
    masks=np.load(_file(source,'threshold_mask','mask'),mmap_mode='r',allow_pickle=False)
    _validate_contract(contract,shape=masks.shape)
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True);dest=out/'clean_full_mask.npy'
    result=np.lib.format.open_memmap(dest,mode='w+',dtype=bool,shape=masks.shape)
    statistics=[]
    try:
        if masks.dtype!=bool:raise ValueError('Threshold masks must be boolean')
        for b in range(len(masks)):
            if contract['block_valid'][b]:
                final,stat=cleanup_mask(masks[b],p['min_foreground_pixels'],p['max_hole_area_exclusive'],ctx.package_root)
                result[b]=final
            else:
                result[b]=False;stat=dict(post_cleanup_foreground_pixels=0,filled_pixels=0)
            statistics.append(dict(block_index=b,valid=contract['block_valid'][b],unknown_reason=contract['block_reasons'][b],
                mask_full_area_px2=int(result[b].sum()),**stat))
        result.flush()
    finally:_close(result);_close(masks)
    cp=out/'contract.json';_json(cp,contract);sp=out/'cleanup_statistics.json';_json(sp,statistics)
    return _result(dict(full_mask=dest,contract=cp,statistics=sp),dict(contract=contract))


def execute_roi(ctx):
    import numpy as np
    source=_input(ctx,'mask_cleanup','cleanup');contract=_load_contract(source)
    masks=np.load(_file(source,'full_mask','clean_full_mask'),mmap_mode='r',allow_pickle=False)
    roi_artifact=_input(ctx,'roi','ROI');roi=np.load(_file(roi_artifact,'mask','roi_mask'),allow_pickle=False)
    _validate_contract(contract,shape=masks.shape)
    if roi.dtype!=bool or roi.shape!=masks.shape[1:]:raise ValueError('ROI and full-mask grids differ')
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True);dest=out/'analysis_ROI_mask.npy'
    result=np.lib.format.open_memmap(dest,mode='w+',dtype=bool,shape=masks.shape);statistics=[]
    try:
        if masks.dtype!=bool:raise ValueError('Complete mask must be boolean')
        for b in range(len(masks)):
            result[b]=masks[b]&roi
            block=contract.get('expected_block_frame_ids',contract['block_frame_ids'])[b]
            statistics.append(dict(block_index=b,original_frame_start=min(block),original_frame_stop_exclusive=max(block)+1,
                mask_full_area_px2=int(masks[b].sum()),mask_ROI_area_px2=int(result[b].sum()),
                mask_full_area_um2=float(masks[b].sum())*(contract['pixel_size_nm']/1000)**2,
                mask_ROI_area_um2=float(result[b].sum())*(contract['pixel_size_nm']/1000)**2,
                valid=contract['block_valid'][b],unknown_reason=contract['block_reasons'][b]))
        result.flush()
    finally:_close(result);_close(masks)
    cp=out/'contract.json';_json(cp,contract)
    sp=out/'per_block_statistics.csv'
    with sp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    return _result(dict(roi_mask=dest,contract=cp,statistics=sp),dict(contract=contract,roi_pixels=int(roi.sum())))


def _native_main(argv=None):
    """Preserve infrastructure/cancellation status through the environment boundary."""
    argv=list(sys.argv[1:] if argv is None else argv)
    if len(argv)!=2 or argv[0]!='--native-worker':
        raise SystemExit('Use the public trak_pipeline entry point; --native-worker accepts an owned request only.')
    try:
        _native_worker(argv[1]);return 0
    except KeyboardInterrupt:
        print('Native Nellie worker interrupted; owned checkpoints are retained.',file=sys.stderr,flush=True)
        return 130
    except Exception as error:
        import traceback
        traceback.print_exc()
        if isinstance(error,(ModuleNotFoundError,ImportError)):
            return 74
        if isinstance(error,OSError) and getattr(error,'errno',None) in (5,6,19,28,30,57,60,74,121,122):
            return 74
        return 1


if __name__=='__main__':raise SystemExit(_native_main())
