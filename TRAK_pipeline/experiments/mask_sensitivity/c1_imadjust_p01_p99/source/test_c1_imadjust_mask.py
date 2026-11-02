"""Isolated per-frame C1 MATLAB-style contrast test; masks only."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,math,os,platform,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);args=parser.parse_args()
cfg_path=Path(args.config).resolve();cfg=json.loads(cfg_path.read_text());run=Path(cfg["run_directory"]).resolve()
if not run.is_relative_to(Path("/Volumes/My Book")):raise ValueError("Outputs must stay on My Book")
for key,folder in {"TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp","MPLCONFIGDIR":"matplotlib","NUMBA_CACHE_DIR":"numba","XDG_CACHE_HOME":"xdg"}.items():
    dest=run/"cache"/folder;dest.mkdir(parents=True,exist_ok=True);os.environ[key]=str(dest)
os.environ.update(PYTHONDONTWRITEBYTECODE="1",PYDANTIC_DISABLE_PLUGINS="__all__",MPLBACKEND="Agg",QT_QPA_PLATFORM="offscreen",
    OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import tempfile;tempfile.tempdir=str(run/"cache/tmp")
import numpy as np
import tifffile
sys.path.insert(0,str(run/"source"))
from support50_postmin32_core import process_counts
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".partial")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8");os.replace(tmp,p)
def rows(path,values):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def descriptor(path):
    p=Path(path).resolve();s=p.stat();r=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if s.st_size<=1024*1024:r["sha256"]=sha(p)
    return r
def log(text):
    line=datetime.now(timezone.utc).isoformat()+" | "+text
    with (run/"logs/numerical.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
    print(line,flush=True)
def adjust(frame):
    a=frame.astype(np.float64);low0,high0=float(a.min()),float(a.max())
    if high0==low0:z=np.clip(a,0,1)
    else:
        delta=1/(high0-low0);z=np.clip(delta*a-low0*delta,0,1)
    bins=np.floor(z*65535+.5).astype(np.int64)
    hist=np.bincount(bins.ravel(),minlength=65536);cdf=np.cumsum(hist,dtype=np.float64)/z.size
    li=int(np.searchsorted(cdf,.01,side="right"));hi=int(np.searchsorted(cdf,.99,side="left"))
    fallback=li==hi;lo,up=(0.,1.) if fallback else (li/65535,hi/65535)
    transformed=np.floor(np.clip((z-lo)/(up-lo),0,1)*65535+.5).astype(np.uint16)
    return transformed,dict(frame_min_ADU=low0,frame_max_ADU=high0,stretch_low_bin=li,stretch_high_bin=hi,
        stretch_low_normalized=lo,stretch_high_normalized=up,flat_limit_fallback=fallback,
        effective_low_ADU=low0+lo*(high0-low0),effective_high_ADU=low0+up*(high0-low0),
        original_zero_pixels=int((frame==0).sum()),new_zero_pixels_nonzero_input=int(((transformed==0)&(frame!=0)).sum()),
        output_zero_pixels=int((transformed==0).sum()),output_saturated_pixels=int((transformed==65535).sum()))
# Bounded semantic checks use artificial integer frames; no MATLAB or cross-system gate.
assert adjust(np.zeros((3,4),np.uint16))[0].max()==0
assert adjust(np.full((3,4),400,np.uint16))[0].min()==65535
ramp=np.arange(65536,dtype=np.uint16).reshape(256,256);ramped,info=adjust(ramp)
assert info["stretch_low_bin"]==655 and info["stretch_high_bin"]==64880
assert ramped[0,0]==0 and ramped[-1,-1]==65535
source=Path(cfg["source_output"]);c1=Path(cfg["source_C1"])
roi=np.load(source/"roi/roi_mask.npy",allow_pickle=False);vertices=np.load(source/"roi/vertices.npy",allow_pickle=False)
job=json.loads((source/"job.json").read_text())
if roi.dtype!=bool or roi.shape!=(684,428) or int(roi.sum())!=job["ROI_pixels"]:raise ValueError("ROI differs")
protected=[c1,Path(cfg["source_ROI"]),source/"job.json",source/"COMPLETED.json",source/"classification_summary.json",
    source/"masks/clean_full_mask.npy",source/"masks/analysis_ROI_mask.npy",source/"masks/foreground_count.npy",
    source/"roi/roi_mask.npy",source/"roi/vertices.npy"]
before=[descriptor(p) for p in protected]
binding=sha(cfg_path)
records=dict(config_sha256=binding,protected_sources=before,source_C1=cfg["source_C1"])
dump(run/"source_records.json",records)
np.save(run/"masks/roi_mask.npy",roi);np.save(run/"masks/roi_vertices_xy.npy",vertices)
native_env=dict(os.environ);native_env.pop("PYTHONPYCACHEPREFIX",None)
cleanup_site=Path(cfg["runtime"]["cleanup_python"]).parent.parent/"lib/python3.11/site-packages"
query='import json,importlib.metadata as m;print(json.dumps({k:m.version(k) for k in ["nellie","numpy","scipy","tifffile"]}))'
native_versions=json.loads(subprocess.check_output([cfg["runtime"]["nellie_python"],"-B","-c",query],text=True,env=native_env,cwd=run))
if native_versions["nellie"]!="1.0.4":raise ValueError("Existing Nellie method version differs")
matlab_files=[Path("/Applications/MATLAB_R2024a.app/toolbox/images/images")/n for n in ("mat2gray.m","stretchlim.m","imhist.m","imadjust.m","im2uint16.m")]
dump(run/"environment.json",dict(python=sys.version,platform=platform.platform(),
    analysis_versions={k:importlib.metadata.version(k) for k in ("numpy","scipy","tifffile")},
    native_versions=native_versions,matlab_source_reference=[descriptor(p) for p in matlab_files if p.is_file()],
    implementation="MATLAB R2024a source-matched histogram semantics; not bitwise MATLAB-verified",
    native_bootstrap="Reused existing TIFF-only dependency bootstrap; no packages installed or numerical algorithms substituted",
    cache_environment={k:os.environ[k] for k in ("TMPDIR","MPLCONFIGDIR","NUMBA_CACHE_DIR","XDG_CACHE_HOME")},
    source_files=[descriptor(p) for p in sorted((run/"source").glob("*.py"))]))
old_full=np.load(source/"masks/clean_full_mask.npy",mmap_mode="r",allow_pickle=False)
old_roi=np.load(source/"masks/analysis_ROI_mask.npy",mmap_mode="r",allow_pickle=False)
finals=[];counts_all=[];old_final=[];zero_parts=[];mapping=[];limits_rows=[];comparison=[];threshold_rows=[];sample_checks=[];summaries=[]
started=time.perf_counter()
log("Starting isolated C1 test: per-frame full-field MATLAB-style 1%/99% stretch, 150 frames, existing raw baseline masks.")
with tifffile.TiffFile(c1) as movie:
    if len(movie.pages)!=2500 or movie.pages[0].shape!=(684,428):raise ValueError("C1 grid/time differs")
    for local_block,start in enumerate(cfg["block_starts"]):
        original_ids=np.arange(start,start+50,dtype=np.int32);block=start//50
        raw=np.stack([movie.pages[int(f)].asarray() for f in original_ids])
        if raw.dtype!=np.uint16:raise ValueError("Expected original uint16")
        transformed=[]
        for frame,value in zip(original_ids,raw):
            new,limits=adjust(value);transformed.append(new)
            limits_rows.append(dict(original_frame=int(frame),block_index=block,**limits))
        transformed=np.stack(transformed)
        for kind,values in (("original",raw),("adjusted",transformed)):
            name=run/"inputs"/f"C1_{kind}_frames{start:04d}_{start+49:04d}.ome.tif"
            if not name.exists():
                tifffile.imwrite(name,values,ome=True,photometric="minisblack",
                    metadata=dict(axes="TYX",PhysicalSizeX=.117,PhysicalSizeY=.117,PhysicalSizeXUnit="µm",PhysicalSizeYUnit="µm",TimeIncrement=.02,TimeIncrementUnit="s"))
        zeros=np.packbits((raw==0).reshape(50,-1),axis=1,bitorder="little");zero_parts.append(zeros)
        for j,frame in enumerate(original_ids):
            f=int(frame);mapping.append(dict(page_index=len(mapping),original_frame=f,time_s=f*.02,
                mask_block=block,local_block_index=local_block,original_frame_start=start,original_frame_stop_exclusive=start+50))
            if f in cfg["sample_frames"]:
                with tifffile.TiffFile(run/"inputs"/f"C1_adjusted_frames{start:04d}_{start+49:04d}.ome.tif") as saved:
                    if not np.array_equal(saved.pages[j].asarray(),transformed[j]):raise ValueError("Transformed input readback mismatch")
                expected,_=adjust(movie.pages[f].asarray())
                if not np.array_equal(expected,transformed[j]):raise ValueError("Sample transform mismatch")
                sample_checks.append(dict(original_frame=f,formula_and_mapping_match=True,
                    original_pixels_sha256=hashlib.sha256(raw[j].tobytes()).hexdigest(),
                    adjusted_pixels_sha256=hashlib.sha256(transformed[j].tobytes()).hexdigest(),
                    original_zero_pixels=int((raw[j]==0).sum())))
        masks=[]
        for offset in (0,25):
            f0=start+offset;tag=f"frames{f0:04d}_{f0+24:04d}";permanent=run/"masks/native_batches"/tag
            permanent.mkdir(parents=True,exist_ok=True);packed_path=permanent/"broad_packed.npz";marker=permanent/"COMPLETED.json"
            if marker.exists() and packed_path.exists():
                saved=json.loads(marker.read_text())
                if saved["config_sha256"]!=binding:raise ValueError("Native cache config mismatch")
                with np.load(packed_path,allow_pickle=False) as z:
                    broad=np.unpackbits(z["packed"],axis=1,count=684*428,bitorder="little").reshape(25,684,428).astype(bool)
                log(f"Reused native chunk {f0}-{f0+24}")
            else:
                scratch=run/"_work"/tag
                if scratch.exists():shutil.rmtree(scratch)
                scratch.mkdir()
                tifffile.imwrite(scratch/"input.ome.tif",transformed[offset:offset+25],ome=True,photometric="minisblack",
                    metadata=dict(axes="TYX",PhysicalSizeX=.117,PhysicalSizeY=.117,PhysicalSizeXUnit="µm",PhysicalSizeYUnit="µm",TimeIncrement=.02,TimeIncrementUnit="s"))
                t0=time.perf_counter();log(f"Nellie broad starting: original frames {f0}-{f0+24}")
                command=[cfg["runtime"]["nellie_python"],"-B","-u",str(run/"source/native_bootstrap.py"),str(cleanup_site),
                    str(run/"source/nellie_worker.py"),str(run/"_work"),str(scratch),str(permanent/"native_dependency_runtime.json")]
                with (permanent/"nellie.log").open("w") as logfile:
                    subprocess.run(command,cwd=run,env=native_env,stdout=logfile,stderr=subprocess.STDOUT,check=True)
                broad=np.load(scratch/"broad_mask.npy",allow_pickle=False)
                if broad.shape!=(25,684,428) or broad.dtype!=bool:raise ValueError("Native broad dimensions differ")
                parameters=json.loads((scratch/"actual_parameters.json").read_text())
                if parameters["native_min_area"]!=4:raise ValueError("Native broad area differs from default")
                np.savez_compressed(packed_path,packed=np.packbits(broad.reshape(25,-1),axis=1,bitorder="little"),
                    shape=np.array(broad.shape),original_frame_ids=np.arange(f0,f0+25,dtype=np.int32),bitorder=np.array("little"))
                for name in ("thresholds.json","actual_parameters.json"):shutil.copy2(scratch/name,permanent/name)
                dump(marker,dict(config_sha256=binding,original_frame_start=f0,original_frame_stop_exclusive=f0+25,
                    elapsed_s=time.perf_counter()-t0,completed_utc=datetime.now(timezone.utc).isoformat()))
                shutil.rmtree(scratch);log(f"Nellie broad complete: {f0}-{f0+24}; {time.perf_counter()-t0:.1f} s")
            native_thresholds=json.loads((permanent/"thresholds.json").read_text())
            for i,row in enumerate(native_thresholds):threshold_rows.append(dict(row,original_frame=f0+i,block_index=block))
            masks.append(broad)
        broad=np.concatenate(masks);support=broad.sum(axis=0,dtype=np.uint8)
        np.savez_compressed(run/"masks/broad_packed"/f"b{block:03d}.npz",
            packed=np.packbits(broad.reshape(50,-1),axis=1,bitorder="little"),shape=np.array(broad.shape),
            original_frame_ids=original_ids,bitorder=np.array("little"))
        final=process_counts(support,support_minimum=10,minimum_foreground=32,hole_limit=64,frame_count=50)["final_mask"]
        old=np.asarray(old_full[block]);oldr=np.asarray(old_roi[block])
        if not np.array_equal(oldr,old&roi):raise ValueError("Original ROI mask intersection mismatch")
        finals.append(final);counts_all.append(support);old_final.append(old.copy())
        for scope,gate in (("ROI",roi),("full",np.ones_like(roi))):
            a=old&gate;b=final&gate;n0=int(a.sum());n1=int(b.sum());inter=int((a&b).sum());union=int((a|b).sum())
            added=int((b&~a).sum());removed=int((a&~b).sum())
            if n0+added-removed!=n1:raise ValueError("Mask difference conservation failed")
            comparison.append(dict(scope=scope,block_index=block,original_frame_start=start,original_frame_stop_exclusive=start+50,
                old_pixels=n0,new_pixels=n1,area_old_um2=n0*.117**2,area_new_um2=n1*.117**2,
                delta_pixels=n1-n0,delta_area_um2=(n1-n0)*.117**2,intersection_pixels=inter,union_pixels=union,
                added_pixels=added,removed_pixels=removed,iou=inter/union if union else 1.,dice=2*inter/(n0+n1) if n0+n1 else 1.))
        summary=dict(original_block=block,original_frames=[start,start+49],ROI_old_pixels=int(oldr.sum()),ROI_new_pixels=int((final&roi).sum()))
        summaries.append(summary);dump(run/"checkpoints"/f"block_{block:03d}.json",dict(config_sha256=binding,**summary))
        log(f"Support block {block} complete: ROI pixels {summary['ROI_old_pixels']} -> {summary['ROI_new_pixels']}")
np.save(run/"masks/foreground_count.npy",np.stack(counts_all))
np.save(run/"masks/clean_full_mask.npy",np.stack(finals))
np.save(run/"masks/analysis_ROI_mask.npy",np.stack(finals)&roi)
np.save(run/"masks/baseline_clean_full_mask.npy",np.stack(old_final))
np.savez_compressed(run/"masks/original_zero_regions_packed.npz",packed=np.concatenate(zero_parts),shape=np.array([150,684,428]),
    original_frame_ids=np.array([r["original_frame"] for r in mapping],np.int32),bitorder=np.array("little"))
rows(run/"data/mask_comparison.csv",comparison);rows(run/"data/frame_mapping.csv",mapping)
rows(run/"data/preprocessing_limits.csv",limits_rows);rows(run/"data/native_thresholds.csv",threshold_rows)
if before!=[descriptor(p) for p in protected]:raise ValueError("Original source files changed")
records.update(protected_sources_unchanged=True,completed_utc=datetime.now(timezone.utc).isoformat())
dump(run/"source_records.json",records)
dump(run/"validation.json",dict(sampled_frames=sample_checks,n_samples=len(sample_checks),full_blocks=3,frames=150,
    no_input_transformations_outside_C1=True,original_sources_unchanged=True,ROI_intersection_matches=True,
    preprocessing_synthetic_checks=3,original_zero_validity_separate_from_contrast_created_zeros=True,
    classification_calls=0,localization_calls=0,TARDIS_calls=0))
dump(run/"masks/metadata.json",dict(axes="TYX",shape=[3,684,428],original_blocks=[0,24,49],
    original_frame_to_block="Original f maps to full-movie f//50; local test index is explicit in frame_mapping.csv",
    pixel_size_nm=117,recipe=cfg["recipe"],original_zero_regions="original_zero_regions_packed.npz",
    preprocessing=cfg["preprocessing"],registration_status=cfg["registration_status"]))
dump(run/"NUMERICAL_COMPLETE.json",dict(complete=True,frames=150,blocks=summaries,elapsed_s=time.perf_counter()-started))
log(f"Numerical complete: three blocks / 150 frames; {time.perf_counter()-started:.1f} s.")
