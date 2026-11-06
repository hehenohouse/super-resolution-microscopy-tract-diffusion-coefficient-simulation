#!/usr/bin/env python3
"""Refresh C1 displays only, preserving every saved scientific artifact.

Uses a new style-keyed presentation cache/gallery.  It never schedules stages,
computes masks, reclassifies points, averages images or calculates TARDIS.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid


def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(tmp,p)
    finally: tmp.unlink(missing_ok=True)


def setup_environment(common, extra_site=None, source_only=False):
    root=common/'validation/C1_display_update_20261104/runtime'
    for name,rel in [('TMPDIR','tmp'),('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('NUMBA_CACHE_DIR','numba')]:
        p=root/rel;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    tempfile.tempdir=str(root/'tmp')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONNOUSERSITE='1')
    sys.dont_write_bytecode=True
    # Preserve the working existing system NumPy before adding optional packages.
    import numpy
    if extra_site:
        site=Path(extra_site).resolve()
        if not site.is_dir():raise FileNotFoundError('Existing package directory missing: '+str(site))
        sys.path.append(str(site))
    if source_only:
        original=importlib.machinery.SourceFileLoader.get_code
        def source_code(loader,fullname):
            if 'site-packages' in loader.path:
                return compile(Path(loader.path).read_bytes(),loader.path,'exec',dont_inherit=True)
            return original(loader,fullname)
        importlib.machinery.SourceFileLoader.get_code=source_code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--cell',action='append',required=True)
    parser.add_argument('--existing-site-packages',type=Path)
    parser.add_argument('--source-only-imports',action='store_true')
    args=parser.parse_args(argv)
    common=Path(__file__).resolve().parents[1]
    drive=common.parent
    if not os.path.ismount(str(drive)):raise OSError('External data drive is not mounted')
    release=args.release.resolve()
    if not release.is_relative_to(common/'releases') or not (release/'release_manifest.json').is_file():
        raise ValueError('Use an existing frozen release on this same drive')
    setup_environment(common,args.existing_site_packages,args.source_only_imports)
    sys.path.insert(0,str(release/'src'))
    from trak_pipeline.config import load_config,discover_samples,portable_input
    from trak_pipeline.cache import CacheStore,resolve_portable
    from trak_pipeline.graph import default_specs,stage_parameters,expected_identity
    from trak_pipeline.identity import digest
    from trak_pipeline.types import Artifact,StageResult
    from trak_pipeline import runtime
    def forbidden(*a,**kw):raise AssertionError('C1 display refresh may not launch a stage or subprocess')
    runtime.invoke=forbidden
    config=load_config(args.config)
    family=digest({s.stage:dict(method_id=s.method_id,method_version=s.method_version,parameters=stage_parameters(config,s))for s in default_specs(config)if s.scientific})
    samples={s.sample_id:s for s in discover_samples(config)}
    audit=common/'validation/C1_display_update_20261104';audit.mkdir(parents=True,exist_ok=True)
    source=common/'tools/c1_display_renderer.py'
    module_spec=importlib.util.spec_from_file_location('_c1_display_renderer',source)
    renderer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(renderer)
    overall=dict(started_utc=utc(),release=str(release.relative_to(common)),config=str(args.config.resolve().relative_to(common)),
        requested_C1_display_policy={k:config['style'].get(k)for k in ('C1_display_mode','C1_display_ADU','C1_display_percentiles','C1_display_sample_frames')},
        C2_display_ADU=config['style']['C2_display_ADU'],
        scientific_stage_calls=0,new_cells_started=0,cells=[])
    expected_specs=default_specs(config)
    for sid in args.cell:
        if sid not in samples:raise ValueError('Unknown cell: '+sid)
        sample=samples[sid];store=CacheStore(config,sample)
        state=Path(config['control_root'])/'modular/state/cells'/sid/(family+'.json')
        report=read(state)
        if not report.get('science_complete') or not report.get('publication_complete'):
            raise ValueError('Only fully completed cells may enter this display refresh: '+sid)
        if report['method_family']!=family:raise ValueError('Scientific method family changed')
        old=report['artifacts'];available={};markers=[];described_paths=set()
        def rebase(text):
            text=re.sub('/+','/',str(text).replace('\\','/'))
            return str(portable_input(text,config))
        for name in ('C1','C2','roi'):
            payload=dict(old[name]);payload['files']={k:rebase(v)for k,v in payload['files'].items()}
            available[name]=Artifact(**payload)
            described_paths.update(payload['files'].values())
            if any(not Path(v).is_file()for v in payload['files'].values()):
                raise FileNotFoundError('Saved source/ROI geometry missing: '+name)
        if not all(k in available['roi'].files for k in ('mask','vertices')):
            raise FileNotFoundError('ROI raster must already be materialized; refresh will not compute it')
        for spec in expected_specs:
            if not spec.scientific:continue
            if spec.stage not in old:raise FileNotFoundError('Missing saved scientific artifact: '+spec.stage)
            marker=store.root/spec.stage/old[spec.stage]['id']/'artifact.json'
            saved=read(marker);markers.append(marker)
            if not saved.get('complete'):raise ValueError('Incomplete science commit: '+str(marker))
            art=dict(saved['artifact']);art['files']={k:str(resolve_portable(v['location'],config['data_root'],release))for k,v in saved['descriptors'].items()}
            inputs={d:available[d]for d in spec.dependencies}
            expected=expected_identity(spec,stage_parameters(config,spec),inputs)
            if art['id']!=expected or art['id']!=old[spec.stage]['id']:
                raise ValueError('Scientific cache identity differs: '+spec.stage)
            for descriptor in saved['descriptors'].values():
                p=resolve_portable(descriptor['location'],config['data_root'],release)
                if descriptor['type']=='file':
                    if not p.is_file()or p.stat().st_size!=descriptor['fingerprint']['bytes']:
                        raise FileNotFoundError('Saved science file missing or size differs: '+str(p))
                    described_paths.add(str(p))
                else:
                    if not p.is_dir():raise FileNotFoundError(str(p))
                    for rel,fp in descriptor['entries'].items():
                        child=p/rel
                        if not child.is_file()or child.stat().st_size!=fp['bytes']:
                            raise FileNotFoundError('Saved science directory entry missing/size differs: '+str(child))
                        described_paths.add(str(child))
            available[spec.stage]=Artifact(**art)
        spec=next(s for s in expected_specs if s.stage=='presentation')
        params=stage_parameters(config,spec);inputs={d:available[d]for d in spec.dependencies}
        identity=store.key(spec,params,inputs);output=sample.directory/('Analysis_Figures_modular_'+identity[:12])
        old_id=old['presentation']['id'];old_marker=store.root/'presentation'/old_id/'artifact.json'
        old_commit=read(old_marker)
        old_gallery=resolve_portable(old_commit['descriptors']['directory']['location'],config['data_root'],release)
        if old_gallery==output and old_id==identity:
            current=store.lookup(spec,params,inputs)
            if current is None:raise ValueError('Current display commit failed its saved descriptor check')
            saved_publication=read(output/'publication.json')
            overall['cells'].append(dict(sample_id=sid,status='reused',figures=str(output),presentation_id=identity,
                C1_display_ADU=saved_publication.get('C1_display_ADU'),
                C1_display_policy=saved_publication.get('C1_display_policy'),scientific_stage_calls=0));continue
        before_markers={str(p):sha(p)for p in markers}
        def stats_snapshot(paths):
            return {str(p):(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)for p in paths}
        scientific_snapshot=stats_snapshot(described_paths)
        old_gallery_snapshot=stats_snapshot(p for p in old_gallery.rglob('*')if p.is_file()and not p.name.startswith('._'))
        backup=audit/(sid+'_completion_before.json')
        if not backup.exists():backup.write_bytes(state.read_bytes())
        existing=store.lookup(spec,params,inputs)
        if existing is None:
            output.mkdir(parents=True,exist_ok=True)
            for p in old_gallery.iterdir():
                if p.is_file()and not p.name.startswith('._'):shutil.copy2(p,output/p.name)
            settings=dict(sample_id=sid,C1=str(sample.C1_file),C2=str(sample.C2_file),stats_dir=available['statistics'].files['directory'],
                c1_mean_path=available['c1_mean'].files['mean'],full_mask_path=available['mask_cleanup'].files['full_mask'],
                output_dir=str(output),style=config['style'],pixel_size_nm=config['acquisition']['pixel_size_nm'],
                frame_interval_s=config['acquisition']['frame_interval_s'],frames=config['acquisition']['frames'],
                sample_frames=config['sample_frames'])
            print(sid+' | render 11 C1 images from saved science; no upstream calls',flush=True)
            details=renderer.render_cell(settings)
            if details.get('scientific_stage_calls')!=0 or len(details.get('rendered',[]))!=11:
                raise AssertionError('Unexpected display renderer scope')
            actual=sorted(p.name for p in output.glob('*.png')if not p.name.startswith('._'))
            expected_pngs=sorted(p.name for p in old_gallery.glob('*.png')if not p.name.startswith('._'))
            if actual!=expected_pngs:raise ValueError('New display gallery must retain the full existing PNG roster')
            publication=dict(figures=str(output),scientific_stage_calls=0,new_cells_started=0,missing=[],images=actual,
                adopted_readonly=False,C1_display_ADU=details['C1_display_ADU'],C2_display_ADU=details['C2_display_ADU'],
                C1_display_policy=details['C1_display_policy'],requested_C1_display_policy=overall['requested_C1_display_policy'],
                display_refresh=dict(previous_presentation_id=old_id,rendered=details['rendered'],reused_images=sorted(set(actual)-set(details['rendered'])),
                    renderer_sha256=sha(source),driver_sha256=sha(__file__),science_modified=False),renderer_report=details)
            atomic(output/'publication.json',publication)
            items=''.join('<li><a href="'+html.escape(name)+'">'+html.escape(name)+'</a><br><img src="'+html.escape(name)+'" width="700"></li>'for name in actual)
            note=f"C1 display {details['C1_display_ADU']} ADU ({details['C1_display_policy']['mode']}); C2 display {details['C2_display_ADU']} ADU. One C1 range per cell, shared by all representative frames and the saved mean overlay. Display only: raw pixels, masks, localizations, classification and TARDIS unchanged. Scientific stage calls: 0."
            (output/'INDEX.html').write_text('<html><body><h1>'+html.escape(sid)+'</h1><p>'+html.escape(note)+'</p><ul>'+items+'</ul></body></html>',encoding='utf-8')
            result=StageResult(dict(directory=output,figures=output,metadata=output/'publication.json'),publication)
            current=store.commit(spec,params,inputs,result)
        else:
            current=existing;publication=read(output/'publication.json');details=publication.get('renderer_report',{})
        if before_markers!={str(p):sha(p)for p in markers}:raise AssertionError('A scientific cache marker changed')
        if scientific_snapshot!=stats_snapshot(described_paths):raise AssertionError('A saved scientific product/source changed')
        if old_gallery_snapshot!=stats_snapshot(old_gallery_snapshot):raise AssertionError('The original gallery changed')
        new_report=dict(report);new_report['artifacts']=dict(report['artifacts']);new_report['artifacts']['presentation']=asdict(current)
        new_report['display_refresh']=dict(updated_utc=utc(),previous_presentation_id=old_id,presentation_id=identity,
            C1_display_ADU=publication['C1_display_ADU'],C1_display_policy=publication.get('C1_display_policy'),
            requested_C1_display_policy=overall['requested_C1_display_policy'],
            scientific_stage_calls=0,renderer_sha256=sha(source),raw_scientific_artifacts_unchanged=True)
        atomic(state,new_report)
        row=dict(sample_id=sid,status='published',figures=str(output),previous_presentation_id=old_id,presentation_id=identity,
            png_count=sum(not p.name.startswith('._')for p in output.glob('*.png')),rendered_images=details.get('rendered',[]),scientific_stage_calls=0,
            C1_display_ADU=publication['C1_display_ADU'],C1_display_policy=publication.get('C1_display_policy'),
            scientific_artifact_ids_unchanged=True,scientific_file_stats_unchanged=True,old_gallery_unchanged=True)
        overall['cells'].append(row);atomic(audit/(sid+'_display_refresh.json'),row)
        print(sid+' | saved C1 display gallery; scientific calls 0',flush=True)
    if any(name.startswith('trak_pipeline.stages.')for name in sys.modules):
        raise AssertionError('Display refresh unexpectedly imported a scientific-stage implementation')
    overall['ended_utc']=utc();overall['complete']=True
    atomic(audit/'refresh_result.json',overall)
    print(json.dumps(overall,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
