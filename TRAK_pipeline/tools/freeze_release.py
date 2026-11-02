#!/usr/bin/env python3
"""Freeze one common release. Source checksums audit publication, not science caches."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]


def _write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def _files(root):
    for name in ('src','tools','docs','tests'):
        folder=root/name
        if not folder.is_dir():continue
        for path in sorted(folder.rglob('*')):
            parts=path.relative_to(root).parts
            if any(x.startswith('._') or x in ('__pycache__','.DS_Store','.pytest_cache') for x in parts):continue
            if path.suffix in ('.pyc','.pyo'):continue
            if path.is_symlink():raise ValueError('Frozen source cannot reference a mutable symlink: '+str(path))
            if path.is_file():yield path,path.relative_to(root)
    for name in ('README.md','CHANGELOG.md','IMPLEMENTATION_INTERFACES.json'):
        path=root/name
        if path.is_file():yield path,Path(name)
    for path in sorted((root/'configs').glob('*.json')):
        if not path.name.startswith('._'):yield path,Path('config_snapshot')/path.name


def freeze(root,release_id,activate=False,dry_run=False):
    root=Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}',release_id):raise ValueError('Use a simple release ID without slashes or spaces')
    files=list(_files(root));dest=root/'releases'/release_id
    if dest.exists():raise FileExistsError('Frozen releases are immutable; choose a new release ID: '+str(dest))
    report=dict(release=str(dest),files=len(files),bytes=sum(p.stat().st_size for p,_ in files),activate=activate,dry_run=dry_run)
    if dry_run:return report
    if not any(rel==Path('src/trak_pipeline/cli.py') for _,rel in files):raise ValueError('Cannot freeze an incomplete package without its CLI')
    parent=root/'releases';parent.mkdir(parents=True,exist_ok=True)
    lock=parent/'.freeze.lock'
    try:lock.mkdir()
    except FileExistsError:raise BlockingIOError('Another freeze operation owns '+str(lock))
    staging=parent/('_building_'+release_id+'_'+uuid.uuid4().hex)
    try:
        staging.mkdir()
        records=[]
        for source,relative in files:
            target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            records.append(dict(relative_path=str(relative).replace(os.sep,'/'),bytes=target.stat().st_size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        manifest=dict(schema_version=1,release_id=release_id,created_utc=datetime.now(timezone.utc).isoformat(),
            scientific_cache_identity='Explicit per-method version, canonical scientific parameters and upstream scientific artifact IDs; source checksums are audit only',
            runtime_configuration='Mutable common configs outside the release; config_snapshot is provenance only',files=records)
        _write(staging/'release_manifest.json',manifest)
        if dest.exists():raise FileExistsError('A release with that ID appeared during freeze')
        os.replace(staging,dest)
        if activate:
            _write(root/'CURRENT_RELEASE.json',dict(schema_version=1,release='releases/'+release_id,
                manifest_sha256=hashlib.sha256((dest/'release_manifest.json').read_bytes()).hexdigest(),
                activated_utc=datetime.now(timezone.utc).isoformat()))
        return report
    finally:
        if staging.exists():shutil.rmtree(staging)
        lock.rmdir()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root',default=str(ROOT));parser.add_argument('--release-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_modular_v1'))
    parser.add_argument('--activate',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv);print(json.dumps(freeze(args.common_root,args.release_id,args.activate,args.dry_run),indent=2));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError) as error:print(str(error),file=sys.stderr);raise SystemExit(2)
