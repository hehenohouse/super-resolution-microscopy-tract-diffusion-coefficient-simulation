#!/usr/bin/env python3
"""Stable, dependency-free loader for the active common frozen release."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def active_release(root=ROOT):
    root=Path(root).resolve();pointer=root/'CURRENT_RELEASE.json'
    if not pointer.is_file():return root,False,None
    value=json.loads(pointer.read_text(encoding='utf-8'));relative=Path(value['release'])
    if relative.is_absolute():raise ValueError('Active release must be relative to the common package')
    release=(root/relative).resolve()
    if not release.is_relative_to(root/'releases'):raise ValueError('Active release escapes the common releases directory')
    manifest=release/'release_manifest.json'
    if not manifest.is_file():raise FileNotFoundError('Active release manifest is missing: '+str(manifest))
    if value.get('manifest_sha256') and hashlib.sha256(manifest.read_bytes()).hexdigest()!=value['manifest_sha256']:
        raise ValueError('Active release manifest checksum differs from the published pointer')
    return release,True,json.loads(manifest.read_text(encoding='utf-8'))


def verify_release(release,manifest):
    """Small frozen code/config audit, never a scientific-cache identity."""
    release=Path(release).resolve()
    for record in manifest.get('files',[]):
        path=(release/record['relative_path']).resolve()
        if not path.is_relative_to(release):raise ValueError('Invalid release audit path')
        if not path.is_file() or path.stat().st_size!=record['bytes']:
            raise ValueError('Frozen release file missing or changed: '+record['relative_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Frozen release checksum mismatch: '+record['relative_path'])


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    release,frozen,manifest=active_release()
    if frozen:verify_release(release,manifest)
    if argv==['--check-release'] or argv==['--print-release']:
        print(json.dumps(dict(package_root=str(ROOT),release=str(release),frozen=frozen,
            source_audit_passed=bool(frozen),scientific_steps_started=0),indent=2));return 0
    source=release/'src'
    if not (source/'trak_pipeline/cli.py').is_file():raise FileNotFoundError('Common CLI is missing: '+str(source/'trak_pipeline/cli.py'))
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    os.environ['TRAK_COMMON_ROOT']=str(ROOT)
    os.environ['TRAK_ACTIVE_RELEASE']=str(release)
    old=os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH']=str(source)+(os.pathsep+old if old else '')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(source))
    if not frozen:print('DEVELOPMENT SOURCE: no frozen release is active.',flush=True)
    from trak_pipeline.cli import main as cli_main
    result=cli_main(argv)
    return int(result or 0)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,ImportError) as error:
        print('Shared pipeline launch failed: '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(2)
