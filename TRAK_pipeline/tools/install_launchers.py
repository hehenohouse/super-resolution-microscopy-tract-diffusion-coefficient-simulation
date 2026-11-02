#!/usr/bin/env python3
"""Install thin dataset launchers only when --apply is explicit; back up originals."""
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

WIN_ACTIONS={
 'START_BATCH_WINDOWS.cmd':'start','RESUME_BATCH_WINDOWS.cmd':'resume',
 'STOP_AFTER_CURRENT_CELL_WINDOWS.cmd':'stop','PLAN_BATCH_WINDOWS.cmd':'plan',
 'REFRESH_FIGURES_WINDOWS.cmd':'refresh','REFRESH_SUMMARIES_WINDOWS.cmd':'refresh',
 'UPDATE_SUMMARIES_WINDOWS.cmd':'summarize',
 'UPDATE_CELL_METRICS_WINDOWS.cmd':'summarize','UPDATE_INTRA_SUMMARIES_WINDOWS.cmd':'summarize',
}
MAC_ACTIONS={
 'START_BATCH.command':'start','START_BATCH_MAC.command':'start',
 'RESUME_BATCH_MAC.command':'resume','STOP_AFTER_CURRENT_CELL_MAC.command':'stop',
 'PLAN_BATCH_MAC.command':'plan','REFRESH_FIGURES_MAC.command':'refresh',
 'REFRESH_SUMMARIES_MAC.command':'refresh','UPDATE_SUMMARIES_MAC.command':'summarize',
 'UPDATE_CELL_METRICS_MAC.command':'summarize','UPDATE_INTRA_SUMMARIES_MAC.command':'summarize',
}


def _atomic(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_bytes(data);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def windows_launcher(molecule,action):
    return ('@echo off\nsetlocal\n'
        'set "TRAK_SHARED_CONTROL=%~dp0..\\..\\_TRAK_pipeline\\tools\\shared_control.ps1"\n'
        'if not exist "%TRAK_SHARED_CONTROL%" (\n'
        '  echo Shared pipeline is unavailable. Connect the complete My Book directory structure.\n'
        '  pause\n  exit /b 2\n)\n'
        'powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TRAK_SHARED_CONTROL%" '
        '-Molecule "'+molecule+'" -Action "'+action+'" %*\n'
        'set "TRAK_PIPELINE_EXIT=%ERRORLEVEL%"\necho.\necho Exit code: %TRAK_PIPELINE_EXIT%\npause\n'
        'exit /b %TRAK_PIPELINE_EXIT%\n').replace('\n','\r\n').encode('utf-8')


def mac_launcher(molecule,action):
    command={'start':'run','resume':'run','stop':'stop','plan':'plan','refresh':'refresh','summarize':'summarize'}[action]
    flags=' --resume' if action=='resume' else ' --completed-only' if action in ('refresh','summarize') else ''
    text='''#!/bin/zsh
# Thin common-pipeline launcher; all controlled output remains on the data drive.
set -u
set -o pipefail
batch_control="${0:A:h}"
batch_volume="${batch_control:h:h}"
batch_shared="$batch_volume/_TRAK_pipeline"
if ! /sbin/mount | /usr/bin/grep -Fq " on $batch_volume ("; then
  print -u2 'The data volume is not mounted. No pipeline output was written.'
  if [[ -t 0 ]]; then read '?Press Return to close.'; fi
  exit 2
fi
batch_python_file="$batch_control/SCIENCE_PYTHON.txt"
if [[ ! -f "$batch_python_file" ]]; then
  batch_python_file="$batch_volume/Xinran_FullsetTRAK2/_batch_pipeline/SCIENCE_PYTHON.txt"
fi
if [[ ! -f "$batch_python_file" || ! -f "$batch_shared/tools/launch.py" ]]; then
  print -u2 'The existing Mac science environment or shared pipeline is unavailable.'
  if [[ -t 0 ]]; then read '?Press Return to close.'; fi
  exit 2
fi
IFS= read -r batch_python < "$batch_python_file"
if [[ ! -x "$batch_python" ]]; then
  print -u2 "The configured existing Python is unavailable: $batch_python"
  if [[ -t 0 ]]; then read '?Press Return to close.'; fi
  exit 2
fi
batch_runtime="$batch_control/modular/runtime"
mkdir -p "$batch_runtime/tmp" "$batch_runtime/matplotlib" "$batch_runtime/numba" "$batch_control/logs"
export TMPDIR="$batch_runtime/tmp" TEMP="$batch_runtime/tmp" TMP="$batch_runtime/tmp"
export MPLCONFIGDIR="$batch_runtime/matplotlib" NUMBA_CACHE_DIR="$batch_runtime/numba"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
batch_log="$batch_control/logs/modular_@ACTION@_$(/bin/date -u +%Y%m%dT%H%M%SZ)_$$.log"
print 'Common protein pipeline | @MOLECULE@ | @ACTION@'
print 'Press Control-C to stop and preserve owned checkpoints.'
/usr/bin/caffeinate -i -m -s "$batch_python" -B -s -u "$batch_shared/tools/launch.py" @COMMAND@ --config "$batch_shared/configs/@MOLECULE@.json"@FLAGS@ "$@" 2>&1 | /usr/bin/tee "$batch_log"
batch_exit=${pipestatus[1]}
print "Pipeline exit code: $batch_exit"
print "Log: $batch_log"
if [[ -t 0 ]]; then read '?Press Return to close.'; fi
exit "$batch_exit"
'''
    return text.replace('@MOLECULE@',molecule).replace('@ACTION@',action).replace('@COMMAND@',command).replace('@FLAGS@',flags).encode('utf-8')


def installation_plan(root,molecules):
    root=Path(root).resolve();plans=[]
    for molecule in molecules:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',molecule):raise ValueError('Invalid molecule configuration name')
        config_path=root/'configs'/(molecule+'.json');config=json.loads(config_path.read_text(encoding='utf-8'))
        data_root=(config_path.parent/config['data_root']).resolve()
        if data_root.parent!=root.parent:raise ValueError('Dataset and common package must remain sibling folders on the same data drive')
        control=data_root/config.get('control_relative','_batch_pipeline')
        if not control.is_dir():raise FileNotFoundError('Dataset control folder is unavailable: '+str(control))
        files={name:windows_launcher(molecule,action) for name,action in WIN_ACTIONS.items()}
        files.update({name:mac_launcher(molecule,action) for name,action in MAC_ACTIONS.items()})
        # Existing alternate refresh launchers are also made strictly presentation-only.
        for path in control.iterdir():
            name=path.name
            if name.startswith('._'):continue
            if name.upper().startswith('REFRESH_'):
                if name.endswith('.cmd'):files[name]=windows_launcher(molecule,'refresh')
                elif name.endswith('.command'):files[name]=mac_launcher(molecule,'refresh')
        files['README_MODULAR_PIPELINE.txt']=(
            'COMMON MODULAR PROTEIN PIPELINE\n\n'
            'START_BATCH: compute missing scientific stages and publish results.\n'
            'RESUME_BATCH: continue the same configured scientific stages.\n'
            'STOP_AFTER_CURRENT_CELL: request a clean stop after the current cell.\n'
            'PLAN_BATCH: show cache reuse/recompute/missing decisions without computing science.\n'
            'REFRESH_FIGURES / REFRESH_SUMMARIES: render saved scientific artifacts only; no hidden science.\n'
            'UPDATE_SUMMARIES: compute hierarchy statistics from completed cells, then publish.\n\n'
            'Keep this dataset beside _TRAK_pipeline on the external drive. Existing SETUP_WINDOWS.cmd is preserved.\n'
            'Existing scientific results and legacy releases are unchanged. See _TRAK_pipeline/README.md.\n'
        ).encode('utf-8')
        plans.append(dict(molecule=molecule,data_root=data_root,control=control,files=files))
    return plans


def install(root,molecules,apply=False):
    root=Path(root).resolve();plans=installation_plan(root,molecules)
    result=dict(applied=apply,datasets=[dict(molecule=p['molecule'],control=str(p['control']),files=list(p['files'])) for p in plans])
    if not apply:return result
    if not (root/'CURRENT_RELEASE.json').is_file():raise ValueError('Activate a tested frozen common release before installing launchers')
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    for plan in plans:
        control=plan['control'];backup=control/'modular/legacy_launchers'/timestamp;backup.mkdir(parents=True)
        records=[];written=[]
        for name,data in plan['files'].items():
            target=control/name;existed=target.is_file()
            if existed:shutil.copy2(target,backup/name)
            records.append(dict(name=name,existed_before=existed,old_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if existed else None,
                new_sha256=hashlib.sha256(data).hexdigest()))
        record=dict(schema_version=1,molecule=plan['molecule'],backed_up_utc=timestamp,backup_relative='modular/legacy_launchers/'+timestamp,common_root_relative='../../_TRAK_pipeline',active_release=json.loads((root/'CURRENT_RELEASE.json').read_text(encoding='utf-8')),files=records,
            unchanged=['SETUP_WINDOWS.cmd','VALIDATE_WINDOWS.cmd','windows software and environments','legacy scientific releases and outputs'])
        _atomic(backup/'backup_manifest.json',(json.dumps(record,indent=2)+'\n').encode('utf-8'))
        try:
            for name,data in plan['files'].items():
                _atomic(control/name,data);written.append(name)
                if name.endswith('.command'):(control/name).chmod(0o755)
            _atomic(control/'modular/launcher_installation.json',(json.dumps(record,indent=2)+'\n').encode('utf-8'))
        except BaseException:
            for name in reversed(written):
                original=backup/name
                if original.is_file():_atomic(control/name,original.read_bytes())
                else:(control/name).unlink(missing_ok=True)
            raise
        result['datasets'][len([p for p in result['datasets'] if 'backup' in p])]['backup']=str(backup)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root',default=str(ROOT));parser.add_argument('--molecules',nargs='+',default=['TRAK1','TRAK2'])
    group=parser.add_mutually_exclusive_group();group.add_argument('--apply',action='store_true');group.add_argument('--dry-run',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(install(args.common_root,args.molecules,args.apply),indent=2));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError) as error:print(str(error),file=sys.stderr);raise SystemExit(2)
