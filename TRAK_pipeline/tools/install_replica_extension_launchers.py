#!/usr/bin/env python3
"""Install only new BR4/BR5 launchers; ordinary batch launchers are untouched."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import uuid

NAMES={'START_NEW_REPLICAS':'start','RESUME_NEW_REPLICAS':'resume','STOP_AFTER_CURRENT_CELL_NEW_REPLICAS':'stop','PLAN_NEW_REPLICAS':'plan','UPDATE_FIVE_REPLICA_SUMMARIES':'summarize','REFRESH_FIVE_REPLICA_FIGURES':'refresh'}

def write(path,text):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temp.write_bytes(text);os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def windows(action,root_alias):
    prefix='%~dp0_TRAK_pipeline\\' if root_alias else '%~dp0..\\..\\'
    manifest='%~dp0_TRAK_pipeline\\batches\\20261102_BR4_BR5\\batch_manifest.json' if root_alias else '%~dp0batch_manifest.json'
    text='@echo off\nsetlocal\nset "TRAK_EXTENSION_CONTROL='+prefix+'tools\\replica_extension_control.ps1"\nset "TRAK_EXTENSION_MANIFEST='+manifest+'"\n'
    text+='if not exist "%TRAK_EXTENSION_CONTROL%" (\n echo The complete My Book pipeline folder is unavailable.\n pause\n exit /b 2\n)\n'
    text+='powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TRAK_EXTENSION_CONTROL%" -Manifest "%TRAK_EXTENSION_MANIFEST%" -Action "'+action+'"\n'
    text+='set "TRAK_EXTENSION_EXIT=%ERRORLEVEL%"\necho.\necho Exit code: %TRAK_EXTENSION_EXIT%\npause\nexit /b %TRAK_EXTENSION_EXIT%\n'
    return text.replace('\n','\r\n').encode('ascii')


def mac(action,root_alias):
    common='${0:A:h}/_TRAK_pipeline' if root_alias else '${0:A:h:h:h}'
    text='''#!/bin/zsh
set -u
set -o pipefail
batch_common="@COMMON@"
batch_volume="${batch_common:h}"
batch_manifest="$batch_common/batches/20261102_BR4_BR5/batch_manifest.json"
batch_python_file="$batch_volume/Xinran_FullsetTRAK2/_batch_pipeline/SCIENCE_PYTHON.txt"
if [[ ! -f "$batch_python_file" || ! -f "$batch_manifest" ]]; then
 print -u2 'Connect My Book with the existing installed Mac environment and complete pipeline.'
 if [[ -t 0 ]]; then read '?Press Return to close.'; fi
 exit 2
fi
IFS= read -r batch_python < "$batch_python_file"
if [[ ! -x "$batch_python" ]]; then
 print -u2 "Existing Mac Python unavailable: $batch_python"
 if [[ -t 0 ]]; then read '?Press Return to close.'; fi
 exit 2
fi
batch_runner=$("$batch_python" -I -B -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);m=json.loads(p.read_text());assert m.get("ready_for_execution"),"Batch is not marked ready";c=(p.parent/m["common_root_relative"]).resolve();print(c/m["release_relative"]/"tools/run_replica_extension_batch.py")' "$batch_manifest") || exit 2
batch_runtime="${batch_manifest:h}/runtime"
mkdir -p "$batch_runtime/tmp" "$batch_runtime/matplotlib" "$batch_runtime/numba" "${batch_manifest:h}/logs"
export TMPDIR="$batch_runtime/tmp" TEMP="$batch_runtime/tmp" TMP="$batch_runtime/tmp"
export MPLCONFIGDIR="$batch_runtime/matplotlib" NUMBA_CACHE_DIR="$batch_runtime/numba"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONUTF8=1 PYTHONIOENCODING=utf-8
batch_log="${batch_manifest:h}/logs/combined_@ACTION@_$(/bin/date -u +%Y%m%dT%H%M%SZ)_$$.log"
print 'TRAK1 + TRAK2 | new BR4/BR5 batch | @ACTION@'
print 'Only156 new cells; all367 cells enter five-replica summaries.'
/usr/bin/caffeinate -i -m -s "$batch_python" -B -s -u "$batch_runner" --manifest "$batch_manifest" --action @ACTION@ 2>&1 | /usr/bin/tee "$batch_log"
batch_exit=${pipestatus[1]}
print "Exit code: $batch_exit"
print "Log: $batch_log"
if [[ -t 0 ]]; then read '?Press Return to close.'; fi
exit "$batch_exit"
'''
    return text.replace('@COMMON@',common).replace('@ACTION@',action).encode('utf-8')


def install(manifest,apply=False):
    path=Path(manifest).resolve();m=json.loads(path.read_text());batch=path.parent;common=(batch/m['common_root_relative']).resolve()
    if not batch.is_relative_to(common/'batches'):raise ValueError('Manifest is outside common batches directory')
    names=[]
    for folder,alias in ((batch,False),(common.parent,True)):
        for base,action in NAMES.items():
            for suffix,data in (('_WINDOWS.cmd',windows(action,alias)),('_MAC.command',mac(action,alias))):
                target=folder/(base+suffix);names.append(str(target))
                if apply:
                    write(target,data)
                    if suffix.endswith('.command'):target.chmod(0o755)
    result=dict(applied=apply,files=names,ordinary_launchers_modified=False,new_cells_started=0)
    if apply:write(batch/'launcher_installation.json',(json.dumps(result,indent=2)+'\n').encode())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--apply',action='store_true');a=p.parse_args();print(json.dumps(install(a.manifest,a.apply),indent=2))
if __name__=='__main__':main()
