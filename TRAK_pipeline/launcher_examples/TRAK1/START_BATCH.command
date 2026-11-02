#!/bin/zsh
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
batch_log="$batch_control/logs/modular_start_$(/bin/date -u +%Y%m%dT%H%M%SZ)_$$.log"
print 'Common protein pipeline | TRAK1 | start'
print 'Press Control-C to stop and preserve owned checkpoints.'
/usr/bin/caffeinate -i -m -s "$batch_python" -B -s -u "$batch_shared/tools/launch.py" run --config "$batch_shared/configs/TRAK1.json" "$@" 2>&1 | /usr/bin/tee "$batch_log"
batch_exit=${pipestatus[1]}
print "Pipeline exit code: $batch_exit"
print "Log: $batch_log"
if [[ -t 0 ]]; then read '?Press Return to close.'; fi
exit "$batch_exit"
