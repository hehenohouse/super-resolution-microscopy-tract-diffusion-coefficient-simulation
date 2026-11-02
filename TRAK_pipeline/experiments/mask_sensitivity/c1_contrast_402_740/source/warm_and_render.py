#!/usr/bin/env python3
"""Warm the existing viewing environment, then await this isolated run's masks."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.metadata,json,os,sys,time,traceback
RUN=Path(__file__).resolve().parents[1]
RUNTIME=RUN/'runtime/rendering';RUNTIME.mkdir(parents=True,exist_ok=True)
for key,folder in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg'),('XDG_CONFIG_HOME','xdg_config'),('XDG_DATA_HOME','xdg_data')]:
 p=RUNTIME/folder;p.mkdir(exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',MPLBACKEND='Agg')
sys.dont_write_bytecode=True
(RUN/'logs').mkdir(exist_ok=True)
class Tee:
 def __init__(self,original,file):self.original=original;self.file=file
 def write(self,value):self.original.write(value);self.file.write(value);self.file.flush();return len(value)
 def flush(self):self.original.flush();self.file.flush()
def utc():return datetime.now(timezone.utc).isoformat()
with (RUN/'logs/rendering.log').open('a',encoding='utf-8') as log:
 oldout,olderr=sys.stdout,sys.stderr;sys.stdout=Tee(oldout,log);sys.stderr=Tee(olderr,log)
 try:
  start=time.perf_counter();print(utc(),'RENDER_ENV_WARM_START',flush=True)
  import numpy as np
  import tifffile
  from PIL import Image,ImageDraw,ImageFont
  import matplotlib
  matplotlib.use('Agg',force=True)
  from matplotlib import pyplot as plt
  from render_contrast_test import render,font_path
  font=font_path();ImageFont.truetype(font,20).getbbox('C1 contrast test')
  versions=dict(warmed_utc=utc(),warm_seconds=time.perf_counter()-start,python=sys.version,python_executable=sys.executable,render_python=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','tifffile','Pillow','matplotlib']},font_path=font)
  (RUNTIME/'warmed_versions.json').write_text(json.dumps(versions,indent=2)+'\n',encoding='utf-8')
  (RUN/'render_runtime.json').write_text(json.dumps(versions,indent=2)+'\n',encoding='utf-8')
  (RUN/'command_render.json').write_text(json.dumps(dict(command=[sys.executable,'-B','-s','-u',str(RUN/'source/render_contrast_test.py'),'--config',str(RUN/'config.json')],orchestrator_command=[sys.executable,'-B','-s','-u',str(Path(__file__).resolve())],actual_runtime='existing science environment for pure viewing only; numerical config unchanged'),indent=2)+'\n',encoding='utf-8')
  print(utc(),'RENDER_ENV_WARM_COMPLETE',versions['warm_seconds'],flush=True)
  while not (RUN/'NUMERICAL_COMPLETE.json').is_file():
   print(utc(),'WAITING_FOR_NUMERICAL_COMPLETE',flush=True);time.sleep(15)
  print(utc(),'CONTRAST_RENDER_START',flush=True);render(RUN/'config.json');print(utc(),'CONTRAST_RENDER_RETURNED',flush=True)
 except BaseException:
  traceback.print_exc();raise
 finally:sys.stdout,sys.stderr=oldout,olderr
