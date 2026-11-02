"""Context-bound bridge for frozen helper functions. No scientific cache policy."""
from pathlib import Path
import hashlib,json,os,sys
from trak_pipeline.io import atomic_json,replace,fingerprint,close_memmaps,safe,utc
from trak_pipeline.runtime import configure
from trak_pipeline.compat import platform_runtime as platform_api
BatchInfrastructureError=platform_api.BatchInfrastructureError
RELEASE=Path(__file__).resolve().parent
SAMPLE_FRAMES=[0,49,200,625,1200,1250,1875,2000,2450,2499]
def _root():
    value=os.environ.get('TRAK_DATA_ROOT')
    if not value:raise BatchInfrastructureError('An explicit dataset context is required')
    return Path(value)
def guard_path(path):return platform_api.guard_path(path,_root())
def atomic_replace(temp,target):guard_path(temp);guard_path(target);replace(temp,target)
def dump(path,value):atomic_json(path,value,_root())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def descriptor(path):
    p=Path(path);f=fingerprint(p);d=dict(path=str(p),bytes=f['bytes'],mtime_ns=p.stat().st_mtime_ns)
    d['sha256'if f['fingerprint_kind']=='full_small_file'else'sampled_sha256']=f['content_digest']
    return d
def verify_descriptor(d):
    try:
        current=descriptor(d['path'])
        keys=[k for k in['sha256','sampled_sha256']if k in d]
        return bool(keys)and current['bytes']==d['bytes']and all(current.get(k)==d[k]for k in keys)
    except(OSError,ValueError,KeyError):return False
def configure_environment(runtime_dir):
    c=dict(data_root=str(_root()),control_root=os.environ.get('TRAK_CONTROL_ROOT',str(_root()/'_batch_pipeline')),runtime={})
    configure(c);return dict(os.environ)
def vendor_paths(*names):
    for name in names:
        p=str(RELEASE/'vendor'/name)
        if p not in sys.path:sys.path.insert(0,p)
def resolve_runtime_paths(value=None):
    if value is None:raise BatchInfrastructureError('Explicit runtime configuration required')
    return dict(value)
def load_job(path):
    job=json.loads(Path(path).read_text(encoding='utf-8'));job['_job_path']=str(Path(path))
    os.environ['TRAK_DATA_ROOT']=job['data_root'];os.environ['TRAK_CONTROL_ROOT']=job['control_root']
    configure_environment(job['runtime_dir']);return job
def stage_done(job,stage):return False
def commit_stage(job,stage,files=(),**extra):
    raise RuntimeError('Frozen helpers cannot commit scientific stages; use the modular artifact store')
def log(job,stage,**values):print(json.dumps(dict(sample_id=job['sample_id'],stage=stage,**safe(values))),flush=True)
