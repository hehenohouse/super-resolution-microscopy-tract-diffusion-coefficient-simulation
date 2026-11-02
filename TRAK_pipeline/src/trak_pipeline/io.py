"""Drive-owned I/O. Source integrity is separate from scientific cache identity."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,math,os,tempfile,time

def utc():return datetime.now(timezone.utc).isoformat()
def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v)for v in value]
    if isinstance(value,Path):return str(value)
    if hasattr(value,'tolist'):return safe(value.tolist())
    if hasattr(value,'item'):return safe(value.item())
    if isinstance(value,float)and not math.isfinite(value):return None
    return value

def guard(path,root):
    from .compat.platform_runtime import guard_path
    return guard_path(path,root)

def replace(temp,target):
    for attempt in range(7):
        try:os.replace(temp,target);return
        except OSError as e:
            if os.name!='nt' or (getattr(e,'winerror',None)not in(5,32,33)and not isinstance(e,PermissionError))or attempt==6:raise
            time.sleep(min(.8,.05*2**attempt))

def atomic_bytes(path,data,root=None):
    path=guard(path,root)if root else Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb')as stream:stream.write(data);stream.flush()
        replace(name,path)
    finally:Path(name).unlink(missing_ok=True)

def atomic_json(path,data,root=None):atomic_bytes(path,(json.dumps(safe(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'),root)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def fingerprint(path):
    path=Path(path);st=path.stat();h=hashlib.sha256()
    with path.open('rb')as stream:
        if st.st_size<=2*1024*1024:
            for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
            kind='full_small_file'
        else:
            h.update(stream.read(65536));stream.seek(max(0,st.st_size-65536));h.update(stream.read(65536));kind='first_last_64KiB'
    return dict(bytes=st.st_size,content_digest=h.hexdigest(),fingerprint_kind=kind)

def close_memmaps(*arrays):
    for a in arrays:
        m=getattr(a,'_mmap',None)
        if m is not None:
            try:
                if getattr(a,'mode',None)!='r':a.flush()
            finally:m.close()
