"""Platform processes, locks and drive-owned runtime caches; no scientific logic."""
from pathlib import Path
from contextlib import contextmanager
import json,os,subprocess,sys,time,uuid,threading,traceback
from .compat import platform_runtime as platform
from .io import atomic_json,read_json,guard,utc

_CHILDREN = {}
_CHILD_LOCK = threading.RLock()
_CANCELLED = threading.Event()

def reset_cancellation():
    with _CHILD_LOCK:
        if _CHILDREN:raise RuntimeError('Cannot reset cancellation while child stages are active')
        _CANCELLED.clear()

def terminate_children():
    """Stop owned process trees, including workers launched by background threads."""
    with _CHILD_LOCK:
        _CANCELLED.set()
        children = list(_CHILDREN.values())
    for process in children:
        try:
            if process.poll() is None:
                platform.terminate_tree(process)
        except (OSError, ProcessLookupError):
            pass

CACHE_NAMES={'TMPDIR':'tmp','TEMP':'tmp','TMP':'tmp','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','XDG_CACHE_HOME':'xdg_cache','XDG_CONFIG_HOME':'xdg_config','XDG_DATA_HOME':'xdg_data','JGO_CACHE_DIR':'jgo','M2_REPO':'m2','CJDK_CACHE_DIR':'cjdk','PIP_CACHE_DIR':'pip'}
def configure(config):
    root=Path(config['data_root']);platform.assert_drive_root(root)
    runtime=guard(Path(config['control_root'])/'modular/runtime',root);runtime.mkdir(parents=True,exist_ok=True)
    for key,name in CACHE_NAMES.items():
        p=runtime/name;p.mkdir(exist_ok=True);os.environ[key]=str(p)
    import tempfile
    tempfile.tempdir=str(runtime/'tmp')
    os.environ.update(TRAK_DATA_ROOT=str(root),TRAK_CONTROL_ROOT=str(config['control_root']),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    sys.dont_write_bytecode=True
    java=config.get('runtime',{}).get('java_home')
    if java:
        os.environ['JAVA_HOME']=java;binary=str(Path(java)/'bin')
        if os.environ.get('PATH','').split(os.pathsep)[0]!=binary:os.environ['PATH']=binary+os.pathsep+os.environ.get('PATH','')
    from .kernels import paths
    paths()
    return runtime

def validate_runtime(config):
    """Check executable presence only; no parity replay, pilot or package install."""
    values=config.get('runtime',{})
    for name in ('science_python','detector_python','nellie_python','cleanup_python'):
        if not values.get(name)or not Path(values[name]).is_file():raise platform.BatchInfrastructureError('Installed runtime is missing: '+name)
    if not values.get('fiji')or not Path(values['fiji']).is_dir():raise platform.BatchInfrastructureError('Installed Fiji directory is missing')
    java=Path(values.get('java_home',''))/'bin'/('java.exe'if os.name=='nt'else'java')
    if not java.is_file():raise platform.BatchInfrastructureError('Installed Java is missing: '+str(java))

def process_main(fn):
    """Preserve shared-environment/drive failures across native subprocess layers."""
    try:return int(fn()or 0)
    except KeyboardInterrupt:
        terminate_children();return 130
    except (platform.BatchInfrastructureError,ImportError):
        traceback.print_exc();terminate_children();return 74
    except subprocess.CalledProcessError as error:
        if error.returncode==74:traceback.print_exc();terminate_children();return 74
        if error.returncode==130:terminate_children();return 130
        raise
    except OSError as error:
        if error.errno in(5,6,19,28,30,57,60,74,121,122):
            traceback.print_exc();terminate_children();return 74
        raise

@contextmanager
def batch_lock(config,kind='run'):
    root=Path(config['data_root']);parent=guard(Path(config['control_root'])/'modular/state',root);parent.mkdir(parents=True,exist_ok=True)
    path=parent/(kind+'.lock');token=uuid.uuid4().hex
    process=platform.process_identity(os.getpid())
    if process is None:raise RuntimeError('Cannot identify lock owner')
    owner=dict(host=platform.host_identity(),process=process,token=token,started_utc=utc())
    try:path.mkdir()
    except FileExistsError:
        record=path/'owner.json'
        if not record.is_file():raise BlockingIOError('Incomplete lock; inspect '+str(path))
        old=read_json(record)
        if old.get('host')!=owner['host']or platform.same_process(old):raise BlockingIOError('Another live or inaccessible operation owns '+str(path))
        os.replace(path,path.with_name(path.name+'.stale_'+uuid.uuid4().hex));path.mkdir()
    atomic_json(path/'owner.json',owner,root)
    try:yield
    finally:
        record=path/'owner.json'
        if record.is_file()and read_json(record).get('token')==token:
            record.unlink();path.rmdir()

def invoke(config,command,log_path,cwd=None):
    configure(config);root=Path(config['data_root']);log_path=guard(log_path,root);log_path.parent.mkdir(parents=True,exist_ok=True)
    print('COMMAND_STAGE '+json.dumps(command),flush=True)
    env=dict(os.environ)
    with log_path.open('w',encoding='utf-8')as log:
        with _CHILD_LOCK:
            if _CANCELLED.is_set():raise KeyboardInterrupt('Stage spawn cancelled')
            process=platform.spawn_stage(list(map(str,command)),cwd or config['package_root'],log,env)
            _CHILDREN[id(process)]=process
        try:
            with log_path.open(encoding='utf-8',errors='replace')as reader:
                while process.poll()is None:
                    for line in reader:print(line,end='',flush=True)
                    time.sleep(.2)
                for line in reader:print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            platform.terminate_tree(process);raise
        finally:
            with _CHILD_LOCK:_CHILDREN.pop(id(process),None)
            close=getattr(process,'close',None)
            if close:close()
    if code==74:raise platform.BatchInfrastructureError('Shared software or drive failure; see '+str(log_path))
    if code==130:raise KeyboardInterrupt('Child stage was interrupted')
    if code:raise subprocess.CalledProcessError(code,command)
