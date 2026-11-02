"""Platform isolation for the separate Windows x64 release; no science kernels."""
from pathlib import Path
import contextlib,ctypes,hashlib,os,platform,signal,struct,subprocess,threading,time,uuid

class BatchInfrastructureError(RuntimeError): pass
IS_WINDOWS=os.name=='nt'

def host_identity():
    token=platform.node()+'|'+str(uuid.getnode())
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Cryptography',0,winreg.KEY_READ|winreg.KEY_WOW64_64KEY) as key:
                token=str(winreg.QueryValueEx(key,'MachineGuid')[0])
        except OSError:pass
    return dict(os=platform.system(),architecture=platform.machine(),pointer_bits=struct.calcsize('P')*8,
                host_id=hashlib.sha256(token.encode('utf-8')).hexdigest()[:20])

def platform_key():
    h=host_identity();return ('windows_x64' if IS_WINDOWS else 'posix_packaging_test')+'_'+h['host_id']

def executable(directory,name):
    return Path(directory)/(name+'.exe' if IS_WINDOWS else name)

def _windows_normalized_path(value):
    """Normalize equivalent DOS/extended spellings after physical resolution.

    Windows non-strict realpath may retain the extended prefix for a missing
    target while stripping it for its existing parent/root. A concurrent mkdir
    can trigger that difference; it does not mean the target left the root.
    """
    import ntpath,re
    text=os.fspath(value).replace('/','\\')
    if '\x00' in text:raise BatchInfrastructureError('NUL is not valid in a Windows file path')
    if text.startswith('\\\\?\\'):
        suffix=text[4:]
        if suffix[:4].casefold()=='unc\\':text='\\\\'+suffix[4:]
        elif re.match(r'^[A-Za-z]:\\',suffix):text=suffix
        else:raise BatchInfrastructureError('Unsupported Windows device path: '+str(value))
    if text.startswith('\\\\.\\'):
        raise BatchInfrastructureError('Windows device paths are not output paths: '+str(value))
    text=ntpath.normpath(text)
    if not ntpath.isabs(text) or not ntpath.splitdrive(text)[0]:
        raise BatchInfrastructureError('Expected an absolute Windows file path: '+str(value))
    return text


def _windows_path_is_within(path,root):
    """Compare complete components, never a string prefix; no drive fallback."""
    import ntpath
    try:
        target=ntpath.normcase(_windows_normalized_path(path))
        base=ntpath.normcase(_windows_normalized_path(root))
        return ntpath.commonpath([target,base])==base
    except (ValueError,TypeError,BatchInfrastructureError):return False


def assert_drive_root(data_root):
    root=Path(data_root).resolve()
    if IS_WINDOWS:root=Path(_windows_normalized_path(root))
    if not root.is_dir():raise BatchInfrastructureError('Portable data root unavailable; refusing fallback: '+str(root))
    if IS_WINDOWS:
        anchor=Path(root.anchor)
        system=os.environ.get('SystemDrive','C:').rstrip('\\/').casefold()
        if str(anchor).rstrip('\\/').casefold()==system:
            raise BatchInfrastructureError('Data root is on the Windows system drive; external-drive execution required')
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetDriveTypeW.argtypes=[ctypes.c_wchar_p];k.GetDriveTypeW.restype=ctypes.c_uint
        if k.GetDriveTypeW(str(anchor)) not in (2,3):
            raise BatchInfrastructureError('External local data volume is unavailable')
    else:
        mount=root
        while mount.parent!=mount and not mount.is_mount():mount=mount.parent
        if not mount.is_mount() or mount==Path('/'):
            raise BatchInfrastructureError('Packaging test root must be on an independently mounted external volume')
    return root

def guard_path(path,data_root):
    root=assert_drive_root(data_root);resolved=Path(path).absolute().resolve()
    if IS_WINDOWS:
        target=Path(_windows_normalized_path(resolved))
        allowed=_windows_path_is_within(target,root)
    else:
        target=resolved;allowed=target.is_relative_to(root)
    if not allowed:
        raise BatchInfrastructureError('Write outside portable data root: '+str(path)+
            ' | resolved_root='+repr(str(root))+' | resolved_target='+repr(str(target)))
    return target

def _kernel():
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    for name,args,result in [
        ('OpenProcess',[w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
        ('GetExitCodeProcess',[w.HANDLE,ctypes.POINTER(w.DWORD)],w.BOOL),
        ('GetProcessTimes',[w.HANDLE,ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME)],w.BOOL),
        ('CloseHandle',[w.HANDLE],w.BOOL),
        ('WaitForSingleObject',[w.HANDLE,w.DWORD],w.DWORD),
        ('TerminateProcess',[w.HANDLE,w.UINT],w.BOOL),
        ('GenerateConsoleCtrlEvent',[w.DWORD,w.DWORD],w.BOOL)]:
        f=getattr(k,name);f.argtypes=args;f.restype=result
    return k,w

def process_identity(pid):
    """Non-signalling liveness query. NEVER calls Windows os.kill(pid,0)."""
    if IS_WINDOWS:
        k,w=_kernel();handle=k.OpenProcess(0x1000|0x100000,False,int(pid))
        if not handle:
            error=ctypes.get_last_error()
            if error in (87,1168):return None
            return dict(pid=int(pid),alive=True,inaccessible=True)
        try:
            code=w.DWORD()
            if not k.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
            if code.value!=259:return None
            created,exited,kernel,user=(w.FILETIME() for _ in range(4))
            if not k.GetProcessTimes(handle,*[ctypes.byref(v) for v in [created,exited,kernel,user]]):
                raise ctypes.WinError(ctypes.get_last_error())
            return dict(pid=int(pid),alive=True,creation_time_100ns=(created.dwHighDateTime<<32)|created.dwLowDateTime)
        finally:k.CloseHandle(handle)
    try:
        # Non-signalling POSIX process query; ps is packaging-test-only.
        p=subprocess.run(['ps','-p',str(int(pid)),'-o','lstart=,stat='],capture_output=True,text=True,check=False)
        text=p.stdout.strip()
        if not text or text.split()[-1].startswith('Z'):return None
        return dict(pid=int(pid),alive=True,start=' '.join(text.split()[:-1]))
    except OSError as e:raise BatchInfrastructureError('Cannot query process identity: '+str(e)) from e

def same_process(owner):
    if owner.get('host')!=host_identity():return False
    current=process_identity(owner['process']['pid'])
    if current is None:return False
    if current.get('inaccessible'):return True
    return current==owner['process']

class WindowsProcess:
    """Create suspended -> assign Job Object -> resume; descendants cannot escape."""
    def __init__(self,args,cwd,stdout,env):
        if not IS_WINDOWS:raise RuntimeError('WindowsProcess requires Windows')
        import msvcrt
        k,w=_kernel();self.args=list(map(str,args));self.returncode=None;self.pid=None;self._lock=threading.RLock();self._w=w
        class IO(ctypes.Structure):
            _fields_=[(x,ctypes.c_ulonglong) for x in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
        class BASIC(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
              ('LimitFlags',w.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
              ('ActiveProcessLimit',w.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class EXTENDED(ctypes.Structure):
            _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),
              ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        class STARTUP(ctypes.Structure):
            _fields_=[('cb',w.DWORD),('lpReserved',w.LPWSTR),('lpDesktop',w.LPWSTR),('lpTitle',w.LPWSTR),
              ('dwX',w.DWORD),('dwY',w.DWORD),('dwXSize',w.DWORD),('dwYSize',w.DWORD),('dwXCountChars',w.DWORD),
              ('dwYCountChars',w.DWORD),('dwFillAttribute',w.DWORD),('dwFlags',w.DWORD),('wShowWindow',w.WORD),
              ('cbReserved2',w.WORD),('lpReserved2',ctypes.POINTER(ctypes.c_ubyte)),('hStdInput',w.HANDLE),
              ('hStdOutput',w.HANDLE),('hStdError',w.HANDLE)]
        class PROCESS(ctypes.Structure):
            _fields_=[('hProcess',w.HANDLE),('hThread',w.HANDLE),('dwProcessId',w.DWORD),('dwThreadId',w.DWORD)]
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
        k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];k.SetInformationJobObject.restype=w.BOOL
        k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
        k.TerminateJobObject.argtypes=[w.HANDLE,w.UINT];k.TerminateJobObject.restype=w.BOOL
        k.ResumeThread.argtypes=[w.HANDLE];k.ResumeThread.restype=w.DWORD
        k.GetCurrentProcess.restype=w.HANDLE
        k.DuplicateHandle.argtypes=[w.HANDLE,w.HANDLE,w.HANDLE,ctypes.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD];k.DuplicateHandle.restype=w.BOOL
        k.CreateProcessW.argtypes=[w.LPCWSTR,w.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,w.BOOL,w.DWORD,ctypes.c_void_p,w.LPCWSTR,ctypes.POINTER(STARTUP),ctypes.POINTER(PROCESS)]
        k.CreateProcessW.restype=w.BOOL
        self._k=k;self._process=None;self._job=k.CreateJobObjectW(None,None)
        if not self._job:raise BatchInfrastructureError('CreateJobObjectW failed: '+str(ctypes.WinError(ctypes.get_last_error())))
        info=EXTENDED();info.BasicLimitInformation.LimitFlags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self._job,9,ctypes.byref(info),ctypes.sizeof(info)):
            k.CloseHandle(self._job);self._job=None;raise BatchInfrastructureError('Cannot configure Windows process Job Object')
        duplicates=[];null_input=None;pi=PROCESS()
        try:
            def inheritable(handle):
                duplicate=w.HANDLE()
                if not k.DuplicateHandle(k.GetCurrentProcess(),handle,k.GetCurrentProcess(),ctypes.byref(duplicate),0,True,2):
                    raise ctypes.WinError(ctypes.get_last_error())
                duplicates.append(duplicate);return duplicate
            out=inheritable(msvcrt.get_osfhandle(stdout.fileno()))
            null_input=open(os.devnull,'rb');inp=inheritable(msvcrt.get_osfhandle(null_input.fileno()))
            startup=STARTUP();startup.cb=ctypes.sizeof(startup);startup.dwFlags=0x100
            startup.hStdInput=inp;startup.hStdOutput=out;startup.hStdError=out
            environment='\0'.join(str(a)+'='+str(b) for a,b in sorted(env.items(),key=lambda item:item[0].upper()))+'\0\0'
            envblock=ctypes.create_unicode_buffer(environment)
            command=ctypes.create_unicode_buffer(subprocess.list2cmdline(self.args))
            flags=0x00000004|0x00000200|0x00000400 # suspended, new process group, Unicode env
            if not k.CreateProcessW(self.args[0],command,None,None,True,flags,envblock,str(cwd),ctypes.byref(startup),ctypes.byref(pi)):
                raise ctypes.WinError(ctypes.get_last_error())
            self._process=pi.hProcess;self.pid=int(pi.dwProcessId)
            if not k.AssignProcessToJobObject(self._job,self._process):
                raise ctypes.WinError(ctypes.get_last_error())
            if k.ResumeThread(pi.hThread)==0xffffffff:raise ctypes.WinError(ctypes.get_last_error())
        except Exception as e:
            if pi.hProcess:k.TerminateProcess(pi.hProcess,2)
            self.close()
            raise BatchInfrastructureError('Windows stage process creation/isolation failed: '+str(e)) from e
        finally:
            if pi.hThread:k.CloseHandle(pi.hThread)
            for h in duplicates:k.CloseHandle(h)
            if null_input is not None:null_input.close()

    def poll(self):
        with self._lock:
            if self.returncode is not None:return self.returncode
            code=self._w.DWORD()
            if not self._k.GetExitCodeProcess(self._process,ctypes.byref(code)):
                raise BatchInfrastructureError('GetExitCodeProcess failed')
            if code.value!=259 or self._k.WaitForSingleObject(self._process,0)==0:
                self.returncode=int(code.value);self.close()
            return self.returncode

    def wait(self,timeout=None):
        # A short polling wait avoids another controller thread closing a HANDLE
        # while WaitForSingleObject is blocked on it during cancellation.
        started=time.monotonic()
        while True:
            code=self.poll()
            if code is not None:return code
            if timeout is not None and time.monotonic()-started>=timeout:
                raise subprocess.TimeoutExpired(self.args,timeout)
            time.sleep(.05)

    def terminate_tree(self,grace_s=8):
        if self.poll() is not None:return
        self._k.GenerateConsoleCtrlEvent(1,self.pid) # CTRL_BREAK_EVENT; cooperative stage handler
        try:self.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with self._lock:
                if self.poll() is not None:return
                if not self._k.TerminateJobObject(self._job,130):raise BatchInfrastructureError('TerminateJobObject failed')
            self.wait(timeout=10)

    def close(self):
        with self._lock:
            if self._job:self._k.CloseHandle(self._job);self._job=None
            if self._process:self._k.CloseHandle(self._process);self._process=None

def spawn_stage(args,cwd,stdout,env):
    if IS_WINDOWS:return WindowsProcess(args,cwd,stdout,env)
    return subprocess.Popen(args,cwd=cwd,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True,env=env)

def terminate_tree(process,grace_s=8):
    if IS_WINDOWS:return process.terminate_tree(grace_s)
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=10)

def install_stop_handlers():
    def stop(signum,frame):raise KeyboardInterrupt('Requested stop signal '+str(signum))
    for name in ['SIGINT','SIGTERM','SIGBREAK'] if IS_WINDOWS else ['SIGINT','SIGTERM','SIGHUP']:
        if hasattr(signal,name):signal.signal(getattr(signal,name),stop)

@contextlib.contextmanager
def inhibit_sleep():
    if IS_WINDOWS:
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.SetThreadExecutionState.argtypes=[ctypes.c_uint];k.SetThreadExecutionState.restype=ctypes.c_uint
        if not k.SetThreadExecutionState(0x80000000|0x00000001):raise BatchInfrastructureError('Cannot inhibit Windows sleep')
        try:yield
        finally:k.SetThreadExecutionState(0x80000000)
    else:yield
