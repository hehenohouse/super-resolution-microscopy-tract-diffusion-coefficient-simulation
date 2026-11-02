
import sys,json,runpy,types
from pathlib import Path
site,worker,work,batch,record=sys.argv[1:]
sys.path.insert(0,site)
import numpy,scipy,scipy.ndimage,tifffile
sys.path.pop(0)
class UnusedTIFFDependency(types.ModuleType):
    def __getattr__(self,name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError('Unexpected access to TIFF-unused optional dependency '+self.__name__+'.'+name)
# Nellie __init__ eagerly imports these modules, but its frozen TIFF Filter/Label
# path does not call them. Fail immediately if an unexpected use is attempted.
skimage_proxy=types.ModuleType('skimage')
for optional in ['filters','morphology','measure']:
    setattr(skimage_proxy,optional,UnusedTIFFDependency('skimage.'+optional))
sys.modules['skimage']=skimage_proxy
sys.modules['nd2']=UnusedTIFFDependency('nd2')
# Standard absence behavior for unused optional Dask integration in Pint.
sys.modules['dask']=None
# Import the unmodified Nellie root, but do not eagerly load unused Network/Markers.
import nellie
from importlib.machinery import ModuleSpec
segmentation=types.ModuleType('nellie.segmentation')
segmentation.__path__=[str(Path(nellie.__file__).parent/'segmentation')]
segmentation.__package__='nellie.segmentation'
segmentation.__spec__=ModuleSpec('nellie.segmentation',loader=None,is_package=True)
sys.modules['nellie.segmentation']=segmentation
nellie.segmentation=segmentation
modules=[numpy,scipy,scipy.ndimage,tifffile]
runtime={m.__name__:{'file':m.__file__,'version':getattr(m,'__version__',None)} for m in modules}
runtime['direct_segmentation_namespace_path']=segmentation.__path__
runtime['unused_segmentation_initializer_classes_not_imported']=['Markers','Network']
runtime['TIFF_unused_optional_import_policy']=['skimage.filters','skimage.morphology','skimage.measure','nd2','dask (optional, unavailable)']
Path(record).write_text(json.dumps(runtime,indent=2))
print('Using identical hydrated numerical dependencies; TIFF-unused optional imports fail closed; frozen Nellie worker unchanged',flush=True)
sys.argv=[worker,work,batch]
runpy.run_path(worker,run_name='__main__')
