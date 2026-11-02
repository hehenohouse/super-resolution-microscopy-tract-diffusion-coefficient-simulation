"""Protein identity and dataset paths are configuration, never algorithm globals."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import json,os,re
from .types import Sample
from .io import read_json

PACKAGE_ROOT=Path(__file__).resolve().parents[2]
REPRESENTATIVE_FRAMES=[0,49,200,625,1200,1250,1875,2000,2450,2499]

def natural(v):return tuple(int(x)if x.isdigit()else x.lower()for x in re.split(r'(\d+)',str(v)))
def _path(value,base):
    p=Path(value)
    return(p if p.is_absolute()else base/p).resolve()

def portable_input(value,config):
    """Rebase declared dataset-relative paths; never search for a best match."""
    root=Path(config['data_root']);text=str(value).replace('\\','/')
    marker='/'+root.name+'/'
    if marker in text:text=text.rsplit(marker,1)[1]
    elif text.startswith(root.name+'/'):text=text[len(root.name)+1:]
    else:
        p=Path(text)
        if p.is_absolute():
            resolved=p.resolve()
            if resolved.is_relative_to(root):return resolved
            raise ValueError('Saved input is outside this dataset: '+text)
        if re.match(r'^[A-Za-z]:',text):raise ValueError('Cannot rebase undeclared Windows dataset path: '+text)
    p=(root/text).resolve()
    if not p.is_relative_to(root):raise ValueError('Input escapes dataset root')
    return p

def load_config(path,overrides=None):
    path=Path(path).resolve();c=read_json(path)
    if c.get('schema_version')!=1:raise ValueError('Unsupported dataset configuration schema')
    c=deepcopy(c);c['_config_path']=str(path);c['package_root']=str(PACKAGE_ROOT)
    c['data_root']=str(_path(c['data_root'],path.parent))
    c['control_root']=str(Path(c['data_root'])/c.get('control_relative','_batch_pipeline'))
    if not Path(c['data_root']).is_dir():raise FileNotFoundError('Dataset drive/root is unavailable: '+c['data_root'])
    if overrides:
        for stage,values in overrides.get('methods',{}).items():
            existing=c['methods'].setdefault(stage,{})
            existing.setdefault('parameters',{}).update(values.get('parameters',values))
        c.setdefault('style',{}).update(overrides.get('style',{}))
    acq=c['acquisition']
    if acq.get('axes')!='TYX' or len(acq.get('shape',[]))!=3:raise ValueError('Acquisition requires explicit TYX shape')
    if acq['shape'][0]!=acq['frames']or min(acq['shape'])<1 or acq['pixel_size_nm']<=0 or acq['frame_interval_s']<=0:raise ValueError('Invalid acquisition contract')
    for stage,method in c.get('methods',{}).items():
        parameters=method.get('parameters',{})
        for key,expected in [('frames',acq['frames']),('shape',acq['shape'][1:]),('pixel_size_nm',acq['pixel_size_nm']),('frame_interval_s',acq['frame_interval_s'])]:
            if key in parameters and parameters[key]!=expected:raise ValueError('Conflicting acquisition field in '+stage+': '+key)
    c['sample_frames']=[int(f)for f in c.get('sample_frames',REPRESENTATIVE_FRAMES)]
    if len(set(c['sample_frames']))!=10 or any(f<0 or f>=acq['frames']for f in c['sample_frames']):raise ValueError('Exactly ten in-range representative frames are required')
    platform='windows'if os.name=='nt'else'mac';profile=c.get('runtime_profiles',{}).get(platform,{})
    runtime={}
    if profile.get('reference_file'):
        reference=_path(profile['reference_file'],path.parent)
        override=profile.get('active_override')
        if override and _path(override,path.parent).is_file():reference=_path(override,path.parent)
        raw=read_json(reference)
        base=_path(profile.get('reference_base',str(reference.parent)),path.parent)
        runtime={k:str(_path(v,base))for k,v in raw.items()if not k.startswith('_')}
    runtime.update({k:str(_path(v,path.parent))for k,v in profile.get('paths',{}).items()})
    c['runtime']=runtime
    legacy=c.get('legacy_releases',{}).get(platform)
    c['legacy_release']=str(_path(legacy,path.parent))if legacy else None
    return c

def sample_from_row(row,c):
    directory=portable_input(row.get('source_directory_relative')or row['source_directory'],c)
    metadata=dict(row)
    metadata.update(molecule=c['molecule'],sample_id=row['sample_id'],sample_name=row.get('sample_name',row['sample_id']))
    for k in ['C1_file','C2_file','ROI_file']:
        if k not in row:raise ValueError('Sample inventory lacks '+k)
    return Sample(row['sample_id'],directory,metadata,portable_input(row['C1_file'],c),portable_input(row['C2_file'],c),portable_input(row['ROI_file'],c))

def discover_samples(c,cell=None):
    if 'samples'in c:rows=c['samples']
    else:
        inventory=Path(c['data_root'])/c.get('inventory_relative','_batch_pipeline/inventory.json')
        if inventory.is_file():
            rows=read_json(inventory)
            if isinstance(rows,dict):rows=rows.get('cells',rows.get('samples',[]))
        else:
            rows=[];root=Path(c['data_root'])
            conditions=c.get('conditions')or sorted(p.name for p in root.iterdir()if p.is_dir()and not p.name.startswith(('_','.','Analysis','analysis')))
            for condition in conditions:
                folder=root/condition
                if not folder.is_dir():continue
                for br in sorted((p for p in folder.iterdir()if p.is_dir()),key=lambda p:natural(p.name)):
                    match=re.search(c.get('replica_pattern',r'(BR\d+)$'),br.name)
                    if not match:continue
                    replica=match.group(1)
                    for directory in sorted((p for p in br.iterdir()if p.is_dir()),key=lambda p:natural(p.name)):
                        files=[p for p in directory.iterdir()if p.is_file()and not p.name.startswith('.')]
                        channels=[[p for p in files if p.name.startswith(prefix)and p.suffix.lower()in('.tif','.tiff')]for prefix in[c.get('channel_prefixes',{}).get('C1','C1-'),c.get('channel_prefixes',{}).get('C2','C2-')]]
                        rois=[p for p in files if p.suffix.lower()=='.roi']
                        if not any(channels)and not rois:continue
                        if any(len(a)!=1 for a in channels)or len(rois)!=1:raise ValueError('Expected unique C1,C2,ROI at '+str(directory))
                        m=re.search(c.get('cell_pattern',r'_(T\d+)(?:f)?$'),directory.name)
                        if not m:raise ValueError('Cell identifier missing at '+str(directory))
                        name=m.group(1);sid='_'.join([condition,c['molecule'],replica,name])
                        acquisition=''
                        try:acquisition=datetime.strptime(directory.name[:6],'%m%d%y').date().isoformat()
                        except ValueError:pass
                        rows.append(dict(sample_id=sid,sample_name=' '.join([condition,c['molecule'],replica,'cell',name]),genotype=condition,BR=replica,cell=name,acquisition_date=acquisition,source_directory=str(directory),C1_file=str(channels[0][0]),C2_file=str(channels[1][0]),ROI_file=str(rois[0])))
    result=[sample_from_row(row,c)for row in rows if row.get('status')!='invalid_input']
    if len({s.sample_id for s in result})!=len(result):raise ValueError('Duplicate sample IDs in inventory')
    if cell:
        target=Path(cell).resolve()
        result=[s for s in result if s.sample_id==cell or s.directory==target]
        if len(result)!=1:raise ValueError('Cell selection must match exactly one inventory sample: '+cell)
    return result
