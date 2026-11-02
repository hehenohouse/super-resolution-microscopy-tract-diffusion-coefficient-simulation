"""One environment-isolated stage call; never dispatches an upstream dependency."""
from pathlib import Path
from dataclasses import asdict
import argparse,importlib,json,sys,traceback
from .types import Artifact,Sample,StageContext
from .io import read_json,atomic_json,safe,utc

DISPATCH={
 'detection':('localization','execute_detection'),'fit':('localization','execute_fit'),
 'broad':('mask','execute_broad'),'support_counts':('mask','execute_support_counts'),
 'mask_threshold':('mask','execute_threshold'),'mask_cleanup':('mask','execute_cleanup'),
 'mask_roi':('mask','execute_roi'),'assignment':('assignment','execute_assignment'),
 'distances':('distances','execute'),'decomposition':('tardis','execute_decomposition'),
 'bootstrap':('tardis','execute_bootstrap'),'c1_mean':('statistics','execute_c1_mean'),
 'statistics':('statistics','execute'),'presentation':('presentation','execute'),
}
def context_from(value):
    s=dict(value['sample'])
    for key in ['directory','C1_file','C2_file','ROI_file','legacy_output','legacy_figures']:
        if s.get(key)is not None:s[key]=Path(s[key])
    return StageContext(value['config'],Sample(**s),value['parameters'],{k:Artifact(**v)for k,v in value['inputs'].items()},Path(value['output_dir']),Path(value['package_root']),value.get('runtime',{}))
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);args=p.parse_args(argv)
    request=read_json(args.request);ctx=context_from(request);stage=request['stage']
    from .runtime import configure,terminate_children,platform
    platform.install_stop_handlers()
    configure(ctx.config)
    module,function=DISPATCH[stage]
    # Only this dispatch path is permitted to invoke a scientific stage.
    print(json.dumps(dict(event='stage_call',stage=stage,scientific=stage!='presentation',utc=utc())),flush=True)
    try:
        fn=getattr(importlib.import_module('.stages.'+module,'trak_pipeline'),function)
        result=fn(ctx)
        atomic_json(request['result_path'],dict(files={k:str(v)for k,v in result.files.items()},metadata=safe(result.metadata)),ctx.config['data_root'])
    except BaseException:
        traceback.print_exc();raise
    finally:
        terminate_children()
if __name__=='__main__':main()
