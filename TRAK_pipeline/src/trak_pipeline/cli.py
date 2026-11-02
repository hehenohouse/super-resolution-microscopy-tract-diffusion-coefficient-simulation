"""Explicit planning, computation, aggregation, and presentation commands."""
from pathlib import Path
import argparse,json,sys,traceback
from .config import load_config,discover_samples
from .io import safe,atomic_json,utc
from . import runtime

def parser():
    p=argparse.ArgumentParser(description='Portable protein analysis pipeline; no implicit scientific refresh.')
    p.add_argument('command',choices=['plan','run','summarize','refresh','stop'])
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--cell',help='Exact sample ID or cell directory')
    p.add_argument('--completed-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--target',help='Stop after this explicit pipeline stage')
    p.add_argument('--validate-legacy',action='store_true',help='Read-only fixed-ten-frame legacy audit; no science rerun')
    p.add_argument('--skip-summary',action='store_true')
    p.add_argument('--force-stage',action='append',default=[],help='Explicit run-only recomputation of this stage; upstream compatible caches are retained')
    p.add_argument('--set',action='append',default=[],metavar='STAGE.PARAMETER=JSON',help='Scientific parameter override; plan reports affected stages')
    p.add_argument('--style',action='append',default=[],metavar='KEY=JSON',help='Presentation override; does not invalidate scientific artifacts')
    p.add_argument('--report',type=Path,help='Optional JSON report, required to be on the dataset drive')
    return p

def _value(text):
    try:return json.loads(text)
    except json.JSONDecodeError:return text

def overrides(args):
    result={'methods':{},'style':{}}
    for item in args.set:
        left,sep,value=item.partition('=');stage,dot,key=left.partition('.')
        if not sep or not dot or not key:raise ValueError('--set requires STAGE.PARAMETER=JSON')
        result['methods'].setdefault(stage,{}).setdefault('parameters',{})[key]=_value(value)
    for item in args.style:
        key,sep,value=item.partition('=')
        if not sep or not key:raise ValueError('--style requires KEY=JSON')
        result['style'][key]=_value(value)
    return result

def main(argv=None):
    args=parser().parse_args(argv)
    runtime.platform.install_stop_handlers()
    try:
        config=load_config(args.config,overrides(args))
        if args.force_stage:
            from .graph import default_specs
            if args.command!='run' or any(s not in {x.stage for x in default_specs(config)}for s in args.force_stage):raise ValueError('--force-stage accepts known stages in explicit run commands only')
        if args.command in ('refresh','summarize')and not args.completed_only:
            raise ValueError(args.command+' requires --completed-only; it never starts a cell')
        if args.command=='stop':
            from .runner import _state
            path=_state(config)/'STOP_AFTER_CURRENT_CELL'
            atomic_json(path,{'requested_utc':utc(),'mode':'after_current_cell'},config['data_root'])
            result={'stop_after_current_cell':True,'path':str(path),'scientific_stage_calls':0}
        else:
            samples=discover_samples(config,args.cell)
            if args.command=='plan':
                from .legacy import LegacyReader
                from .graph import build_plan
                reader=LegacyReader(config);rows=[]
                for sample in samples:
                    old=reader.read_sample(sample,validate=args.validate_legacy)
                    plan=build_plan(config,sample,old)
                    rows.append({'sample_id':sample.sample_id,'stages':[{k:v for k,v in r.items()if k!='parameters'}for r in plan],'legacy_audit':reader.reports.get(sample.sample_id)})
                result={'mode':'read_only_plan','cells':rows,'scientific_stage_calls':0,'new_cells_started':0}
            else:
                from .runner import run_dataset,refresh_dataset,summarize_dataset
                runtime.configure(config)
                # One mutation operation per dataset, regardless of its command.
                with runtime.batch_lock(config),runtime.platform.inhibit_sleep():
                    if args.command=='run':result=run_dataset(config,samples,validate=args.validate_legacy,target=args.target,skip_summary=args.skip_summary,resume=args.resume,force_stages=args.force_stage)
                    elif args.command=='summarize':result=summarize_dataset(config,discover_samples(config),publish=True)
                    else:result=refresh_dataset(config,samples)
        if args.report:atomic_json(args.report,result,config['data_root'])
        print(json.dumps(safe(result),indent=2,allow_nan=False),flush=True)
        if args.command=='refresh'and not result.get('complete',True):return 1
        if any(r.get('status')in('failed','publication_incomplete')for r in result.get('cells',[])if isinstance(r,dict))or result.get('summary_failures'):return 1
        return 0
    except KeyboardInterrupt:
        runtime.terminate_children();print('Interrupted; committed stage checkpoints are retained.',file=sys.stderr);return 130
    except BlockingIOError as error:
        print('BATCH BUSY: '+str(error),file=sys.stderr);return 75
    except runtime.platform.BatchInfrastructureError as error:
        runtime.terminate_children();print('BATCH STOPPED: '+str(error),file=sys.stderr);return 74
    except Exception as error:
        runtime.terminate_children();traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
