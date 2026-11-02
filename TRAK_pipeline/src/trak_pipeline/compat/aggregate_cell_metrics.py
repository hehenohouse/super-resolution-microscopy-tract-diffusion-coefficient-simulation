#!/usr/bin/env python3
"""Read committed scalar cell metrics and publish portable hierarchy summaries.

Never executes localization, masks, classification, tracking or TARDIS.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,html,json,math,os,sys,traceback,uuid
RELEASE=Path(__file__).resolve().parent
CONTROL=RELEASE.parents[1] if RELEASE.parent.name=='releases' else RELEASE.parents[1]
DATA_ROOT=CONTROL.parent
METRICS=('on_percentage','first_mask_ROI_area_um2','mean_ROI_area_um2')


def utc():return datetime.now(timezone.utc).isoformat()
def guarded(path):
    p=Path(path).resolve()
    if not DATA_ROOT.is_dir() or not CONTROL.is_dir() or not p.is_relative_to(DATA_ROOT.resolve()):raise OSError(19,'Portable drive output unavailable: '+str(path))
    return p

def safe(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if hasattr(value,'tolist'):return safe(value.tolist())
    if hasattr(value,'item'):return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def digest(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=guarded(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temp.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
def descriptor(path):
    path=Path(path);return dict(relative_path=path.relative_to(DATA_ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path))
def valid_descriptor(record):
    try:
        p=guarded(DATA_ROOT/record['relative_path']);return p.is_file() and p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
    except (KeyError,ValueError,OSError):return False

def scalar(value):
    if value is None:return ''
    if isinstance(value,bool):return 'true' if value else 'false'
    if isinstance(value,float):return format(value,'.17g') if math.isfinite(value) else ''
    if isinstance(value,(dict,list,tuple)):return json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    return str(value)
def write_csv(path,rows,preferred):
    path=guarded(path);temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');fields=list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    try:
        with temp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({key:scalar(row.get(key)) for key in fields})
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


class MetricsLock:
    def __init__(self):self.path=CONTROL/'state/cell_metrics.lock';self.owner=None
    def __enter__(self):
        from platform_runtime import host_identity,process_identity,same_process
        guarded(self.path);self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.path.mkdir()
        except FileExistsError:
            try:owner=json.loads((self.path/'owner.json').read_text())
            except (OSError,ValueError):raise RuntimeError('Incomplete cell-metrics lock; no files removed')
            if owner.get('host')!=host_identity() or same_process(owner):raise RuntimeError('Another or inaccessible cell-metrics writer owns the lock')
            os.replace(self.path,self.path.with_name(self.path.name+'.stale_'+uuid.uuid4().hex));self.path.mkdir()
        process=process_identity(os.getpid())
        if process is None:raise RuntimeError('Cannot identify cell-metrics writer')
        self.owner=dict(host=host_identity(),process=process,started_utc=utc());atomic_json(self.path/'owner.json',self.owner);return self
    def __exit__(self,*args):
        import shutil
        p=self.path/'owner.json'
        if p.is_file() and json.loads(p.read_text())==self.owner:shutil.rmtree(self.path)


def stats_rows(node):
    groups=node.get('condition_records',[]) if node['level']=='comparison' else [dict(condition=node.get('condition'),BR=node.get('BR'),metrics=node.get('metric_summaries',{}))]
    rows=[]
    for group in groups:
        for metric in METRICS:
            stat=group.get('metrics',{}).get(metric,{})
            rows.append(dict(level=node['level'],node_key=node['key'],condition=group.get('condition'),BR=group.get('BR'),metric=metric,**stat))
    return rows

def BR_rows(node):
    rows=[]
    for br in node.get('BR_records',[]):
        for metric in METRICS:rows.append(dict(condition=br['condition'],BR=br['BR'],metric=metric,**br['metrics'].get(metric,{})))
    return rows

def plot_limits(nodes):
    high=1.
    for node in nodes:
        for record in node.get('cell_records',[]):
            for metric in METRICS[1:]:
                value=record.get(metric)
                if value is not None:high=max(high,float(value))
        for group in [node.get('metric_summaries',{}),*[r.get('metrics',{}) for r in node.get('BR_records',[])],*[r.get('metrics',{}) for r in node.get('condition_records',[])]]:
            for metric in METRICS[1:]:
                stat=group.get(metric,{})
                if stat.get('mean') is not None:high=max(high,float(stat['mean'])+float(stat.get('sd') or 0.))
    return dict(on_percentage=[0.,100.],first_mask_ROI_area_um2=[0.,high*1.08],mean_ROI_area_um2=[0.,high*1.08])


def update_node(node,metrics_id,version,limits):
    from cell_metrics_plots import render_node
    parent=guarded(node['directory']);figures=parent/'Analysis_Figures_Cell_Metrics';data=parent/'analysis_results'/('cell_metrics_'+metrics_id)
    payload={k:v for k,v in node.items() if k not in ('directory','data_directory')};signature=digest(dict(version=version,node=payload,plot_limits=limits))
    complete_path=data/'COMPLETE.json'
    if complete_path.is_file():
        old=json.loads(complete_path.read_text())
        if old.get('complete') and old.get('signature')==signature and old.get('artifacts') and all(valid_descriptor(r) for r in old['artifacts']):
            return dict(key=node['key'],level=node['level'],status='unchanged',progress=node['progress'],figures=str(figures),data=str(data),signature=signature)
    if node.get('blocked'):raise ValueError('Conflicting committed scalar results block '+node['key'])
    data.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    atomic_json(complete_path,dict(complete=False,status='UPDATING',signature=signature,started_utc=utc()))
    write_csv(data/'cell_metrics.csv',node.get('cell_records',[]),['sample_id','genotype','BR','cell','molecule','acquisition_date','source_run_id','status','On','Not_on','unknown','QC_excluded','selected','ROI_candidates','on_denominator',*METRICS])
    write_csv(data/'summary_statistics.csv',stats_rows(node),['level','node_key','condition','BR','metric','mean','sd','sem','n','unit_type','units','n_cells_valid','n_BRs_valid'])
    write_csv(data/'BR_statistics.csv',BR_rows(node),['condition','BR','metric','mean','sd','sem','n','unit_type','units','n_cells_valid','n_BRs_valid'])
    node=dict(node,data_directory=str(data));plot_manifest=render_node(node,figures,limits)
    atomic_json(data/'plot_manifest.json',plot_manifest)
    atomic_json(data/'metadata.json',dict(schema_version=1,metrics_id=metrics_id,node_key=node['key'],level=node['level'],molecule=node.get('molecule'),condition=node.get('condition'),BR=node.get('BR'),progress=node['progress'],source_refs=node.get('source_refs',[]),metric_definitions=dict(on_percentage='100*On/(On+Not_on); unknown and QC excluded are outside denominator; whole-movie counts',first_mask_ROI_area_um2='ROI foreground pixel count in support block0 (original frames0-49) times0.117^2',mean_ROI_area_um2='Time-weighted ROI foreground area over original frames0-2499; no position-error expansion'),weighting='Equal cells within BR; equal BR means within condition; never pooled localization-count or area weighting',uncertainty='Descriptive sample SD(ddof1); SEM saved. No formal significance tests.',metric_validity='Independent per metric; not linked to Intra validity. Missing or zero denominators are unavailable, never zero-filled.',version=version,plot_limits=limits,registration_status='exploratory_identity_registration_unverified',scientific_stages_run=[]))
    paths=[data/n for n in ['cell_metrics.csv','summary_statistics.csv','BR_statistics.csv','plot_manifest.json','metadata.json']]+[figures/'Cell_metrics.png',figures/'INDEX.html']
    if any(not p.is_file() for p in paths):raise ValueError('Scalar metrics publication missing output')
    atomic_json(complete_path,dict(complete=True,completed_utc=utc(),metrics_id=metrics_id,signature=signature,progress=node['progress'],artifacts=[descriptor(p) for p in paths],scientific_stages_run=[]))
    return dict(key=node['key'],level=node['level'],status='updated',progress=node['progress'],figures=str(figures),data=str(data),signature=signature)


def publish_global_index(records,metrics_id,molecule):
    rows=[]
    for r in records:
        fig=Path(r['figures'])/'INDEX.html';data=Path(r['data'])/'cell_metrics.csv';summary=Path(r['data'])/'summary_statistics.csv'
        links=[os.path.relpath(p,DATA_ROOT).replace(os.sep,'/') for p in [fig,data,summary]]
        rows.append('<tr><td>'+html.escape(r['key'])+'</td><td>'+html.escape(r['level'])+'</td><td>'+html.escape(str(r['progress'].get('status')) )+'</td>'+''.join('<td><a href="'+html.escape(link,quote=True)+'">'+label+'</a></td>' for link,label in zip(links,['Figures','Cell values','Means and SD/SEM']))+'</tr>')
    text='<!doctype html><meta charset="utf-8"><title>TRAK1 cell metrics</title><style>body{font:16px system-ui;margin:24px;color:#183048}table{border-collapse:collapse}td,th{border:1px solid #ccd5de;padding:9px}img{max-width:100%}</style><h1>TRAK1: On percentage and ROI mitochondrial area</h1><p>On percentage =100×On/(On+Not-on), excluding unknown and QC-excluded observations. First area describes the first support mask (frames0-49). Areas are ROI-only and have no localization-error expansion.</p><p>BR means use equal cell weights; condition means use equal BR weights. Missing data are not zero-filled. Descriptive SD is not a significance test.</p><p><a href="INTRA_RESULTS.html">Existing Intra curve galleries</a></p><a href="Analysis_Figures_Cell_Metrics/Cell_metrics.png"><img src="Analysis_Figures_Cell_Metrics/Cell_metrics.png"></a><table><tr><th>Sample/group</th><th>Level</th><th>Status</th><th>Pictures</th><th>Full-precision values</th><th>Statistics</th></tr>'+''.join(rows)+'</table><p>Version '+html.escape(metrics_id)+' | Updated UTC '+utc()+'</p>'
    text=text.replace('TRAK1',html.escape(molecule))
    target=guarded(DATA_ROOT/'CELL_METRICS_RESULTS.html');temp=target.with_name('.'+target.name+'.tmp');temp.write_text(text,encoding='utf-8');os.replace(temp,target)


def io_failure(error):return isinstance(error,OSError) and (error.errno in [5,6,13,19,27,28,30,57,60,74,121,122] or not DATA_ROOT.is_dir())
def run(config_path,report_path=None):
    from batch_common import configure_environment
    configure_environment(CONTROL/('windows/runtime/cell_metrics' if os.name=='nt' else 'runtime/cell_metrics'))
    from cell_metrics_core import build_metrics
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg.get('on_denominator')!='On+Not_on':raise ValueError('On denominator must exclude unknown and QC excluded')
    code={n:sha(RELEASE/n) for n in ['aggregate_cell_metrics.py','cell_metrics_core.py','cell_metrics_plots.py']};version=dict(config=cfg,code_SHA256=code);metrics_id='v1_'+digest(version)[:12]
    result=build_metrics(CONTROL,RELEASE,cfg);limits=plot_limits(result['nodes']);records=[];pending_cells=[];errors=list(result.get('errors',[]))
    for i,node in enumerate(result['nodes'],1):
        if node['level']=='cell' and not node['progress']['completed'] and not node['progress']['failed']:
            pending_cells.append(dict(sample_id=node['key'],reason='Awaiting committed cell results; values remain unavailable in hierarchy tables'))
            continue
        try:records.append(update_node(node,metrics_id,version,limits))
        except (KeyboardInterrupt,SystemExit):raise
        except Exception as error:
            if io_failure(error):raise
            errors.append(dict(node=node['key'],error=str(error),traceback=traceback.format_exc()));print('CELL_METRICS_NODE_ERROR '+node['key']+' '+str(error),flush=True)
        if i%10==0 or i==len(result['nodes']):
            print('CELL_METRICS '+str(i)+'/'+str(len(result['nodes']))+' nodes; '+str(len(errors))+' errors',flush=True)
            atomic_json(CONTROL/'cell_metrics_progress.json',dict(updated_utc=utc(),processed_nodes=i,total_nodes=len(result['nodes']),errors=len(errors),metrics_id=metrics_id,scientific_stages_run=[]))
    publish_global_index(records,metrics_id,cfg['molecule'])
    report=dict(all_passed=not errors,completed_utc=utc(),metrics_id=metrics_id,version=version,expected_cells=result.get('expected_cells',len(result.get('cells',[]))),nodes=records,errors=errors,discovery=result.get('discovery'),completed_only=True,pending_cells=pending_cells,scientific_stages_run=[],new_cells_started=0,plot_limits=limits)
    atomic_json(CONTROL/'cell_metrics_status.json',report)
    if report_path:atomic_json(report_path,report)
    with guarded(CONTROL/'BATCH_LOG.txt').open('a',encoding='utf-8') as f:f.write(utc()+' | Cell metrics '+metrics_id+': '+str(sum(r['status']=='updated' for r in records))+' updated, '+str(sum(r['status']=='unchanged' for r in records))+' unchanged; '+str(len(errors))+' errors. Saved scalar caches only; no science stages.\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--completed-only',action='store_true',required=True);p.add_argument('--report',type=Path);args=p.parse_args()
    try:
        from platform_runtime import assert_drive_root
        assert_drive_root(DATA_ROOT)
        with MetricsLock():report=run(args.config,args.report)
        print(json.dumps(dict(all_passed=report['all_passed'],metrics_id=report['metrics_id'],nodes=len(report['nodes']),expected_cells=report['expected_cells'],errors=report['errors'],new_cells_started=0)),flush=True)
        return 0 if report['all_passed'] else 1
    except (KeyboardInterrupt,SystemExit):raise
    except Exception as error:traceback.print_exc();return 74 if io_failure(error) else 1

if __name__=='__main__':raise SystemExit(main())
