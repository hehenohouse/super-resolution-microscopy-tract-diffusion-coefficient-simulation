#!/usr/bin/env python3
"""Read committed cell curves and update portable hierarchical Intra summaries.

This entrypoint never runs localization, segmentation, tracking or TARDIS.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, gzip, hashlib, io, json, math, os, sys, traceback, uuid

RELEASE=Path(__file__).resolve().parent
CONTROL=RELEASE.parents[1] if RELEASE.parent.name=='releases' else RELEASE.parent
DATA_ROOT=CONTROL.parent
IO_ERROR=74
SUPPORTED_CONFIG={'schema_version': 1, 'data_root': '@portable_data_root', 'scientific_fingerprint': '58bef6b94294f648f15e8637a744e71c6dd59a3fa0fd4e40c483a98d204cb7a3', 'molecule': 'TRAK1', 'condition_order': ['WT', 'TDK', 'MDK'], 'replicas': ['BR1', 'BR2', 'BR3'], 'scope': 'full', 'frames': 2500, 'tau_frames': 1, 'frame_interval_s': 0.02, 'pixel_size_nm': 117.0, 'distance_max_um': 3.0, 'distance_bins': 300, 'signed_area_min_abs': 0.01, 'positive_peak_min_exclusive': 0.0, 'groups': ['On', 'Not-on'], 'normalizations': ['signed', 'area', 'peak'], 'common_on_off_and_view_cohort': True, 'cell_weighting': 'equal', 'BR_weighting': 'equal', 'BR_area_order': 'normalize_equal_cell_signed_mean', 'BR_peak_internal_order': 'mean_of_cell_positive_peak_normalized_curves', 'condition_area_order': 'mean_of_BR_signed_area_normalized_canonical_curves', 'condition_peak_order': 'mean_of_BR_positive_peak_normalized_canonical_curves', 'SD_ddof': 1, 'uncertainty': 'descriptive_between_BR_SD; SEM_saved_not_plotted; no_time_bootstrap', 'zoom_max_um': 0.5, 'cross_condition_layout': 'On_NotOn_two_panels_WT_TDK_MDK_overlay', 'top_shared_y_limits': 'all_condition_means_plus_minus_SD; same_On_NotOn_full_zoom_within_view', 'progress_updates': 'after_each_committed_cell; completed_only_startup', 'source_conflicts': 'block_affected_summaries; never_choose_latest', 'new_cell_execution': False, 'image_format': 'PNG', 'data_retention': 'compact_full_precision_summaries_and_membership_no_point_data_or_bootstrap_duplication'}


class SummaryBusy(RuntimeError):pass


class SummaryLock:
    """Serialize summary writers without acquiring a cell-analysis lock."""
    def __init__(self):self.path=CONTROL/'state/intra_summary.lock';self.owner=None
    def __enter__(self):
        from platform_runtime import host_identity,process_identity,same_process
        guarded(self.path);self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.path.mkdir()
        except FileExistsError:
            try:owner=json.loads((self.path/'owner.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):raise SummaryBusy('Summary lock is incomplete; no files were removed')
            if owner.get('host')!=host_identity() or same_process(owner):
                raise SummaryBusy('A live, inaccessible or other-host summary writer owns the lock')
            stale=self.path.with_name(self.path.name+'.stale_'+uuid.uuid4().hex)
            os.replace(self.path,stale);self.path.mkdir()
        process=process_identity(os.getpid())
        if process is None:raise RuntimeError('Cannot identify summary writer process')
        self.owner=dict(host=host_identity(),process=process,started_utc=utc())
        atomic_json(self.path/'owner.json',self.owner);return self
    def __exit__(self,*args):
        import shutil
        owner=self.path/'owner.json'
        if owner.is_file() and json.loads(owner.read_text(encoding='utf-8'))==self.owner:shutil.rmtree(self.path)


def validate_config(config):
    unknown=set(config)-set(SUPPORTED_CONFIG)-{'enabled'}
    if unknown:raise ValueError('Unsupported summary configuration fields: '+str(sorted(unknown)))
    for key,expected in SUPPORTED_CONFIG.items():
        if key in ('molecule','scientific_fingerprint'):continue
        if config.get(key)!=expected:raise ValueError('Unsupported fixed summary setting: '+key)
    if config.get('molecule') not in ('TRAK1','TRAK2'):raise ValueError('Unsupported summary molecule')
    if not isinstance(config.get('scientific_fingerprint'),str) or len(config['scientific_fingerprint'])!=64 or any(c not in '0123456789abcdef' for c in config['scientific_fingerprint']):raise ValueError('Invalid scientific fingerprint')
    if not isinstance(config.get('enabled',True),bool):raise ValueError('enabled must be boolean')


def utc():return datetime.now(timezone.utc).isoformat()

def guarded(path):
    target=Path(path).absolute().resolve()
    if not DATA_ROOT.is_dir() or not CONTROL.is_dir():raise OSError(19,'External data root unavailable')
    if not target.is_relative_to(DATA_ROOT.resolve()):raise ValueError('Summary write outside portable data root: '+str(path))
    return target


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe(value):
    if isinstance(value,Path):
        try:return value.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
        except ValueError:return str(value)
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if hasattr(value,'tolist'):return safe(value.tolist())
    if hasattr(value,'item'):return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def atomic_json(path,value):
    path=guarded(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def digest(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def descriptor(path):
    path=Path(path);return dict(relative_path=path.resolve().relative_to(DATA_ROOT.resolve()).as_posix(),bytes=path.stat().st_size,sha256=sha(path))


def valid_descriptor(record):
    try:
        path=guarded(DATA_ROOT/record['relative_path'])
        return path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
    except (KeyError,ValueError,FileNotFoundError):return False


def number(value):
    if value is None:return ''
    value=float(value)
    return format(value,'.17g') if math.isfinite(value) else ''


def save_curves(node,path):
    path=guarded(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    fields=['level','condition','BR','spatial_group','normalization','series_kind','series_id','distance_left_um','distance_right_um','distance_mid_um','value','SD','SEM','n','units']
    try:
        with temp.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(fields)
                    for group in ['On','Not-on']:
                        for view in ['signed','area','peak']:
                            part=node['groups'][group][view]
                            units='dimensionless' if view=='peak' else 'um^-1'
                            series=[]
                            if part.get('mean') is not None:
                                series.append(('mean',node['key'],part['mean'],part.get('sd'),part.get('sem'),part.get('n',len(part.get('series',{})))))
                            for sid,curve in part.get('series',{}).items():
                                if curve is not None:series.append(('member',sid,curve,part.get('series_sd',{}).get(sid),part.get('series_sem',{}).get(sid),part.get('series_n',{}).get(sid,'')))
                            for kind,sid,curve,sd,sem,n in series:
                                for i,x in enumerate(node['x']):
                                    writer.writerow([node['level'],node.get('condition') or '',node.get('BR') or '',group,view,kind,sid,number(node['edges'][i]),number(node['edges'][i+1]),number(x),number(curve[i]),number(None if sd is None else sd[i]),number(None if sem is None else sem[i]),n,units])
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def io_failure(error):
    return isinstance(error,OSError) and (error.errno in [5,6,13,19,27,28,30,57,60,74,121,122] or not DATA_ROOT.is_dir())


def update_node(node,summary_id,version):
    from intra_summary_plots import render_node
    parent=guarded(node['directory'])
    figures=parent/'Analysis_Figures_Intra'
    data=parent/'analysis_results'/('intra_summary_'+summary_id)
    state_path=data/'COMPLETE.json'
    stable=dict(signature_payload=node.get('signature_payload'),progress=node['progress'],members=node.get('members',[]),exclusions=node.get('exclusions',[]),groups=node['groups'],blocked=node.get('blocked'),source_refs=node.get('source_refs',[]))
    signature=digest(dict(version=version,node=stable))
    if state_path.is_file():
        old=json.loads(state_path.read_text(encoding='utf-8'))
        if old.get('complete') is True and old.get('signature')==signature and old.get('artifacts') and all(valid_descriptor(a) for a in old['artifacts']):
            return dict(key=node['key'],level=node['level'],status='unchanged',progress=node['progress'],figures=str(figures),data=str(data),signature=signature)
    data.mkdir(parents=True,exist_ok=True)
    atomic_json(state_path,dict(complete=False,status='UPDATING',signature=signature,started_utc=utc()))
    save_curves(node,data/'curves_long.csv.gz')
    atomic_json(data/'members_and_exclusions.json',dict(members=node.get('members',[]),exclusions=node.get('exclusions',[]),progress=node['progress'],common_On_NotOn_and_all_views=True))
    plot_manifest=render_node(node,figures)
    atomic_json(data/'plot_manifest.json',plot_manifest)
    atomic_json(data/'metadata.json',dict(schema_version=1,summary_id=summary_id,version=version,node_key=node['key'],level=node['level'],molecule=node.get('molecule'),condition=node.get('condition'),BR=node.get('BR'),progress=node['progress'],available=node.get('available'),blocked=node.get('blocked'),source_refs=node.get('source_refs',[]),signature_payload=node.get('signature_payload'),series_metadata={g:{v:{'n':part.get('n',0),'series_ids':list(part.get('series',{}))} for v,part in views.items()} for g,views in node['groups'].items()},edges_um=node['edges'],distance_mid_um=node['x'],plot_rule='Full-range normalization; no smoothing/clipping; top shared units and Y limits',uncertainty='Descriptive between-BR sample SD; SEM saved. Not time bootstrap; no p values.',calibration_status='manufacturer_nominal',registration_status='exploratory_identity_registration_unverified',new_scientific_analysis=False))
    paths=[data/'curves_long.csv.gz',data/'members_and_exclusions.json',data/'plot_manifest.json',data/'metadata.json',figures/'INDEX.html']
    paths+=sorted(p for p in figures.glob('*.png') if not p.name.startswith('.'))
    if len(paths)!=11:raise ValueError('Expected six PNG files and compact summary records')
    atomic_json(state_path,dict(complete=True,signature=signature,completed_utc=utc(),summary_id=summary_id,progress=node['progress'],artifacts=[descriptor(p) for p in paths]))
    return dict(key=node['key'],level=node['level'],status='updated',progress=node['progress'],figures=str(figures),data=str(data),signature=signature)


def publish_global_index(report,molecule):
    """Publish stable reader navigation without touching cached cell curves."""
    import html,urllib.parse
    from intra_summary_core import portable_path
    def link(path):
        return urllib.parse.quote(Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix(),safe='/')
    rows=[]
    for node in report.get('nodes',[]):
        figures=portable_path(DATA_ROOT,node.get('figures_relative') or node['figures'])
        data=portable_path(DATA_ROOT,node['data'])
        progress=node.get('progress',{})
        cells=str(progress.get('completed',0))+'/'+str(progress.get('expected',0))
        values=[node['key'],node.get('level',''),cells,progress.get('included',0),progress.get('status','')]
        targets=[(figures/'INDEX.html','On / Not-on panels'),
                 (figures/'On_NotOn_Overlay/INDEX.html','Same-axis overlays'),
                 (data/'curves_long.csv.gz','Full-precision curves')]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+
                    ''.join('<td><a href="'+link(path)+'">'+label+'</a></td>'for path,label in targets)+'</tr>')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(molecule)+
          ' Intra summaries</title><style>body{font:17px system-ui;margin:30px;color:#203746}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccd5df}img{max-width:100%;height:auto}a{color:#165d96}</style><h1>'+html.escape(molecule)+
          ': Intra summaries</h1><p>On and Not-on share the same valid cell cohort. Equal cell weights within BR; equal BR weights within condition. Signed, area-normalized and peak-normalized views each have full and zoom plots.</p><p><a href="CELL_METRICS_RESULTS.html">On percentage and ROI mitochondrial areas</a> | <a href="CELL_METRICS_TABLES.html">Reader Excel tables</a> | <a href="_batch_pipeline/windows/INDEX.html">All cell figures</a></p>'+
          '<a href="Analysis_Figures_Intra/Intra_area_zoom.png"><img src="Analysis_Figures_Intra/Intra_area_zoom.png" alt="Area-normalized Intra zoom"></a>'+
          '<a href="Analysis_Figures_Intra/On_NotOn_Overlay/Intra_On_NotOn_area_zoom.png"><img src="Analysis_Figures_Intra/On_NotOn_Overlay/Intra_On_NotOn_area_zoom.png" alt="On/Not-on same-axis zoom"></a>'+
          '<table><tr><th>Group</th><th>Level</th><th>Completed cells</th><th>Included cells</th><th>Status</th><th>Panels</th><th>Overlays</th><th>Saved curves</th></tr>'+''.join(rows)+
          '</table><p>Between-BR shading is descriptive sample SD. No formal significance test. Nominal camera calibration and C1/C2 identity registration remain unverified.</p><p>Updated UTC '+utc()+'</p></html>')
    target=guarded(DATA_ROOT/'INTRA_RESULTS.html');temp=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temp.write_text(page,encoding='utf-8');os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()


def run(config_path,report_path=None):
    from batch_common import configure_environment
    configure_environment(CONTROL/('windows/runtime/intra_summaries' if os.name=='nt' else 'runtime/intra_summaries'))
    from intra_summary_core import build_summaries
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(config)
    if not config.get('enabled',True):
        report=dict(all_passed=True,status='disabled',nodes=[],completed_only=True,new_cells_started=0,scientific_stages_run=[])
        atomic_json(CONTROL/'intra_summary_status.json',report)
        return report
    code={name:sha(RELEASE/name) for name in ['aggregate_intra.py','intra_summary_core.py','intra_summary_plots.py']}
    version=dict(schema_version=1,config=config,code_SHA256=code)
    summary_id='v1_'+digest(version)[:12]
    result=build_summaries(CONTROL,RELEASE,config)
    records=[];errors=[]
    for node in result['nodes']:
        node['molecule']=config['molecule']
        p=node['progress']
        try:
            if node.get('blocked'):errors.append(dict(key=node['key'],error='Source conflict blocks this summary; no mixed curve published'))
            record=update_node(node,summary_id,version);records.append(record)
            print(json.dumps(safe(dict(summary_node=record['key'],status=record['status'],progress=record['progress'])),allow_nan=False),flush=True)
        except (KeyboardInterrupt,SystemExit):raise
        except Exception as error:
            if io_failure(error):raise
            detail=dict(key=node['key'],error=str(error),traceback=traceback.format_exc())
            errors.append(detail);print('SUMMARY_ERROR '+json.dumps(detail),flush=True)
    report=dict(all_passed=not errors,completed_utc=utc(),summary_id=summary_id,version=version,completed_only=True,new_cells_started=0,scientific_stages_run=[],nodes=records,errors=errors,discovery=result.get('discovery'),expected_cells=len(result['inventory']))
    target=CONTROL/'intra_summary_status.json'
    atomic_json(target,report)
    publish_global_index(report,config['molecule'])
    if report_path is not None:atomic_json(report_path,report)
    line=utc()+' | Intra hierarchy summary '+summary_id+': '+str(sum(r['status']=='updated' for r in records))+' updated, '+str(sum(r['status']=='unchanged' for r in records))+' unchanged, '+str(len(errors))+' errors; committed cached curves only; no new cells.\n'
    with guarded(CONTROL/'BATCH_LOG.txt').open('a',encoding='utf-8') as stream:stream.write(line)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--completed-only',action='store_true',required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    try:
        from platform_runtime import assert_drive_root
        assert_drive_root(DATA_ROOT)
        with SummaryLock():report=run(args.config,args.report)
        print(json.dumps(safe(dict(all_passed=report['all_passed'],summary_id=report.get('summary_id'),nodes=report.get('nodes',[]),new_cells_started=0)),allow_nan=False),flush=True)
        return 0 if report['all_passed'] else 1
    except (KeyboardInterrupt,SystemExit):raise
    except SummaryBusy as error:
        print('SUMMARY_BUSY '+str(error),file=sys.stderr);return 75
    except Exception as error:
        traceback.print_exc()
        return IO_ERROR if io_failure(error) else 1


if __name__=='__main__':raise SystemExit(main())
