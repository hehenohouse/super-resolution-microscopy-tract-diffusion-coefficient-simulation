"""Top-level reader pages built from the committed modular (five-replica) summary manifest.

Publication only: reads saved summary products and writes INTRA_RESULTS.html,
CELL_METRICS_RESULTS.html and CELL_METRICS_TABLES.html at the data root. It replaces the
older pages written by the BR1-BR3 aggregators, whose tables and curve links no longer match
the figures. It also removes macOS AppleDouble files (``._*``) and stale overlay manifests
from summary figure folders and drops ``._*`` entries from their INDEX pages. No science runs.
"""
from pathlib import Path
import csv,html,json,os,re,urllib.parse
from ..io import atomic_bytes,read_json,utc

LEVEL_ORDER={'comparison':0,'condition':1,'BR':2,'cell':3}
CONDITION_ORDER={'WT':0,'TDK':1,'MDK':2}
FIGURE_DIRS=('Analysis_Figures_Intra','Analysis_Figures_Intra/On_NotOn_Overlay','Analysis_Figures_Cell_Metrics')
METRIC_LABELS={'on_percentage':'On percentage (%)','first_mask_ROI_area_um2':'First-mask ROI area (µm²)','mean_ROI_area_um2':'Mean ROI area (µm²)'}
STYLE=('body{font:16px system-ui;margin:28px;color:#203746;max-width:1200px}table{border-collapse:collapse;margin:12px 0 24px}'
       'td,th{padding:8px 14px;border-bottom:1px solid #ccd5df;text-align:left;vertical-align:top}th{background:#f2f5f8}'
       'img{max-width:100%;height:auto;margin:8px 0}a{color:#165d96}.note{color:#52606d;font-size:14px}nav a{margin-right:18px}')
SUPERSEDED=('Older folders analysis_results/intra_summary_v1_*, intra_summary_v2_* and cell_metrics_v1_* hold the earlier '
            'BR1–BR3 summaries. They are superseded by the folders linked here and are kept unchanged for provenance.')
_APPLEDOUBLE_ENTRY=re.compile(r'<li><a href="\._[^"]*">.*?</li>',re.S)


def _manifest_path(config,manifest=None):
    root=Path(config['data_root']).resolve()
    return Path(manifest) if manifest else Path(config.get('control_root',root/'_batch_pipeline'))/'modular_summaries/summary_manifest.json'


def _sort_key(node):
    key=node['key'];condition=key.split('/')[0].split('_')[0]
    numbers=[int(n) for n in re.findall(r'\d+',key.split('/')[-1])]
    return (LEVEL_ORDER.get(node['level'],9),CONDITION_ORDER.get(condition,9),condition,numbers,key)


def _link(root,path):
    return urllib.parse.quote(Path(path).resolve().relative_to(root).as_posix(),safe='/')


def _a(root,path,label):
    path=Path(path)
    if not path.exists():return '<span class="note">'+html.escape(label)+' (missing)</span>'
    return '<a href="'+_link(root,path)+'">'+html.escape(label)+'</a>'


def _write(root,name,text):
    atomic_bytes(root/name,text.encode('utf-8'),root)


def _page(title,body):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+html.escape(title)+'</title><style>'+STYLE+'</style><h1>'+html.escape(title)+'</h1>'+body+'</html>')


def _nav():
    return ('<nav><a href="INTRA_RESULTS.html">Intra summaries</a><a href="CELL_METRICS_RESULTS.html">On percentage and ROI area</a>'
            '<a href="CELL_METRICS_TABLES.html">Excel tables</a></nav>')


def _footer(summary_id,updated):
    return ('<p class="note">'+html.escape(SUPERSEDED)+'</p><p class="note">Summary '+html.escape(summary_id)+
            ' | Saved numeric summary only; no scientific recomputation | Updated UTC '+html.escape(updated)+'</p>')


def _clean_figure_dirs(directory,summary_id):
    removed=[]
    for name in FIGURE_DIRS:
        folder=directory/name
        if not folder.is_dir():continue
        for path in folder.iterdir():
            stale_overlay=path.name=='overlay_manifest.json' and folder.name=='On_NotOn_Overlay'
            if stale_overlay:
                try:stale_overlay=read_json(path).get('summary_id')!=summary_id
                except (OSError,ValueError):stale_overlay=True
            if (path.name.startswith('._') and path.is_file()) or stale_overlay:
                path.unlink();removed.append(path)
        index=folder/'INDEX.html'
        if index.is_file():
            text=index.read_text(encoding='utf-8')
            cleaned=_APPLEDOUBLE_ENTRY.sub('',text)
            if cleaned!=text:atomic_bytes(index,cleaned.encode('utf-8'));removed.append(index.with_name('INDEX.html (._ entries)'))
    return removed


def _read_csv(path):
    with open(path,newline='',encoding='utf-8')as stream:return list(csv.DictReader(stream))


def _fmt(value,digits=2):
    try:return f'{float(value):.{digits}f}'
    except (TypeError,ValueError):return '—'


def write_overview_pages(config,manifest=None):
    root=Path(config['data_root']).resolve();path=_manifest_path(config,manifest)
    if not path.is_file():raise FileNotFoundError('Saved summary manifest is missing: '+str(path))
    from .summaries import _portable
    saved=read_json(path);summary_id=saved['summary_id'];molecule=str(config.get('molecule') or config.get('dataset_id') or '')
    nodes=sorted(saved['nodes'],key=_sort_key);updated=utc();removed=[]
    for node in nodes:
        node['_dir']=_portable(root,node['directory']);node['_data']=_portable(root,node['data'])
        removed+=_clean_figure_dirs(node['_dir'],summary_id)
    groups=[n for n in nodes if n['level']!='cell'];cells=[n for n in nodes if n['level']=='cell']
    top=next((n for n in groups if n['level']=='comparison'),groups[0] if groups else None)

    # Intra
    rows=[]
    for node in (n for n in groups if n['intra']):
        intra=node['_data']/'intra_node.json';progress=read_json(intra).get('progress',{}) if intra.is_file() else {}
        brs=str(progress.get('valid_BRs','—'))+'/'+str(progress.get('expected_BRs','—'))
        values=[node['key'],node['level'],str(progress.get('completed',0))+'/'+str(progress.get('expected',0)),progress.get('included',0),brs,progress.get('status','missing')]
        links=[_a(root,node['_dir']/'Analysis_Figures_Intra/INDEX.html','On / Not-on panels'),
               _a(root,node['_dir']/'Analysis_Figures_Intra/On_NotOn_Overlay/INDEX.html','Same-axis overlays'),
               _a(root,node['_data']/'intra_curves_long.csv.gz','Full-precision curves')]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+''.join('<td>'+l+'</td>' for l in links)+'</tr>')
    hero=''
    if top is not None:
        for rel,alt in [('Analysis_Figures_Intra/Intra_area_zoom.png','Area-normalized Intra, zoom'),('Analysis_Figures_Intra/On_NotOn_Overlay/Intra_On_NotOn_area_zoom.png','On / Not-on same-axis, zoom')]:
            if (top['_dir']/rel).is_file():hero+='<a href="'+_link(root,top['_dir']/rel)+'"><img src="'+_link(root,top['_dir']/rel)+'" alt="'+alt+'"></a>'
    body=(_nav()+'<p>On and Not-on share the same valid cell cohort. Equal cell weights within BR; equal BR weights within condition. '
          'Signed, area-normalized and peak-normalized views each have full and zoom plots.</p>'+hero+
          '<table><tr><th>Group</th><th>Level</th><th>Completed cells</th><th>Included cells</th><th>Valid BRs</th><th>Status</th><th>Panels</th><th>Overlays</th><th>Saved curves</th></tr>'+''.join(rows)+'</table>'
          '<p class="note">Between-BR shading is descriptive sample SD. No formal significance test. Nominal camera calibration and C1/C2 identity registration remain unverified.</p>'+_footer(summary_id,updated))
    _write(root,'INTRA_RESULTS.html',_page(molecule+': Intra summaries',body))

    # Cell metrics: condition means, then every summary node
    stats=[]
    for node in (n for n in groups if n['level']=='condition'):
        source=node['_data']/'summary_statistics.csv'
        if source.is_file():stats+=_read_csv(source)
    stat_rows=''.join('<tr><td>'+html.escape(r.get('condition',''))+'</td><td>'+html.escape(METRIC_LABELS.get(r.get('metric'),r.get('metric','')))+'</td><td>'+_fmt(r.get('mean'))+'</td><td>'+_fmt(r.get('sd'))+'</td><td>'+_fmt(r.get('sem'))+'</td><td>'+html.escape(str(r.get('n_BRs_valid','')))+'</td><td>'+html.escape(str(r.get('n_cells_valid','')))+'</td></tr>' for r in stats)
    rows=[]
    for node in groups:
        links=[_a(root,node['_dir']/'Analysis_Figures_Cell_Metrics/INDEX.html','Figures'),_a(root,node['_data']/'cell_metrics.csv','Cell values'),
               _a(root,node['_data']/'summary_statistics.csv','Means and SD/SEM'),_a(root,node['_data']/'BR_statistics.csv','BR statistics')]
        rows.append('<tr><td>'+html.escape(node['key'])+'</td><td>'+html.escape(node['level'])+'</td>'+''.join('<td>'+l+'</td>' for l in links)+'</tr>')
    hero=''
    if top is not None and (top['_dir']/'Analysis_Figures_Cell_Metrics/Cell_metrics.png').is_file():
        image=_link(root,top['_dir']/'Analysis_Figures_Cell_Metrics/Cell_metrics.png');hero='<a href="'+image+'"><img src="'+image+'" alt="Cell metrics"></a>'
    body=(_nav()+'<p>On percentage = 100×On/(On+Not-on), excluding unknown and QC-excluded observations. First area describes the first support mask (frames 0–49). '
          'Areas are ROI-only and have no localization-error expansion. BR means use equal cell weights; condition means use equal BR weights. Missing data are not zero-filled. Descriptive SD is not a significance test.</p>'+hero+
          '<h2>Condition means</h2><table><tr><th>Condition</th><th>Metric</th><th>Mean</th><th>SD</th><th>SEM</th><th>BRs</th><th>Cells</th></tr>'+stat_rows+'</table>'
          '<h2>All summary groups</h2><table><tr><th>Group</th><th>Level</th><th>Pictures</th><th>Full-precision values</th><th>Statistics</th><th>Per BR</th></tr>'+''.join(rows)+'</table>'+_footer(summary_id,updated))
    _write(root,'CELL_METRICS_RESULTS.html',_page(molecule+': On percentage and ROI mitochondrial area',body))

    # Excel tables for every node, cells included
    rows=[]
    for node in groups+cells:
        data=node['_data'];counts=[_a(root,data/'localization_counts.csv','Cell localization counts')]
        if node['level']!='cell':counts+=[_a(root,data/'localization_count_per_cell_summary.csv','Counts per cell'),_a(root,data/'localization_count_means.csv','Localization count means')]
        rows.append('<tr><td>'+html.escape(node['level'])+'</td><td>'+html.escape(node['key'])+'</td><td>'+_a(root,node['_dir']/'Analysis_Figures_Cell_Metrics/Cell_Metrics.xlsx','Excel workbook')+'</td><td>'+'<br>'.join(counts)+'</td></tr>')
    body=(_nav()+'<p>Total localizations = On + Not-on + Unknown (QC-selected). QC excluded is separate. On percentage = On / (On + Not-on). '
          'First-mask area describes frames 0–49; mean ROI area describes the full movie. Cell equal weighting within BR; BR equal weighting within condition. On SD is in percentage points.</p>'
          '<table><tr><th>Level</th><th>Group</th><th>Workbook</th><th>Numerical source</th></tr>'+''.join(rows)+'</table>'+_footer(summary_id,updated))
    _write(root,'CELL_METRICS_TABLES.html',_page(molecule+' cell metrics: Excel tables',body))
    return dict(summary_id=summary_id,pages=[str(root/n) for n in ('INTRA_RESULTS.html','CELL_METRICS_RESULTS.html','CELL_METRICS_TABLES.html')],
                summary_groups=len(groups),cells=len(cells),removed=[str(p) for p in removed],scientific_stage_calls=0)
