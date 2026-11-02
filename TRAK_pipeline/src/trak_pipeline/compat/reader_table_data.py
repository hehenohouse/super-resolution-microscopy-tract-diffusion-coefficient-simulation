"""Prepare reader-facing workbook data from existing scalar-metric CSVs only.

Workbook authoring and formatting belong to the spreadsheet adapter. Original
scientific data and CSV files are never modified by this module.
"""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,re
from localization_count_data import cell_count_records, summary_count_values, count_mean_records, AVERAGING_LABELS

METRICS=('on_percentage','first_mask_ROI_area_um2','mean_ROI_area_um2')
CONDITIONS=('WT','TDK','MDK')
CELL_HEADERS=['Condition','BR','Cell','Protein','Acquisition date','Total localizations','On count','Not-on count','On percentage','First-mask ROI area (µm²)','Mean ROI area (µm²)','Unknown count','QC excluded count']
CELL_TYPES=['text','text','text','text','date','integer','integer','integer','percent','decimal','decimal','integer','integer']
CELL_WIDTHS=[12,9,10,12,16,23,16,16,18,27,26,17,20]
COND_HEADERS=['Condition','Protein','Cells with valid On %','BRs with valid On %','Cells with counts','Localizations per cell','On per cell','Not-on per cell','Unknown per cell','QC excluded per cell','On mean (%)','On SD (pp)','First-mask ROI area mean (µm²)','First-mask area SD (µm²)','Mean ROI area mean (µm²)','Mean area SD (µm²)']
BR_HEADERS=['Condition','BR','Protein','Cells with valid On %','Cells with counts','Localizations per cell','On per cell','Not-on per cell','Unknown per cell','QC excluded per cell','On mean (%)','On SD (pp)','First-mask ROI area mean (µm²)','First-mask area SD (µm²)','Mean ROI area mean (µm²)','Mean area SD (µm²)']
SUMMARY_TYPES=['text','text','integer','integer','integer','decimal','decimal','decimal','decimal','decimal','percent','decimal','decimal','decimal','decimal','decimal']
BR_TYPES=['text','text','text','integer','integer','decimal','decimal','decimal','decimal','decimal','percent','decimal','decimal','decimal','decimal','decimal']
SUMMARY_WIDTHS=[13,12,22,21,20,27,21,23,23,25,18,16,29,26,28,24]
BR_WIDTHS=[13,9,12,22,20,27,21,23,23,25,18,16,29,26,28,24]
MEAN_HEADERS=['Condition','BR','Averaging','Cells','Replicas','Total localizations mean','On count mean','Not-on count mean']
MEAN_TYPES=['text','text','text','integer','integer','decimal','decimal','decimal']
MEAN_WIDTHS=[13,9,28,10,12,27,23,25]


def _path(root,value):
    root=Path(root).resolve();text=str(value).replace('\\','/');native=Path(text)
    if native.is_absolute() and native.is_relative_to(root):return native
    marker='/'+root.name+'/'
    if marker in text:text=text.rsplit(marker,1)[1]
    elif native.is_absolute() or re.match(r'^[A-Za-z]:',text):raise ValueError('Source is outside the portable data root')
    result=(root/text).resolve()
    if not result.is_relative_to(root):raise ValueError('Source path escapes the portable data root')
    return result


def _read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def _number(value):
    if value in (None,''):return None
    parsed=float(value)
    return parsed if math.isfinite(parsed) else None


def _int(value):
    parsed=_number(value)
    if parsed is None:return None
    if parsed!=int(parsed):raise ValueError('Count is not an integer: '+str(value))
    return int(parsed)


def _fraction(value):
    parsed=_number(value)
    return parsed/100.0 if parsed is not None else None


def _natural(value):return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)',str(value)))


def _condition_order(condition):
    return CONDITIONS.index(condition) if condition in CONDITIONS else len(CONDITIONS)


def _cell_order(row):return (_condition_order(row.get('genotype','')), _natural(row.get('BR','')), _natural(row.get('cell','')))


def _sheet(name,title,subtitle,headers,rows,types,widths,notes=''):
    if any(len(row)!=len(headers) for row in rows):raise ValueError('Reader-facing row/header width mismatch')
    return dict(name=name,title=title,subtitle=subtitle,notes=notes,headers=headers,rows=rows,column_types=types,widths=widths,header_row=6,data_start_row=7)


def _pivot(rows,keys):
    groups=defaultdict(dict)
    for row in rows:
        metric=row['metric']
        if metric not in METRICS:continue
        key=tuple(row.get(field,'') for field in keys)
        if metric in groups[key]:raise ValueError('Duplicate saved summary metric: '+str(key)+' '+metric)
        groups[key][metric]=row
    return groups


def _count_notes(grouped):
    lines=[]
    for identity,metrics in grouped.items():
        cellcounts=[_int(metrics.get(metric,{}).get('n_cells_valid')) for metric in METRICS]
        brcounts=[_int(metrics.get(metric,{}).get('n_BRs_valid')) for metric in METRICS]
        if len(set(cellcounts))>1 or len(set(brcounts))>1:
            lines.append('/'.join(identity)+': valid cells (On / first area / mean area) '+ '/'.join('NA' if value is None else str(value) for value in cellcounts)+'; valid BRs '+ '/'.join('NA' if value is None else str(value) for value in brcounts)+'.')
    return ' '.join(lines)


def _summary_values(metrics):
    on=metrics.get('on_percentage',{});first=metrics.get('first_mask_ROI_area_um2',{});mean=metrics.get('mean_ROI_area_um2',{})
    return [_fraction(on.get('mean')),_number(on.get('sd')),_number(first.get('mean')),_number(first.get('sd')),_number(mean.get('mean')),_number(mean.get('sd'))]


def build_reader_tables(node_record,data_root):
    """Return reader-facing sheets from saved cell/count/summary CSV records."""
    data_root=Path(data_root).resolve();data=_path(data_root,node_record.get('data',node_record.get('data_dir','')))
    level=node_record['level'];key=node_record['key']
    cells=sorted(_read_csv(data/'cell_metrics.csv'),key=_cell_order)
    count_records=cell_count_records(cells)
    summary=_read_csv(data/'summary_statistics.csv');brrows=_read_csv(data/'BR_statistics.csv')
    molecules=sorted({row.get('molecule','') for row in cells if row.get('molecule')});protein=', '.join(molecules) or node_record.get('molecule') or data_root.name.rsplit('_',1)[-1]
    tables=[]
    if level in ('condition','comparison'):
        grouped=_pivot(summary,['condition']);rows=[]
        for identity,metrics in sorted(grouped.items(),key=lambda item:_condition_order(item[0][0])):
            on=metrics.get('on_percentage',{})
            rows.append([identity[0],protein,_int(on.get('n_cells_valid')),_int(on.get('n_BRs_valid'))]+summary_count_values(count_records,identity[0])+_summary_values(metrics))
        notes=_count_notes(grouped)
        tables.append(_sheet('Condition Summary',protein+' | condition summary','Count columns are per-cell means across the condition; On% and area means give each BR equal weight, with SD between BR means.',COND_HEADERS,rows,SUMMARY_TYPES,SUMMARY_WIDTHS,notes))
    if level in ('BR','condition','comparison'):
        grouped=_pivot(brrows,['condition','BR']);rows=[]
        for identity,metrics in sorted(grouped.items(),key=lambda item:(_condition_order(item[0][0]),_natural(item[0][1]))):
            on=metrics.get('on_percentage',{})
            rows.append([identity[0],identity[1],protein,_int(on.get('n_cells_valid'))]+summary_count_values(count_records,identity[0],identity[1])+_summary_values(metrics))
        notes=_count_notes(grouped)
        tables.append(_sheet('BR Summary',protein+' | biological replicate summary','Per-cell count columns give each cell with saved counts equal weight; On% and area means use their valid cells, with SD between cell values.',BR_HEADERS,rows,BR_TYPES,BR_WIDTHS,notes))
    means=count_mean_records(count_records,level)
    if means:
        meanrows=[[row['condition'],row['BR'],AVERAGING_LABELS[row['averaging_code']],row['cells'],row['replicas'],row['total_localizations_mean'],row['on_count_mean'],row['not_on_count_mean']] for row in means]
        tables.append(_sheet('Localization Means',protein+' | localization count means','BR means give each available cell equal weight; condition means explicitly show pooled-cell and equal-replica averaging.',MEAN_HEADERS,meanrows,MEAN_TYPES,MEAN_WIDTHS,'Total = On + Not-on + Unknown; QC excluded is separate. Only cells with saved counts contribute. All-cell mean pools cells; Equal-replica mean averages BR cell means equally. These count means do not change On-percentage or area summary means.'))
    rows=[]
    for row in cells:
        rows.append([row.get('genotype'),row.get('BR'),row.get('cell'),row.get('molecule'),row.get('acquisition_date') or None,_int(row.get('selected')),_int(row.get('On')),_int(row.get('Not_on')),_fraction(row.get('on_percentage')),_number(row.get('first_mask_ROI_area_um2')),_number(row.get('mean_ROI_area_um2')),_int(row.get('unknown')),_int(row.get('QC_excluded'))])
    name='Cell Summary' if level=='cell' else 'Cells'
    tables.append(_sheet(name,protein+' | '+key+' | cell measurements','One row per cell. Empty numeric cells mean unavailable; values retain their saved precision.',CELL_HEADERS,rows,CELL_TYPES,CELL_WIDTHS))
    common='Total localizations = On + Not-on + Unknown (QC-selected observations); QC excluded is separate. Per-cell count columns are summed counts divided by Cells with counts, with each available cell weighted equally. On % = On / (On + Not-on). First mask: frames 0–49 (1 s support block); mean area: time-weighted over all 2,500 frames.'
    weights=' Cells have equal weight within each BR; BRs have equal weight within each condition. On SD is in percentage points (pp).'
    old=tables[0]['notes'];tables[0]['notes']=common+weights+(' '+old if old else '')
    for table in tables[1:]:
        if not table['notes']:table['notes']='Definitions and weighting are provided on the first worksheet.'
    return tables


def _fingerprint(root,path):
    path=Path(path);data=path.read_bytes()
    return dict(relative_path=path.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def build_all(control):
    """Prepare inventory-derived workbook jobs from current saved metric summaries; perform no analysis."""
    control=Path(control).resolve();root=control.parent;status_path=control/'cell_metrics_status.json'
    status=json.loads(status_path.read_text());jobs=[];sources={str(status_path):_fingerprint(root,status_path)}
    for node in status['nodes']:
        data=_path(root,node['data']);figures=_path(root,node['figures']);sheets=build_reader_tables(node,root)
        for filename in ['cell_metrics.csv','summary_statistics.csv','BR_statistics.csv']:
            path=data/filename;sources[str(path)]=_fingerprint(root,path)
        jobs.append(dict(molecule=status.get('version',{}).get('config',{}).get('molecule') or root.name.rsplit('_',1)[-1],key=node['key'],level=node['level'],output_path=str(figures/'Cell_Metrics.xlsx'),data_dir=str(data),sheets=sheets,scientific_data_unchanged=True))
    if len({job['output_path'] for job in jobs})!=len(jobs):raise ValueError('Duplicate workbook destination')
    return dict(schema_version=1,workbook_filename='Cell_Metrics.xlsx',jobs=jobs,job_count=len(jobs),sheet_count=sum(len(job['sheets']) for job in jobs),data_root=str(root),source_fingerprints=list(sources.values()),source_tables_preserved=True,scientific_stages_run=[],new_cells_started=0,column_type_formats=dict(text='General',date='yyyy-mm-dd',integer='#,##0',percent='0.00%',decimal='0.00'),notes='Reader-facing selections of saved scalar metric CSVs; no recalculation or scientific filtering.')
