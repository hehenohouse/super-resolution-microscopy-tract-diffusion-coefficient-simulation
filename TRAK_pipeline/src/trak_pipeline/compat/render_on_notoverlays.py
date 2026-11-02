#!/usr/bin/env python3
"""Overlay saved On/Not-on hierarchy curves, without scientific recalculation.

Writes only the On_NotOn_Overlay subfolder of each existing hierarchy gallery.
Parent galleries, source CSVs, metadata and completion markers remain read-only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import uuid

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parents[1] if HERE.parent.name=='releases' else HERE.parent
DATA_ROOT=CONTROL.parent
CONDITIONS=('WT','TDK','MDK')
GROUPS=('On','Not-on')
VIEWS=('signed','area','peak')
COLORS={'On':'#00A582','Not-on':'#9658CB'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def owned(path):
    path=Path(path).resolve()
    if not DATA_ROOT.is_dir() or not CONTROL.is_dir() or not path.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError('Drive-owned output path unavailable: '+str(path))
    from platform_runtime import assert_drive_root
    assert_drive_root(DATA_ROOT)
    return path


def atomic_text(path,text):
    path=owned(path);temp=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    temp.write_text(text,encoding='utf-8');os.replace(temp,path)


def source_ref(path):
    return dict(relative_path=Path(path).relative_to(DATA_ROOT).as_posix(),bytes=Path(path).stat().st_size,sha256=sha(path))


def load_node(parent,summary_id):
    source=parent/'analysis_results'/('intra_summary_'+summary_id)
    metadata_path=source/'metadata.json';csv_path=source/'curves_long.csv.gz';complete=source/'COMPLETE.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('summary_id')!=summary_id:raise ValueError('Summary ID mismatch: '+str(source))
    if not json.loads(complete.read_text(encoding='utf-8')).get('complete'):raise ValueError('Uncommitted hierarchy source: '+str(source))
    with gzip.open(csv_path,'rt',encoding='utf-8',newline='')as stream:rows=list(csv.DictReader(stream))
    return dict(parent=parent,source=source,metadata=metadata,rows=rows,refs=[source_ref(p)for p in [csv_path,metadata_path,complete]])


def select_curve(node,group,view,condition=None):
    import numpy as np
    compare=node['metadata']['level']=='comparison'
    rows=[r for r in node['rows']if r['spatial_group']==group and r['normalization']==view and
          r['series_kind']==('member'if compare else'mean') and(not compare or r['series_id']==condition)]
    if not rows:return None
    rows.sort(key=lambda r:float(r['distance_mid_um']))
    if len(rows)!=300:raise ValueError('Expected 300 saved bins: '+node['metadata']['node_key']+' '+group+' '+view)
    x=np.array([float(r['distance_mid_um'])for r in rows]);values=np.array([float(r['value'])for r in rows])
    if not np.allclose(x,(np.arange(300)+.5)*.01,rtol=0,atol=1e-12)or not np.isfinite(values).all():raise ValueError('Saved curve grid or values invalid')
    sds=[r.get('SD','')for r in rows]
    if all(s==''for s in sds):sd=None
    elif any(s==''for s in sds):raise ValueError('Saved BR SD is only partially available')
    else:
        sd=np.array([float(s)for s in sds])
        if not np.isfinite(sd).all()or(sd<0).any():raise ValueError('Saved SD invalid')
    ns={int(float(r['n']))for r in rows}
    if len(ns)!=1:raise ValueError('Saved cohort size changes across bins')
    return dict(x=x,values=values,sd=sd,n=ns.pop(),units=rows[0]['units'])


def curve_limits(curves):
    import numpy as np
    lows=[0.];highs=[0.]
    for curve in curves:
        if curve is None:continue
        spread=curve['sd']if curve['sd']is not None else 0.
        lows.append(float(np.min(curve['values']-spread)));highs.append(float(np.max(curve['values']+spread)))
    low,high=min(lows),max(highs)
    if low==high:return(-.05,.05)
    margin=.07*(high-low)
    return(low-margin,high+margin)


def render_node(node,summary_id):
    import matplotlib.pyplot as plt
    metadata=node['metadata'];comparison=metadata['level']=='comparison';progress=metadata['progress']
    destination=owned(node['parent']/'Analysis_Figures_Intra/On_NotOn_Overlay');destination.mkdir(parents=True,exist_ok=True)
    status=progress['status'];label=metadata['node_key'].replace('/',' / ')
    products=[];manifest_views={}
    for view in VIEWS:
        panels=list(CONDITIONS)if comparison else[None]
        curves={(panel,group):select_curve(node,group,view,panel)for panel in panels for group in GROUPS}
        limits=curve_limits(curves.values());manifest_views[view]=dict(y_limits=list(limits),normalization_range_um=[0,3],smoothing=False,negative_bins_retained=True)
        for range_name,xmax in [('full',3.),('zoom',.5)]:
            figure,axes=plt.subplots(1,len(panels),figsize=(16.5,5.6)if comparison else(11,6.4),squeeze=False,sharey=True)
            for ax,panel in zip(axes[0],panels):
                found=False;cohort_n=[]
                for group in GROUPS:
                    curve=curves[(panel,group)]
                    if curve is None:continue
                    found=True;cohort_n.append(curve['n']);ax.plot(curve['x'],curve['values'],color=COLORS[group],lw=2.2,label=group)
                    if curve['sd']is not None and(comparison or metadata['level']=='condition'):
                        ax.fill_between(curve['x'],curve['values']-curve['sd'],curve['values']+curve['sd'],color=COLORS[group],alpha=.16,linewidth=0)
                if not found:ax.text(.5,.5,'Unavailable: no eligible shared cohort',ha='center',va='center',transform=ax.transAxes)
                if comparison:title=(panel or'')+(' | valid BRs: '+str(cohort_n[0])if cohort_n else' | unavailable')
                elif metadata['level']=='condition':title=label+' | valid BRs: '+str(progress['valid_BRs'])+'/'+str(progress['expected_BRs'])
                else:title=label+' | included cells: '+str(progress['included'])+'/'+str(progress['expected'])
                ax.set_title(title,fontsize=12);ax.set_xlim(0,xmax);ax.set_ylim(*limits);ax.axhline(0,color='#777777',lw=.8)
                ax.set_xlabel('Distance (μm)',fontsize=12);ax.grid(alpha=.17)
                if found:ax.legend(frameon=True,fontsize=11)
            axes[0][0].set_ylabel('Peak-normalized signed shape (dimensionless)'if view=='peak'else('Signed Intra density (μm⁻¹)'if view=='signed'else'Signed-area-normalized shape (μm⁻¹)'),fontsize=12)
            view_label={'signed':'signed Intra','area':'signed-area normalization','peak':'positive-peak normalization'}[view]
            figure.suptitle(metadata.get('molecule','TRAK1')+' | On versus Not-on | '+view_label,fontsize=16,y=.975)
            figure.text(.5,.92,status+' | '+str(progress['completed'])+'/'+str(progress['expected'])+' cells completed; '+str(progress['included'])+' included',ha='center',fontsize=11,color='#52616D')
            figure.text(.5,.055,'Normalization and positive peaks use 0–3 μm. Shading: ±1 sample SD across BRs when available; no significance test.',ha='center',fontsize=10)
            figure.text(.5,.022,'On/Not-on denotes spatial compatibility. Manufacturer-nominal localization SE; C1/C2 identity registration independently unverified.',ha='center',fontsize=9,color='#52616D')
            figure.subplots_adjust(top=.83,bottom=.17,left=.085 if not comparison else.065,right=.985,wspace=.09)
            name='Intra_On_NotOn_'+view+'_'+range_name+'.png';path=destination/name;temp=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
            figure.savefig(temp,format='png',dpi=160,facecolor='white');plt.close(figure);os.replace(temp,path)
            products.append(dict(name=name,sha256=sha(path),bytes=path.stat().st_size,view=view,display_range_um=[0,xmax],y_limits=list(limits)))
    sources=node['refs'];links=[]
    for ref in sources:
        path=DATA_ROOT/ref['relative_path'];link=os.path.relpath(path,destination).replace(os.sep,'/')
        links.append('<li><a href="'+html.escape(link,quote=True)+'">'+html.escape(path.name)+'</a></li>')
    body=['<!doctype html><meta charset="utf-8"><title>On / Not-on overlay — '+html.escape(label)+'</title><style>body{font:16px system-ui;margin:24px;color:#17212b}img{max-width:100%;height:auto}figure{margin:24px 0}small{color:#52616d}</style>',
        '<h1>'+html.escape(label)+' — On / Not-on overlays</h1>',
        '<p>'+html.escape(status)+' | completed '+str(progress['completed'])+'/'+str(progress['expected'])+' cells; included '+str(progress['included'])+'; valid BRs '+str(progress['valid_BRs'])+'/'+str(progress['expected_BRs'])+'</p>',
        '<p>Saved canonical curves only; no scientific recalculation, smoothing, reweighting or renormalization. Negative bins retained. Between-BR sample SD is distinct from per-point nominal localization SE and within-movie time bootstrap.</p>',
        '<h2>Full-precision sources</h2><ul>'+''.join(links)+'</ul>']
    for product in products:body.append('<figure><figcaption>'+html.escape(product['name'])+'</figcaption><a href="'+product['name']+'"><img src="'+product['name']+'"></a></figure>')
    body.append('<small>Spatial compatibility does not prove mitochondrial binding or definitive cytoplasmic localization. C1/C2 identity registration remains unverified; no formal significance claims.</small>')
    atomic_text(destination/'INDEX.html','\n'.join(body))
    manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),summary_id=summary_id,node_key=metadata['node_key'],level=metadata['level'],progress=progress,
        sources=sources,views=manifest_views,images=products,curves_duplicated=False,scientific_stages_run=[],new_cells_started=0,
        source_completion_and_parent_gallery_unchanged=True,interpretation='Existing descriptive curves only; spatial compatibility and nominal calibration; no significance test')
    atomic_text(destination/'overlay_manifest.json',json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return dict(node_key=metadata['node_key'],gallery_relative=destination.relative_to(DATA_ROOT).as_posix(),images=len(products))


def portable_path(root,value):
    text=str(value).replace('\\','/');native=Path(text)
    if native.is_absolute() and native.is_relative_to(root):return native
    marker='/'+root.name+'/'
    if marker in text:text=text.rsplit(marker,1)[1]
    elif native.is_absolute() or (len(text)>1 and text[1]==':'):raise ValueError('Summary source is outside the portable data root')
    result=(root/text).resolve()
    if not result.is_relative_to(root):raise ValueError('Summary source escapes the portable data root')
    return result


def run(control,summary_id=None):
    """Render only committed canonical nodes; skip source-identical overlays."""
    global CONTROL,DATA_ROOT
    CONTROL=Path(control).resolve();DATA_ROOT=CONTROL.parent;owned(CONTROL)
    status_path=CONTROL/'intra_summary_status.json'
    if not status_path.is_file():raise FileNotFoundError('Canonical Intra status is absent: '+str(status_path))
    status=json.loads(status_path.read_text(encoding='utf-8'));summary_id=summary_id or status.get('summary_id')
    if not summary_id and status.get('status')!='disabled':raise ValueError('Canonical Intra summary ID is absent')
    results=[];errors=[];missing=[];code_sha=sha(Path(__file__))
    for item in status.get('nodes',[]):
        try:
            source=portable_path(DATA_ROOT,item['data']);metadata_path=source/'metadata.json'
            canonical=[metadata_path,source/'curves_long.csv.gz',source/'COMPLETE.json']
            absent=[str(path) for path in canonical if not path.is_file()]
            if absent:missing.extend(absent);raise FileNotFoundError('Missing canonical summary files: '+', '.join(absent))
            parent=portable_path(DATA_ROOT,item['figures']).parent
            node=load_node(parent,summary_id)
            signature=hashlib.sha256(json.dumps(dict(code_SHA256=code_sha,source_refs=node['refs']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            destination=owned(parent/'Analysis_Figures_Intra/On_NotOn_Overlay');manifest_path=destination/'overlay_manifest.json'
            previous=json.loads(manifest_path.read_text())if manifest_path.is_file()else{}
            unchanged=previous.get('signature')==signature and len(previous.get('images',[]))==6
            if unchanged:
                for image in previous['images']:
                    path=destination/image['name']
                    if not path.is_file() or path.stat().st_size!=image['bytes'] or sha(path)!=image['sha256']:unchanged=False;break
                unchanged=unchanged and (destination/'INDEX.html').is_file()
            if unchanged:result=dict(node_key=node['metadata']['node_key'],gallery_relative=destination.relative_to(DATA_ROOT).as_posix(),images=6,status='unchanged')
            else:
                result=render_node(node,summary_id);result['status']='updated'
                manifest=json.loads(manifest_path.read_text());manifest['signature']=signature;manifest['code_SHA256']=code_sha
                atomic_text(manifest_path,json.dumps(manifest,indent=2,allow_nan=False)+'\n')
            if any(not (DATA_ROOT/ref['relative_path']).is_file() or sha(DATA_ROOT/ref['relative_path'])!=ref['sha256']for ref in node['refs']):raise ValueError('Canonical source changed during overlay refresh; retry required')
            results.append(result);print('OVERLAY '+result['node_key']+' | '+result['status'],flush=True)
        except OSError as error:
            if error.errno in (5,6,19,28,30,57,60,74,121,122) or not DATA_ROOT.is_dir():raise
            errors.append(dict(node=item.get('key'),error=str(error),missing_path=getattr(error,'filename',None)))
        except (ValueError,KeyError,TypeError)as error:errors.append(dict(node=item.get('key'),error=str(error)))
    report=dict(schema_version=1,all_passed=not errors,completed_utc=datetime.now(timezone.utc).isoformat(),summary_id=summary_id,nodes=results,errors=errors,missing_paths=sorted(set(missing)),
        total_png=6*len(results),protected_sources_and_parent_galleries_unchanged=True,scientific_stages_run=[],new_cells_started=0,
        source='Committed canonical Intra summaries; no new membership or normalization calculation')
    atomic_text(CONTROL/'on_not_overlay_status.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    global CONTROL,DATA_ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control',type=Path,default=CONTROL);parser.add_argument('--summary-id');parser.add_argument('--completed-only',action='store_true',required=True);args=parser.parse_args()
    CONTROL=args.control.resolve();DATA_ROOT=CONTROL.parent
    runtime=owned(CONTROL/'runtime/on_not_overlay')
    for variable,folder in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','cache')]:
        target=runtime/folder;target.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(target)
    os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    try:report=run(CONTROL,args.summary_id)
    except OSError as error:
        print('OVERLAY_ERROR '+str(error),flush=True)
        return 74 if error.errno in (5,6,19,28,30,57,60,74,121,122) or not DATA_ROOT.is_dir()else 1
    return 0 if report['all_passed']else 1


if __name__=='__main__':raise SystemExit(main())
