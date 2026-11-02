"""Cached point-assignment sensitivity plots; no scientific recomputation.

Public entry: render_outputs(manifest_path). All outputs go to the new run's
figure/data folders. Source masks, localizations, QC, and figures are read only.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import html
import importlib.util
import json
import os
import sys
import uuid

REPRESENTATIVE_FRAMES = [0,49,200,625,1200,1250,1875,2000,2450,2499]
LABELS = {'mito-compatible':'On','non-mito-compatible':'Not-on','unknown':'unknown',
          'On':'On','Not-on':'Not-on','Not_on':'Not-on'}


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)


def _true(value):return value is True or str(value).lower() in ('true','1')


def _atomic_json(path, value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _write_csv(path,rows,fields):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def _save_image(image,path):
    path=Path(path);temporary=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.tmp.png')
    try:image.save(temporary,format='PNG');os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _save_figure(figure,path,plt):
    path=Path(path);temporary=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.tmp.png')
    try:figure.savefig(temporary,dpi=180,facecolor='white');os.replace(temporary,path)
    finally:
        plt.close(figure)
        if temporary.exists():temporary.unlink()


def _font(job):
    choices=[job.get('font_path'),job.get('fit',{}).get('font_path'),'/System/Library/Fonts/Supplemental/Arial.ttf','C:/Windows/Fonts/arial.ttf']
    for choice in choices:
        if choice and Path(choice).is_file():return str(choice)
    raise FileNotFoundError('No approved display font found in job or system Arial paths')


def render_outputs(manifest_path):
    manifest_path=Path(manifest_path).resolve();manifest=_json(manifest_path)
    output=Path(manifest['output']).resolve();figures=Path(manifest['figures']).resolve()
    source=Path(manifest['source_output']).resolve();release=Path(manifest['source_release']).resolve()
    # Write only into the isolated run directories, never the source result.
    if output==source or figures==source or source.is_relative_to(figures):raise ValueError('Sensitivity outputs must be independent of source result')
    output.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True);sources=output/'plot_sources';sources.mkdir(exist_ok=True)
    cache=output/'runtime';cache.mkdir(exist_ok=True);os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import tifffile
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    pc=_module('point_stack_core',release/'vendor/mito_tardis/point_stack_core.py')
    sf=_module('assignment_supplemental_figures',release/'vendor/supplemental_figures.py')
    renderer=_module('assignment_frozen_render_stage',release/'render_stage.py')
    renderer.np=np;renderer.pc=pc;renderer.sf=sf;renderer.Image=Image;renderer.ImageDraw=ImageDraw;renderer.ImageFont=ImageFont
    job=_json(source/'job.json');job=dict(job,crop_xyxy_px=manifest['crop_xyxy_px'],font_path=_font(job),sample_name=manifest.get('sample_name',job['sample_name']))
    methods=manifest['methods'];summary=_json(output/'summary.json');by_method={m['key']:m for m in summary['methods']}
    vertices=np.load(source/'roi/vertices.npy',allow_pickle=False)
    full=np.load(source/'masks/clean_full_mask.npy',mmap_mode='r',allow_pickle=False)
    with gzip.open(output/'data/pointwise_comparison.csv.gz','rt',encoding='utf-8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if _true(r.get('spatial_selected',r.get('selected',False)))]
    xy=np.asarray([[float(row['fit_x_px']),float(row['fit_y_px'])] for row in rows],float).reshape(-1,2)
    frames=np.asarray([int(row['frame']) for row in rows],int)
    crop=job['crop_xyxy_px'];scale=4;width=(crop[2]-crop[0])*scale;height=(crop[3]-crop[1])*scale
    roi=pc.roi_alpha(vertices,crop,scale);base=pc.composite(np.zeros((height,width,3),np.uint8),roi,pc.ROI_COLOR)
    colors={'On':pc.ON,'Not-on':pc.OFF,'unknown':np.asarray([160,160,160],float)}
    labels={method['key']:np.asarray([LABELS[row[method['field']]] for row in rows]) for method in methods}
    pooled_counts=[];files=[];point_hash=hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest()
    for method in methods:
        key=method['key'];groups={g:labels[key]==g for g in ('On','Not-on','unknown')}
        counts={g:int(which.sum()) for g,which in groups.items()}
        expected=by_method[key]
        if counts['On']!=expected['On'] or counts['Not-on']!=expected['Not_on'] or counts['unknown']!=expected['unknown'] or sum(counts.values())!=expected['selected']:
            raise ValueError('Pooled plotted counts do not match saved method summary: '+key)
        alpha={g:pc.marker_alpha((height,width),pc.display_centers(xy[which],crop,scale),.20*scale) for g,which in groups.items()}
        # On/Not-on blending is commutative; unknown is retained as a gray layer.
        combined=pc.combined_layer(base,alpha['On'],alpha['Not-on']);combined=pc.composite(combined,alpha['unknown'],colors['unknown'])
        panels=[pc.composite(base,alpha['On'],colors['On']),pc.composite(base,alpha['Not-on'],colors['Not-on']),combined]
        titles=[f"On | {counts['On']:,} localizations",f"Not-on | {counts['Not-on']:,} localizations",f"All selected | {len(rows):,}; unknown {counts['unknown']:,}"]
        pictures=[];display_job=dict(job,sample_name=job['sample_name']+' | '+method['label'])
        for panel,title in zip(panels,titles):
            picture=renderer.labeled_panel(panel,title,display_job,scale=scale)
            draw=ImageDraw.Draw(picture);draw.rectangle([0,height+80,picture.width,picture.height],fill='black')
            sf.draw_text(draw,(12,height+90),'Original QC and centers; repeated observations retained; fixed-size dots.',job['font_path'],picture.width-24,17)
            sf.draw_text(draw,(12,height+116),'Identity registration unverified; dot size is not precision or molecular size.',job['font_path'],picture.width-24,17)
            pictures.append(picture)
        gap=16;triptych=Image.new('RGB',(3*width+2*gap,max(p.height for p in pictures)),'#101317')
        for i,picture in enumerate(pictures):triptych.paste(picture,(i*(width+gap),0))
        filename=f'pooled_{key}_On_NotOn_All_triptych.png';_save_image(triptych,figures/filename);files.append(filename)
        pooled_counts.extend(dict(method=key,label=method['label'],group=g,count=counts[g]) for g in counts)
        print('Distance sensitivity pooled figure:',key,counts,flush=True)
    _write_csv(sources/'pooled_plotted_counts.csv',pooled_counts,['method','label','group','count'])
    # Saved scalar counts only; no reclassification is done in plotting.
    fig,axes=plt.subplots(1,2,figsize=(13.6,5.5));x=np.arange(len(methods));percentages=[by_method[m['key']]['On_percentage'] for m in methods]
    bars=axes[0].bar(x,percentages,color=['#607d8b','#27b98a','#137b60']);axes[0].set_xticks(x,[m['label'].replace(' close ','\nclose ') for m in methods],fontsize=9)
    axes[0].set(ylim=(0,100),ylabel='On / (On + Not-on) (%)',title='On fraction across assignment rules')
    for bar,value in zip(bars,percentages):
        if value is not None:axes[0].text(bar.get_x()+bar.get_width()/2,float(value)+1,f'{float(value):.2f}%',ha='center',fontsize=10)
    flips=summary.get('flips',[]);fx=np.arange(len(flips));span=.25
    for offset,field,color,label in [(-span,'On_to_Not_on','#e78b45','On → Not-on'),(0,'Not_on_to_On','#27b98a','Not-on → On'),(span,'other_transition_count','#999999','Other / unknown')]:
        values=[row.get(field,0) for row in flips];bars=axes[1].bar(fx+offset,values,width=span,color=color,label=label)
        for bar,value in zip(bars,values):
            if value:axes[1].text(bar.get_x()+bar.get_width()/2,float(value),f'{value:,}',ha='center',va='bottom',fontsize=8)
    axes[1].set_xticks(fx,[str(r['comparison']).replace('Original k=1 SE ellipse','SE k=1').replace('1-pixel close (117 nm)','1 px').replace('2-pixel close (234 nm)','2 px').replace('_vs_',' → ') for r in flips],fontsize=8);axes[1].set(ylabel='Selected localizations',title='Bidirectional classification changes');axes[1].legend(fontsize=9)
    for axis in axes:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    fig.suptitle(job['sample_name']+' | fixed-distance assignment sensitivity',fontsize=16)
    fig.text(.5,.02,'Same coordinates, errors, QC, and mask. Unknown is outside the On-percentage denominator. No TARDIS recomputation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]);name='On_percentage_and_bidirectional_flips.png';_save_figure(fig,figures/name,plt);files.append(name)
    _write_csv(sources/'On_percentage_plot_source.csv',[dict(method=m['key'],label=m['label'],On=by_method[m['key']]['On'],Not_on=by_method[m['key']]['Not_on'],unknown=by_method[m['key']]['unknown'],On_percentage=by_method[m['key']]['On_percentage']) for m in methods],['method','label','On','Not_on','unknown','On_percentage'])
    if flips:_write_csv(sources/'flip_plot_source.csv',flips,list(flips[0]))
    # Ten original C1 pages; all method panels use exactly the same gray pixels.
    qscale=2;qw=(crop[2]-crop[0])*qscale;qh=(crop[3]-crop[1])*qscale;qroi=pc.roi_alpha(vertices,crop,qscale)
    page_records=[]
    with tifffile.TiffFile(job['C1_file']) as movie:
        for frame in REPRESENTATIVE_FRAMES:
            raw=movie.pages[frame].asarray();block=frame//50;mask=full[block]
            gray=np.repeat(pc.gray_crop(raw,crop,qscale,[407,651])[...,None],3,axis=2)
            mask_alpha=pc.mask_contour_alpha(mask,crop,qscale);c1=pc.composite(pc.composite(gray,mask_alpha,pc.MASK_COLOR),qroi,pc.ROI_COLOR)
            canvas=Image.new('RGB',(3*qw+32,qh+150),'black');draw=ImageDraw.Draw(canvas)
            sf.draw_text(draw,(12,8),f"{job['sample_name']} | C1 + original full-mask contour | frame {frame} | {frame*.02:.2f} s | block {block}",job['font_path'],canvas.width-24,23)
            which_frame=frames==frame;counts_frame={}
            for i,method in enumerate(methods):
                key=method['key'];flabels=labels[key][which_frame];points=xy[which_frame]
                alpha={g:pc.marker_alpha((qh,qw),pc.display_centers(points[flabels==g],crop,qscale),3*qscale,stroke=1.25) for g in ('On','Not-on','unknown')}
                panel=pc.combined_layer(c1,alpha['On'],alpha['Not-on']);panel=pc.composite(panel,alpha['unknown'],colors['unknown'])
                counts={g:int((flabels==g).sum()) for g in ('On','Not-on','unknown')};counts_frame[key]=counts
                px=i*(qw+16);canvas.paste(Image.fromarray(panel),(px,80))
                sf.draw_text(draw,(px+8,45),method['label']+f" | On {counts['On']}; Not-on {counts['Not-on']}; unknown {counts['unknown']}",job['font_path'],qw-16,18)
                bx,by=pc.display_centers([[crop[0]+14,crop[3]-6]],crop,qscale)[0];length=2000/117*qscale
                draw.line([(px+bx,80+by),(px+bx+length,80+by)],fill='white',width=2)
                draw.text((px+bx,80+by-20),'2 um',font=ImageFont.truetype(job['font_path'],15),fill='white')
            sf.draw_text(draw,(12,qh+91),'C1: 407–651 ADU | Red: unchanged full-mask boundary | Gray: ROI | Green: On | Purple: Not-on | Gray points: unknown',job['font_path'],canvas.width-24,17)
            sf.draw_text(draw,(12,qh+116),'Circles: radius 3 native pixels for display only; not localization errors. Original QC-selected centers only; C1/C2 identity registration unverified.',job['font_path'],canvas.width-24,17)
            filename=f'frame_{frame:04d}_C1_mask_assignment_comparison.png';_save_image(canvas,figures/filename);files.append(filename)
            page_records.append(dict(frame=frame,time_s=frame*.02,mask_block=block,counts=counts_frame,C1_pixels_SHA256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),png=filename))
    result=dict(sample_id=manifest.get('sample_id'),sample_name=job['sample_name'],source_manifest=str(manifest_path),files=files,
        pooled_counts=pooled_counts,selected_count=len(rows),selected_centers_SHA256=point_hash,representative_frames=page_records,
        crop_xyxy_px=crop,pooled_display_scale=4,pooled_marker_radius_native_px=.20,representative_display_scale=2,
        representative_circle_radius_native_px=3,raw_C1_display_ADU=[407,651],full_mask_unchanged=True,
        unknown_display_color='gray; excluded from On percentage denominator',repeated_observations_retained=True,
        On_Not_on_blending='commutative combined_layer',scientific_stages_run=[],new_TARDIS_calculations=0)
    _atomic_json(output/'plot_manifest.json',result)
    links=''.join('<li><a href="'+html.escape(n,quote=True)+'">'+html.escape(n)+'</a></li>' for n in files)
    preview=''.join('<h2>'+html.escape(m['label'])+'</h2><a href="pooled_'+m['key']+'_On_NotOn_All_triptych.png"><img src="pooled_'+m['key']+'_On_NotOn_All_triptych.png" style="width:100%"></a>' for m in methods)
    workbook_relative=os.path.relpath(output/'Distance_Comparison.xlsx',figures).replace(os.sep,'/')
    workbook_link='<p><a href="'+html.escape(workbook_relative,quote=True)+'">Download Distance_Comparison.xlsx</a></p>'
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>Assignment-distance sensitivity</title><style>body{font:16px system-ui;margin:28px;max-width:1600px}img{height:auto}a{color:#2166ac}</style><h1>'+html.escape(job['sample_name'])+': assignment-distance sensitivity</h1><p>Identical source QC-selected points and masks. Only the assignment rule changes. Repeated localization observations remain retained. No TARDIS recomputation.</p>'+workbook_link+preview+'<h2>All figures</h2><ul>'+links+'</ul></html>'
    index=figures/'INDEX.html';temporary=index.with_name('.INDEX.'+uuid.uuid4().hex+'.tmp')
    try:temporary.write_text(document,encoding='utf-8');os.replace(temporary,index)
    finally:
        if temporary.exists():temporary.unlink()
    print('Distance sensitivity figures complete:',len(files),flush=True)
    return result
