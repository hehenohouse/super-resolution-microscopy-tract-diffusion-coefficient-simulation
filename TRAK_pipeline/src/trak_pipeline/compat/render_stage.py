#!/usr/bin/env python3
"""Static PNG deliverables from canonical data; no fitting or pair recomputation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

RELEASE = Path(__file__).resolve().parent
REPRESENTATIVE_FRAMES = [0,49,200,625,1200,1250,1875,2000,2450,2499]


def resolve_font(job):
    """Read an explicit/font-runtime path or a font installed on this host."""
    runtime=job.get('runtime',{})
    if not isinstance(runtime,dict):
        runtime={}
    candidates=[job.get('font_path'),runtime.get('font_path')]
    windows_dir=os.environ.get('WINDIR')
    if windows_dir:
        candidates.append(str(Path(windows_dir)/'Fonts'/'arial.ttf'))
    for candidate in candidates:
        if candidate and Path(str(candidate)).is_file():
            return str(Path(str(candidate)))
    import matplotlib
    fallback=Path(matplotlib.get_data_path())/'fonts'/'ttf'/'DejaVuSans.ttf'
    if not fallback.is_file():
        raise FileNotFoundError('No configured, Windows Arial, or installed Matplotlib font is available')
    return str(fallback)


def setup(job):
    from analysis_stage import configure, load_science
    configure(job); load_science()
    sys.path.insert(0, str(RELEASE/'vendor'))
    global np, pd, Image, ImageDraw, ImageFont, tifffile, sf, pc, plt
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    import tifffile
    import supplemental_figures as sf
    import point_stack_core as pc
    import matplotlib.pyplot as plt
    job['font_path']=resolve_font(job)


class ViewRoot:
    """Route frozen plot reads to canonical tables and PNG writes to flat figures."""
    def __init__(self, analysis, figures):
        self.analysis, self.figures = Path(analysis), Path(figures)
    def __truediv__(self, key):
        path = Path(key)
        return self.figures.joinpath(*path.parts[1:]) if path.parts[0]=='figures' else self.analysis/path


def public_png_files(folder):
    """Exclude Finder/AppleDouble metadata from the public figure inventory."""
    return [path for path in sorted(Path(folder).glob('*.png'))
            if not path.name.startswith('.') and path.is_file()]


def save_png(image, path):
    from batch_common import atomic_replace
    path=Path(path)
    temporary=path.with_name('.'+path.name+'.tmp.png')
    try:
        image.save(temporary,format='PNG')
        # Pillow has closed its output handle before the bounded Windows replace.
        atomic_replace(temporary,path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass  # Leave owned hidden scratch for the next safe cleanup.
    with Image.open(path) as saved:
        saved.load()
        if saved.size != image.size:
            raise ValueError('PNG dimensions changed')


def labeled_panel(panel, title, job, scale=4, mean_background=False):
    width, height = panel.shape[1], panel.shape[0]
    result = Image.new('RGB', (width, height+150), 'black')
    result.paste(Image.fromarray(panel), (0,80))
    draw = ImageDraw.Draw(result)
    font = resolve_font(job)
    sf.draw_text(draw, (12,8), job['sample_name']+' | all original frames 0-2499 pooled', font, width-24, 23)
    sf.draw_text(draw, (12,43), title, font, width-24, 24)
    first_footer = ('C1 raw-ADU arithmetic mean; fixed display 407-651 ADU; all pixels and zero values included.'
                    if mean_background else 'Frozen QC and spatial labels; repeated observations retained; fixed-size dots.')
    for y, text in [(height+90, first_footer),
                    (height+116, 'Identity registration unverified; dot size is not precision or molecular size.')]:
        sf.draw_text(draw, (12,y), text, font, width-24, 17)
    crop=job['crop_xyxy_px']
    bx, by = pc.display_centers([[crop[0]+14, crop[3]-6]],crop,scale)[0]
    length=min(2000/117*scale, max(10,width-2*bx))
    draw.line([(bx,80+by),(bx+length,80+by)],fill='white',width=3)
    draw.text((bx,80+by-23),f'{length/scale*117/1000:g} um',font=ImageFont.truetype(font,17),fill='white')
    return result


def four_panels(job, audit, detections, full, vertices, figures, sources, c1_pages):
    from analysis_stage import dump, tagged
    selected_frames=set(REPRESENTATIVE_FRAMES)
    fits=audit[audit.frame.astype(int).isin(selected_frames)].copy()
    det=detections[detections.frame.astype(int).isin(selected_frames)].copy()
    ids=set(fits.candidate_index.astype(str))
    det=det[det.candidate_index.astype(str).isin(ids)]
    if set(det.candidate_index.astype(str)) != ids or det.candidate_index.duplicated().any():
        raise ValueError('Representative ROI candidate identity mismatch')
    spec=dict(job, display_scale=2, display_C1_ADU=[407,651], display_C2_ADU=[406,554],
              pixel_size_nm=117, frame_interval_s=.02, TrackMate_circle_radius_native_px=3.,
              font_path=resolve_font(job))
    if 'postmin32_pointwise_label' not in fits:
        fits['postmin32_pointwise_label']=fits['pointwise_label']
    records=[]; rows=[]
    with tifffile.TiffFile(job['C2_file']) as movie:
        for frame in REPRESENTATIVE_FRAMES:
            f=fits[fits.frame.astype(int).eq(frame)].to_dict('records')
            d=det[det.frame.astype(int).eq(frame)].to_dict('records')
            by_id={str(row['candidate_index']):row for row in d}
            for row in f:
                match=by_id[str(row['candidate_index'])]
                if row.get('source_detection_uid','') != match.get('source_detection_uid',''):
                    raise ValueError('Representative source detection UID mismatch')
            raw2=movie.pages[frame].asarray()
            picture, coordinates, info=sf.render_frame(c1_pages[frame],raw2,full[frame//50],vertices,f,d,spec,frame)
            path=figures/f'frame_{frame:04d}_four_panels.png'; save_png(picture,path)
            origins=np.asarray(info['panel_origins_xy'])
            if len(np.unique(origins[:,0]))!=4 or len(np.unique(origins[:,1]))!=1:
                raise ValueError('Four-panel layout must be four horizontal columns')
            rows.extend(coordinates); records.append(dict(info,PNG=path.name))
    tagged(pd.DataFrame(rows),job).to_csv(sources/'representative_display_coordinates.csv.gz',index=False,compression='gzip')
    dump(sources/'four_panel_manifest.json',records)
    return records


def histograms(job, selected, figures, sources):
    from analysis_stage import dump, tagged
    labels=selected.tardis_group.to_numpy()
    masks={'All':np.ones(len(selected),bool),'On':labels=='On','Not-on':labels=='Not-on'}
    colors={'All':'#555555',**sf.COLORS}; edges=np.arange(0,205,5,dtype=float)
    values={field:selected[field].to_numpy(float) for field in ['x_se_nm','y_se_nm','photons_e']}
    photons=values['photons_e']; good=photons[np.isfinite(photons)&(photons>=0)]
    photon_edges=np.linspace(0,max(1.,float(good.max())) if len(good) else 1.,101)
    counts={}; outside={}; rows=[]
    fig,axes=plt.subplots(1,3,figsize=(16,5.3))
    for axis,field in zip(axes,values):
        bins=photon_edges if field=='photons_e' else edges
        for group,which in masks.items():
            h,info=sf.histogram_counts(values[field][which],bins)
            key=field+'__'+group; counts[key]=h; outside[key]=info
            axis.stairs(h,bins,color=colors[group],label=f"{group} (outside {info['below_range']+info['above_range']}; NA {info['nonfinite']})")
            rows.extend(dict(field=field,group=group,bin_index=i,left=float(bins[i]),right=float(bins[i+1]),count=int(n)) for i,n in enumerate(h))
        axis.set(xlabel='Nominal photoelectron-equivalent signal (e)' if field=='photons_e' else field+' (nm)',ylabel='Localizations',
                 xlim=(bins[0],bins[-1]),xscale='linear',yscale='linear')
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle(job['sample_name']+' | all QC-selected localizations | linear histograms')
    fig.tight_layout();fig.savefig(figures/'SE_and_signal_linear_histograms.png',dpi=180);plt.close(fig)
    maps={}
    for group,which in masks.items():
        maps[group],outside['SE_2D__'+group]=sf.histogram2d_counts(values['x_se_nm'][which],values['y_se_nm'][which],edges)
        counts['SE_2D__'+group]=maps[group]
    maximum=max(1,max(int(a.max()) for a in maps.values()))
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.3),constrained_layout=True)
    for axis,group in zip(axes,masks):
        picture=axis.pcolormesh(edges,edges,maps[group].T,shading='flat',cmap='viridis',vmin=0,vmax=maximum)
        info=outside['SE_2D__'+group]
        axis.set(xlabel='X SE (nm)',ylabel='Y SE (nm)',xlim=(0,200),ylim=(0,200),aspect='equal',
                 title=f"{group}: shown {info['displayed']}; outside {info['outside_range']}; NA {info['nonfinite']}")
    fig.colorbar(picture,ax=axes,label='Localizations per 5 nm x 5 nm bin (shared linear scale)')
    fig.suptitle(job['sample_name']+' | nominal per-axis model SE; no precision threshold')
    fig.savefig(figures/'SE_2D_All_On_NotOn_shared_linear.png',dpi=180);plt.close(fig)
    np.savez_compressed(sources/'histogram_counts_and_edges.npz',SE_edges_nm=edges,photons_edges_e=photon_edges,**counts)
    tagged(pd.DataFrame(rows),job).to_csv(sources/'histogram_1D_bin_counts.csv',index=False)
    rows2=[dict(group=group,x_bin=i,y_bin=j,x_left_nm=edges[i],x_right_nm=edges[i+1],y_left_nm=edges[j],y_right_nm=edges[j+1],count=int(maps[group][i,j]))
           for group in masks for i in range(len(edges)-1) for j in range(len(edges)-1)]
    tagged(pd.DataFrame(rows2),job).to_csv(sources/'histogram_2D_bin_counts.csv',index=False)
    metadata=dict(outside_counts=outside,SE_edges_nm=edges.tolist(),photon_edges_e=photon_edges.tolist(),
        shared_2D_vmin=0,shared_2D_vmax=maximum,color_normalization='linear shared across All/On/Not-on',
        All_includes_unknown=True,selected_count=len(selected),raw_values_source='data/localizations.csv.gz; original values retained',
        precision_selection='Original QC and fitted-center ROI only; no SE-size cutoff',signal_units='manufacturer-nominal photoelectron equivalents')
    dump(sources/'histogram_metadata.json',metadata)
    from axis_histograms import render_axis_histograms
    render_axis_histograms(sources/'histogram_counts_and_edges.npz', sources/'histogram_metadata.json', figures, job['sample_name'])
    return metadata


def c1_average(job, sources):
    """Exactly one full raw C1 traversal; uint64 sum then float64 division."""
    from analysis_stage import dump, array_sha
    frames=int(job.get('frames',2500));representatives={};page_rows=[]
    with tifffile.TiffFile(job['C1_file']) as movie:
        if len(movie.pages)!=frames:
            raise ValueError('C1 frame count mismatch')
        summed=np.zeros(movie.pages[0].shape,np.uint64)
        for frame,page in enumerate(movie.pages):
            raw=page.asarray()
            if raw.dtype!=np.uint16 or raw.shape!=summed.shape:
                raise ValueError('C1 page shape/dtype mismatch')
            summed+=raw
            if frame in REPRESENTATIVE_FRAMES:
                representatives[frame]=raw
                page_rows.append(dict(frame=frame,pixel_SHA256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()))
            if (frame+1)%250==0:print('C1_AVERAGE_FRAME',frame+1,flush=True)
    mean64=summed/frames;mean32=mean64.astype(np.float32)
    np.save(sources/'C1_mean_frames0000_2499.npy',mean32,allow_pickle=False)
    dump(sources/'C1_average2500_metadata.json',dict(source_C1=job['C1_file'],source_frames=frames,shape=list(summed.shape),
        all_source_pixels_included=True,zero_exclusion=False,alignment=False,filtering=False,
        source_page_hash_scope='Only the 10 approved representative pages; no full sequence fingerprint',
        representative_page_hashes=page_rows,sum_uint64_C_bytes_SHA256=array_sha(summed),
        mean_float64_C_bytes_SHA256=array_sha(mean64),mean_float32_C_bytes_SHA256=array_sha(mean32)))
    pd.DataFrame(page_rows).to_csv(sources/'C1_page_pixel_hashes.csv',index=False)
    return mean32,representatives


def count_grid(xy,crop,scale):
    x0,y0,x1,y1=crop
    xedges=np.linspace(x0-.5,x1-.5,(x1-x0)*scale+1)
    yedges=np.linspace(y0-.5,y1-.5,(y1-y0)*scale+1)
    result=np.histogram2d(xy[:,1],xy[:,0],bins=(yedges,xedges))[0].astype(np.uint32)
    if int(result.sum())!=len(xy):raise ValueError('Pooled crop loses a selected center')
    return result


def pooled(job, selected, mean, roi, vertices, figures, sources):
    from analysis_stage import dump
    crop,scale=job['crop_xyxy_px'],4
    width,height=(crop[2]-crop[0])*scale,(crop[3]-crop[1])*scale
    xy={group:selected.loc[selected.tardis_group.eq(group),['fit_x_px','fit_y_px']].to_numpy(float) for group in ['On','Not-on','unknown']}
    alphas={group:pc.marker_alpha((height,width),pc.display_centers(points,crop,scale),.2*scale) for group,points in xy.items()}
    maps={group:count_grid(points,crop,scale) for group,points in xy.items()}
    maps['All']=count_grid(selected[['fit_x_px','fit_y_px']].to_numpy(float),crop,scale)
    np.testing.assert_array_equal(maps['All'],maps['On']+maps['Not-on']+maps['unknown'])
    outline=pc.roi_alpha(vertices,crop,scale)
    base=pc.composite(np.zeros((height,width,3),np.uint8),outline,pc.ROI_COLOR)
    combined=pc.combined_layer(base,alphas['On'],alphas['Not-on'])
    combined=pc.composite(combined,alphas['unknown'],[160,160,160])
    panels=[pc.composite(base,alphas['On'],pc.ON),pc.composite(base,alphas['Not-on'],pc.OFF),combined]
    names=['pooled_On','pooled_NotOn','pooled_All_selected']
    titles=[f'On | {len(xy["On"]):,} localizations',f'Not-on | {len(xy["Not-on"]):,} localizations',f'All selected | {len(selected):,}; unknown {len(xy["unknown"]):,}']
    pictures=[]
    for panel,name,title in zip(panels,names,titles):
        picture=labeled_panel(panel,title,job,scale);save_png(picture,figures/(name+'.png'));pictures.append(picture)
    gap=16;triptych=Image.new('RGB',(sum(p.width for p in pictures)+2*gap,max(p.height for p in pictures)),'#101317')
    x=0
    for picture in pictures:triptych.paste(picture,(x,0));x+=picture.width+gap
    save_png(triptych,figures/'pooled_On_NotOn_All_triptych.png')
    gray=np.repeat(pc.gray_crop(mean,crop,scale,[407,651])[...,None],3,axis=2)
    x0,y0,x1,y1=crop;inside=np.repeat(np.repeat(roi[y0:y1,x0:x1],scale,0),scale,1);gray[~inside]=0
    average_base=pc.composite(gray,outline,pc.ROI_COLOR)
    overlay=pc.composite(average_base,alphas['Not-on'],pc.OFF)
    save_png(labeled_panel(overlay,f'Not-on | {len(xy["Not-on"]):,} centers | C1 mean of all 2500 raw frames',job,scale,mean_background=True),figures/'NotOn_on_C1_mean2500.png')
    np.savez_compressed(sources/'pooled_count_maps.npz',**maps,crop_xyxy_px=np.array(crop),nm_per_bin=np.array(117/scale))
    dump(sources/'pooled_metadata.json',dict(selected_count=len(selected),group_counts={k:len(v) for k,v in xy.items()},
        counts_preserve_repeated_observations=True,All_includes_unknown=True,dot_radius_native_px=.2,display_scale=scale,
        crop_xyxy_px=crop,mean_source='C1_mean_frames0000_2499.npy',source_centers='data/localizations.csv.gz',
        no_implicit_precision_threshold=True))


def tardis_figures(job, selected, analysis, figures, sources):
    from analysis_stage import tagged,dump
    import plots
    import postmin32_extras as extras
    c=json.loads((analysis/'metadata.json').read_text(encoding='utf-8'))['config_rules']
    summary=pd.read_csv(analysis/'per_scope_summary.csv',float_precision='round_trip')
    frames=pd.read_csv(analysis/'per_frame_counts.csv',float_precision='round_trip')
    effects=pd.read_csv(analysis/'descriptive_shape_differences.csv',float_precision='round_trip').to_dict('records')
    root=ViewRoot(analysis,figures)
    def read_csv(path,*args,**kwargs):
        if Path(path).name=='statistical_input.csv':
            columns=kwargs.get('usecols');return selected[columns].copy() if columns else selected.copy()
        return pd.read_csv(path,*args,**kwargs)
    namespace=dict(plots.all_figures.__globals__);namespace['pd']=types.SimpleNamespace(read_csv=read_csv)
    unchanged=types.FunctionType(plots.all_figures.__code__,namespace,plots.all_figures.__name__,plots.all_figures.__defaults__,plots.all_figures.__closure__)
    unchanged(root,c,summary,frames,effects)
    curves={group:pd.read_csv(analysis/'distribution'/group/'full.csv',float_precision='round_trip') for group in ['On','Not-on']}
    statistics=[extras._statistics(group,curves[group],c) for group in curves]
    tagged(pd.DataFrame(statistics),job).to_csv(sources/'intra_descriptive_statistics.csv',index=False)
    extras._annotated_intra(root,c,curves,statistics)
    area=pd.read_csv(analysis/'mask_area_by_frame.csv',float_precision='round_trip')
    fig,axes=plt.subplots(2,1,figsize=(13.5,8),sharex=True)
    for axis,scope,color in zip(axes,['ROI','full'],['#246B9E','#697F50']):
        values=area[scope+'_mask_area_um2'].to_numpy()
        axis.step(np.r_[area.frame.to_numpy(),c['frames']],np.r_[values,values[-1]],where='post',color=color,lw=1.1,label='50-frame support mask')
        axis.axhline(float(values.mean()),ls='--',lw=.8,color=color,label='Full-movie mean')
        axis.set(title=('Inside original cell ROI' if scope=='ROI' else 'Full image before ROI clipping')+f' | mean {values.mean():.2f} um²; median {np.median(values):.2f} um²',ylabel='Mask area (um²)',xlim=(0,c['frames']))
        axis.grid(alpha=.2);axis.legend(fontsize=9)
    seconds=axes[0].secondary_xaxis('top',functions=(lambda x:x*.02,lambda x:x/.02));seconds.set_xlabel('Acquisition time (s)')
    axes[1].set_xlabel('Original frame (0-based; steps end at exclusive frame 2500)')
    fig.suptitle(job['sample_name']+' | support50>=10 / post-min32 | Native mask area',fontsize=13)
    fig.subplots_adjust(left=.085,right=.97,top=.79,bottom=.16,hspace=.43)
    fig.text(.5,.075,'Area = foreground pixels x (0.117 um)^2. Masks update every 50 frames; no extra smoothing or endpoint fallback.',ha='center',fontsize=9)
    fig.text(.5,.035,'Binary mask area has no localization-error expansion and does not establish segmentation accuracy.',ha='center',fontsize=9)
    fig.savefig(figures/'09_mask_area.png',dpi=180);plt.close(fig)
    expected=['01_counts.png','02_full_decomposition.png','04_segment_shapes.png','05_shape_difference.png','06_time_summary.png','07_background_diagnostics.png','08_intra_with_statistics.png','09_mask_area.png']
    expected += ['03_intra_'+scope+'.png' for scope in summary.scope.unique()]
    if len(expected)!=19 or not all((figures/name).exists() for name in expected):raise ValueError('Missing TARDIS figure')
    return expected


def run(job):
    started=time.perf_counter();setup(job)
    from analysis_stage import selected_table,sha,dump,identity
    from batch_common import guard_path,descriptor,commit_stage
    if list(job['sample_frames']) != REPRESENTATIVE_FRAMES:raise ValueError('Representative-frame contract changed')
    output=guard_path(job['output']);figures=guard_path(job['figures']);figures.mkdir(parents=True,exist_ok=True)
    sources=output/'data/render_sources';sources.mkdir(parents=True,exist_ok=True)
    analysis=output/'data/tardis'
    if not json.loads((analysis/'COMPLETE.json').read_text(encoding='utf-8'))['all_passed']:raise ValueError('TARDIS analysis is not complete')
    input_paths=[output/'data/localizations.csv.gz',output/'data/all_detections.csv.gz',output/'masks/clean_full_mask.npy',
                 output/'masks/analysis_ROI_mask.npy',output/'masks/roi_mask.npy',output/'masks/roi_vertices_xy.npy',Path(job['C1_file']),Path(job['C2_file'])]
    before=[descriptor(p) for p in input_paths]
    audit=pd.read_csv(input_paths[0],dtype=str,keep_default_na=False)
    c=json.loads((analysis/'metadata.json').read_text(encoding='utf-8'))['config_rules'];selected,_=selected_table(audit,c)
    detections=pd.read_csv(output/'data/all_detections.csv.gz',dtype=str,keep_default_na=False)
    full=np.load(output/'masks/clean_full_mask.npy',allow_pickle=False)
    roi=np.load(output/'masks/roi_mask.npy',allow_pickle=False);vertices=np.load(output/'masks/roi_vertices_xy.npy',allow_pickle=False)
    mean,pages=c1_average(job,sources)
    representatives=four_panels(job,audit,detections,full,vertices,figures,sources,pages)
    histogram=histograms(job,selected,figures,sources)
    pooled(job,selected,mean,roi,vertices,figures,sources)
    names=tardis_figures(job,selected,analysis,figures,sources)
    after=[descriptor(p) for p in input_paths]
    if before!=after:raise ValueError('Protected rendering input changed')
    pngs=public_png_files(figures)
    if len(pngs)!=38:raise ValueError(f'Expected 38 flat PNGs, found {len(pngs)}')
    images=[]
    for path in pngs:
        with Image.open(path) as picture:picture.load();images.append(dict(name=path.name,size=list(picture.size),sha256=sha(path)))
    dump(sources/'render_metadata.json',dict(identity=identity(job),images=images,representative_frames=REPRESENTATIVE_FRAMES,
        input_fingerprints_before=before,input_fingerprints_after=after,all_passed=True,actual_visual_review_pending=True,
        actual_visual_review_passed=False,only_PNG_viewing_artifacts=True,TARDIS_figures=names,
        histogram=histogram,four_horizontal_columns=True,elapsed_seconds=time.perf_counter()-started))
    if 'scientific_fingerprint' in job:
        commit_stage(job,'render',pngs+[p for p in sorted(sources.rglob('*')) if p.is_file() and not any(part.startswith('.') for part in p.relative_to(sources).parts)],numerical_complete=True,visual_review_pending=True)
    print('RENDER_COMPLETE',str(figures),flush=True)
    return figures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    from batch_common import load_job,stage_done
    job=load_job(args.job)
    if stage_done(job,'render'):print('RENDER_ALREADY_COMPLETE',flush=True)
    else:run(job)


if __name__=='__main__':main()
