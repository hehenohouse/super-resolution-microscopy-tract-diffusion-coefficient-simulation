"""Pure publisher: plot saved numerics, never calculate missing science.

No dependency on scientific execute functions exists in this module. Missing
statistics/decomposition/bands are reported; refresh never creates them.
"""
from __future__ import annotations
import html, json, os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from ..types import StageResult
from ..display import resolve_c1_display
from ..io import replace as atomic_replace
from .distances import dump,load_json,GROUPS
from .statistics import REPRESENTATIVE_FRAMES

COLORS={'All':'#555555','On':'#00B894','Not-on':'#A45BDB','unknown':'#AAAAAA'}


def required_cell_pngs(frames=2500,segment_frames=250):
    names={f'frame_{f:04d}_four_panels.png' for f in REPRESENTATIVE_FRAMES if f<frames}
    names.update(['SE_and_signal_linear_histograms.png','SE_2D_All_On_NotOn_shared_linear.png','SE_X_All_On_NotOn_linear.png','SE_Y_All_On_NotOn_linear.png','pooled_On.png','pooled_NotOn.png','pooled_All_selected.png','pooled_On_NotOn_All_triptych.png','NotOn_on_C1_mean2500.png','01_counts.png','02_full_decomposition.png','04_segment_shapes.png','05_shape_difference.png','06_time_summary.png','07_background_diagnostics.png','08_intra_with_statistics.png','09_mask_area.png','03_intra_full.png'])
    names.update(f'03_intra_f{s:04d}_{s+segment_frames-1:04d}.png' for s in range(0,frames,segment_frames))
    return names


def _directory(ctx,stage):
    if stage not in ctx.inputs:raise FileNotFoundError('Saved '+stage+' artifact missing; refresh does not run scientific stages')
    path=ctx.inputs[stage].files.get('directory')
    if not path or not Path(path).is_dir():raise FileNotFoundError('Saved '+stage+' directory missing; refresh does not run scientific stages')
    return Path(path)


def _read(path):
    if not Path(path).is_file():raise FileNotFoundError('Missing scientific product: '+str(path)+'; run an explicit scientific stage, not refresh')
    return pd.read_csv(path,float_precision='round_trip')


def _finish(fig,path,footer=None):
    import matplotlib.pyplot as plt
    if footer:fig.text(.5,.015,footer,ha='center',fontsize=8,color='white' if sum(fig.get_facecolor()[:3])<1.5 else 'black')
    fig.tight_layout(rect=(0,.055 if footer else 0,1,.95));temp=Path(path).with_name('.'+Path(path).name+'.tmp.png');fig.savefig(temp,dpi=180);plt.close(fig);atomic_replace(temp,path)


def _curves(decomp,c):
    scopes=['full']+[f'f{s:04d}_{s+c["segment_frames"]-1:04d}' for s in range(0,c['frames'],c['segment_frames'])]
    return {scope:{group:_read(decomp/'distribution'/group/(scope+'.csv')) for group in GROUPS} for scope in scopes}


def _scatter(ax,coords,vertices,crop,title,mean=None,pixel_size_nm=117,display_range=(407,651),dot_radius_px=.2):
    x0,y0,x1,y1=crop
    ax.set_facecolor('black')
    if mean is not None:ax.imshow(mean,cmap='gray',vmin=display_range[0],vmax=display_range[1],origin='upper',extent=(-.5,mean.shape[1]-.5,mean.shape[0]-.5,-.5),interpolation='nearest')
    ax.plot(vertices[:,0],vertices[:,1],color='#777777',lw=.4)
    for group,xy in coords.items():
        if len(xy):ax.scatter(xy[:,0],xy[:,1],s=7.5*dot_radius_px**2,color=COLORS[group],linewidths=0,rasterized=True)
    ax.set(xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5),aspect='equal',title=title);ax.axis('off')
    bar=2000/pixel_size_nm;ax.plot([x0+12,x0+12+bar],[y1-6,y1-6],color='white',lw=2);ax.text(x0+12,y1-9,'2 µm',color='white',fontsize=8)


def execute(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True)
    style=ctx.config.get('style',{})
    COLORS.update({'On':style.get('on_color','#00B894'),'Not-on':style.get('not_on_color','#A45BDB'),'unknown':style.get('unknown_color','#AAAAAA')})
    plt.rcParams.update({'font.size':style.get('font_size',10)})
    stats=_directory(ctx,'statistics');decomp=_directory(ctx,'decomposition');bs=_directory(ctx,'bootstrap')
    metadata=load_json(stats/'metadata.json') if (stats/'metadata.json').exists() else ctx.input('statistics').metadata
    c=metadata.get('config_rules') or load_json(decomp/'metadata.json')['config_rules']
    histpath=stats/'histogram_counts_and_edges.npz';histmeta=stats/'histogram_metadata.json'
    for path in (histpath,histmeta,stats/'per_frame_counts.csv',stats/'mask_area_by_frame.csv',stats/'pooled_coordinates.npz'):
        if not path.is_file():raise FileNotFoundError('Missing scientific product: '+str(path)+'; refresh calls no science')
    with np.load(histpath,allow_pickle=False) as saved:h={k:saved[k].copy() for k in saved.files}
    hm=load_json(histmeta);curves=_curves(decomp,c);summary=_read(decomp/'per_scope_summary.csv');frame=_read(stats/'per_frame_counts.csv');area=_read(stats/'mask_area_by_frame.csv');scopes=list(curves)
    figdir=out;figdir.mkdir(exist_ok=True);sample=c.get('sample_name',ctx.sample_id)
    for field,axisname in [('x_se_nm','X'),('y_se_nm','Y')]:
        fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
        for ax,group in zip(axes,('All',*GROUPS)):
            ax.stairs(h[field+'__'+group],h['SE_edges_nm'],color=COLORS[group]);info=hm['outside_counts'][field+'__'+group];ax.set(title=f"{group} | shown {info['displayed']}; outside {info['below_range']+info['above_range']}; NA {info['nonfinite']}",xlabel=axisname+' nominal model SE (nm)',ylabel='Localizations',xlim=(0,200));ax.grid(alpha=.2)
        fig.suptitle(sample+' | '+axisname+' SE | no precision threshold');_finish(fig,figdir/f'SE_{axisname}_All_On_NotOn_linear.png','Manufacturer-nominal model SE; independent per-axis ranges.')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,field,bins in zip(axes,('x_se_nm','y_se_nm','photons_e'),(h['SE_edges_nm'],h['SE_edges_nm'],h['photons_edges_e'])):
        for group in ('All',*GROUPS):ax.stairs(h[field+'__'+group],bins,color=COLORS[group],label=group)
        ax.set(xlabel=field+' (nm)' if field!='photons_e' else 'Nominal signal (electron equivalents)',ylabel='Localizations');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(sample+' | linear nominal uncertainty and signal');_finish(fig,figdir/'SE_and_signal_linear_histograms.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,group in zip(axes,('All',*GROUPS)):
        image=ax.pcolormesh(h['SE_edges_nm'],h['SE_edges_nm'],h['SE_2D__'+group].T,cmap='viridis',vmin=0,vmax=hm['shared_2D_vmax']);info=hm['outside_counts']['SE_2D__'+group];ax.set(title=f"{group}: shown {info['displayed']}; outside {info['outside_range']}; NA {info['nonfinite']}",xlabel='X SE (nm)',ylabel='Y SE (nm)',aspect='equal')
    fig.colorbar(image,ax=axes,label='Localizations per 5 nm × 5 nm bin');fig.suptitle(sample+' | shared linear scale');fig.savefig(figdir/'SE_2D_All_On_NotOn_shared_linear.png',dpi=180);plt.close(fig)
    with np.load(stats/'pooled_coordinates.npz',allow_pickle=False) as saved:pooled={k:saved[k].copy() for k in saved.files}
    # Resolve the C1 viewing window once per cell. Raw pixels and all scientific
    # inputs stay read-only; the request's style/parameters remain untouched.
    requested_style=dict(style)
    if requested_style.get('C1_display_mode','fixed')=='roi_percentile':
        import tifffile
        sample_frames=requested_style.get('C1_display_sample_frames',ctx.config.get('sample_frames',REPRESENTATIVE_FRAMES))
        with tifffile.TiffFile(ctx.sample.C1_file) as source_c1:
            c1_display=resolve_c1_display(requested_style,frame_ids=sample_frames,
                frame_reader=lambda f:source_c1.pages[f].asarray(),roi_mask=pooled['roi_mask'])
    else:
        c1_display=resolve_c1_display(requested_style)
    style=dict(requested_style,C1_display_ADU=c1_display['limits_ADU'])
    c1_footer=(f"C1 display {style['C1_display_ADU'][0]:g}-{style['C1_display_ADU'][1]:g} ADU "
               f"({c1_display['mode']}); fixed within this cell.\nDisplay only; raw pixels, masks, labels and analysis unchanged.")
    dump(out/'display_ranges.json',dict(C1=c1_display,C2=dict(mode='fixed',limits_ADU=style.get('C2_display_ADU',[406,554])),scientific_stage_calls=0))
    crop=pooled['crop_xyxy_px'];vertices=pooled['vertices'];pictures=[]
    for name,groups in [('pooled_On',('On',)),('pooled_NotOn',('Not-on',)),('pooled_All_selected',('On','Not-on','unknown'))]:
        fig,ax=plt.subplots(figsize=(7,9),facecolor='black');_scatter(ax,{g:pooled[g] for g in groups},vertices,crop,sample+' | '+', '.join(f'{g}: {len(pooled[g]):,}' for g in groups),pixel_size_nm=c['pixel_size_nm'],dot_radius_px=style.get('dot_radius_px',.2));ax.title.set_color('white');_finish(fig,figdir/(name+'.png'))
    fig,axes=plt.subplots(1,3,figsize=(18,9),facecolor='black')
    for ax,groups in zip(axes,[('On',),('Not-on',),('On','Not-on','unknown')]):_scatter(ax,{g:pooled[g] for g in groups},vertices,crop,', '.join(f'{g}: {len(pooled[g]):,}' for g in groups),pixel_size_nm=c['pixel_size_nm'],dot_radius_px=style.get('dot_radius_px',.2));ax.title.set_color('white')
    _finish(fig,figdir/'pooled_On_NotOn_All_triptych.png')
    meanpath=ctx.inputs.get('c1_mean').files.get('mean') if 'c1_mean' in ctx.inputs else None
    if not meanpath or not Path(meanpath).is_file():raise FileNotFoundError('Missing saved C1 mean; refresh never averages images')
    mean=np.load(meanpath,allow_pickle=False).copy();mean[~pooled['roi_mask']]=0
    fig,ax=plt.subplots(figsize=(7,9),facecolor='black');_scatter(ax,{'Not-on':pooled['Not-on']},vertices,crop,f'{sample} | Not-on {len(pooled["Not-on"]):,} | mean C1, all raw frames',mean,pixel_size_nm=c['pixel_size_nm'],display_range=style.get('C1_display_ADU',[407,651]),dot_radius_px=style.get('dot_radius_px',.2));ax.title.set_color('white');_finish(fig,figdir/'NotOn_on_C1_mean2500.png',c1_footer)
    fig,ax=plt.subplots(figsize=(12,4))
    for key,color in [('On',COLORS['On']),('Not_on',COLORS['Not-on']),('QC_excluded','#F39416'),('unknown','#888888')]:ax.plot(frame.frame,frame[key],lw=.6,label=key,color=color)
    ax.set(xlabel='Original frame',ylabel='Localizations per frame');ax.legend();fig.suptitle(sample);_finish(fig,figdir/'01_counts.png')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,field in zip(axes.flat,('total','background','inter','intra','shape')):
        for group in GROUPS:ax.plot(curves['full'][group].distance_mid_um,curves['full'][group][field+'_density_um_inv'],label=group,color=COLORS[group])
        ax.set(title=field,xlabel='Distance (µm)',ylabel='Density (µm⁻¹)',xlim=(0,3));ax.axhline(0,color='gray',lw=.5);ax.legend()
    for group in GROUPS:
        for field,ls in [('x_se_nm','-'),('y_se_nm','--')]:axes.flat[5].stairs(h[field+'__'+group],h['SE_edges_nm'],label=group+' '+field,color=COLORS[group],linestyle=ls)
    axes.flat[5].set(title='Nominal SE composition',xlabel='Model SE (nm)',ylabel='Localizations');axes.flat[5].legend(fontsize=7);fig.suptitle(sample+' | TARDIS decomposition');_finish(fig,figdir/'02_full_decomposition.png','Signed negative bins retained; channel identity registration unverified.')
    for scope in scopes:
        band=_read(bs/f'paired_bands_{scope}_L25.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,field in zip(axes,('intra','shape')):
            for group in GROUPS:
                curve=curves[scope][group];ax.plot(curve.distance_mid_um,curve[field+'_density_um_inv'],color=COLORS[group],label=group);key=group.replace('-','_')+'_'+field;ax.fill_between(band.distance_mid_um,band[key+'_low'],band[key+'_high'],color=COLORS[group],alpha=.15)
            ax.set(title=field,xlim=(0,.5),xlabel='Distance (µm)',ylabel='Density (µm⁻¹)');ax.axhline(0,color='gray',lw=.5);ax.legend()
        fig.suptitle(sample+' | '+scope);_finish(fig,figdir/f'03_intra_{scope}.png','Exploratory pointwise L25 ranges; no whole-curve significance test.')
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for ax,scope in zip(axes.flat,[s for s in scopes if s!='full']):
        for group in GROUPS:ax.plot(curves[scope][group].distance_mid_um,curves[scope][group].shape_density_um_inv,color=COLORS[group],label=group)
        ax.set(title=scope,xlim=(0,.5));ax.axhline(0,color='gray',lw=.5)
    axes.flat[0].legend();fig.suptitle('Ten time windows | saved signed-area-normalized curves');_finish(fig,figdir/'04_segment_shapes.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for length,ls in [(10,':'),(25,'-'),(50,'--')]:
        band=_read(bs/f'paired_bands_full_L{length}.csv')
        for ax,limit in zip(axes,((0,.5),(0,3))):
            ax.plot(band.distance_mid_um,band.shape_point_difference,color='#246B9E',linestyle=ls,lw=1 if length==25 else .4,label=f'L{length}');ax.plot(band.distance_mid_um,band.shape_difference_low,color='#7E9CB2',linestyle=ls,lw=.5);ax.plot(band.distance_mid_um,band.shape_difference_high,color='#7E9CB2',linestyle=ls,lw=.5)
            if length==25:ax.fill_between(band.distance_mid_um,band.shape_difference_low,band.shape_difference_high,color='#246B9E',alpha=.18)
            ax.set(xlim=limit,xlabel='Distance (µm)',ylabel='On − Not-on density (µm⁻¹)');ax.axhline(0,color='gray',lw=.5)
    axes[0].legend();fig.suptitle('Area-normalized signed shape difference');_finish(fig,figdir/'05_shape_difference.png','Exploratory paired ranges; not formal inference.')
    scope_stats=_read(stats/'scope_localization_statistics.csv');join_keys=['group','scope','start_frame','stop_exclusive'];extra=[k for k in scope_stats if k not in summary or k in join_keys];merged=summary.merge(scope_stats[extra],on=join_keys,how='left');period=merged[merged.scope!='full']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for group in GROUPS:
        d=period[period.group==group]
        for ax,key in zip(axes.flat,('points','x_se_nm_median','y_se_nm_median','beta','signed_area','tail_main_ratio')):
            ax.plot(d.center_s,d[key],'-o',ms=3,color=COLORS[group],label=group);ax.set(title=key,xlabel='Time (s)');ax.legend(fontsize=8)
    fig.suptitle('Descriptive time windows');_finish(fig,figdir/'06_time_summary.png')
    adj=_read(stats/'mask_adjacent_pair_metrics.csv');tails=_read(stats/'tail_normalized_diagnostics.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].plot(adj.frame_after*c['frame_interval_s'],adj.IoU,'-o',ms=2);axes[0].set(title='Adjacent ROI mask IoU',xlabel='Time (s)',ylim=(0,1))
    for group in GROUPS:
        d=tails[tails.group==group];axes[1].plot(d.distance_mid_um,d.total_count_tail_mass,color=COLORS[group],label=group+' Total');axes[1].plot(d.distance_mid_um,d.background_ordered_nonzero_count_tail_mass,color=COLORS[group],ls='--',label=group+' B')
    axes[1].set(title='Saved tail-normalized compatibility',xlabel='Distance (µm)');axes[1].legend(fontsize=8);_finish(fig,figdir/'07_background_diagnostics.png')
    intra=_read(stats/'intra_descriptive_statistics.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,field in zip(axes,('intra','shape')):
        for group in GROUPS:ax.plot(curves['full'][group].distance_mid_um,curves['full'][group][field+'_density_um_inv'],color=COLORS[group],label=group)
        ax.set(title=field,xlim=(0,.5),xlabel='Distance (µm)',ylabel='Density (µm⁻¹)');ax.legend()
    def number(value):return f'{float(value):.1f}' if value is not None and np.isfinite(float(value)) else 'NA'
    for i,row in enumerate(intra.itertuples()):
        annotation=f'{row.group}: signed mean {number(row.signed_first_moment_nm)} nm; half-area {number(row.signed_half_area_location_nm)} nm; peak bin {number(row.mode_bin_center_nm)} nm'
        for ax in axes:ax.text(.98,.95-i*.09,annotation,transform=ax.transAxes,ha='right',va='top',fontsize=7,color=COLORS[row.group])
    notes='; '.join(f"{r.group}: {r.statistics_status}" for r in intra.itertuples());_finish(fig,figdir/'08_intra_with_statistics.png',notes)
    areastats=metadata['area_statistics'];fig,axes=plt.subplots(2,1,figsize=(13.5,8),sharex=True)
    for ax,scope,color in zip(axes,('ROI','full'),('#246B9E','#697F50')):
        s=next(row for row in areastats if row['scope']==scope);ax.step(area.frame,area[scope+'_mask_area_um2'],where='post',color=color)
        if s['mean_um2'] is not None:ax.axhline(s['mean_um2'],ls='--',color=color)
        meantext=f"{s['mean_um2']:.2f}" if s['mean_um2'] is not None else 'unavailable';mediantext=f"{s['median_um2']:.2f}" if s['median_um2'] is not None else 'unavailable'
        ax.set(title=f'{scope}: mean {meantext} µm²; median {mediantext} µm²',ylabel='Mask area (µm²)')
    axes[1].set_xlabel('Original frame');fig.suptitle(sample+' | mask area');_finish(fig,figdir/'09_mask_area.png','Binary mask area within support blocks; no position-error expansion.')
    plot_ctx=replace(ctx,config=dict(ctx.config,style=style))
    _four_frames(plot_ctx,stats,figdir,pooled,c)
    expected=required_cell_pngs(c['frames'],c['segment_frames']);actual={p.name for p in figdir.glob('*.png')if not p.name.startswith('._')};missing=sorted(expected-actual)
    entries=''.join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a><br><img src="{html.escape(name)}" width="700"></li>' for name in sorted(actual))
    display_note=(f"C1 display: {c1_display['limits_ADU']} ADU; mode {c1_display['mode']}. "
                  "One window per cell, shared by representative frames and saved C1 mean. Display only; raw pixels and scientific analysis unchanged.")
    (figdir/'INDEX.html').write_text('<html><body><h1>'+html.escape(sample)+'</h1><p>Pure refresh from saved scientific products. Scientific stage calls: 0.</p><p>'+html.escape(display_note)+'</p><ul>'+entries+'</ul></body></html>',encoding='utf-8')
    result=dict(figures=str(figdir),scientific_stage_calls=0,missing=missing,images=sorted(actual),adopted_readonly=False,
                C1_display=c1_display,C1_display_ADU=c1_display['limits_ADU'],C1_display_policy=c1_display,
                C2_display_ADU=style.get('C2_display_ADU',[406,554]),C2_display=dict(mode='fixed',limits_ADU=style.get('C2_display_ADU',[406,554])))
    dump(out/'publication.json',result)
    if missing:raise FileNotFoundError('Incomplete pure publication: '+', '.join(missing))
    return StageResult(dict(directory=figdir,figures=figdir,metadata=out/'publication.json'),result)


def _four_frames(ctx,stats,figdir,pooled,c):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import tifffile
    fits=_read(stats/'representative_fits.csv.gz');selected=_read(stats/'representative_selected.csv.gz');fullpath=None
    for stage in ('cleanup','mask_cleanup','mask'):
        if stage in ctx.inputs:fullpath=ctx.inputs[stage].files.get('full_mask') or ctx.inputs[stage].files.get('mask')
        if fullpath:break
    if not fullpath:raise FileNotFoundError('Saved full mask missing for four-panel display')
    full=np.load(fullpath,mmap_mode='r',allow_pickle=False);crop=pooled['crop_xyxy_px'];x0,y0,x1,y1=map(int,crop);v=pooled['vertices']
    saved_area=_read(stats/'mask_area_by_frame.csv').set_index('frame');mapping=saved_area.block_index.to_dict();mask_valid=saved_area.mask_valid.to_dict() if 'mask_valid' in saved_area else {f:True for f in mapping}
    if set(range(c['frames']))-set(mapping):raise ValueError('Explicit saved frame-to-mask mapping is incomplete')
    style=ctx.config.get('style',{});range1=style.get('C1_display_ADU',[407,651]);range2=style.get('C2_display_ADU',[406,554])
    frames=[f for f in REPRESENTATIVE_FRAMES if f<c['frames']]
    with tifffile.TiffFile(ctx.sample.C1_file) as c1,tifffile.TiffFile(ctx.sample.C2_file) as c2:
        for frame in frames:
            raw1=c1.pages[frame].asarray();raw2=c2.pages[frame].asarray();fig,axes=plt.subplots(1,4,figsize=(18,7));part=fits[fits.frame.astype(int)==frame];sel=selected[selected.frame.astype(int)==frame]
            for i,ax in enumerate(axes):
                ax.imshow(raw1 if i==0 else raw2,cmap='gray',vmin=range1[0] if i==0 else range2[0],vmax=range1[1] if i==0 else range2[1],origin='upper',interpolation='nearest');ax.plot(v[:,0],v[:,1],color='#888888',lw=.4);ax.set(xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5));ax.axis('off')
            block=mapping[frame]
            if not np.isfinite(block) or int(block)!=block or block<0 or block>=len(full):raise ValueError('Representative frame has no valid saved mask mapping')
            if mask_valid[frame]:axes[0].contour(full[int(block)],levels=[.5],colors=['#FF5555'],linewidths=.5)
            else:axes[0].text(.5,.5,'Mask unavailable / unknown',transform=axes[0].transAxes,ha='center',color='red')
            for row in part.itertuples():
                # Saved native candidate centers, never guessed from refined ones.
                d=row._asdict();dx=d.get('detection_x_px',d.get('detector_x_px',d.get('x_px')));dy=d.get('detection_y_px',d.get('detector_y_px',d.get('y_px')))
                if dx is None or dy is None:
                    dx=d.get('candidate_x_px');dy=d.get('candidate_y_px')
                if dx is None or dy is None:raise ValueError('Saved native TrackMate centers missing for pure four-panel display')
                axes[2].add_patch(Circle((float(dx),float(dy)),3,fill=False,color='#FFB347',lw=.5))
            for row in sel.itertuples():axes[3].errorbar(row.fit_x_px,row.fit_y_px,xerr=row.x_se_nm/c['pixel_size_nm'],yerr=row.y_se_nm/c['pixel_size_nm'],fmt='.',ms=1,color=COLORS[row.tardis_group],elinewidth=.5)
            uid=next((key for key in ('source_detection_uid','detection_uid','uid') if key in part),None)
            if uid is None:raise ValueError('Saved representative fits lack stable detection UIDs')
            selected_uids=set(sel[uid].astype(str))
            excluded=part.loc[~part[uid].astype(str).isin(selected_uids)]
            for row in excluded.itertuples():
                d=row._asdict();x=d.get('fit_x_px');y=d.get('fit_y_px')
                if x is None or y is None or not np.isfinite([x,y]).all():x=d.get('x_px');y=d.get('y_px')
                if x is not None and y is not None and np.isfinite([x,y]).all():axes[3].plot(x,y,'x',color='#F39416',ms=2,mew=.5)
            if len(excluded):axes[3].text(.02,.02,'Orange ×: QC excluded',transform=axes[3].transAxes,color='#F39416',fontsize=7)

            for ax,title in zip(axes,('C1 + mask','C2 raw','C2 + TrackMate centers','C2 + refined centers / X,Y SE')):ax.set_title(title,fontsize=10)
            fig.suptitle(f"{ctx.sample_id} | original frame {frame} | {frame*c['frame_interval_s']:.2f} s | nominal SE; identity registration unverified")
            display_footer=(f'C1 display {range1[0]:g}-{range1[1]:g} ADU; '
                            f'C2 display {range2[0]:g}-{range2[1]:g} ADU. '
                            'Fixed within this cell; display only.')
            _finish(fig,figdir/f'frame_{frame:04d}_four_panels.png',display_footer)

refresh=execute
