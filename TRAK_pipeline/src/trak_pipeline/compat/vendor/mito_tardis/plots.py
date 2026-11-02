"""Scientific figures; percentile shading is exploratory and pointwise only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'On':'#159C76', 'Not-on':'#9854CA'}


def finish(fig, path, footer=None):
    if footer:
        fig.text(.5,.015,footer,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.95))
    else:
        fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(path,dpi=180)
    plt.close(fig)


def all_figures(out,c,summary,frames,effects):
    plt.style.use('default')
    mask_label=c.get('mask_label','W5/min32')
    sample_name=c.get('sample_name','TDKd3')
    fig,ax=plt.subplots(figsize=(13,4))
    for key,color in [('On',COLORS['On']),('Not-on',COLORS['Not-on']),('QC_excluded','#F39416')]:
        ax.plot(frames.frame,frames[key],color=color,lw=.6,label=key)
    ax.set(xlabel='Original frame (0-based)',ylabel='Localizations per frame',xlim=(0,2499))
    ax.legend();ax.grid(alpha=.2)
    fig.suptitle(f'{sample_name} | {mask_label} k=1 groups | Unsmoothed counts')
    finish(fig,out/'figures/01_counts.png','Identity registration unverified; original QC and spatial labels retained.')

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fields=['total','background','inter','intra','shape']
    titles=['Total: same-group adjacent-frame pairs','B: ordered same-frame nonzero pairs',
            'Inter = beta B','Signed Intra = Total - beta B','Intra / signed area (0-3 um)']
    for group in COLORS:
        d=pd.read_csv(out/'distribution'/group/'full.csv')
        s=summary[(summary.group==group)&(summary.scope=='full')].iloc[0]
        for ax,field,title in zip(axes.flat,fields,titles):
            ax.plot(d.distance_mid_um,d[field+'_density_um_inv'],color=COLORS[group],lw=1.2,
                    label=f'{group} | beta={s.beta:.4f}' if field=='intra' else group)
            ax.set(title=title,xlim=(0,3),xlabel='Distance (um)',ylabel='Density (1/um)')
            ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=8)
        selected=pd.read_csv(out/'statistical_input.csv',usecols=['tardis_group','x_se_nm','y_se_nm'])
        selected=selected[selected.tardis_group==group]
        for axis,ls in [('x_se_nm','-'),('y_se_nm','--')]:
            axes.flat[5].hist(selected[axis],bins=np.linspace(0,200,51),histtype='step',density=True,
                              color=COLORS[group],ls=ls,label=f'{group} {axis[:1].upper()}')
    axes.flat[5].set(title='Nominal position SE: group composition',xlabel='Position SE (nm)',
                    ylabel='Density within displayed 0-200 nm');axes.flat[5].legend(fontsize=8)
    fig.suptitle(f'{sample_name} | {mask_label} | tau=20 ms | Group-specific background and beta')
    finish(fig,out/'figures/02_full_decomposition.png','Negative bins retained; group/ROI boundary-crossing pairs excluded. Nominal SE is not calibrated accuracy.')

    for scope in summary.scope.unique():
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        band=pd.read_csv(out/'bootstrap'/f'paired_bands_{scope}_L25.csv')
        unavailable=[]
        for group in COLORS:
            d=pd.read_csv(out/'distribution'/group/(scope+'.csv'))
            s=summary[(summary.group==group)&(summary.scope==scope)].iloc[0]
            if not s.shape_valid:unavailable.append(group+': '+str(s.shape_status))
            for ax,field,title in zip(axes,['intra','shape'],['Signed Intra','Intra normalized by signed area on 0-3 um']):
                key=group.replace('-','_')+'_'+field
                v=d[field+'_density_um_inv']
                ax.plot(d.distance_mid_um,v,color=COLORS[group],lw=1.4,label=group)
                ax.fill_between(band.distance_mid_um,band[key+'_low'],band[key+'_high'],color=COLORS[group],alpha=.16)
                ax.set(xlim=(0,.5),xlabel='Distance (um)',ylabel='Density (1/um)',title=title)
                ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.2);ax.legend()
        fig.suptitle(f'{sample_name} | {scope} | On vs Not-on | {mask_label}, k=1, tau=20 ms')
        footer='Display 0-0.5 um; normalization uses 0-3 um. Shading: exploratory paired L25 pointwise percentile range.'
        if unavailable:footer='UNAVAILABLE: '+'; '.join(unavailable)
        finish(fig,out/'figures'/f'03_intra_{scope}.png',footer)

    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for ax,scope in zip(axes.flat,[s for s in summary.scope.unique() if s!='full']):
        missing=[]
        for group in COLORS:
            d=pd.read_csv(out/'distribution'/group/(scope+'.csv'))
            s=summary[(summary.group==group)&(summary.scope==scope)].iloc[0]
            ax.plot(d.distance_mid_um,d.shape_density_um_inv,color=COLORS[group],label=group,lw=1)
            if not s.shape_valid:missing.append(group+' unavailable')
        ax.set(title=scope,xlim=(0,.5),xlabel='Distance (um)',ylabel='Signed shape (1/um)')
        if missing:ax.text(.5,.8,'\n'.join(missing),transform=ax.transAxes,ha='center',fontsize=8)
        ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Ten 250-frame windows | No pairs across window boundaries')
    finish(fig,out/'figures/04_segment_shapes.png','All shapes normalized on 0-3 um; sparse tails or near-zero area shown as unavailable.')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for length,ls in [(10,':'),(25,'-'),(50,'--')]:
        band=pd.read_csv(out/'bootstrap'/f'paired_bands_full_L{length}.csv')
        for ax,xlim in zip(axes,[(0,.5),(0,3)]):
            ax.plot(band.distance_mid_um,band.shape_point_difference,color='#246B9E',ls=ls,
                    label='Full point estimate' if length==25 else None,lw=1.2 if length==25 else .3)
            if length==25:
                ax.fill_between(band.distance_mid_um,band.shape_difference_low,band.shape_difference_high,
                                color='#246B9E',alpha=.18,label='Exploratory L25 pointwise range')
            else:
                ax.plot(band.distance_mid_um,band.shape_difference_low,color='#7E9CB2',ls=ls,lw=.65,label=f'L{length} range')
                ax.plot(band.distance_mid_um,band.shape_difference_high,color='#7E9CB2',ls=ls,lw=.65)
            ax.set(xlim=xlim,xlabel='Distance (um)',ylabel='On - Not-on shape density (1/um)')
            ax.axhline(0,c='gray',lw=.7);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Area-normalized signed shape difference | Shared valid replicate IDs')
    finish(fig,out/'figures/05_shape_difference.png','Exploratory pointwise ranges are not simultaneous bands or a whole-curve significance test. No p value calculated.')

    periods=summary[summary.scope!='full'].copy()
    periods['center_s']=(periods.start_frame+periods.stop_exclusive-1)/2*c['frame_interval_s']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for group in COLORS:
        d=periods[periods.group==group]
        for ax,key,title in zip(axes.flat,
            ['points','x_se_nm_median','y_se_nm_median','beta','signed_area','tail_main_ratio'],
            ['Group localizations','Nominal X SE median (nm)','Nominal Y SE median (nm)',
             'Group-specific beta','Signed Intra area','Signed tail/main area ratio']):
            ax.plot(d.center_s,d[key],'-o',c=COLORS[group],ms=3,label=group)
            bad=~d.shape_valid
            for x in d.loc[bad,'center_s']:ax.axvline(x,c=COLORS[group],ls=':',alpha=.3,lw=.7)
            ax.set(title=title,xlabel='Acquisition time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Descriptive time windows | Missing derived values mean unavailable estimates')
    finish(fig,out/'figures/06_time_summary.png','SE describes localization-model variability; no error subtraction, diffusion correction or biological-state fitting.')

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    mask=pd.read_csv(out/'mask_adjacent_pair_metrics.csv')
    candidates=c.get('mask_diagnostic_candidates',[['unsmoothed_min32','#AAAAAA','Unsmoothed min32'],['W5','#246B9E','W5/min32']])
    for candidate,color,label in candidates:
        part=mask[mask.candidate==candidate].sort_values('frame_after')
        if len(part):
            if c.get('mask_diagnostics_block_boundaries_only',False):
                axes[0].plot(part.frame_after*c['frame_interval_s'],part.IoU,'-o',lw=.6,ms=2,c=color,label=label)
            else:
                axes[0].plot(part.frame_after*c['frame_interval_s'],part.IoU,lw=.6,c=color,label=label)
    axes[0].set(xlabel='Acquisition time (s)',ylabel='Adjacent ROI mask IoU',ylim=(0,1),title=c.get('mask_agreement_title','Mask temporal agreement'))
    axes[0].legend(fontsize=8)
    for group in COLORS:
        d=pd.read_csv(out/'distribution'/group/'full.csv');tail=d.distance_mid_um.between(2.1,3)
        t=d.loc[tail,'total_count'];b=d.loc[tail,'background_ordered_nonzero_count']
        axes[1].plot(d.loc[tail,'distance_mid_um'],t/t.sum(),c=COLORS[group],label=group+' Total tail')
        axes[1].plot(d.loc[tail,'distance_mid_um'],b/b.sum(),c=COLORS[group],ls='--',label=group+' B tail')
    axes[1].set(xlabel='Distance (um)',ylabel='Tail-normalized mass per bin',title='Background tail-shape compatibility diagnostic')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Dynamic spatial selection diagnostics')
    finish(fig,out/'figures/07_background_diagnostics.png',c.get('mask_diagnostic_footer','Shared W5 windows increase adjacent mask agreement. These diagnostics do not establish background-model validity.'))
