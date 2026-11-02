"""Scientific figures from saved full-film distributions and time-block intervals."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from full_core import key_name

COLORS=['#3487b8','#e58a29','#3c9a65','#aa65a5','#606060']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,out,name):
    fig.savefig(out/'figures'/name,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)


def label(g):return f'Box {g["box"]}, {g["factor"]:g}x'


def distribution_figure(out,c,scope,title,with_bands=True):
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    keys=['total','background','inter','intra','shape']
    titles=['Total (area = 1)','Background B (area = 1)','Weighted Inter = beta B',
            'Signed Intra = Total - beta B','Intra / signed area','Nominal localization precision']
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=pd.read_csv(out/'distribution'/name/(scope+'.csv'))
        band=pd.read_csv(out/'bootstrap/bands'/f'{name}_{scope}_L25.csv')
        for ax,key in zip(axes.ravel(),keys):
            x=d.distance_mid_um;y=d[key+'_density_um_inv']
            ax.plot(x,y,lw=1.2,color=color,label=label(g))
            if with_bands and key in ['intra','shape']:
                ax.fill_between(x,band[key+'_ci_low'],band[key+'_ci_high'],color=color,alpha=.09)
        precision=pd.read_csv(out/'precision_histograms.csv')
        p=precision.loc[(precision.group==name)&(precision.scope==scope)&(precision.axis=='lpx_nm')]
        ax=axes[1,2]
        if p['count'].sum():ax.plot(p.right_nm,p['count'].cumsum()/p['count'].sum(),color=color,label=label(g))
    for i,(ax,t) in enumerate(zip(axes.ravel(),titles)):
        ax.set_title(t)
        if i<5:
            ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Distance (um)',ylabel='Density / signed contribution (1/um)',
                 xlim=(0,.45 if i>=3 else 3))
        else:ax.set(xlabel='Nominal X CRLB SD (nm)',ylabel='CDF within displayed 0–500 nm histogram',xlim=(0,200),ylim=(0,1))
    axes[0,0].legend(fontsize=8);axes[1,2].legend(fontsize=8)
    fig.suptitle(title+'\nSame QC, fixed thresholds; nominal calibration; shading = pointwise 95% time-block interval (L=25)',fontsize=13)
    save(fig,out,f'01_{scope}_distributions.png')


def make_figures(out,c,summary,boot,paired,perframe):
    distribution_figure(out,c,'full','T1 C2: full 2500-frame parameter sensitivity')
    for s in range(0,2500,250):
        scope=f'f{s:04d}_{s+249:04d}'
        distribution_figure(out,c,scope,f'T1 C2: original frames {s}–{s+249}, only within-segment pairs')
    periods=summary.loc[summary.scope!='full']
    fig,axes=plt.subplots(3,2,figsize=(13.4,11),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=periods.loc[periods.group==name].sort_values('start_frame')
        b=boot.loc[(boot.group==name)&(boot.block_frames==25)].set_index('scope').loc[d.scope]
        x=d.time_mid_s.to_numpy()
        axes[0,0].plot(x,d.points/250,'o-',ms=4,color=color,label=label(g))
        axes[0,1].plot(x,d.retained_fraction,'o-',ms=4,color=color)
        for ax,key in [(axes[1,0],'lpx_nm'),(axes[1,1],'lpy_nm')]:
            ax.plot(x,d[key+'_median'],'o-',ms=4,color=color)
            ax.fill_between(x,d[key+'_q25'],d[key+'_q75'],color=color,alpha=.10)
        for ax,key in [(axes[2,0],'beta'),(axes[2,1],'tail_main_ratio')]:
            ax.plot(x,d[key],'o-',ms=4,color=color)
            ax.fill_between(x,b[key+'_ci_low'].to_numpy(),b[key+'_ci_high'].to_numpy(),color=color,alpha=.10)
    titles=['Retained points per original frame','Retained / threshold-passing candidates',
            'Nominal X CRLB: median and point IQR','Nominal Y CRLB: median and point IQR',
            'Beta: point estimate and time-block interval','Signed tail/main: time-block interval']
    ylabels=['Points / frame','Fraction','Nominal nm','Nominal nm','Beta','Area(0.10–0.25) / area(0–0.10)']
    for ax,title,ylabel in zip(axes.ravel(),titles,ylabels):ax.set(title=title,xlabel='Original time (s), 250-frame window centers',ylabel=ylabel,xlim=(0,50))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Time-window sensitivity, same five configurations\nCRLB shading is variability among points; beta/tail shading is paired time-block uncertainty',fontsize=12)
    save(fig,out,'02_time_window_summary.png')
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=periods.loc[periods.group==name].sort_values('start_frame')
        for ax,key in zip(axes,['lpx_nm_median','lpy_nm_median']):
            ax.plot(d[key],d.tail_main_ratio,'o-',ms=4,color=color,label=label(g),alpha=.9)
    for ax,axis in zip(axes,['X','Y']):ax.set(xlabel=f'Median nominal {axis} CRLB SD (nm)',ylabel='Signed tail/main area ratio',
             title=f'{axis} precision vs Intra tail; points = 250-frame windows')
    axes[0].legend(fontsize=8)
    fig.suptitle('Temporal association is not a cause determination\nConnected points follow original time; no error subtraction, trajectory linking or deconvolution',fontsize=12)
    save(fig,out,'03_precision_tail_association.png')
    fig,axes=plt.subplots(2,3,figsize=(14.8,8.4),layout='constrained')
    fields=['candidates','retained','exclude_model_parameters','nonpositive_backend_amplitude','negative_background','exclude_detector_zeros']
    for g,color in zip(c['groups'],COLORS):
        d=perframe.loc[(perframe.box==g['box'])&(perframe.factor==g['factor'])].sort_values('frame')
        for ax,key in zip(axes.ravel(),fields):
            v=d[key] if key in ['candidates','retained'] else d[key]/d.candidates.replace(0,np.nan)
            ax.plot(d.time_s,v.rolling(50,center=True,min_periods=1).mean(),color=color,label=label(g),lw=1)
    for ax,key in zip(axes.ravel(),fields):ax.set(xlabel='Original time (s)',ylabel='Count / frame' if key in ['candidates','retained'] else 'Fraction of candidates',title=key.replace('_',' '),xlim=(0,50))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Per-frame detection and reason diagnostics (50-frame display mean)\nReason fractions overlap; negative width alone is diagnostic, not a QC exclusion',fontsize=12)
    save(fig,out,'04_frame_qc_reasons.png')
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8),layout='constrained')
    comparisons=[key_name(9,.75)+' - '+key_name(9,1.),key_name(9,1.25)+' - '+key_name(9,1.)]
    for ax,metric in zip(axes,['beta','tail_main_ratio','tail_mass']):
        for index,comp in enumerate(comparisons):
            d=paired.loc[(paired.scope=='full')&(paired.comparison==comp)&(paired.metric==metric)].sort_values('block_frames')
            x=d.block_frames.to_numpy()+(-.8 if index==0 else .8);color=COLORS[0 if index==0 else 2]
            ax.vlines(x,d.ci_low,d.ci_high,color=color,lw=2)
            ax.scatter(x,d.point_difference,color=color,label='0.75x - 1x' if index==0 else '1.25x - 1x',s=30)
        ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Bootstrap block length (frames)',ylabel='Paired configuration difference',title=metric.replace('_',' '),xticks=[10,25,50])
    axes[0].legend(fontsize=8)
    fig.suptitle('Full-film paired differences: sensitivity to time-block length\nMarkers = full-film difference; lines = paired percentile 95% intervals; block edges excluded in resampling',fontsize=12)
    save(fig,out,'05_block_length_sensitivity.png')
    acf=pd.read_csv(out/'temporal_acf.csv');fig,axes=plt.subplots(1,2,figsize=(12.4,4.9),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        for ax,field in zip(axes,['retained','retained_median_lpx_nm']):
            d=acf.loc[(acf.box==g['box'])&(acf.factor==g['factor'])&(acf.series==field)]
            ax.plot(d.lag_s,d.detrended_acf,color=color,label=label(g))
    for ax,title in zip(axes,['Retained point counts','Median nominal X CRLB']):
        ax.axhline(0,color='gray',lw=.7);ax.axhline(.1,color='gray',lw=.5,ls=':');ax.axhline(-.1,color='gray',lw=.5,ls=':')
        for seconds in [.2,.5,1.]:ax.axvline(seconds,color='gray',lw=.5,ls='--')
        ax.set(xlabel='Original lag (s)',ylabel='Within-window detrended correlation',title=title,xlim=(0,2))
    axes[0].legend(fontsize=8)
    fig.suptitle('Temporal correlation diagnostic for the declared bootstrap blocks\nLinear trend removed separately within each 250-frame window; no cross-window products',fontsize=12)
    save(fig,out,'06_temporal_correlation.png')
