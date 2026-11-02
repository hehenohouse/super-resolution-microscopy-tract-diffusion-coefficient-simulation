"""ROI and signed curve figures, distinguishing pointwise and family uncertainty."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from roi_core import key_name

COLORS = ['#3487b8','#e58a29','#3c9a65','#aa65a5','#606060']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def label(g):
    return f'Box {g["box"]}, {g["factor"]:g}x'


def save(fig, out, name):
    fig.savefig(out/'figures'/name, dpi=155, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def roi_overlay(out, image, vertices, mat_vertices, selected):
    fig, axes = plt.subplots(1, 2, figsize=(12,7), layout='constrained')
    lo, hi = np.percentile(image[image>0], [1,99.7])
    for ax in axes:
        ax.imshow(image, origin='upper', cmap='gray', vmin=lo, vmax=hi)
        ax.plot(*np.vstack([vertices,vertices[0]]).T, color='#ffdd36', lw=1.8, label='Original ImageJ cell ROI')
        ax.plot(*np.vstack([mat_vertices,mat_vertices[0]]).T, color='#f541cb', ls='--', lw=1, label='MAT ROI: exact same vertices')
        ax.set(xlabel='Native C2 x (pixels)', ylabel='Native C2 y (pixels)')
        ax.legend(fontsize=8,loc='upper right')
    axes[0].set_title('Original C2 mean: frames 0-49; no intensity correction')
    pts = selected.loc[selected.frame<50]
    axes[1].scatter(pts.x,pts.y,s=3,c='#39cfff',alpha=.5,linewidths=0)
    axes[1].set_xlim(vertices[:,0].min()-15, vertices[:,0].max()+15)
    axes[1].set_ylim(vertices[:,1].max()+15, vertices[:,1].min()-15)
    axes[1].set_title('ROI zoom with cached reference QC localization centers')
    fig.suptitle('T1 C2 cell ROI: no coordinate transform, no mitochondria mask',fontsize=13)
    save(fig,out,'01_roi_overlay.png')


def count_figure(out,c,perframe):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        d=perframe.loc[(perframe.box==g['box'])&(perframe.factor==g['factor'])]
        for ax,key in zip(axes.ravel(),['whole_qc_pass','retained','roi_fraction_of_qc_pass','retained_fraction']):
            y=d[key].rolling(50,center=True,min_periods=1).mean()
            ax.plot(d.time_s,y,c=color,lw=1,label=label(g))
    titles=['Original whole-field QC points','QC points inside common ROI',
            'ROI fraction of whole-field QC points','QC pass fraction of ROI candidate centers']
    for ax,title in zip(axes.ravel(),titles):
        ax.set(title=title,xlabel='Original time (s)',ylabel='Points/frame' if ax in axes[0] else 'Fraction',xlim=(0,50))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('ROI selection counts: 50-frame display mean; all 2500 original frame counts saved',fontsize=12)
    save(fig,out,'02_per_frame_counts.png')


def distributions(out,c,summary,scope):
    fig,axes=plt.subplots(2,3,figsize=(15,8.4),layout='constrained')
    keys=['total','background','inter','intra','shape']
    titles=['Total: area 1 on 0-3 um','ROI background B: area 1 on 0-3 um','Weighted Inter = beta B',
            'Signed Intra = Total - beta B','Intra / signed area on 0-3 um','Nominal localization uncertainty']
    unavailable=[]
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=pd.read_csv(out/'distribution'/name/(scope+'.csv'))
        b=pd.read_csv(out/'bootstrap/bands'/f'{name}_{scope}_L25.csv')
        row=summary.loc[(summary.group==name)&(summary.scope==scope)].iloc[0]
        for ax,key in zip(axes.ravel(),keys):
            ax.plot(d.distance_mid_um,d[key+'_density_um_inv'],lw=1.1,c=color,label=label(g))
            if key in ['intra','shape']:
                ax.fill_between(d.distance_mid_um,b[key+'_ci_low'],b[key+'_ci_high'],color=color,alpha=.08)
        kept=pd.read_csv(out/'selected'/(name+'.csv'),usecols=['frame','lpx_nm'])
        kept=kept.loc[kept.frame.between(row.start_frame,row.end_frame)]
        vals=np.sort(kept.lpx_nm.dropna().to_numpy())
        if len(vals):axes[1,2].plot(vals,np.arange(1,len(vals)+1)/len(vals),c=color,label=label(g))
        if not row.shape_valid:unavailable.append(label(g)+': '+row.shape_status.replace('unavailable_',''))
    for k,(ax,title) in enumerate(zip(axes.ravel(),titles)):
        ax.set_title(title)
        if k<5:
            ax.axhline(0,c='gray',lw=.6)
            ax.set(xlabel='Distance (um)',ylabel='Signed density (1/um)',xlim=(0,.45 if k in [3,4] else 3))
        else:ax.set(xlabel='Nominal X CRLB SD (nm)',ylabel='CDF of all ROI QC points',xlim=(0,200),ylim=(0,1))
    axes[0,0].legend(fontsize=8)
    if unavailable:axes[0,2].text(.03,.98,'Beta-dependent curves unavailable:\n'+'\n'.join(unavailable),transform=axes[0,2].transAxes,va='top',fontsize=7,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    scope_note='formal full-movie comparison on 0-3 um' if scope=='full' else 'descriptive window, no formal curve test'
    fig.suptitle(f'T1 C2 common cell ROI: {scope}; shape normalized on 0-3 um; {scope_note}\nIntra display zoom 0-0.45 um; shading is NOMINAL POINTWISE 95% interval',fontsize=11)
    save(fig,out,f'03_{scope}_distributions.png')


def summaries(out,c,summary,boot):
    d=summary.loc[summary.scope!='full'];fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);a=d.loc[d.group==name].sort_values('start_frame')
        b=boot.loc[(boot.group==name)&(boot.block_frames==25)].set_index('scope').loc[a.scope]
        x=a.time_mid_s
        axes[0,0].plot(x,a.points/250,'o-',c=color,label=label(g),ms=4)
        axes[0,1].plot(x,a.roi_fraction_of_whole_qc,'o-',c=color,ms=4)
        for ax,key in [(axes[1,0],'lpx_nm'),(axes[1,1],'lpy_nm')]:
            ax.plot(x,a[key+'_median'],'o-',c=color,ms=4)
            ax.fill_between(x,a[key+'_q25'],a[key+'_q75'],color=color,alpha=.1)
        for ax,key in [(axes[2,0],'beta'),(axes[2,1],'tail_main_ratio')]:
            ax.plot(x,a[key],'o-',c=color,ms=4)
            ax.fill_between(x,b[key+'_ci_low'],b[key+'_ci_high'],color=color,alpha=.1)
    titles=['ROI QC points per frame','ROI fraction of whole-field QC points','Nominal X CRLB: median + point IQR',
            'Nominal Y CRLB: median + point IQR','ROI beta + time-block interval','ROI signed tail/main + time-block interval']
    for ax,title in zip(axes.ravel(),titles):ax.set(title=title,xlabel='Original time (s), 250-frame windows',xlim=(0,50))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Descriptive time windows: missing beta/Intra points mean insufficient support\nCRLB IQR describes point variability; other shading is statistical uncertainty',fontsize=12)
    save(fig,out,'04_time_window_summary.png')


def full_shapes(out,c,summary):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=pd.read_csv(out/'distribution'/name/'full.csv')
        for ax in axes:ax.plot(d.distance_mid_um,d.shape_density_um_inv,c=color,lw=1.3,label=label(g))
    for ax,limit in zip(axes,[3,.45]):
        ax.axhline(0,c='gray',lw=.6);ax.set(xlabel='Distance (um)',ylabel='Signed-area normalized density (1/um)',xlim=(0,limit))
    axes[0].legend(fontsize=8);axes[0].set_title('Entire fixed normalization and test range')
    axes[1].set_title('Display zoom only: same 0-3 um normalization')
    fig.suptitle('ROI full-movie shapes: negative bins retained; not a nonnegative PDF\nFormal comparison uses paired centered maxT, not overlap of pointwise shadows',fontsize=12)
    save(fig,out,'05_full_shapes.png')


def simultaneous_figures(out,c,pairs,inference,primary):
    fig,axes=plt.subplots(5,2,figsize=(14,16),layout='constrained')
    mids=(np.arange(c['distance_bins'])+.5)*c['distance_max_um']/c['distance_bins']
    gated=primary.get('available',False) and primary['p_global']<=c['alpha']
    for ax,row,(i,j) in zip(axes.ravel(),pairs.itertuples(),inference['pair_order']):
        difference=inference['differences'][row.pair_index]
        ax.plot(mids,difference,c='#247db3',lw=1)
        if gated:
            ax.fill_between(mids,inference['lower'][row.pair_index],inference['upper'][row.pair_index],color='#247db3',alpha=.18)
        ax.axhline(0,c='gray',lw=.7)
        ax.set(xlabel='Distance (um)',ylabel='Difference (1/um)',xlim=(0,3),
               title=f'{label(c["groups"][i])} - {label(c["groups"][j])}\n'+
                     (f'family maxT adjusted p={row.adjusted_p:.4g}' if gated else 'descriptive; post-hoc gate not passed'))
    caution=(out/'simulation_validation.json').exists() and not __import__('json').loads((out/'simulation_validation.json').read_text())['nominal_alpha_in_null_rate_interval']
    fig.suptitle('Whole-curve differences, full 0-3 um range\n'+('NOMINAL 95% family bands: ten pairs and all 300 bins; L25' if gated else 'No formal post-hoc claims')+
        ('\nToy calibration: 9% null rejection, 91% coverage; weak significance claims are unreliable' if caution else ''),fontsize=12)
    save(fig,out,'06_ten_pair_simultaneous_bands.png')
    # Separate zoom does not change the full-range critical value or inferential domain.
    for ax in axes.ravel():ax.set_xlim(0,.45)
    fig.suptitle('Whole-curve differences, DISPLAY ZOOM 0-0.45 um; test domain remains 0-3 um\n'+('NOMINAL 95% family bands: ten pairs and all 300 bins; L25' if gated else 'No formal post-hoc claims')+
        ('\nToy calibration: 9% null rejection, 91% coverage; weak significance claims are unreliable' if caution else ''),fontsize=11)
    # Original save closes figure but the object can still be exported with new limits.
    save(fig,out,'06_ten_pair_simultaneous_bands_zoom.png')


def block_comparison(out,tests):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    d=tests.sort_values('block_frames')
    for field,lab,color in [('p_global','All real adjacent pairs','#237bb1'),('boundary_restricted_p','Block-interior control','#d18a32')]:
        axes[0].plot(d.block_frames,d[field],'o-',label=lab,c=color)
    axes[0].axhline(.05,c='gray',ls='--',lw=.8);axes[0].set(yscale='log',xlabel='Block length (frames)',ylabel='Omnibus p (log scale)',xticks=[10,25,50]);axes[0].legend(fontsize=8)
    axes[1].plot(d.block_frames,d.max_pair_centering_offset,'o-',label='Bootstrap mean - full point',c='#237bb1')
    axes[1].plot(d.block_frames,d.max_pair_boundary_change,'o-',label='Block control - full point',c='#d18a32')
    axes[1].set(xlabel='Block length (frames)',ylabel='Maximum contrast shift (1/um)',xticks=[10,25,50]);axes[1].legend(fontsize=8)
    fig.suptitle('L25 is primary; L10/L50 and block-interior estimates are sensitivity checks',fontsize=12)
    save(fig,out,'07_block_length_controls.png')


def qc_diagnostics(out,c,perframe,summary):
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    fields=['raw_negative_width','nonpositive_backend_amplitude','negative_background',
            'exclude_detector_zeros','exclude_psf_width','exclude_residual_overlap']
    for g,color in zip(c['groups'],COLORS):
        d=perframe.loc[(perframe.box==g['box'])&(perframe.factor==g['factor'])]
        for ax,field in zip(axes.ravel(),fields):
            fraction=d[field]/d.candidates.replace(0,np.nan)
            ax.plot(d.time_s,fraction.rolling(50,center=True,min_periods=1).mean(),c=color,label=label(g),lw=1)
    for ax,field in zip(axes.ravel(),fields):ax.set(title=field.replace('_',' '),xlabel='Original time (s)',ylabel='Fraction of ROI candidate centers',xlim=(0,50))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('ROI-specific QC reasons: categories overlap; not a false-positive rate\nNegative width alone is a sign diagnostic, not a QC exclusion; 50-frame display mean',fontsize=12)
    save(fig,out,'08_roi_qc_reasons.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        name=key_name(g['box'],g['factor']);d=pd.read_csv(out/'selected'/(name+'.csv'))
        for ax,key in zip(axes[0],['lpx_nm','lpy_nm']):
            a=np.sort(d[key].dropna().to_numpy())
            if len(a):ax.plot(a,np.arange(1,len(a)+1)/len(a),c=color,label=label(g))
        for ax,key in zip(axes[1],['sx','sy']):
            hist,edges=np.histogram(d[key],bins=np.linspace(.5,5.5,61),density=True)
            ax.plot((edges[:-1]+edges[1:])/2,hist,c=color,label=label(g))
    for ax,axis in zip(axes[0],['X','Y']):ax.set(title=f'Nominal {axis} CRLB distribution',xlabel='Nominal CRLB SD (nm)',ylabel='CDF of all ROI QC points',xlim=(0,200),ylim=(0,1))
    for ax,axis in zip(axes[1],['X','Y']):ax.set(title=f'Gaussian sigma {axis}',xlabel='Camera pixels',ylabel='Density (1/pixel)',xlim=(.5,5.5))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Point uncertainty and fitted widths: manufacturer_nominal, no error correction',fontsize=12)
    save(fig,out,'09_precision_and_widths.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for g,color in zip(c['groups'],COLORS):
        d=summary.loc[(summary.group==key_name(g['box'],g['factor']))&(summary.scope!='full')].sort_values('start_frame')
        for ax,key in zip(axes,['lpx_nm_median','lpy_nm_median']):ax.plot(d[key],d.tail_main_ratio,'o-',c=color,label=label(g),ms=4)
    for ax,axis in zip(axes,['X','Y']):ax.set(xlabel=f'Median nominal {axis} CRLB SD (nm)',ylabel='Signed tail/main area ratio')
    axes[0].legend(fontsize=8)
    fig.suptitle('ROI time-window association; gaps mean unavailable tail estimates\nAssociation does not identify noise or true-motion causation',fontsize=12)
    save(fig,out,'10_precision_tail_association.png')
