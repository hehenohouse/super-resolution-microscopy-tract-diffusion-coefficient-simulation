"""Scientific figures and within-frame parameter stability for the pilot."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pilot_core import model_images

plt.style.use('default')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def key_name(box,factor):
    return f'box{box}_threshold{factor:.2f}'.replace('.','p')


def frame_counts(ids,locs,config):
    result=pd.DataFrame({'frame':ids,'time_s':ids*config['frame_interval_s']})
    for name,mask in [('candidates',np.ones(len(locs),bool)),('converged',locs.fit_converged),
                      ('usable_model_fits',locs.usable_model_fit),('failed',~locs.fit_converged),
                      ('qc_unflagged',locs.qc_unflagged_model_fit),('requires_qc_review',~locs.qc_unflagged_model_fit),
                      ('suspect_overlap',locs.usable_model_fit&locs.suspect_overlap)]:
        result[name]=result.frame.map(locs.loc[mask].groupby('frame').size()).fillna(0).astype(int)
    result['failed_fraction']=result.failed.div(result.candidates.where(result.candidates>0))
    result['qc_review_fraction']=result.requires_qc_review.div(result.candidates.where(result.candidates>0))
    return result


def match_locations(reference,other,radius):
    """Mutual nearest neighbours in the SAME original frame; never across time gaps."""
    rows=[]
    for frame,ref in reference.groupby('frame'):
        cmp=other.loc[other.frame==frame]
        if cmp.empty:continue
        a=ref[['x','y']].to_numpy();b=cmp[['x','y']].to_numpy()
        distance,index=cKDTree(b).query(a)
        _,reverse=cKDTree(a).query(b)
        for i in np.flatnonzero((distance<=radius)&(reverse[index]==np.arange(len(a)))):
            ra=ref.iloc[i];rb=cmp.iloc[index[i]]
            rows.append({'frame':int(frame),'reference_candidate_id':int(ra.candidate_id),
                         'comparison_candidate_id':int(rb.candidate_id),'distance_px':float(distance[i]),
                         'dx_px':float(rb.x-ra.x),'dy_px':float(rb.y-ra.y),
                         'relative_sx_change':float((rb.sx-ra.sx)/ra.sx),
                         'relative_sy_change':float((rb.sy-ra.sy)/ra.sy),
                         'reference_sx_px':float(ra.sx),'comparison_sx_px':float(rb.sx),
                         'reference_sy_px':float(ra.sy),'comparison_sy_px':float(rb.sy),
                         'reference_overlap_flag':bool(ra.suspect_overlap),'comparison_overlap_flag':bool(rb.suspect_overlap)})
    return pd.DataFrame(rows,columns=['frame','reference_candidate_id','comparison_candidate_id','distance_px',
                    'dx_px','dy_px','relative_sx_change','relative_sy_change','reference_sx_px','comparison_sx_px',
                    'reference_sy_px','comparison_sy_px','reference_overlap_flag','comparison_overlap_flag'])


def stability_tables(results,config,out):
    reference=results[(config['reference_box'],config['reference_threshold_factor'])]
    ref=reference.loc[reference.usable_model_fit]
    summaries=[]
    cleanref=reference.loc[reference.qc_unflagged_model_fit]
    for (box,factor),locs in results.items():
        good=locs.loc[locs.usable_model_fit]
        matches=match_locations(ref,good,config['stability_match_radius_px'])
        matches.to_csv(out/(key_name(box,factor)+'_matches_to_reference.csv'),index=False)
        isolated=matches.loc[~matches.reference_overlap_flag&~matches.comparison_overlap_flag]
        cleanmatches=match_locations(cleanref,locs.loc[locs.qc_unflagged_model_fit],config['stability_match_radius_px'])
        cleanmatches.to_csv(out/(key_name(box,factor)+'_unflagged_matches_to_reference.csv'),index=False)
        summary={'box':box,'threshold_factor':factor,'matched':len(matches),
                 'reference_usable':len(ref),'comparison_usable':len(good),
                 'reference_matched_fraction':len(matches)/len(ref) if len(ref) else np.nan,
                 'comparison_matched_fraction':len(matches)/len(good) if len(good) else np.nan,
                 'isolated_heuristic_matches':len(isolated),'unflagged_matches':len(cleanmatches),
                 'reference_unflagged':len(cleanref)}
        for tag,table in [('all',matches),('no_overlap_flag',isolated),('unflagged',cleanmatches)]:
            for name in ['distance_px','relative_sx_change','relative_sy_change']:
                values=table[name].to_numpy(dtype=float)
                summary[f'{tag}_median_abs_{name}']=float(np.median(np.abs(values))) if len(values) else np.nan
                summary[f'{tag}_p90_abs_{name}']=float(np.quantile(np.abs(values),.9)) if len(values) else np.nan
        summaries.append(summary)
    table=pd.DataFrame(summaries)
    table.to_csv(out/'parameter_stability.csv',index=False)
    return table


def save_figure(fig,out,name):
    fig.savefig(out/name,dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_overlays(ids,movie,results,config,out):
    box=config['reference_box'];ref=results[(box,1.)];low=results[(box,.75)]
    fig,axes=plt.subplots(1,3,figsize=(12.3,8.3),layout='constrained')
    images=dict(zip(ids,movie))
    for ax,frame in zip(axes,config['threshold_reference_frames']):
        image=images[frame];vmin,vmax=np.percentile(image,[2,99.85])
        ax.imshow(image,cmap='gray',vmin=vmin,vmax=vmax,origin='upper',interpolation='nearest')
        ax.grid(False)
        locs=ref.loc[ref.frame==frame];good=locs.loc[locs.usable_model_fit];failed=locs.loc[~locs.fit_converged]
        extra=low.loc[(low.frame==frame)&~low.candidate_id.isin(locs.candidate_id)]
        ax.scatter(extra.candidate_x,extra.candidate_y,s=15,facecolors='none',edgecolors='#ffd441',linewidths=.55,label='Only at 0.75x threshold')
        ax.scatter(locs.candidate_x,locs.candidate_y,s=17,facecolors='none',edgecolors='#23d9ed',linewidths=.6,label='Reference candidates')
        ax.scatter(good.x,good.y,s=17,c='#ff5757',marker='+',linewidths=.7,label='Usable MLE positions')
        ax.scatter(failed.candidate_x,failed.candidate_y,s=20,c='#e179ff',marker='x',linewidths=.8,label='Backend failed')
        suspicious=locs.loc[~locs.qc_unflagged_model_fit]
        ax.scatter(suspicious.candidate_x,suspicious.candidate_y,s=21,facecolors='none',edgecolors='#c374ed',marker='s',linewidths=.6,label='QC review required')
        ax.set_title(f'Original frame {frame} | {frame*.02:.2f} s\n{len(locs)} candidates, {len(good)} usable fits')
        ax.set_xlabel('x (camera pixels)');ax.set_ylabel('y (camera pixels)')
        ax.set_xlim(-.5,image.shape[1]-.5);ax.set_ylim(image.shape[0]-.5,-.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('T1 C2: source images and independently detected localizations\nReference: 9-pixel box, baseline threshold; display contrast only',fontsize=13)
    save_figure(fig,out,'01_source_overlays.png')


def plot_patches(ids,movie,locs,box,config,out):
    images=dict(zip(ids,movie));chosen=[]
    for frame in config['threshold_reference_frames']:
        group=locs.loc[locs.frame==frame]
        if group.empty:continue
        good=group.loc[group.qc_unflagged_model_fit]
        representative=good.iloc[(good.deviance_per_dof-good.deviance_per_dof.median()).abs().argmin()] if len(good) else group.iloc[0]
        chosen.append((representative,'typical model fit'))
        flagged=group.loc[~group.qc_unflagged_model_fit&(group.candidate_id!=representative.candidate_id)]
        if len(flagged):
            chosen.append((flagged.sort_values('model_residual_rms_e',ascending=False).iloc[0],'flagged / inspect'))
        elif len(good)>1:
            weak=good.loc[good.candidate_id!=representative.candidate_id].sort_values('photons').iloc[0]
            chosen.append((weak,'weaker model fit'))
    if not chosen:return
    fig,axes=plt.subplots(len(chosen),3,figsize=(10.6,2.35*len(chosen)),squeeze=False,layout='constrained')
    selection=[];half=box//2;cam=config['camera']
    for (row,case),axs in zip(chosen,axes):
        frame=int(row.frame);x=int(row.candidate_x);y=int(row.candidate_y)
        raw=images[frame][y-half:y+half+1,x-half:x+half+1].astype(float)
        theta=np.array([[row.photons,row.x-x+half,row.y-y+half,row.sx,row.sy,row.bg]])
        model=model_images(theta,box)[0]
        data=(raw-cam['baseline_adu'])*cam['sensitivity_e_per_adu']/cam['em_gain']
        model_adu=model*cam['em_gain']/cam['sensitivity_e_per_adu']+cam['baseline_adu']
        residual=data-model
        extent=(x-half-.5,x+half+.5,y+half+.5,y-half-.5)
        vmin,vmax=float(raw.min()),float(raw.max());bound=max(1.,float(np.nanmax(np.abs(residual))))
        for ax,array,title,cmap,mn,mx in [(axs[0],raw,'Original TIFF (ADU)','gray',vmin,vmax),
                (axs[1],model_adu,'Gaussian fit (ADU)','gray',vmin,vmax),
                (axs[2],residual,'Residual (nominal e/pixel)','RdBu_r',-bound,bound)]:
            im=ax.imshow(array,cmap=cmap,vmin=mn,vmax=mx,extent=extent,origin='upper',interpolation='nearest')
            ax.grid(False)
            ax.scatter([row.x],[row.y],marker='+',s=40,c='#50d4ef',linewidths=1)
            ax.set_title(title,fontsize=10);ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)')
            fig.colorbar(im,ax=ax,shrink=.77)
        axs[0].text(-.2,1.2,f'Frame {frame}: {case}\nstate={row.fit_state_label}, sx/sy={row.sx:.2f}/{row.sy:.2f} px',transform=axs[0].transAxes,fontsize=10)
        selection.append({'frame':frame,'candidate_id':int(row.candidate_id),'case':case,'fit_state':int(row.fit_state),
                          'suspect_overlap':bool(row.suspect_overlap),'suspect_psf_width':bool(row.suspect_psf_width)})
    fig.suptitle('Reference fits: model and unfiltered-source residuals\nUniform nominal noise only; flags are inspection prompts',fontsize=13)
    pd.DataFrame(selection).to_csv(out/'patch_examples.csv',index=False)
    save_figure(fig,out,'02_fit_patches_and_residuals.png')


def plot_counts(ids,results,config,out):
    counts={key:frame_counts(ids,locs,config) for key,locs in results.items()}
    fig,axes=plt.subplots(2,3,figsize=(13,7.3),layout='constrained')
    for col,(start,end) in enumerate(config['sample_segments']):
        for factor,color in [(.75,'#3379b6'),(1.,'#e88d21'),(1.25,'#469c66')]:
            part=counts[(9,factor)].loc[lambda a:a.frame.between(start,end)]
            axes[0,col].plot(part.time_s,part.candidates,color=color,lw=1,label=f'Candidates: {factor:g}x')
        ref=counts[(9,1.)].loc[lambda a:a.frame.between(start,end)]
        axes[0,col].plot(ref.time_s,ref.converged,color='#888888',ls=':',lw=1,label='Converged: reference')
        axes[0,col].plot(ref.time_s,ref.usable_model_fits,'k--',lw=1,label='Usable: reference')
        axes[0,col].plot(ref.time_s,ref.qc_unflagged,color='#a029aa',lw=1,label='Unflagged: reference')
        for box,color in [(7,'#3379b6'),(9,'#e88d21'),(11,'#469c66')]:
            part=counts[(box,1.)].loc[lambda a:a.frame.between(start,end)]
            axes[1,col].plot(part.time_s,part.failed_fraction*100,color=color,lw=1,label=f'Box {box}, baseline')
        axes[1,col].plot(ref.time_s,ref.qc_review_fraction*100,color='#a029aa',ls='--',lw=1,label='Reference: QC review required')
        axes[0,col].set_title(f'Original frames {start}–{end}');axes[0,col].set_ylabel('Points / frame')
        axes[1,col].set_ylabel('Backend failure / QC review (%)')
        axes[1,col].set_ylim(-2,102)
        for row in [0,1]:axes[row,col].set_xlabel('Original time (s)');axes[row,col].grid(alpha=.15)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Detection and convergence across sampled time segments\nSegments retain their source frame numbers; no interpolation over gaps',fontsize=13)
    save_figure(fig,out,'03_counts_and_failure_rates.png')
    return counts


def plot_distributions(results,config,out):
    locs=results[(9,1.)];good=locs.loc[locs.qc_unflagged_model_fit];precision=good
    fig,axes=plt.subplots(2,2,figsize=(11.7,8),layout='constrained')
    sets=[(axes[0,0],good,['sx','sy'],'Gaussian PSF sigma (pixels)',1.),
          (axes[0,1],precision,['lpx_nm','lpy_nm'],'Nominal model CRLB standard deviation (nm)',.99),
          (axes[1,0],good,['photons'],'Nominal detected photoelectron equivalents',.99),
          (axes[1,1],good,['deviance_per_dof'],'Shifted-Poisson deviance / (box pixels - 6)',.99)]
    for ax,table,keys,label,quantile in sets:
        for key in keys:
            v=table[key].to_numpy();v=v[np.isfinite(v)&(v>=0)]
            if not len(v):continue
            upper=float(np.quantile(v,quantile)) if quantile else float(v.max())
            ax.hist(v,bins=np.linspace(0,max(upper,1e-6),60),histtype='step',lw=1.6,label=f'{key}: median {np.median(v):.2f}')
            ax.set_xlim(0,max(upper,1e-6))
        ax.set_xlabel(label);ax.set_ylabel('Fits / bin');ax.legend(fontsize=9);ax.grid(alpha=.13)
        if quantile and quantile<1:ax.set_title('Display up to 99th percentile; all rows remain in tables',fontsize=9)
    fig.suptitle(f'T1 C2 reference: {len(good):,} heuristic-unflagged fits\nManufacturer-nominal estimates; subset is not experimentally validated',fontsize=13)
    save_figure(fig,out,'04_width_signal_precision_distributions.png')
    admissible=locs.loc[locs.usable_model_fit]
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),layout='constrained')
    for ax,keys,title in [(axes[0],['sx','sy'],'Gaussian sigma (pixels)'),(axes[1],['lpx_nm','lpy_nm'],'Nominal CRLB SD (nm)'),(axes[2],['photons'],'Nominal detected electrons')]:
        for key in keys:
            values=admissible[key].to_numpy();values=values[np.isfinite(values)&(values>0)]
            if len(values):ax.hist(np.log10(values),bins=65,histtype='step',lw=1.5,label=key)
        ax.set_xlabel('log10 '+title);ax.set_ylabel('Fits / bin');ax.legend(fontsize=9)
    fig.suptitle(f'All {len(admissible):,} admissible reference fits, including flagged extremes\nLog scale exposes degenerate fits; positive finite values only',fontsize=12)
    save_figure(fig,out,'04b_all_admissible_log_distributions.png')


def plot_stability(stability,out):
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.6),layout='constrained')
    labels=[f'B{row.box:g}\n{row.threshold_factor:g}x' for row in stability.itertuples()]
    metrics=[('reference_matched_fraction',100,'Reference fits matched (%)'),
             ('all_median_abs_distance_px',117,'Median matched position change (nm)'),
             ('all_median_abs_relative_sx_change',100,'Median absolute PSF width change (%)')]
    for ax,(field,scale,label) in zip(axes,metrics):
        ax.bar(np.arange(len(labels)),stability[field]*scale,color=['#548fb8' if not (r.box==9 and r.threshold_factor==1) else '#e9922c' for r in stability.itertuples()])
        ax.set_xticks(np.arange(len(labels)),labels,fontsize=8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        if 'width' in label:
            ax.plot(np.arange(len(labels)),stability.all_median_abs_relative_sy_change*100,'k.-',label='Y width')
            ax.legend(['Y width','X width'],fontsize=8)
    fig.suptitle('Nine configurations versus box 9, baseline threshold\nMutual nearest neighbours within 1 pixel, same original frame',fontsize=12)
    save_figure(fig,out,'05_parameter_stability.png')


def plot_thresholds(candidate_tables,threshold_stats,config,out):
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),layout='constrained')
    for ax,box in zip(axes,config['box_sizes']):
        values=candidate_tables[box].loc[lambda t:t.frame.isin(config['threshold_reference_frames']),'net_gradient'].to_numpy()
        base=threshold_stats[box]['baseline_threshold_adu_gradient']
        upper=max(base*2,float(np.quantile(values,.99)))
        ax.hist(values,bins=np.linspace(0,upper,100),histtype='step',color='black')
        for factor,color in [(.75,'#3379b6'),(1.,'#e88d21'),(1.25,'#469c66')]:ax.axvline(base*factor,color=color,label=f'{factor:g}x: {base*factor:.1f}')
        ax.set_yscale('log');ax.set_title(f'Box {box}');ax.set_xlabel('Raw ADU net-gradient sum');ax.set_ylabel('Candidates / bin');ax.legend(fontsize=8)
    fig.suptitle('Threshold reference frames: 0, 1250, 2499\nBaseline = candidate-score median + 6 x (1.4826 x MAD)',fontsize=12)
    save_figure(fig,out,'06_threshold_reference_distributions.png')


def make_qc(ids,movie,results,candidate_tables,threshold_stats,config,out):
    out=Path(out);out.mkdir(exist_ok=True)
    stability=stability_tables(results,config,out)
    counts=plot_counts(ids,results,config,out)
    for key,table in counts.items():table.to_csv(out/(key_name(*key)+'_per_frame_counts.csv'),index=False)
    plot_overlays(ids,movie,results,config,out)
    plot_patches(ids,movie,results[(9,1.)],9,config,out)
    plot_distributions(results,config,out)
    plot_stability(stability,out)
    plot_thresholds(candidate_tables,threshold_stats,config,out)
    return stability
