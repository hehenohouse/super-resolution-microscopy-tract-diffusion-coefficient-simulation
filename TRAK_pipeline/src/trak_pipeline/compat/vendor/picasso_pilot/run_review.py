#!/usr/bin/env python3
"""Sign-aware QC review and continuous-segment TARDIS sensitivity analysis."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pilot_core import FIELD_UNITS,dump_json,model_images,save_locs,sha256_file
from pilot_qc import key_name
from pilot_review import TARDIS_ROOT,analyze_continuous_subset,review_cached_fit
from tardis_tau1.outputs import save_result

plt.style.use('default')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


class Tee:
    def __init__(self,stream,log):self.stream=stream;self.log=log
    def write(self,value):self.stream.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.stream.flush();self.log.flush()


def save_figure(fig,path):
    fig.savefig(path,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)


def representative(table,field='photons'):
    """Select the median finite value in a declared stratum, not the best fit."""
    if table.empty:return None
    values=table[field].replace([np.inf,-np.inf],np.nan)
    if values.notna().any():return table.loc[(values-values.median()).abs().idxmin()]
    return table.iloc[len(table)//2]


def patch_figure(cases,movie,frames,config,out,stem):
    cases=[c for c in cases if c[1] is not None]
    if not cases:return []
    fig,axes=plt.subplots(len(cases),3,figsize=(10.8,2.7*len(cases)),squeeze=False,layout='constrained')
    index={int(f):i for i,f in enumerate(frames)};half=4;selected=[]
    for (label,row),axs in zip(cases,axes):
        frame=int(row.frame);cx=int(row.candidate_x);cy=int(row.candidate_y)
        raw=movie[index[frame],cy-half:cy+half+1,cx-half:cx+half+1].astype(float)
        theta=np.array([[row.photons,row.x-cx+half,row.y-cy+half,row.sx,row.sy,row.bg]])
        model=model_images(theta,9)[0]
        sensitivity=config['camera']['sensitivity_e_per_adu']/config['camera']['em_gain']
        predicted_adu=model/sensitivity+config['camera']['baseline_adu']
        residual=(raw-config['camera']['baseline_adu'])*sensitivity-model
        lo=float(np.min(raw));hi=float(np.max(raw));bound=max(1.,float(np.nanmax(np.abs(residual))))
        extent=(cx-half-.5,cx+half+.5,cy+half+.5,cy-half-.5)
        for ax,a,title,cmap,vmin,vmax in [(axs[0],raw,'Original TIFF (ADU)','gray',lo,hi),
                (axs[1],predicted_adu,'Gaussian model (ADU)','gray',lo,hi),
                (axs[2],residual,'Residual (nominal e/pixel)','RdBu_r',-bound,bound)]:
            im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,extent=extent,origin='upper',interpolation='nearest')
            ax.scatter([row.x],[row.y],marker='+',s=35,c='#28ceda',lw=1)
            ax.set_xlim(cx-half-.5,cx+half+.5);ax.set_ylim(cy+half+.5,cy-half-.5)
            ax.set_xticks([cx-half,cx,cx+half]);ax.set_yticks([cy-half,cy,cy+half])
            ax.tick_params(labelsize=9);ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)');ax.set_title(title,fontsize=10)
            fig.colorbar(im,ax=ax,shrink=.75)
        text=f'{label} | frame {frame}, id {int(row.candidate_id)}\nA={row.backend_amplitude_e_per_pixel:.2f}, bg={row.bg:.2f}; signed sx/sy={row.backend_sx_signed_px:.2f}/{row.backend_sy_signed_px:.2f} px'
        axs[0].text(-.13,1.22,text,transform=axs[0].transAxes,fontsize=9)
        selected.append({'category':label,'frame':frame,'candidate_id':int(row.candidate_id),
                         'threshold':float(row.threshold_factor),'backend_amplitude_e_per_pixel':float(row.backend_amplitude_e_per_pixel),
                         'bg_e_per_pixel':float(row.bg),'signed_sx_px':float(row.backend_sx_signed_px),
                         'signed_sy_px':float(row.backend_sy_signed_px),'canonical_sx_px':float(row.sx),
                         'canonical_sy_px':float(row.sy),'unflagged':bool(row.qc_unflagged_model_fit),
                         'parameter_issues':str(row.fit_parameter_issues),'image':stem})
    fig.suptitle('T1 C2: inspect model, source and residual together\nNominal calibration; sign normalization preserves the fitted image and X/Y',fontsize=12)
    save_figure(fig,out/(stem+'.png'))
    return selected


def qc_examples(tables,movie,frames,config,out):
    ref=tables[(9,1.)];low=tables[(9,.75)];high=tables[(9,1.25)]
    good=ref.loc[ref.qc_unflagged_model_fit]
    normal_cases=[(f'Unflagged, segment {a}–{b}',representative(good.loc[good.frame.between(a,b)])) for a,b in config['sample_segments']]
    selected=patch_figure(normal_cases,movie,frames,config,out,'01_normal_examples')
    other=[('Negative Gaussian amplitude',representative(ref.loc[ref.backend_amplitude_e_per_pixel<=0],'backend_amplitude_e_per_pixel')),
           ('Negative background',representative(ref.loc[ref.bg<0],'bg')),
           ('Fitted center outside box',representative(ref.loc[ref.fit_parameter_issues.str.contains('center_outside_fitting_box')],'fit_deviance')),
           ('Sign-only but abnormal |width|',representative(ref.loc[ref.sign_only_parameter_issue&ref.suspect_psf_width],'sx'))]
    selected+=patch_figure(other,movie,frames,config,out,'02_other_parameter_examples')
    extra=low.loc[low.frame.between(0,49)&~low.candidate_id.isin(ref.candidate_id)]
    dropped=ref.loc[ref.frame.between(0,49)&ref.qc_unflagged_model_fit&~ref.candidate_id.isin(high.candidate_id)]
    eligible=extra.loc[extra.qc_unflagged_model_fit]
    other_extra=extra.loc[~extra.qc_unflagged_model_fit]
    changed=[('Low-threshold added, unflagged',representative(eligible)),
             ('Low-threshold added, weaker example',representative(eligible.loc[eligible.photons<=eligible.photons.quantile(.25)])),
             ('Low-threshold added, QC flagged',representative(other_extra,'fit_deviance')),
             ('Reference point removed at 1.25x',representative(dropped))]
    selected+=patch_figure(changed,movie,frames,config,out,'03_threshold_changed_examples')
    pd.DataFrame(selected).to_csv(out/'example_selection.csv',index=False)
    detector_padding_examples(ref,movie,frames,config,out)


def detector_padding_examples(ref,movie,frames,config,out):
    rescued=ref.loc[ref.rescued_by_sign_canonicalization]
    cases=[('Width sign fixed, zero in detector halo',representative(rescued.loc[rescued.suspect_zero_detection_support])),
           ('Width sign fixed, no zero in detector halo',representative(rescued.loc[~rescued.suspect_zero_detection_support]))]
    cases=[c for c in cases if c[1] is not None]
    fig,axes=plt.subplots(len(cases),4,figsize=(14.7,3.15*len(cases)),squeeze=False,layout='constrained')
    index={int(f):i for i,f in enumerate(frames)};rows=[]
    for (label,row),axs in zip(cases,axes):
        f=int(row.frame);x=int(row.candidate_x);y=int(row.candidate_y)
        context=movie[index[f],y-5:y+6,x-5:x+6].astype(float);raw=context[1:-1,1:-1]
        model=model_images(np.array([[row.photons,row.x-x+4,row.y-y+4,row.sx,row.sy,row.bg]]),9)[0]
        sensitivity=config['camera']['sensitivity_e_per_adu']/config['camera']['em_gain']
        model_adu=model/sensitivity+config['camera']['baseline_adu'];residual=(raw-config['camera']['baseline_adu'])*sensitivity-model
        bound=max(1,float(np.max(np.abs(residual))));low=float(raw.min());high=float(raw.max())
        for col,(a,title,cmap,vmin,vmax,half) in enumerate([(context,'Detector context (ADU)','gray',float(context.min()),float(context.max()),5),
                (raw,'Fitting patch (ADU)','gray',low,high,4),(model_adu,'Gaussian fit (ADU)','gray',low,high,4),
                (residual,'Fit residual (nominal e)','RdBu_r',-bound,bound,4)]):
            ax=axs[col];im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,origin='upper',extent=(x-half-.5,x+half+.5,y+half+.5,y-half-.5))
            ax.set_xticks([x-half,x,x+half]);ax.set_yticks([y-half,y,y+half]);ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)');ax.set_title(title,fontsize=9)
            ax.scatter(row.x,row.y,marker='+',c='#1bc9d6',s=30);fig.colorbar(im,ax=ax,shrink=.72)
            if col==0:ax.add_patch(plt.Rectangle((x-4.5,y-4.5),9,9,fill=False,edgecolor='#ed6f41',lw=1))
        axs[0].text(-.1,1.18,f'{label}\nFrame {f}, id {int(row.candidate_id)}',transform=axs[0].transAxes,fontsize=9)
        rows.append({'frame':f,'candidate_id':int(row.candidate_id),'zero_in_fit_patch':int(row.source_zero_pixels),
                     'zero_in_detector_support':int(row.source_zero_pixels_detection_support),'label':label})
    fig.suptitle('Net-gradient detection reads one pixel beyond the fitting patch\nOrange rectangle = 9-pixel fit; unchanged source pixels, no masking or padding correction',fontsize=12)
    save_figure(fig,out/'07_detection_support_padding.png')
    pd.DataFrame(rows).to_csv(out/'detection_support_examples.csv',index=False)


def cohort_summary(tables,out):
    ref=tables[(9,1.)].loc[lambda t:t.frame.between(0,49)]
    low=tables[(9,.75)].loc[lambda t:t.frame.between(0,49)]
    high=tables[(9,1.25)].loc[lambda t:t.frame.between(0,49)]
    cohorts={'low_added':low.loc[~low.candidate_id.isin(ref.candidate_id)],
             'reference_removed_by_high':ref.loc[~ref.candidate_id.isin(high.candidate_id)],
             'reference_retained_at_high':high}
    summaries=[]
    for name,table in cohorts.items():
        table.to_csv(out/(name+'_all_candidates.csv'),index=False)
        unflagged=table.loc[table.qc_unflagged_model_fit]
        row={'cohort':name,'raw_candidates':len(table),'admissible':int(table.usable_model_fit.sum()),
             'unflagged':len(unflagged),'zero_support':int(table.suspect_zero_pixel_support.sum()),
             'raw_unflagged_fraction':float(table.qc_unflagged_model_fit.mean()) if len(table) else None,
             'median_nominal_signal_unflagged':float(unflagged.photons.median()),
             'median_sx_px_unflagged':float(unflagged.sx.median()),'median_sy_px_unflagged':float(unflagged.sy.median()),
             'median_nominal_lpx_nm_unflagged':float(unflagged.lpx_nm.median()),
             'median_nominal_lpy_nm_unflagged':float(unflagged.lpy_nm.median())}
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'threshold_cohort_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.7),layout='constrained')
    colors=['#3b86bc','#ea8a31','#4b9c73']
    for (name,table),color in zip(cohorts.items(),colors):
        part=table.loc[table.qc_unflagged_model_fit]
        axes[0].hist(part.photons,bins=np.linspace(0,2000,45),histtype='step',color=color,label=name)
        axes[1].hist(part.sx,bins=np.linspace(.5,4.5,40),histtype='step',color=color,label=name)
        axes[2].hist(part.lpx_nm,bins=np.linspace(0,250,45),histtype='step',color=color,label=name)
    for ax,label in zip(axes,['Nominal detected signal (e)','Gaussian X sigma (pixels)','Nominal X CRLB SD (nm)']):
        ax.set_xlabel(label);ax.set_ylabel('Unflagged fits / bin')
    axes[0].legend(fontsize=7)
    fig.suptitle('Threshold membership changes, original frames 0–49\nHeuristic-unflagged subsets; displayed ranges omit tails retained in CSV',fontsize=12)
    save_figure(fig,out/'04_threshold_membership_distributions.png')
    return summaries


def membership_pair_components(full,kept_ids,result,out,label,pixel_size_nm):
    """Additive pair components at one common beta; mixed pairs are retained."""
    edges=result.parameters['edges_m'];bins=len(edges)-1
    totals=np.zeros((3,bins),np.int64);backgrounds=np.zeros_like(totals)
    framegroups={int(f):g for f,g in full.groupby('frame')}
    for frame,a in framegroups.items():
        axy=a[['x','y']].to_numpy()*pixel_size_nm*1e-9;achanged=~a.candidate_id.isin(kept_ids).to_numpy()
        for mode,b in [('inter',a),('total',framegroups.get(frame+1))]:
            if b is None:continue
            bxy=b[['x','y']].to_numpy()*pixel_size_nm*1e-9;bchanged=~b.candidate_id.isin(kept_ids).to_numpy()
            distance=np.sqrt(np.sum((axy[:,None,:]-bxy[None,:,:])**2,axis=2))
            category=achanged[:,None].astype(int)+bchanged[None,:].astype(int)
            valid=distance<=edges[-1]
            if mode=='inter':valid&=distance>0
            target=totals if mode=='total' else backgrounds
            for code in [0,1,2]:target[code]+=np.histogram(distance[valid&(category==code)],bins=edges)[0]
    np.testing.assert_array_equal(totals.sum(axis=0),result.counts['total_histogram'])
    np.testing.assert_array_equal(backgrounds.sum(axis=0),result.counts['inter_histogram'])
    p=totals/totals.sum();q=backgrounds/backgrounds.sum();beta=result.distributions['beta']
    intra=p-beta*q
    np.testing.assert_allclose(intra.sum(axis=0),result.distributions['intra_contribution'],rtol=0,atol=1e-14)
    mids=np.asarray(result.parameters['bin_mids_m'])*1e6;density=intra/(np.diff(edges)*1e6)
    names=['kept_kept','mixed','changed_changed'];summaries=[]
    data={'distance_um':mids}
    for i,name in enumerate(names):
        data[name+'_total_count']=totals[i];data[name+'_inter_count']=backgrounds[i];data[name+'_signed_intra_density_um_inv']=density[i]
        summaries.append({'comparison':label,'component':name,'total_pair_count':int(totals[i].sum()),
            'total_pair_fraction':float(p[i].sum()),'inter_pair_fraction':float(q[i].sum()),
            'signed_intra_mass':float(intra[i].sum()),'signed_intra_mass_0_300_nm':float(intra[i,mids<=.3].sum()),
            'common_beta':float(beta)})
    pd.DataFrame(data).to_csv(out/(label+'_pair_components.csv'),index=False)
    return mids,density,summaries


def tardis_figures(results,selected,tables,config,out):
    colors=['#3b86bc','#ee8b22','#489b6e','#ad67ab','#666666']
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.4),layout='constrained')
    for ((box,factor),result),color in zip(results.items(),colors):
        label=f'Box {box}, {factor:g}x (n={result.counts["localizations"]})'
        table=selected[(box,factor)];counts=table.groupby('frame').size().reindex(np.arange(50),fill_value=0)
        axes[0,0].plot(np.arange(50)*.02,counts,color=color,lw=1,label=label)
        mids=result.parameters['bin_mids_m']*1e6;d=result.distributions
        axes[0,1].plot(mids,d['total_density_um_inv'],color=color,lw=1,label=label)
        axes[1,0].plot(mids,d['inter_conditional_density_um_inv'],color=color,lw=1,label=label)
        axes[1,1].plot(mids,d['intra_contribution_density_um_inv'],color=color,lw=1,label=f'B{box} {factor:g}x: beta={d["beta"]:.3f}')
    axes[0,0].set(xlabel='Original time (s)',ylabel='Unflagged points / frame',title='Continuous frames 0–49')
    axes[0,0].legend(fontsize=8)
    for ax,title in [(axes[0,1],'Adjacent-frame Total'),(axes[1,0],'Same-frame background shape'),(axes[1,1],'Signed Intra contribution (zoom)')]:
        ax.set(xlabel='Distance (um)',ylabel='Density / contribution (1/um)',title=title,xlim=(0,.6 if ax is axes[1,1] else 3))
        ax.axhline(0,color='#aaaaaa',lw=.7)
    axes[1,1].legend(fontsize=8)
    fig.suptitle('Exploratory TARDIS tau=1 (20 ms), full target TIFF field\nSame selection rules; nominal calibration; no tracking, mask or uncertainty correction',fontsize=12)
    save_figure(fig,out/'05_continuous_tardis_comparison.png')
    ref=selected[(9,1.)];high=selected[(9,1.25)];low=selected[(9,.75)]
    # Shared fits are identical within one box; memberships, not refitting, differ.
    assert set(ref.candidate_id).issubset(set(low.candidate_id))
    assert set(high.candidate_id).issubset(set(ref.candidate_id))
    for base,other in [(ref,low),(high,ref)]:
        a=base.set_index('candidate_id');b=other.set_index('candidate_id').loc[a.index]
        np.testing.assert_array_equal(a[['frame','x','y']].to_numpy(),b[['frame','x','y']].to_numpy())
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained');summaries=[]
    for ax,label,full,kept,result in [(axes[0],'low_threshold_additions',low,set(ref.candidate_id),results[(9,.75)]),
          (axes[1],'high_threshold_removals',ref,set(high.candidate_id),results[(9,1.)])]:
        mids,dens,rows=membership_pair_components(full,kept,result,out,label,config['pixel_size_nm']);summaries+=rows
        for i,name,color in [(0,'kept–kept','#3b86bc'),(1,'mixed','#489b6e'),(2,'changed–changed','#ee8b22')]:ax.plot(mids,dens[i],color=color,lw=1,label=name)
        ax.plot(mids,dens.sum(axis=0),'k--',lw=1,label='sum = full signed Intra')
        ax.axhline(0,color='#aaaaaa',lw=.7);ax.set(xlim=(0,.6),xlabel='Distance (um)',ylabel='Signed contribution (1/um)',title=label.replace('_',' '));ax.legend(fontsize=8)
    fig.suptitle('Changed detections also create mixed displacement pairs\nComponents use the full set’s common beta and common histogram denominators',fontsize=12)
    save_figure(fig,out/'06_changed_detection_pair_contributions.png')
    pd.DataFrame(summaries).to_csv(out/'pair_membership_summary.csv',index=False)
    return summaries


def write_report(out,stats,cohorts,tardis,pair_summary,padding,elapsed):
    ref=next(s for s in stats if s['box']==9 and s['threshold_factor']==1.)
    lines=['# T1 C2 sign-aware QC and continuous-segment TARDIS review','',
           '**All brightness/CRLB fields remain manufacturer_nominal. No additional source frames were detected or fitted.**','',
           '## Correction to the first pilot’s interpretation','',
           'The previous 1548/5891 (26.3%) “inadmissible” count included negative Gaussian width parameters. The backend model uses amplitude times exp[−0.5(dx²/sx²+dy²/sy²)] + bg. Width sign is degenerate. A negative signed integral `A×2πsxsy` can also be an artifact of that convention; inspect the actual backend amplitude A.', '',
           f"Reference, all 150 frames: {ref['raw_negative_width']} negative-width rows; {ref['raw_nonpositive_signed_integral']} nonpositive signed-integral rows, but only {ref['nonpositive_backend_amplitude']} nonpositive Gaussian amplitudes. {ref['sign_only_parameter_issue']} rows pass other parameter checks after sign normalization; {ref['sign_only_still_width_flagged']} of those remain width-flagged. Revised inadmissible count: {ref['revised_inadmissible']}/{ref['raw_rows']} ({ref['revised_inadmissible']/ref['raw_rows']*100:.1f}%). These are parameter/QC categories, not confirmed false detections or coordinate error rates.", '',
           f"Sign normalization alone gives {ref['canonical_unflagged_before_padding_check']} initially unflagged fits, including {ref['sign_rescued_unflagged']} recovered from the sign convention. However, {ref['sign_rescued_with_zero_detection_support']} of those recovered points read zero source pixels in the detector’s one-pixel halo. After flagging zero-containing detector support, {ref['revised_unflagged']} reference fits remain unflagged. The new detector-support flag catches {ref['newly_flagged_detection_padding']} initially unflagged fits in total. Original X/Y and optimized objective/iterations are unchanged. Model-image maximum difference: {ref['model_max_abs_change_nominal_e_per_pixel']:.3g} nominal e/pixel. No absolute-value operation is applied to a genuinely negative amplitude.", '',
           'Reason totals overlap; use `reason_pattern_counts.csv` for disjoint combinations, not the sum of individual reason counts. The source run and its original report are preserved. Corrected audit tables are in `canonical_fits/`; raw signed backend parameters and v1 fields remain alongside canonical values.', '',
           '## Image inspection','',
           'Image sheets show median examples within declared strata: normal fits from each source segment, negative amplitude/background or outside-box centers, and low-threshold additions/high-threshold removals. A separate detector-support sheet compares sign-recovered points with/without zero pixels in the halo: the raw context can contain a fixed-zero edge outside the fitting patch. The detector’s finite-difference gradients read that edge even when the fitted Gaussian does not. Source pixels are unchanged and the zero origin remains unknown. These examples do not establish molecule/false-positive truth.', '',
           '## Changed detections, continuous frames 0–49','',
           '| Cohort | Raw candidates | Canonical admissible | Heuristic-unflagged | Median nominal signal (e), unflagged | Median nominal X/Y CRLB (nm), unflagged |',
           '|---|---:|---:|---:|---:|---:|']
    for row in cohorts:
        lines.append(f"| {row['cohort']} | {row['raw_candidates']} | {row['admissible']} | {row['unflagged']} | {row['median_nominal_signal_unflagged']:.1f} | {row['median_nominal_lpx_nm_unflagged']:.1f}/{row['median_nominal_lpy_nm_unflagged']:.1f} |")
    lines+=['','These counts explain detection membership changes rather than coordinate changes of shared fits. Same-box shared candidates produce identical positions by construction; that is not independent precision validation. Low-threshold additions and high-threshold removals can represent weak molecules or noise/background; raw-image review and the unverified export/noise model limit classification.','',
            '## Exploratory TARDIS distributions','',
            'Only the single continuous 0–49 segment is analyzed: 50 original frames, 20 ms lag. Main inputs are canonical, admissible, finite-precision fits without width, residual-overlap, fitting-zero or detector-zero flags. The detector support is the fitting square plus its one-pixel axial halo (unused corners excluded). This is a common, inspectable image-quality rule, not a calibrated molecule classifier. Pixel positions use 117 nm/pixel. Existing `tardis_tau1` is imported read-only. Common parameters: 300 bins, 0–3 um, beta least-squares tail 2.1–3 um constrained to [0,1], same-frame ordered background with exact-zero distances removed. No tracking, ROI, mitochondrial mask, drift correction, CRLB-based correction or diffusion fit.', '',
            '| Box | Threshold factor | Input unflagged points | True adjacent frame pairs | Retained Total pairs | Beta | Raw beta | Negative Intra mass |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in tardis:
        lines.append(f"| {s['box']} | {s['threshold_factor']:g} | {s['localizations']} | {s['source_frame_pairs']} | {s['total_retained_pairs']} | {s['beta']:.5f} | {s['beta_raw']:.5f} | {s['negative_intra_mass']:.5f} |")
    lines+=['','### Detector-padding sensitivity control','',
            'The pre-halo-check input is retained as a sensitivity control, not used as the main input. Both sets retain original frame IDs. `detector_padding_sensitivity.csv` records counts and beta before/after the extra source-quality flag. This changes input membership, not fitted coordinates.','',
            '| Box | Threshold factor | Before detector check n | After detector check n | Beta before | Beta after |',
            '|---:|---:|---:|---:|---:|---:|']
    for s in padding:lines.append(f"| {s['box']} | {s['threshold_factor']:g} | {s['before_n']} | {s['after_n']} | {s['before_beta']:.5f} | {s['after_beta']:.5f} |")
    lines+=['','These are signed background-subtracted contributions, not necessarily nonnegative PDFs. Beta clipping, sparse tails and negative bins are recorded per configuration; 50 frames is a short sensitivity test, not a validated diffusion measurement. Adding/removing points also creates/removes mixed pairs with retained points. `qc/pair_membership_summary.csv` and the component figure use one common beta per full set so all kept/mixed/changed components sum exactly to its Intra.','',
            '## Verification, scope and unresolved questions','',
            'Twelve synthetic/adapter tests passed, including all four width-sign combinations, invariant source/model and X/Y CRLB, genuine negative amplitude retention, detection-halo zeros outside the fit with unused corners excluded, and a 49→1225 gap test that does not create a 20 ms pair. Every real TARDIS Total and background histogram bin in five main groups and five padding controls was compared against an independent explicit pair enumerator. Every real frame pair is explicitly adjacent in 0–49. Existing TARDIS code is fingerprinted before and after and unchanged.', '',
            'Input-source pixels, old CSV/HDF5 and first-pilot metadata remain intact. Raw/canonical Gaussian model images and X/Y are verified invariant for every reviewed candidate. Full-distribution pair partitions are independently checked to reconstruct all Total, background and signed Intra bins.', '',
            f'Elapsed review time: {elapsed:.1f} s. Exact command, review config, software versions, source hashes, tests and logs are stored beside this report.','',
            'Still unknown: measured camera offset/sensitivity, per-pixel noise calibration, exposure/mode, TIFF correction/rescaling/crop history, and cause of persistent zero pixels. Low observed/nominal variance and deviance remain unresolved. Empirical ~50 nm and nominal CRLB are different metrics until their definitions and calibration are established. No parameter is optimized to match 50 nm.', '',
            '**Stopped after QC review of the existing 150-frame cache and exploratory TARDIS on frames 0–49. No full 2500-frame run or best-parameter selection.**','',
            'Source: [Picasso v0.11.3 Gaussian backend](https://github.com/jungmannlab/picasso/blob/v0.11.3/picasso/fitting/gaussfit.py).','']
    (out/'REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='review_config.yaml')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    review=yaml.safe_load(Path(args.config).read_text());source=(root/review['source_run']).resolve()
    assert source.is_relative_to(root/'runs') and (source/'COMPLETED.json').exists()
    assert review['tardis_frame_range']==[0,49] and review['new_tiff_frames']==0 and not review['refit']
    config=yaml.safe_load((source/'config.yaml').read_text());started=time.perf_counter()
    out=root/'runs'/('T1_C2_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    log=(out/'run.log').open('w',encoding='utf-8');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    for name in ['canonical_fits','qc','tardis','reproducibility']:(out/name).mkdir()
    command=shlex.join([sys.executable,*sys.argv]);(out/'run_command.txt').write_text(f'cd {shlex.quote(str(root))}\n{command}\n')
    (out/'review_config.yaml').write_text(yaml.safe_dump(review,sort_keys=False));shutil.copy2(source/'config.yaml',out/'original_fit_config.yaml')
    shutil.copy2(source/'software_versions.json',out/'software_versions.json')
    for name in ['run_review.py','pilot_review.py','pilot_core.py','pilot_qc.py','test_pilot.py','test_review.py','requirements.lock.txt']:
        shutil.copy2(root/name,out/'reproducibility'/name)
    tardis_hashes={str(p):sha256_file(p) for p in (TARDIS_ROOT/'tardis_tau1').glob('*.py')}
    original_hashes={str(p):sha256_file(p) for p in list((source/'fits').glob('*'))+[source/'sample_frames.hdf5',source/'RESULTS.md']}
    dump_json(out/'input_provenance.json',{'source_run':str(source),'source_tiff_sha256':json.loads((source/'input_metadata.json').read_text())['source_sha256'],
                                         'cached_file_sha256':original_hashes,'tardis_source_sha256':tardis_hashes,'no_source_tiff_frames_read_or_refitted':True})
    tests=subprocess.run([sys.executable,'-u','-m','unittest','-v','test_pilot','test_review'],cwd=root,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Review validation failed')
    with h5py.File(source/'sample_frames.hdf5') as h:frames=h['original_frame_indices'][()];movie=h['images'][()]
    tables={};stats=[];patterns=[]
    for box in config['box_sizes']:
        for factor in config['threshold_factors']:
            name=key_name(box,factor);original=pd.read_csv(source/'fits'/(name+'.csv'),keep_default_na=False,na_values=['nan','NaN','inf','-inf'])
            signed=np.load(source/'fits'/(name+'_model_parameters.npy'))
            revised,canonical,s=review_cached_fit(original,signed,box,config,frames,movie)
            revised['threshold_factor']=factor;revised['box_size']=box
            tables[(box,factor)]=revised;s.update(box=box,threshold_factor=factor);stats.append(s)
            revised.groupby(['fit_parameter_issues','width_sign_canonicalized'],dropna=False).size().reset_index(name='rows').assign(box=box,threshold_factor=factor).to_csv(out/'qc'/(name+'_reason_patterns.csv'),index=False)
            for (reason,sign),count in revised.groupby(['fit_parameter_issues','width_sign_canonicalized'],dropna=False).size().items():
                patterns.append({'box':box,'threshold_factor':factor,'other_parameter_issues':reason or 'none',
                                 'negative_width_sign':bool(sign),'rows':int(count)})
            params=json.loads((source/'fits'/(name+'.json')).read_text());params.update({'Sign convention':'positive sx/sy, N=A*2*pi*abs(sx)*abs(sy); raw signed backend fields retained',
                           'No refit':True,'Detector support zero check':True,
                           'Detector support':'fit box plus one-pixel axial halo; unused corner pixels excluded',
                           'Original audited input':str(source/'fits'/(name+'.hdf5'))})
            for field in revised.columns:
                if field.startswith('v1_'):FIELD_UNITS[field]='original v1 audit field: '+FIELD_UNITS.get(field[3:],'inspection flag')
            FIELD_UNITS.update({'sign_only_parameter_issue':'negative backend width sign but canonical fit is admissible; may still be QC flagged',
                       'rescued_by_sign_canonicalization':'previously excluded by width sign, passes pre-detector-halo checks after reparameterization; may fail new detector-support check; no new optimization',
                       'unflagged_before_detection_padding_check':'passes earlier fitting-window checks before detector halo check; retained for sensitivity control',
                       'threshold_factor':'multiplier of box-specific baseline net-gradient threshold','box_size':'camera pixels'})
            save_locs(out/'canonical_fits'/(name+'.hdf5'),revised,params)
            (out/'canonical_fits'/(name+'.yaml')).write_text(yaml.safe_dump_all([params],sort_keys=False))
            print(f'{name}: other-parameter inadmissible {s["revised_inadmissible"]}, sign-only {s["sign_only_parameter_issue"]}, unflagged {s["revised_unflagged"]}',flush=True)
    pd.DataFrame(stats).to_csv(out/'sign_and_parameter_summary.csv',index=False);pd.DataFrame(patterns).to_csv(out/'reason_pattern_counts.csv',index=False)
    dump_json(out/'field_units.json',FIELD_UNITS)
    qc_examples(tables,movie,frames,config,out/'qc');cohorts=cohort_summary(tables,out/'qc')
    results={};selected={};tardis_summary=[];padding=[];padding_results={}
    for box,factor in review['comparison_groups']:
        name=key_name(box,factor);print('TARDIS and independent pair check:',name,'frames 0–49',flush=True)
        result,table,pos,pairs,tauconfig=analyze_continuous_subset(tables[(box,factor)],config)
        results[(box,factor)]=result;selected[(box,factor)]=table
        result.metadata['detector_support_zero_check']=True
        save_result(result,out/'tardis'/name,tauconfig,'cell')
        table.to_csv(out/'tardis'/name/'selected_localizations.csv',index=False)
        np.savetxt(out/'tardis'/name/'positions.csv',pos,delimiter=',',header='original_frame,x_m,y_m',comments='',fmt='%.17g')
        pd.DataFrame(pairs,columns=['source_frame','target_frame']).to_csv(out/'tardis'/name/'actual_adjacent_frame_pairs.csv',index=False)
        s={'box':box,'threshold_factor':factor,'localizations':len(table),'source_frame_pairs':len(pairs),
           'total_retained_pairs':int(result.counts['total_retained_pairs']),'beta':float(result.distributions['beta']),
           'beta_raw':float(result.distributions['beta_raw']),'negative_intra_mass':float(result.qc['metrics']['negative_intra_mass']),
           'qc_flags':result.qc['flags'],'independent_histograms_exact':True}
        tardis_summary.append(s);print(json.dumps(s),flush=True)
        before_table=tables[(box,factor)].copy();before_table.qc_unflagged_model_fit=before_table.unflagged_before_detection_padding_check
        before,old_selected,old_pos,old_pairs,_=analyze_continuous_subset(before_table,config)
        before.metadata['detector_support_zero_check']=False
        save_result(before,out/'tardis'/name/'before_detector_padding_check',tauconfig,'cell')
        old_selected.to_csv(out/'tardis'/name/'before_detector_padding_check'/'selected_localizations.csv',index=False)
        padding.append({'box':box,'threshold_factor':factor,'before_n':len(old_selected),'after_n':len(table),
                        'before_beta':float(before.distributions['beta']),'after_beta':s['beta']})
        padding_results[(box,factor)]=(before,result)
    pd.DataFrame(tardis_summary).to_csv(out/'tardis_summary.csv',index=False)
    pd.DataFrame(padding).to_csv(out/'detector_padding_sensitivity.csv',index=False)
    pair_summary=tardis_figures(results,selected,tables,config,out/'qc')
    fig,ax=plt.subplots(figsize=(9.8,4.8),layout='constrained')
    for name,result,color in [('Before detector zero check',padding_results[(9,1.)][0],'#a563ae'),
                              ('After detector zero check',padding_results[(9,1.)][1],'#357eb3')]:
        ax.plot(result.parameters['bin_mids_m']*1e6,result.distributions['intra_contribution_density_um_inv'],color=color,
                label=f'{name}: n={result.counts["localizations"]}, beta={result.distributions["beta"]:.3f}')
    ax.axhline(0,color='gray',lw=.7);ax.set(xlim=(0,.6),xlabel='Distance (um)',ylabel='Signed Intra contribution (1/um)',
          title='Reference box 9, baseline threshold: detector-padding sensitivity\nSame frames 0–49, same fitted coordinates; nominal calibration')
    ax.legend(fontsize=9);save_figure(fig,out/'qc/08_detector_padding_distribution_control.png')
    for path,digest in original_hashes.items():assert sha256_file(path)==digest,'Source audit file changed'
    for path,digest in tardis_hashes.items():assert sha256_file(path)==digest,'TARDIS source changed'
    elapsed=time.perf_counter()-started
    dump_json(out/'validation.json',{'synthetic_tests':12,'passed':True,'all_cached_model_images_and_xy_invariant':True,
                'ten_tardis_main_and_control_results_checked_binwise_against_independent_enumerator':True,'only_true_0_49_adjacent_frame_pairs':True,
                'pair_membership_decomposition_exact':True,'tardis_source_unchanged':True,'original_cache_unchanged':True})
    dump_json(out/'runtime.json',{'seconds':elapsed,'finished_utc':datetime.now(timezone.utc).isoformat(),'new_frames_fitted':0,'tardis_continuous_frame_count':50})
    write_report(out,stats,cohorts,tardis_summary,pair_summary,padding,elapsed)
    dump_json(out/'COMPLETED.json',{'qc_review_complete':True,'continuous_frame_range':[0,49],'full_movie_processed':False})
    print('Completed review:',out,flush=True)


if __name__=='__main__':main()
