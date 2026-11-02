"""Cached numerical inputs for figures; scientific summaries never run in refresh."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from ..types import StageResult
from .distances import dump,load_json,joined_table,rules,kernel,VENDOR,GROUPS,uid_column

REPRESENTATIVE_FRAMES=(0,49,200,625,1200,1250,1875,2000,2450,2499)


def input_file(ctx,stages,names,required=True):
    for stage in stages:
        artifact=ctx.inputs.get(stage)
        if artifact:
            for name in names:
                path=artifact.files.get(name)
                if path and Path(path).is_file(): return Path(path)
    if required: raise FileNotFoundError('Required saved input is missing: '+','.join(stages)+' / '+','.join(names))
    return None


def crop_for(ctx,roi):
    given=ctx.config.get('crop_xyxy_px',ctx.sample.metadata.get('crop_xyxy_px'))
    if given:return list(map(int,given))
    y,x=np.nonzero(roi)
    if not len(x): raise ValueError('Empty authoritative cell ROI')
    return [max(0,int(x.min())-10),max(0,int(y.min())-10),min(roi.shape[1],int(x.max())+11),min(roi.shape[0],int(y.max())+11)]


def _hist(values,edges):
    a=np.asarray(values,float);finite=np.isfinite(a);valid=finite&(a>=edges[0])&(a<=edges[-1]);h=np.histogram(a[valid],edges)[0]
    info=dict(total=len(a),displayed=int(h.sum()),below_range=int((finite&(a<edges[0])).sum()),above_range=int((finite&(a>edges[-1])).sum()),nonfinite=int((~finite).sum()))
    return h,info


def _describe(values,edges,counts):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    modes=np.flatnonzero(counts==counts.max()) if counts.size and counts.max()>0 else []
    return dict(mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,mode_bin_center=float((edges[modes[0]]+edges[modes[0]+1])/2) if len(modes)==1 else None,mode_status='unique_native_histogram_bin' if len(modes)==1 else 'unavailable_tied_or_empty_bins')


def execute_c1_mean(ctx):
    import tifffile
    ctx.output_dir.mkdir(parents=True,exist_ok=True);summed=None;frames=0
    expected=int(ctx.config.get('acquisition',{}).get('frames',2500))
    with tifffile.TiffFile(ctx.sample.C1_file) as movie:
        if len(movie.pages)!=expected: raise ValueError('C1 page count differs from acquisition contract')
        for f,page in enumerate(movie.pages):
            a=page.asarray()
            if a.dtype!=np.uint16: raise ValueError('Raw C1 arithmetic-mean method requires uint16 pixels')
            if summed is None:summed=np.zeros(a.shape,np.uint64)
            if a.shape!=summed.shape: raise ValueError('C1 page shape changed')
            summed+=a;frames+=1
            if frames%250==0 or frames==expected:ctx.progress('C1 raw arithmetic-mean checkpoint',frame=frames)
    mean=(summed.astype(np.float64)/frames).astype(np.float32)
    np.save(ctx.output_dir/'C1_mean.npy',mean,allow_pickle=False)
    metadata=dict(frames=frames,axes='YX',shape=list(mean.shape),units='raw ADU',accumulation='uint64 sum; float64 division; float32 persistence',all_source_pixels_included=True,zeros_included=True,alignment=False,filtering=False)
    dump(ctx.output_dir/'metadata.json',metadata)
    return StageResult(dict(mean=ctx.output_dir/'C1_mean.npy',metadata=ctx.output_dir/'metadata.json'),metadata)


def execute(ctx):
    c=rules(ctx);table,selected,keep=joined_table(ctx);out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True)
    for field in ('photons_e','bg_e_per_pixel','sx_px','sy_px'):
        if field in selected:selected[field]=pd.to_numeric(selected[field],errors='coerce')
    edges=np.asarray(ctx.parameters.get('SE_edges_nm',np.arange(0,205,5)),dtype=float);labels=selected.tardis_group.to_numpy();which={'All':np.ones(len(selected),bool),'On':labels=='On','Not-on':labels=='Not-on'}
    arrays=dict(SE_edges_nm=edges);outside={};description={};histrows=[]
    photons=selected.get('photons_e',pd.Series(np.nan,index=selected.index)).to_numpy(float)
    good=photons[np.isfinite(photons)&(photons>=0)];photon_edges=np.linspace(0,max(1,float(good.max())) if len(good) else 1,int(ctx.parameters.get('signal_bins',100))+1);arrays['photons_edges_e']=photon_edges
    for field,bins in [('x_se_nm',edges),('y_se_nm',edges),('photons_e',photon_edges)]:
        values=selected[field].to_numpy(float) if field in selected else photons
        for group,mask in which.items():
            key=field+'__'+group;counts,info=_hist(values[mask],bins);arrays[key]=counts;outside[key]=info;description[key]=_describe(values[mask],bins,counts)
            histrows.extend(dict(field=field,group=group,left=bins[i],right=bins[i+1],count=int(n)) for i,n in enumerate(counts))
    x=selected.x_se_nm.to_numpy(float);y=selected.y_se_nm.to_numpy(float)
    for group,mask in which.items():
        valid=mask&np.isfinite(x)&np.isfinite(y)&(x>=edges[0])&(x<=edges[-1])&(y>=edges[0])&(y<=edges[-1])
        h=np.histogram2d(x[valid],y[valid],bins=(edges,edges))[0].astype(np.int64);arrays['SE_2D__'+group]=h
        outside['SE_2D__'+group]=dict(total=int(mask.sum()),displayed=int(h.sum()),outside_range=int((mask&np.isfinite(x)&np.isfinite(y)&~valid).sum()),nonfinite=int((mask&~(np.isfinite(x)&np.isfinite(y))).sum()))
    np.savez_compressed(out/'histogram_counts_and_edges.npz',**arrays)
    dump(out/'histogram_metadata.json',dict(outside_counts=outside,descriptive_statistics=description,selected_count=len(selected),shared_2D_vmin=0,shared_2D_vmax=max(1,max(int(arrays['SE_2D__'+g].max()) for g in which)),nominal_uncertainty=True))
    pd.DataFrame(histrows).to_csv(out/'histogram_1D_bin_counts.csv',index=False)
    n=c['frames'];frame=pd.DataFrame(dict(frame=np.arange(n),time_s=np.arange(n)*c['frame_interval_s']))
    for group in (*GROUPS,'unknown'):frame[group.replace('-','_')]=selected.loc[selected.tardis_group.eq(group)].groupby('frame').size().reindex(range(n),fill_value=0).to_numpy()
    ftable=table.assign(frame=pd.to_numeric(table.frame))
    frame['QC_excluded']=ftable.loc[~keep].groupby('frame').size().reindex(range(n),fill_value=0).to_numpy();frame['ROI_candidates']=ftable.groupby('frame').size().reindex(range(n),fill_value=0).to_numpy()
    if not np.array_equal(frame[['On','Not_on','unknown','QC_excluded']].sum(1),frame.ROI_candidates): raise ValueError('Per-frame count conservation failed')
    frame.to_csv(out/'per_frame_counts.csv',index=False)
    roi=np.load(input_file(ctx,('roi','roi_intersection','mask_roi','cleanup'),('mask','roi','roi_mask')),allow_pickle=False)
    full=np.load(input_file(ctx,('cleanup','mask_cleanup','mask'),('full_mask','mask','clean_full_mask')),mmap_mode='r',allow_pickle=False)
    vertices=np.load(input_file(ctx,('roi','cleanup','mask_cleanup'),('vertices','roi_vertices')),allow_pickle=False)
    mapping_path=input_file(ctx,('support_counts','support_count','support','cleanup','roi_intersection','mask_cleanup'),('frame_map','mapping'),required=False)
    cleanup=ctx.inputs.get('mask_cleanup',ctx.inputs.get('cleanup'))
    contract=cleanup.metadata.get('contract',{}) if cleanup else {}
    if cleanup and cleanup.files.get('contract'):contract=load_json(cleanup.files['contract'])
    if mapping_path and mapping_path.suffix=='.npy':mapping=np.load(mapping_path,allow_pickle=False)
    else:
        saved=load_json(mapping_path) if mapping_path else contract
        mapping_spec=saved.get('frame_to_block')
        if mapping_spec is None:raise FileNotFoundError('Explicit original-frame to mask-block mapping is required; no f//window inference')
        if isinstance(mapping_spec,dict):mapping=np.asarray([mapping_spec.get(str(f),mapping_spec.get(f,-1)) for f in range(n)],dtype=int)
        else:mapping=np.asarray(mapping_spec,dtype=int)
    if full.ndim!=3 or roi.shape!=full.shape[1:] or len(mapping)!=n:raise ValueError('Mask/ROI/time-map shapes disagree')
    validmap=(mapping>=0)&(mapping<len(full))
    if contract.get('block_valid') is not None:validmap &= np.asarray(contract['block_valid'],bool)[np.clip(mapping,0,len(full)-1)]
    fullcounts=np.count_nonzero(full,axis=(1,2));roicounts=np.count_nonzero(full&roi,axis=(1,2))
    areas=pd.DataFrame(dict(frame=np.arange(n),time_s=np.arange(n)*c['frame_interval_s'],block_index=mapping,mask_valid=validmap,full_mask_area_pixel2=np.where(validmap,fullcounts[np.clip(mapping,0,len(full)-1)],np.nan),ROI_mask_area_pixel2=np.where(validmap,roicounts[np.clip(mapping,0,len(full)-1)],np.nan)))
    for scope in ('full','ROI'):areas[scope+'_mask_area_um2']=areas[scope+'_mask_area_pixel2']*(c['pixel_size_nm']/1000)**2
    areas.to_csv(out/'mask_area_by_frame.csv',index=False)
    areastats=[dict(scope=scope,first_um2=float(areas[scope+'_mask_area_um2'].iloc[0]),mean_um2=float(areas[scope+'_mask_area_um2'].mean()) if validmap.all() else np.nan,median_um2=float(areas[scope+'_mask_area_um2'].median()) if validmap.all() else np.nan,min_um2=float(areas[scope+'_mask_area_um2'].min()) if validmap.all() else np.nan,max_um2=float(areas[scope+'_mask_area_um2'].max()) if validmap.all() else np.nan,valid_frames=int(validmap.sum()),expected_frames=n,status='available' if validmap.all() else 'unavailable_missing_mask_blocks') for scope in ('ROI','full')]
    pd.DataFrame(areastats).to_csv(out/'mask_area_summary.csv',index=False)
    adj=[]
    for b in range(1,len(full)):
        a=full[b-1]&roi;z=full[b]&roi;union=np.count_nonzero(a|z);adj.append(dict(block_before=b-1,block_after=b,frame_after=int(np.flatnonzero(mapping==b)[0]) if (mapping==b).any() else -1,IoU=float(np.count_nonzero(a&z)/union) if union else 1.))
    pd.DataFrame(adj,columns=['block_before','block_after','frame_after','IoU']).to_csv(out/'mask_adjacent_pair_metrics.csv',index=False)
    crop=crop_for(ctx,roi);scale=4;x0,y0,x1,y1=crop
    grids={};xyarrays={};xedges=np.linspace(x0-.5,x1-.5,(x1-x0)*scale+1);yedges=np.linspace(y0-.5,y1-.5,(y1-y0)*scale+1)
    for group in ('On','Not-on','unknown','All'):
        xy=selected.loc[np.ones(len(selected),bool) if group=='All' else labels==group,['fit_x_px','fit_y_px']].to_numpy(float)
        xyarrays[group]=xy;grids[group]=np.histogram2d(xy[:,1],xy[:,0],bins=(yedges,xedges))[0].astype(np.uint32)
        if int(grids[group].sum())!=len(xy):raise ValueError('Display crop loses selected centers')
    np.savez_compressed(out/'pooled_count_maps.npz',**grids,crop_xyxy_px=np.asarray(crop),nm_per_bin=np.asarray(c['pixel_size_nm']/scale))
    np.savez_compressed(out/'pooled_coordinates.npz',**xyarrays,vertices=vertices,roi_mask=roi,crop_xyxy_px=np.asarray(crop))
    selected.to_csv(out/'selected_localizations.csv.gz',index=False,compression='gzip')
    table.loc[table.frame.astype(int).isin(REPRESENTATIVE_FRAMES)].to_csv(out/'representative_fits.csv.gz',index=False,compression='gzip')
    selected.loc[selected.frame.astype(int).isin(REPRESENTATIVE_FRAMES)].to_csv(out/'representative_selected.csv.gz',index=False,compression='gzip')
    scopes=[]
    for scope,start,stop in kernel().scopes(c):
        for group in GROUPS:
            subset=selected.loc[selected.tardis_group.eq(group)&selected.frame.between(start,stop-1)]
            row=dict(scope=scope,group=group,points=len(subset),start_frame=start,stop_exclusive=stop,center_s=(start+stop-1)/2*c['frame_interval_s'])
            for field in ('x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px'):
                if field in subset:
                    for name,q in [('q10',.1),('median',.5),('q90',.9)]:row[field+'_'+name]=subset[field].quantile(q)
            scopes.append(row)
    pd.DataFrame(scopes).to_csv(out/'scope_localization_statistics.csv',index=False)
    decomp=ctx.inputs.get('decomposition');intrastats=[]
    if decomp:
        extras=kernel()
        for group in GROUPS:
            curve=pd.read_csv(decomp.file('directory')/'distribution'/group/'full.csv',float_precision='round_trip');intrastats.append(extras._statistics(group,curve,c))
        pd.DataFrame(intrastats).to_csv(out/'intra_descriptive_statistics.csv',index=False)
        # Tail normalization is scientific; plotters read these persisted values.
        tail=[]
        for group in GROUPS:
            curve=pd.read_csv(decomp.file('directory')/'distribution'/group/'full.csv');mask=curve.distance_mid_um.between(*c['background_fit_um'])
            part=curve.loc[mask,['distance_mid_um','total_count','background_ordered_nonzero_count']].copy()
            for key in ('total_count','background_ordered_nonzero_count'):
                denom=part[key].sum();part[key+'_tail_mass']=part[key]/denom if denom else np.nan
            part['group']=group;tail.append(part)
        pd.concat(tail).to_csv(out/'tail_normalized_diagnostics.csv',index=False)
    counts=dict(On=int((labels=='On').sum()),Not_on=int((labels=='Not-on').sum()),unknown=int((labels=='unknown').sum()),QC_excluded=int((~keep).sum()),selected=len(selected),ROI_candidates=len(table))
    metrics=dict(counts=counts,on_percentage=100*counts['On']/(counts['On']+counts['Not_on']) if counts['On']+counts['Not_on'] else None,first_mask_ROI_area_um2=areastats[0]['first_um2'],mean_ROI_area_um2=areastats[0]['mean_um2'])
    metadata=dict(config_rules=c,crop_xyxy_px=crop,metrics=metrics,area_statistics=areastats,representative_frames=[f for f in REPRESENTATIVE_FRAMES if f<n],per_axis_SE='manufacturer-nominal model standard error, nm',formal_inference=False,registration_status='identity / unverified',sources={k:v.id for k,v in ctx.inputs.items()})
    dump(out/'metadata.json',metadata)
    return StageResult(dict(directory=out,metadata=out/'metadata.json',histograms=out/'histogram_counts_and_edges.npz',pooled_countmaps=out/'pooled_count_maps.npz',per_frame_counts=out/'per_frame_counts.csv',metrics=out/'metadata.json'),metadata)

execute_statistics=execute
