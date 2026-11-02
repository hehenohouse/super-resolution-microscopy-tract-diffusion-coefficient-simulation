"""Render fixed-distance TARDIS outputs from saved numerical results only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, html, importlib.util, json, math, os, shutil, sys, textwrap

GROUPS=('On','Not-on')
COLORS={'old_k1':'#246B9E','close_1px':'#D57B17','close_2px':'#159C76'}

def _json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)

def _dump(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _descriptor(path):
    path=Path(path);stat=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        if stat.st_size<=2*1024*1024:digest.update(stream.read());kind='complete_small_file'
        else:
            digest.update(stream.read(65536));stream.seek(max(0,stat.st_size-65536));digest.update(stream.read(65536));kind='first_last_64KiB'
    return dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest.hexdigest(),fingerprint_kind=kind)

def _method_finish(original,label):
    def finish(figure,path,footer=None):
        title=getattr(figure,'_suptitle',None)
        if title is not None:
            text=title.get_text().replace(' k=1 groups',' groups').replace(', k=1, tau=',', tau=')
            if label not in text:text+=' | '+label
            title.set_text(textwrap.fill(text,115));title.set_fontsize(12)
        else:figure.suptitle(label,fontsize=12)
        if footer:footer=footer.replace('original QC and spatial labels retained.','original QC retained; saved fixed-distance assignment.')
        return original(figure,path,footer)
    return finish

def _ylim(values):
    import numpy as np
    finite=[array[np.isfinite(array)] for array in values]
    finite=[array for array in finite if len(array)]
    if not finite:return (-1.,1.)
    low=min(0.,min(float(array.min()) for array in finite));high=max(0.,max(float(array.max()) for array in finite))
    span=high-low
    if span<=0:span=1.
    return (low-.04*span,high+.07*span)

def _curve_fields(frame):
    signed=next((name for name in ['intra_density_um_inv','Intra_density_um_inv'] if name in frame),None)
    area=next((name for name in ['shape_density_um_inv','Intra_density_area_normalized','intra_density_area_normalized'] if name in frame),None)
    if not signed or not area:raise ValueError('Saved signed/area-normalized curve fields missing')
    return signed,area

def _write_index(figures,methods,comparison,standard,peaks):
    parts=['<!doctype html><meta charset="utf-8"><title>T5 distance-assignment TARDIS comparison</title>',
    '<style>body{font:16px Arial,sans-serif;margin:28px;max-width:1500px}img{width:100%;height:auto}section{margin-bottom:30px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}a{color:#246B9E}p{line-height:1.5}</style>',
    '<h1>WT TRAK2 BR1 T5: TARDIS by assignment distance</h1>',
    '<p>Original model-SE assignment, 1-pixel proximity (117 nm), and 2-pixel proximity (234 nm). All methods retain the same source localization values and QC. Cross-method lines are descriptive and have no difference-confidence bands.</p>',
    '<h2>Cross-method comparison</h2><div class="grid">']
    for name in comparison:
        parts.append('<section><a href="'+html.escape(name)+'">'+html.escape(name.replace('_',' ').replace('.png',''))+'</a><a href="'+html.escape(name)+'"><img src="'+html.escape(name)+'"></a></section>')
    parts.append('</div>')
    for method in methods:
        key=method['key'];parts.append('<h2>'+html.escape(method['label'])+'</h2>')
        parts.append('<p>Method-specific panels retain their own saved paired time-block bootstrap ranges. These ranges are exploratory and pointwise; no cross-method significance test was performed.</p><ul>')
        for name in standard[key]+[peaks[key]]:
            parts.append('<li><a href="'+html.escape(name)+'">'+html.escape(name.replace(key+'_','').replace('_',' ').replace('.png',''))+'</a></li>')
        parts.append('</ul>')
    parts.append('<p>Localization SE values are unchanged. Geometric proximity thresholds are assignment tolerances, not localization-precision estimates. C1/C2 registration remains independently unverified.</p>')
    (figures/'INDEX.html').write_text('\n'.join(parts),encoding='utf-8')

def render_outputs(config_path):
    """Render all standard and comparison PNGs, never numerical or raw-image stages."""
    config_path=Path(config_path).resolve();cfg=_json(config_path)
    output=Path(cfg.get('output',cfg.get('output_directory'))).resolve();figures=Path(cfg['figures']).resolve()
    cell=Path(cfg.get('cell',cfg.get('cell_directory'))).resolve()
    portable=Path('/Volumes/My Book/Xinran_FullsetTRAK2')
    if not cell.is_relative_to(portable) or not output.is_relative_to(cell) or not figures.is_relative_to(cell):raise ValueError('New plots must remain in this My Book cell')
    figures.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    release=Path(cfg['source_release']).resolve();sys.path.insert(0,str(release))
    spec=importlib.util.spec_from_file_location('distance_frozen_render',release/'render_stage.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    source_output=Path(cfg['source_output']).resolve();job=_json(source_output/'job.json')
    job.update(runtime_dir=str(output/'runtime/plotting'),output=str(output),figures=str(figures),run_id=output.name)
    # Bind only plotting dependencies; no fitting/segmentation or pair kernels are imported.
    cache=output/'runtime/plotting'
    for variable,folder in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
        target=cache/folder;target.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(target)
    os.environ['MPLBACKEND']='Agg';sys.dont_write_bytecode=True
    sys.path.insert(0,str(release/'vendor/mito_tardis'))
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plots
    import analysis_stage
    analysis_stage.np=np;analysis_stage.pd=pd
    render.np=np;render.pd=pd;render.plt=plt
    methods=[dict(method) for method in cfg['methods'] if method['key']!='old_k1']
    if {method['key'] for method in methods}!={'close_1px','close_2px'}:raise ValueError('Expected exactly the 1-pixel and 2-pixel TARDIS methods')
    methods.sort(key=lambda method:['close_1px','close_2px'].index(method['key']))
    baseline=next((dict(method) for method in cfg['methods'] if method['key']=='old_k1'),dict(key='old_k1',label='Original k=1 SE ellipse',field='pointwise_label'))
    baseline['analysis_directory']=str(Path(cfg.get('old_tardis_directory',source_output/'data/tardis')).resolve())
    all_methods=[baseline]+methods
    for method in methods:
        method['analysis_directory']=str(Path(method.get('analysis_directory',output/method['key']/'data/tardis')).resolve())
        marker=Path(method['analysis_directory'])/'COMPLETE.json'
        if not marker.is_file() or not _json(marker).get('all_passed'):raise ValueError('Method numerical output is not complete: '+method['key'])
    classification=Path(cfg['classification_output'])
    comparison_csv=classification/'data/pointwise_comparison.csv.gz' if classification.is_dir() else classification
    if not comparison_csv.is_file():raise FileNotFoundError(comparison_csv)
    cols=['spatial_selected','x_se_nm','y_se_nm']+[method['field'] for method in methods]
    audit=pd.read_csv(comparison_csv,usecols=cols,dtype=str,keep_default_na=False)
    selected=audit[audit.spatial_selected.str.lower().isin(['true','1'])].copy()
    for field in ['x_se_nm','y_se_nm']:selected[field]=selected[field].astype(float)
    if not np.isfinite(selected[['x_se_nm','y_se_nm']].to_numpy()).all():raise ValueError('Saved model SE is nonfinite')
    input_paths=[config_path,comparison_csv]
    for method in all_methods:
        analysis=Path(method['analysis_directory'])
        input_paths += [analysis/'metadata.json',analysis/'per_scope_summary.csv']
        input_paths += sorted((analysis/'distribution').glob('*/*.csv'))
        if method['key']!='old_k1':
            input_paths += sorted((analysis/'bootstrap').glob('paired_bands_*.csv'))
            input_paths += [analysis/name for name in ['per_frame_counts.csv','descriptive_shape_differences.csv','mask_area_by_frame.csv','mask_adjacent_pair_metrics.csv']]
    input_paths=list(dict.fromkeys(input_paths));before=[_descriptor(path) for path in input_paths]
    standard={};peaks={};original_finish=plots.finish
    for method in methods:
        key=method['key'];analysis=Path(method['analysis_directory']);stage=output/key/'_plot_staging';stage.mkdir(parents=True,exist_ok=True)
        source=output/key/'render_sources';source.mkdir(parents=True,exist_ok=True)
        rules=_json(analysis/'metadata.json')['config_rules']
        mask_pairs=pd.read_csv(analysis/'mask_adjacent_pair_metrics.csv')
        requested=[str(item[0]) for item in rules.get('mask_diagnostic_candidates',[])]
        if requested!=['support50_postmin32'] or 'candidate' not in mask_pairs or not mask_pairs.candidate.eq('support50_postmin32').any():
            raise ValueError('Expected saved support50_postmin32 block-boundary diagnostics for '+key)
        current=selected[['x_se_nm','y_se_nm',method['field']]].copy()
        current['tardis_group']=current[method['field']].map({'mito-compatible':'On','non-mito-compatible':'Not-on','unknown':'unknown','On':'On','Not-on':'Not-on'})
        if current.tardis_group.isna().any():raise ValueError('Unknown saved method label')
        perjob=dict(job);perjob['sample_name']=job['sample_name']+' | '+method['label']
        plots.finish=_method_finish(original_finish,method['label'])
        try:names=render.tardis_figures(perjob,current,analysis,stage,source)
        finally:plots.finish=original_finish
        standard[key]=[]
        for name in names:
            dest=figures/(key+'_'+name);(stage/name).replace(dest);standard[key].append(dest.name)
        stage.rmdir()
        print('STANDARD_TARDIS_FIGURES '+key+' '+str(len(names)),flush=True)
    curves={};data_rows=[];reference_x=None
    for method in all_methods:
        key=method['key'];curves[key]={}
        for group in GROUPS:
            frame=pd.read_csv(Path(method['analysis_directory'])/'distribution'/group/'full.csv',float_precision='round_trip')
            if len(frame)!=300:raise ValueError('Expected 300 original distance bins')
            x=frame.distance_mid_um.to_numpy(float);left=frame.distance_left_um.to_numpy(float);right=frame.distance_right_um.to_numpy(float)
            if reference_x is None:reference_x=x.copy()
            else:np.testing.assert_allclose(x,reference_x,rtol=0,atol=1e-14)
            signed_field,area_field=_curve_fields(frame);signed=frame[signed_field].to_numpy(float);area=frame[area_field].to_numpy(float)
            peak=float(np.max(signed)) if np.isfinite(signed).all() else float('nan')
            peak_curve=signed/peak if math.isfinite(peak) and peak>0 else np.full_like(signed,np.nan)
            signed_area=float(np.sum(signed*(right-left))) if np.isfinite(signed).all() else float('nan')
            curves[key][group]=dict(signed=signed,area=area,peak=peak_curve,x=x)
            for view,values,denominator,units in [('signed',signed,1.,'1/um'),('area',area,signed_area,'1/um'),('peak',peak_curve,peak,'dimensionless')]:
                for i,value in enumerate(values):
                    data_rows.append(dict(method=key,method_label=method['label'],group=group,normalization=view,distance_left_um=float(left[i]),distance_right_um=float(right[i]),distance_mid_um=float(x[i]),value=float(value) if math.isfinite(float(value)) else None,denominator=float(denominator) if math.isfinite(denominator) else None,units=units))
    with (output/'comparison_curve_data.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data_rows[0]));writer.writeheader();writer.writerows(data_rows)
    ylimits={view:_ylim([curves[method['key']][group][view] for method in all_methods for group in GROUPS]) for view in ['signed','area','peak']}
    titles=dict(signed='Signed Intra',area='Signed-area-normalized Intra',peak='Positive-peak-normalized Intra')
    comparisons=[]
    for view in ['signed','area','peak']:
        for suffix,xmax in [('full',3.),('zoom',.5)]:
            fig,axes=plt.subplots(1,2,figsize=(12.6,4.9),sharey=True)
            for axis,group in zip(axes,GROUPS):
                unavailable=[]
                for method in all_methods:
                    values=curves[method['key']][group][view]
                    if not np.isfinite(values).all():unavailable.append(method['label']);continue
                    axis.plot(reference_x,values,color=COLORS[method['key']],lw=1.4,label=method['label'])
                axis.axhline(0,color='#777777',lw=.65);axis.grid(alpha=.2)
                axis.set(title=group,xlabel='Distance (µm)',xlim=(0,xmax),ylim=ylimits[view],ylabel='Normalized Intra (dimensionless)' if view=='peak' else 'Density (µm⁻¹)')
                axis.legend(fontsize=8.5)
                if unavailable:axis.text(.98,.98,'Unavailable: '+'; '.join(unavailable),transform=axis.transAxes,ha='right',va='top',fontsize=8)
            fig.suptitle(job['sample_name']+' | '+titles[view]+' | assignment-distance comparison',fontsize=12.5)
            fig.subplots_adjust(left=.07,right=.985,top=.84,bottom=.22,wspace=.16)
            fig.text(.5,.105,'Full 0–3 µm bins define every normalization. Negative bins retained. Same original localization values and QC.',ha='center',fontsize=9)
            fig.text(.5,.055,'Descriptive method overlay; no cross-method confidence bands or significance test. C1/C2 registration remains unverified.',ha='center',fontsize=8.6,color='#52616B')
            name='compare_methods_'+view+'_'+suffix+'.png';fig.savefig(figures/name,dpi=180);plt.close(fig);comparisons.append(name)
    for method in methods:
        key=method['key'];fig,axes=plt.subplots(1,2,figsize=(12.3,4.8),sharey=True)
        limits=_ylim([curves[key][group]['peak'] for group in GROUPS])
        for axis,xmax in zip(axes,[.5,3.]):
            for group,color in [('On','#159C76'),('Not-on','#9854CA')]:
                values=curves[key][group]['peak']
                if np.isfinite(values).all():axis.plot(reference_x,values,color=color,lw=1.5,label=group)
            axis.axhline(0,color='#777777',lw=.65);axis.grid(alpha=.2)
            axis.set(xlim=(0,xmax),ylim=limits,xlabel='Distance (µm)',ylabel='Intra / full-range positive peak',title='0–0.5 µm' if xmax==.5 else '0–3 µm');axis.legend()
        fig.suptitle(job['sample_name']+' | '+method['label']+' | positive-peak normalization',fontsize=12.5)
        fig.tight_layout(rect=(0,.13,1,.92))
        fig.text(.5,.07,'Each group divided by its own positive peak over all 0–3 µm bins; negative values retained. Descriptive lines only.',ha='center',fontsize=9)
        name=key+'_10_peak_normalized.png';fig.savefig(figures/name,dpi=180);plt.close(fig);peaks[key]=name
    _write_index(figures,methods,comparisons,standard,peaks)
    after=[_descriptor(path) for path in input_paths]
    if before!=after:raise ValueError('Protected saved numerical inputs changed during rendering')
    pngs=sorted([path for path in figures.glob('*.png') if not path.name.startswith('.')])
    if len(pngs)!=46:raise ValueError('Expected 46 PNG outputs; found '+str(len(pngs)))
    report=dict(all_passed=True,created_utc=datetime.now(timezone.utc).isoformat(),standard_figures_by_method=standard,peak_figures_by_method=peaks,comparison_figures=comparisons,PNG_count=len(pngs),images=[dict(path=str(path),bytes=path.stat().st_size,sha256=_hash(path)) for path in pngs],shared_ylim_by_normalization={key:list(value) for key,value in ylimits.items()},source_inputs_unchanged=True,source_fingerprints=before,source_script_sha256=_hash(__file__),frozen_render_source_sha256=_hash(release/'render_stage.py'),normalization='Area uses saved shape_density_um_inv; peak uses each full signed curve positive maximum on0-3um; no clipping',cross_method_uncertainty='Descriptive lines only; no invented cross-method uncertainty',scientific_stages_run=[],TARDIS_recomputed=False,classification_recomputed=False,raw_images_read=False,nominal_model_SE_unchanged=True,visual_review_pending=True)
    _dump(output/'PLOTS_COMPLETE.json',report)
    print('PLOTS_COMPLETE '+str(figures),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args();render_outputs(args.config)

if __name__=='__main__':main()
