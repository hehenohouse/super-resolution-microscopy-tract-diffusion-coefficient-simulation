#!/usr/bin/env python3
"""Render saved fixed-distance Not-on groups over the unchanged full-movie C1 mean."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys
from compare_assignment_distance import env_for,guard,descriptor,hash_file,utc
from distance_plots import _module,_save_image,_atomic_json,_font

def render_mean_overlays(config_path):
    config_path=Path(config_path).resolve();c=json.loads(config_path.read_text())
    out=guard(c['output']);figures=guard(c['figures']);source=Path(c['source_output']);release=Path(c['source_release'])
    env_for(out)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    cache=source/'data/render_sources/C1_mean_frames0000_2499.npy'
    meta_path=source/'data/render_sources/C1_average2500_metadata.json'
    audit=out/'data/pointwise_comparison.csv.gz'
    protected=[cache,meta_path,audit,out/'summary.json',source/'roi/vertices.npy',source/'roi/roi_mask.npy']
    before=[descriptor(p) for p in protected]
    meta=json.loads(meta_path.read_text());mean=np.load(cache,allow_pickle=False)
    if mean.shape!=(684,428) or mean.dtype!=np.float32:raise ValueError('Unexpected cached C1 mean')
    mean_hash=hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest()
    if mean_hash!=meta['mean_float32_C_bytes_SHA256'] or meta['source_frames']!=2500 or not meta['all_source_pixels_included'] or meta['zero_exclusion'] or meta['alignment'] or meta['filtering']:
        raise ValueError('Cached background does not match original full-movie arithmetic mean')
    pc=_module('point_stack_core',release/'vendor/mito_tardis/point_stack_core.py')
    sf=_module('mean_overlay_supplemental',release/'vendor/supplemental_figures.py')
    renderer=_module('mean_overlay_frozen_renderer',release/'render_stage.py')
    renderer.np=np;renderer.pc=pc;renderer.sf=sf;renderer.Image=Image;renderer.ImageDraw=ImageDraw;renderer.ImageFont=ImageFont
    job=json.loads((source/'job.json').read_text());job=dict(job,font_path=_font(job),crop_xyxy_px=c['crop_xyxy_px'])
    summary=json.loads((out/'summary.json').read_text());expected={m['key']:m for m in summary['methods']}
    methods=[m for m in c['methods'] if m['key'] in ('close_1px','close_2px')]
    points={m['key']:[] for m in methods};identities={m['key']:[] for m in methods}
    with gzip.open(audit,'rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['spatial_selected'].lower() not in ('true','1'):continue
            for method in methods:
                if row[method['field']]=='non-mito-compatible':
                    points[method['key']].append([float(row['fit_x_px']),float(row['fit_y_px'])])
                    identities[method['key']].append(row['candidate_index'])
    crop=c['crop_xyxy_px'];scale=4;x0,y0,x1,y1=crop
    roi=np.load(source/'roi/roi_mask.npy',allow_pickle=False)
    vertices=np.load(source/'roi/vertices.npy',allow_pickle=False)
    gray=np.repeat(pc.gray_crop(mean,crop,scale,[407,651])[...,None],3,axis=2)
    inside=np.repeat(np.repeat(roi[y0:y1,x0:x1],scale,0),scale,1);gray[~inside]=0
    outline=pc.roi_alpha(vertices,crop,scale);base=pc.composite(gray,outline,pc.ROI_COLOR)
    records=[]
    for method in methods:
        key=method['key'];xy=np.asarray(points[key],float).reshape(-1,2)
        if len(xy)!=expected[key]['Not_on']:raise ValueError('Not-on count differs from classification summary')
        grid=renderer.count_grid(xy,crop,scale)
        if int(grid.sum())!=len(xy):raise ValueError('Display crop loses a localization')
        alpha=pc.marker_alpha(base.shape[:2],pc.display_centers(xy,crop,scale),.20*scale)
        image=pc.composite(base,alpha,pc.OFF)
        display_job=dict(job,sample_name=job['sample_name']+' | '+method['label'])
        panel=renderer.labeled_panel(image,f'Not-on | {len(xy):,} centers | C1 mean of all 2500 raw frames',display_job,scale,mean_background=True)
        filename=f'NotOn_{key}_on_C1_mean2500.png';_save_image(panel,figures/filename)
        with Image.open(figures/filename) as saved:
            saved.load()
            if saved.mode!='RGB' or saved.size!=(1488,1986) or not np.array_equal(np.asarray(saved),np.asarray(panel)):raise ValueError('Saved overlay pixels differ')
        records.append(dict(method=key,label=method['label'],Not_on_count=len(xy),displayed_count_grid_sum=int(grid.sum()),
          candidate_indices_SHA256=hashlib.sha256('\n'.join(identities[key]).encode()).hexdigest(),
          refined_centers_float64_C_bytes_SHA256=hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest(),
          filename=filename,pixel_size=[1488,1986],PNG_SHA256=hash_file(figures/filename)))
    after=[descriptor(p) for p in protected]
    if before!=after:raise ValueError('A protected input changed')
    report=dict(complete=True,completed_utc=utc(),sample_id=c['sample_id'],config=str(config_path),records=records,
      background_source=str(cache),mean_float32_C_bytes_SHA256=mean_hash,source_frames=2500,mean_type='arithmetic raw-ADU mean; all pixels and zeros included',
      same_background_as_original=True,display_C1_ADU=[407,651],crop_xyxy_px=crop,display_scale=4,dot_radius_native_px=.20,scale_bar_um=2,
      ROI_outside_black=True,ROI_outline_gray=True,no_mask_contour=True,no_alignment_or_filtering=True,
      point_selection='saved QC-selected Not-on under the named fixed-radius rule; no new classification or filtering',
      repeated_observations_retained=True,source_fingerprints_before=before,source_fingerprints_after=after,inputs_unchanged=True,
      new_TARDIS_calculations=0,new_fitting_or_classification=0,renderer_source=str(Path(__file__)),renderer_SHA256=hash_file(__file__))
    _atomic_json(out/'mean_overlay_manifest.json',report)
    # Extend only this comparison run's index, preserving its existing content.
    index=figures/'INDEX.html';text=index.read_text()
    begin='<!-- C1_MEAN_OVERLAYS_BEGIN -->';end='<!-- C1_MEAN_OVERLAYS_END -->'
    section=begin+'<h2>Not-on over the mean of all 2500 original C1 frames</h2><p>Identical background, contrast, ROI crop and marker size across rules. No mask contours are shown.</p>'
    for record in records:
        section+='<h3>'+record['label']+' — '+format(record['Not_on_count'],',')+' Not-on centers</h3><a href="'+record['filename']+'"><img src="'+record['filename']+'" style="max-width:700px;width:100%;height:auto"></a>'
    section+=end
    if begin in text:
        first=text.index(begin);last=text.index(end,first)+len(end);text=text[:first]+section+text[last:]
    elif '</body>' in text:text=text.replace('</body>',section+'</body>')
    elif '</html>' in text:text=text.replace('</html>',section+'</html>')
    else:text+=section
    temp=index.with_name('.INDEX.mean.tmp');temp.write_text(text,encoding='utf-8');temp.replace(index)
    archive=out/'source'/Path(__file__).name;archive.write_text(Path(__file__).read_text(),encoding='utf-8')
    with (out/'run.log').open('a',encoding='utf-8') as stream:
        stream.write(utc()+' | Added fixed1px/2px Not-on overlays on unchanged cached C1 mean2500; counts52630/45929; no new classification or TARDIS.\n')
    _atomic_json(out/'MEAN_OVERLAYS_COMPLETE.json',dict(complete=True,completed_utc=utc(),manifest='mean_overlay_manifest.json',PNG_files=[r['filename'] for r in records],classification_unchanged=True))
    print(json.dumps(dict(complete=True,images=records,mean_cache_matches_original=True),indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    render_mean_overlays(parser.parse_args().config)
