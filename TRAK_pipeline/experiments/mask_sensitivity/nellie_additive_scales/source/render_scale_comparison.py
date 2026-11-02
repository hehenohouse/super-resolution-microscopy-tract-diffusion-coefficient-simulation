#!/usr/bin/env python3
"""Publish saved additive-Nellie-scale products only; zero scientific-stage calls."""
import argparse,csv,hashlib,html,json,os,sys,time
from pathlib import Path
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=read_json(args.config);root=Path(c['run_directory']).resolve()
    if not (root/'NUMERICAL_COMPLETE.json').is_file():raise SystemExit('Saved scientific products are incomplete; no rendering started.')
    for key,rel in [('TMPDIR','runtime/tmp'),('MPLCONFIGDIR','runtime/matplotlib'),('NUMBA_CACHE_DIR','runtime/numba'),('PYTHONPYCACHEPREFIX','runtime/pycache')]:
        d=root/rel;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    os.environ['MPLBACKEND']='Agg';os.environ['PYDANTIC_DISABLE_PLUGINS']='__all__';sys.dont_write_bytecode=True
    import numpy as np,tifffile
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from point_stack_core import gray_crop,mask_contour_alpha,roi_alpha,composite
    began=time.perf_counter();out=root/'figures';out.mkdir(exist_ok=True);files=[]
    methods=['A_default','B_add_to_1p5um','C_add_to_2p0um']
    labels=['A: original default scales','B: default + added scales through 1.5 um','C: default + added scales through 2.0 um']
    shortlabels=['Original default','Add scales: 1.5 um','Add scales: 2.0 um']
    masks=[np.load(root/'masks'/name/'clean_full_mask.npy',mmap_mode='r') for name in methods]
    roi_mask=np.load(root/'masks/roi_mask.npy',mmap_mode='r')
    if any(a.shape!=(3,*c['shape_TYX'][1:]) for a in masks):raise ValueError('Saved masks have incompatible shapes.')
    rows=read_csv(root/'data/frame_mapping.csv');frames=[int(r['original_frame']) for r in rows];starts=c['block_starts']
    if frames!=[f for s in [0,1200,2450] for f in range(s,s+50)] or [int(r['page_index']) for r in rows]!=list(range(150)):raise ValueError('Unexpected disjoint-frame mapping.')
    metrics=read_csv(root/'data/method_summary.csv');pairs=read_csv(root/'data/pairwise_mask_comparison.csv');bright=read_csv(root/'data/bright_pixel_diagnostic.csv')
    crop=c['crop_xyxy_px'];scale=c['display_scale'];limits=c['display_original_C1_ADU'];x0,y0,x1,y1=crop
    pw=(x1-x0)*scale;ph=(y1-y0)*scale;margin=20;gap=20;header=112;footer=105;w=3*pw+2*margin+2*gap;h=header+ph+footer
    vertices=np.load(root/'masks/roi_vertices_xy.npy');roi=roi_alpha(vertices,crop,scale)
    gray=np.array([145,145,145],float);red=np.array([255,75,75],float);green=np.array([39,221,165],np.uint8);magenta=np.array([229,82,192],np.uint8)
    fontpath=next((p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('C:/Windows/Fonts/arial.ttf')] if p.is_file()),None)
    fonts={n:ImageFont.truetype(str(fontpath),n) if fontpath else ImageFont.load_default() for n in [16,18,20,23,27]}
    contours=[[mask_contour_alpha(m[j],crop,scale) for m in masks] for j in range(3)]
    def atomic_png(a,path):
        t=path.with_name(path.name+'.tmp');Image.fromarray(a).save(t,format='PNG');t.replace(path)
    def bar(draw,ox,oy):
        yy=oy+ph-22;length=round(2/(c['pixel_size_nm']/1000)*scale);draw.line((ox+16,yy,ox+16+length,yy),fill='white',width=3);draw.text((ox+16,yy-23),'2 um',fill='white',font=fonts[16])
    def render(f,raw):
        j=starts.index((f//50)*50);g=gray_crop(raw,crop,scale,limits);rgb=np.repeat(g[...,None],3,axis=2)
        image=Image.new('RGB',(w,h),'black');draw=ImageDraw.Draw(image)
        draw.text((margin,12),f"{c['sample_name']} | original frame {f} | {f*c['frame_interval_s']:.2f} s | support block {f//50}",font=fonts[27],fill='white')
        draw.text((margin,49),'Raw C1 unchanged | Nellie additive-scale test | support >=10/50 / post-min32 / holes <64 unchanged',font=fonts[20],fill='#DDDDDD')
        for k,title in enumerate(labels):
            ox=margin+k*(pw+gap);panel=composite(composite(rgb,contours[j][k],red),roi,gray);image.paste(Image.fromarray(panel),(ox,header));draw.text((ox,82),title,font=fonts[20],fill='white');bar(draw,ox,header)
        fy=header+ph+12
        draw.text((margin,fy),f'ALL backgrounds: identical ORIGINAL C1 pixels; fixed {limits[0]}-{limits[1]} ADU. Red: full-mask boundary. Gray: original ROI.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+27),'Exact five baseline sigmas retained; only larger sigmas appended. Scale labels are nominal settings, not hard object-size cutoffs.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+54),'DISJOINT SEGMENTS: frames 0-49, 1200-1249, 2450-2499. TIFF page joins are acquisition-time jumps.',font=fonts[18],fill='#FFD37E')
        return np.asarray(image)
    streams=[];rawfirst=[];samplehash={};samples=set(c['sample_frames'])
    try:
        for j,s in enumerate(starts):
            raw=tifffile.TiffFile(root/f'inputs/C1_original_frames{s:04d}_{s+49:04d}.ome.tif');streams.append(raw)
            if len(raw.pages)!=50:raise ValueError('Saved input block page count differs.')
            rawfirst.append(raw.pages[0].asarray())
        def pages():
            for page,f in enumerate(frames):
                j,local=divmod(page,50);a=render(f,streams[j].pages[local].asarray())
                if f in samples:samplehash[page]=hashlib.sha256(a.tobytes()).hexdigest()
                if local==0:
                    name=f'C1_additive_scale_masks_frame{f:04d}.png';atomic_png(a,out/name);files.append(name)
                if (page+1)%50==0:print(f'Rendered {page+1}/150 pages; original frame {f}',flush=True)
                yield a
        tiffname='C1_additive_scale_mask_comparison_150pages.tif';temp=out/(tiffname+'.tmp')
        tifffile.imwrite(temp,pages(),shape=(150,h,w,3),dtype=np.uint8,imagej=True,photometric='rgb',compression='deflate',metadata={'axes':'TYXS','mode':'color','loop':False,'Info':'150 RGB pages from DISJOINT original-frame segments 0-49, 1200-1249, 2450-2499. Actual original frame/time printed per page. Page joins are not consecutive acquisition frames. Within-segment frame interval 0.02 s. All backgrounds are identical original C1 at 407-651 ADU; columns: default scales, default plus larger scales through nominal1.5um, default plus larger scales through nominal2um. The exact original five sigmas are preserved. Scale labels are not hard object-size thresholds. No uniform movie time interval is asserted.'})
        temp.replace(out/tiffname);files.append(tiffname)
    finally:
        for raw in streams:raw.close()
    # Pairwise differences display existing full masks; labels read saved ROI measurements.
    rowhead=36;rowgap=22;dw=2*pw+2*margin+gap;dh=100+3*(ph+rowhead+rowgap)+52;image=Image.new('RGB',(dw,dh),'black');draw=ImageDraw.Draw(image)
    draw.text((margin,10),f"{c['sample_name']} | added-scale mask differences",font=fonts[23],fill='white')
    draw.text((margin,44),'Common = gray | added by test = green | removed from default = magenta | original ROI outline = gray',font=fonts[18],fill='#DDDDDD')
    for k,title in enumerate(['Default -> add scales through 1.5 um','Default -> add scales through 2.0 um']):draw.text((margin+k*(pw+gap),76),title,font=fonts[20],fill='white')
    for j,f in enumerate(starts):
        top=100+j*(ph+rowhead+rowgap)
        for k,test in enumerate([1,2]):
            a=np.asarray(masks[0][j,y0:y1,x0:x1],bool);b=np.asarray(masks[test][j,y0:y1,x0:x1],bool);rgb=np.zeros((*a.shape,3),np.uint8);rgb[a&b]=[92,92,92];rgb[b&~a]=green;rgb[a&~b]=magenta;rgb=np.repeat(np.repeat(rgb,scale,0),scale,1)
            ox=margin+k*(pw+gap);oy=top+rowhead
            row=next(r for r in pairs if r['scope']=='ROI' and int(r['original_frame_start'])==f and r['reference_method']==methods[0] and r['test_method']==methods[test])
            draw.text((ox,top),f"Frame {f}: ROI IoU {float(row['iou']):.3f}; area delta {float(row['delta_percent']):+.2f}%",font=fonts[20],fill='white');image.paste(Image.fromarray(composite(rgb,roi,gray)),(ox,oy));bar(draw,ox,oy)
    draw.text((margin,dh-32),'Difference colors show full masks before ROI clipping; numerical labels are restricted to the original cell ROI.',font=fonts[16],fill='#FFD37E')
    name='C1_additive_scale_pairwise_differences.png';atomic_png(np.asarray(image),out/name);files.append(name)
    # Binary stage views. Each shows the first broad frame, one full50-frame support result and final mask.
    for j,f in enumerate(starts):
        sh=115+3*(ph+rowhead+rowgap)+66;image=Image.new('RGB',(w,sh),'black');draw=ImageDraw.Draw(image)
        draw.text((margin,10),f"{c['sample_name']} | block {f}-{f+49}: saved-stage diagnostic",font=fonts[23],fill='white')
        draw.text((margin,43),'White = existing foreground | gray ROI outline | all panels use the same native crop and spatial scale',font=fonts[18],fill='#DDDDDD')
        for k,title in enumerate([f'Per-frame broad at frame {f}','Support >=10/50, before cleanup','Final mask after min32 / holes <64']):draw.text((margin+k*(pw+gap),79),title,font=fonts[20],fill='white')
        for m,mid in enumerate(methods):
            with np.load(root/'masks'/mid/'broad_packed'/f'b{f//50:03d}.npz') as z:
                shape=tuple(int(v) for v in z['shape']);ids=z['original_frame_ids'];broad=np.unpackbits(z['packed'][0],count=shape[1]*shape[2],bitorder='little').reshape(shape[1:]).astype(bool)
                if int(ids[0])!=f:raise ValueError('Saved broad frame mapping differs.')
            with np.load(root/'masks'/mid/'postprocess_stages.npz') as z:support=np.asarray(z['support_mask'][j],bool)
            top=115+m*(ph+rowhead+rowgap);draw.text((margin,top),labels[m],font=fonts[20],fill='white')
            for k,mask in enumerate([broad,support,masks[m][j]]):
                rgb=np.repeat(gray_crop(mask,crop,scale,[0,1])[...,None],3,axis=2);ox=margin+k*(pw+gap);oy=top+rowhead;image.paste(Image.fromarray(composite(rgb,roi,gray)),(ox,oy));bar(draw,ox,oy)
        draw.text((margin,sh-49),'Broad is one original frame; support and final masks summarize the full50-frame block. They are distinct temporal quantities.',font=fonts[18],fill='#FFD37E')
        draw.text((margin,sh-24),'Saved masks only; no segmentation, thresholding, cleanup or other scientific stage is called by this publisher.',font=fonts[18],fill='#DDDDDD')
        name=f'C1_additive_scale_stage_diagnostic_block{f:04d}.png';atomic_png(np.asarray(image),out/name);files.append(name)
    plt.rcParams.update({'font.size':11,'axes.titlesize':13});fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);colors=['#687C91','#CF8D45','#32AE9C']
    for ax,scope in zip(axes,['ROI','full']):
        for k,method in enumerate(methods):
            selected=sorted([r for r in metrics if r['scope']==scope and r['method']==method],key=lambda r:int(r['original_frame_start']))
            if len(selected)!=3:raise ValueError('Missing area rows for scope/method.')
            ax.bar(np.arange(3)+(k-1)*.25,[float(r['area_um2']) for r in selected],.25,label=shortlabels[k],color=colors[k])
        ax.set_xticks(np.arange(3),[f'{s}-{s+49}' for s in starts]);ax.set_xlabel('Original acquisition-frame block');ax.set_ylabel('Binary mask area (um^2)');ax.set_title('Inside original cell ROI' if scope=='ROI' else 'Full image before ROI clipping');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='upper left');ax.set_ylim(0,max(float(r['area_um2']) for r in metrics if r['scope']==scope)*1.28)
    fig.suptitle(f"{c['sample_name']} | additive-scale mask areas (saved measurements)",fontsize=15)
    name='C1_additive_scale_mask_area_comparison.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180);plt.close(fig);temp.replace(out/name);files.append(name)
    fig,ax=plt.subplots(figsize=(10,4.9),constrained_layout=True)
    for k,method in enumerate(methods):
        selected=sorted([r for r in bright if r['method']==method],key=lambda r:int(r['original_frame']))
        if len(selected)!=3:raise ValueError('Missing saved bright-coverage records.')
        vals=[float(r['coverage_percentage']) for r in selected]
        ax.bar(np.arange(3)+(k-1)*.25,vals,.25,label=shortlabels[k],color=colors[k])
        for xx,yy in zip(np.arange(3)+(k-1)*.25,vals):ax.text(xx,yy+.8,f'{yy:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(np.arange(3),[str(f) for f in starts]);ax.set_xlabel('Original frame');ax.set_ylabel('Saved mask coverage of bright-region pixels (%)');ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='lower left')
    ax.set_title(f"{c['sample_name']} | original bright-pixel coverage diagnostic",fontsize=14)
    fig.text(.5,-.015,'Raw C1 > original full-frame P99, restricted to ROI; same region across methods. Not an accuracy measure.',ha='center',fontsize=9)
    name='C1_additive_scale_bright_region_coverage.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180,bbox_inches='tight');plt.close(fig);temp.replace(out/name);files.append(name)
    # Saved B/C responses only: independent display windows; no baseline response rerun.
    responses={}
    for mid in methods[1:]:
        lookup={}
        for path in sorted((root/'masks'/mid/'native_batches').glob('frames*/sampled_filter_response.npz')):
            with np.load(path) as z:
                for frame,response in zip(z['original_frame_ids'],z['response']):
                    if int(frame) in starts:lookup[int(frame)]=response
        responses[mid]=lookup
    if all(set(responses[mid])==set(starts) for mid in methods[1:]):
        rh=106+2*(ph+rowhead+rowgap)+65;image=Image.new('RGB',(w,rh),'black');draw=ImageDraw.Draw(image)
        draw.text((margin,10),f"{c['sample_name']} | saved added-scale filter responses",font=fonts[23],fill='white')
        draw.text((margin,45),'Display only: max(Frangi response, positive LoG response); own min/max per panel. Not probability or accuracy.',font=fonts[18],fill='#DDDDDD')
        for k,f in enumerate(starts):draw.text((margin+k*(pw+gap),79),f'Original frame {f}',font=fonts[20],fill='white')
        for m,mid in enumerate(methods[1:]):
            top=106+m*(ph+rowhead+rowgap)
            for j,f in enumerate(starts):
                a=np.asarray(responses[mid][f],float);lo=float(np.nanmin(a));hi=float(np.nanmax(a));ox=margin+j*(pw+gap);oy=top+rowhead
                panel=np.repeat(gray_crop(a,crop,scale,[lo,hi if hi>lo else lo+1])[...,None],3,axis=2);image.paste(Image.fromarray(composite(panel,roi,gray)),(ox,oy));bar(draw,ox,oy)
                draw.text((ox,top),f'{shortlabels[m+1]} | display {lo:.3g}-{hi:.3g}',font=fonts[18],fill='white')
        draw.text((margin,rh-48),'A baseline responses were not saved and are unavailable; the baseline was reused without rerunning any filters.',font=fonts[18],fill='#FFD37E')
        draw.text((margin,rh-23),'Independent display normalization prevents amplitude comparisons. These images alone do not identify the source of a mask hole.',font=fonts[18],fill='#DDDDDD')
        name='C1_additive_scale_saved_filter_responses.png';atomic_png(np.asarray(image),out/name);files.append(name)
    checks=[]
    with tifffile.TiffFile(out/tiffname) as saved:
        if len(saved.pages)!=150 or not saved.is_imagej or saved.series[0].shape!=(150,h,w,3):raise AssertionError('TIFF page count, ImageJ flag, or shape differs.')
        for page,expected in sorted(samplehash.items()):
            observed=hashlib.sha256(saved.pages[page].asarray().tobytes()).hexdigest()
            if observed!=expected:raise AssertionError('Lossless sample readback differs.')
            checks.append({'page_index':page,'original_frame':frames[page],'sha256':observed,'match':True})
        ij=saved.imagej_metadata
    titles={'C1_additive_scale_mask_area_comparison.png':'ROI and full-image areas: three saved scale methods','C1_additive_scale_bright_region_coverage.png':'Coverage of fixed original-C1 bright regions','C1_additive_scale_pairwise_differences.png':'Default -> added scales: foreground differences','C1_additive_scale_saved_filter_responses.png':'Saved B/C filter responses (baseline unavailable)',tiffname:'Fiji RGB TIFF:150 pages from three disjoint segments'}
    for f in starts:
        titles[f'C1_additive_scale_masks_frame{f:04d}.png']=f'Three mask methods on identical original C1: frame{f}'
        titles[f'C1_additive_scale_stage_diagnostic_block{f:04d}.png']=f'Saved broad / support / final masks: block{f}-{f+49}'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Nellie additive-scale mask test</title><style>body{font:16px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#172b36}img{max-width:100%;height:auto}article{margin:32px 0}a{color:#086b88}.notice{background:#fff0cc;padding:16px}</style>',f'<h1>{html.escape(c["sample_name"])}: Nellie additive-scale mask test</h1>','<p>All methods use unchanged original C1. A reuses the default five filter sigmas. B preserves all default sigmas and appends larger sigmas through the nominal1.5um setting. C also includes larger sigmas through nominal2.0um. These settings are not hard object-radius cutoffs. Native labeling, support ≥10/50, post-min32, holes &lt;64, original ROI and coordinates are unchanged.</p>','<p class="notice"><strong>Disjoint segments:</strong>frames0–49,1200–1249 and2450–2499. Actual acquisition frame/time are printed on every TIFF page. Page joins are time jumps; no continuous time interval is asserted.</p>','<p>All three mask columns use identical original C1 pixels at fixed407–651ADU. Red contours show full masks before ROI clipping; gray outline is the cell ROI. Difference panels: added green, removed magenta, common gray.</p>']
    for name,title in titles.items():
        if not (out/name).is_file():continue
        body.append(f'<article><h2><a href="{html.escape(name)}">{html.escape(title)}</a></h2>')
        if name.endswith('.png'):body.append(f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(title)}"></a>')
        else:body.append('<p>Open in Fiji. Page1 = original frame0; page51 =1200; page101 =2450.</p>')
        body.append('</article>')
    body+=['<p><a href="../data/method_summary.csv">Saved method areas</a> · <a href="../data/pairwise_mask_comparison.csv">Saved overlap measurements</a> · <a href="../data/bright_pixel_diagnostic.csv">Saved bright-region diagnostic</a> · <a href="../data/frame_mapping.csv">Original-frame mapping</a> · <a href="../README_RENDERING.txt">Rendering notes</a></p>','<p>Mask area, bright-region coverage or filtering-scale size does not establish accuracy. No localization, assignment, TARDIS, filter computation, cleanup or inference is performed by this publisher.</p></html>']
    (out/'INDEX.html').write_text('\n'.join(body),encoding='utf-8')
    (root/'README_RENDERING.txt').write_text('''NELLIE ADDITIVE-SCALE TEST: RENDERING NOTES
All PNGs, RGB TIFF pages, graphs and indexes read saved numerical products only.
Scientific-stage calls =0. No Nellie, fitting, assignment, TARDIS, new filter
responses, mask cleanup, inference or summary-science calculations occur here.
Binary foreground rendering and boundaries are presentation only.

Raw C1 is identical and unprocessed for A, B and C. The five original filter
sigmas are preserved. B and C append larger sigmas through nominal1.5um and2um.
These labels do not mean hard object-radius limits. Original mask cleanup,
support, ROI, coordinates and registration assumptions are unchanged.
A responses are unavailable and are not silently recomputed.

TIFF order combines disjoint original frames0-49,1200-1249 and2450-2499.
Every page records actual frame, time and support block. Page joins are time
jumps. No uniform acquisition interval is asserted for the joined sequence.
C1 display407-651ADU, crop and2x nearest-neighbor display replication are fixed
across mask columns. Scale bars2um; contours antialiased for presentation only.
Difference grids show full-mask pixels before ROI clipping. Numerical labels
read saved ROI measurements. Added=test green; removed=default magenta;
common gray. Broad-stage images show only the first frame of each50-frame
block; support and final products summarize all50frames.

Optional B/C native-response panels show saved max(Frangi, positive LoG)
responses. Each panel has independently labelled min/max display limits;
display amplitudes must not be compared between panels or methods.
Bright-region coverage is read from the scientific-stage saved table, with
identical original-C1 regions across methods. It is not ground truth accuracy.

Only ten specified TIFF pages are read back. No all2500frame replay or full
independent every-page verification occurs. Original inputs, prior scientific
products and default pipeline settings are not modified. C1/C2 identity
registration remains independently unverified.
''',encoding='utf-8')
    manifest={'sample_id':c['sample_id'],'scientific_steps_invoked':0,'tiff':{'relative_path':'figures/'+tiffname,'pages':150,'shape_TYXS':[150,h,w,3],'dtype':'uint8','compression':'deflate','imagej':True,'disjoint_original_segments':[[s,s+50] for s in starts],'metadata':ij},'sample_page_readback':checks,'sample_pages_checked':len(checks),'files':[{'relative_path':'figures/'+name,'bytes':(out/name).stat().st_size} for name in files],'elapsed_seconds':time.perf_counter()-began}
    (root/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'render_complete':True,'scientific_steps_invoked':0,'sample_pages_checked':len(checks),'elapsed_seconds':manifest['elapsed_seconds'],'index':str(out/'INDEX.html')},indent=2),flush=True)
if __name__=='__main__':main()
