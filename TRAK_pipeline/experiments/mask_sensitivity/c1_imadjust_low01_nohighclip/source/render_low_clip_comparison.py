#!/usr/bin/env python3
"""Render saved three-way C1 mask sensitivity products; zero scientific calls."""
import argparse,csv,hashlib,html,json,os,sys,time
from pathlib import Path

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    c=read_json(args.config);root=Path(c['run_directory']).resolve()
    if not (root/'NUMERICAL_COMPLETE.json').is_file():raise SystemExit('Saved scientific products are incomplete; rendering not started.')
    for key,rel in [('TMPDIR','runtime/tmp'),('MPLCONFIGDIR','runtime/matplotlib'),('NUMBA_CACHE_DIR','runtime/numba'),('PYTHONPYCACHEPREFIX','runtime/pycache')]:
        d=root/rel;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    os.environ['MPLBACKEND']='Agg';os.environ['PYDANTIC_DISABLE_PLUGINS']='__all__';sys.dont_write_bytecode=True
    import numpy as np,tifffile
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from point_stack_core import gray_crop,mask_contour_alpha,roi_alpha,composite
    began=time.perf_counter();out=root/'figures';out.mkdir(exist_ok=True);files=[]
    methods=['original_default','double_clip_p01_p99','low_only_p01_max']
    labels=['Original default','P1/P99 clipping','Low 1% only; upper = frame max']
    masks=[np.load(root/'masks'/name,mmap_mode='r') for name in ['baseline_clean_full_mask.npy','prior_double_clip_full_mask.npy','clean_full_mask.npy']]
    if any(a.shape!=(3,*c['shape_TYX'][1:]) for a in masks):raise ValueError('Saved masks have incompatible shapes.')
    rows=read_csv(root/'data/frame_mapping.csv');frames=[int(r['original_frame']) for r in rows];starts=c['block_starts']
    if frames!=[f for s in [0,1200,2450] for f in range(s,s+50)] or [int(r['page_index']) for r in rows]!=list(range(150)):raise ValueError('Unexpected disjoint-frame mapping.')
    metrics=read_csv(root/'data/method_summary.csv');pairs=read_csv(root/'data/pairwise_mask_comparison.csv')
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
        draw.text((margin,49),'C1 preprocessing comparison: unchanged Nellie broad / support >=10/50 / post-min32 / holes <64',font=fonts[20],fill='#DDDDDD')
        for k,title in enumerate(labels):
            ox=margin+k*(pw+gap);panel=composite(composite(rgb,contours[j][k],red),roi,gray);image.paste(Image.fromarray(panel),(ox,header));draw.text((ox,82),title,font=fonts[20],fill='white');bar(draw,ox,header)
        fy=header+ph+12
        draw.text((margin,fy),f'ALL backgrounds: identical ORIGINAL C1 pixels; fixed {limits[0]}-{limits[1]} ADU. Red: full-mask boundary. Gray: original ROI.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+27),'Only C1 preprocessing differs. Identical crop, scale, cleanup parameters and native coordinates. No localization or TARDIS recomputation.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+54),'DISJOINT SEGMENTS: frames 0-49, 1200-1249, 2450-2499. TIFF page joins are acquisition-time jumps.',font=fonts[18],fill='#FFD37E')
        return np.asarray(image)
    prior=Path(c['prior_double_clip_test']);streams=[];rawfirst=[];doublefirst=[];lowfirst=[];samplehash={};samples=set(c['sample_frames'])
    try:
        for j,s in enumerate(starts):
            raw=tifffile.TiffFile(root/f'inputs/C1_original_frames{s:04d}_{s+49:04d}.ome.tif')
            low=tifffile.TiffFile(root/f'inputs/C1_adjusted_frames{s:04d}_{s+49:04d}.ome.tif')
            double=tifffile.TiffFile(prior/f'inputs/C1_adjusted_frames{s:04d}_{s+49:04d}.ome.tif')
            streams.append((raw,low,double))
            if any(len(t.pages)!=50 for t in (raw,low,double)):raise ValueError('Saved input block page count differs.')
            rawfirst.append(raw.pages[0].asarray());lowfirst.append(low.pages[0].asarray());doublefirst.append(double.pages[0].asarray())
        def pages():
            for page,f in enumerate(frames):
                j,local=divmod(page,50);a=render(f,streams[j][0].pages[local].asarray())
                if f in samples:samplehash[page]=hashlib.sha256(a.tobytes()).hexdigest()
                if local==0:
                    name=f'C1_three_method_masks_frame{f:04d}.png';atomic_png(a,out/name);files.append(name)
                if (page+1)%50==0:print(f'Rendered {page+1}/150 pages; original frame {f}',flush=True)
                yield a
        tiffname='C1_three_method_mask_comparison_150pages.tif';temp=out/(tiffname+'.tmp')
        tifffile.imwrite(temp,pages(),shape=(150,h,w,3),dtype=np.uint8,imagej=True,photometric='rgb',compression='deflate',metadata={'axes':'TYXS','mode':'color','loop':False,'Info':'150 RGB pages from DISJOINT original-frame segments 0-49, 1200-1249, 2450-2499. Actual original frame/time printed per page. Page joins are not consecutive acquisition frames. Within-segment frame interval 0.02 s. All backgrounds are identical original C1 at 407-651 ADU; columns: default, P1/P99, low-only P1/max. No uniform movie time interval is asserted.'})
        temp.replace(out/tiffname);files.append(tiffname)
    finally:
        for group in streams:
            for t in group:t.close()
    # Existing input images are displayed separately; adjusted uint16 display is explicit.
    rowhead=36;rowgap=22;previewh=104+3*(rowhead+ph+rowgap)+46
    image=Image.new('RGB',(w,previewh),'black');draw=ImageDraw.Draw(image)
    draw.text((margin,10),f"{c['sample_name']} | saved input previews",font=fonts[27],fill='white')
    draw.text((margin,48),'Original display: 407-651 ADU | Both adjusted inputs: 0-65535 uint16 | display ranges are deliberately different.',font=fonts[18],fill='#DDDDDD')
    for k,title in enumerate(['Original C1: 407-651 ADU','P1/P99 input: 0-65535','Low-only input: 0-65535']):draw.text((margin+k*(pw+gap),78),title,font=fonts[20],fill='white')
    for j,f in enumerate(starts):
        top=104+j*(rowhead+ph+rowgap);draw.text((margin,top),f'Original frame {f} | {f*.02:.2f} s',font=fonts[20],fill='white')
        for k,raw,lim in [(0,rawfirst[j],limits),(1,doublefirst[j],[0,65535]),(2,lowfirst[j],[0,65535])]:
            panel=np.repeat(gray_crop(raw,crop,scale,lim)[...,None],3,axis=2);ox=margin+k*(pw+gap);oy=top+rowhead;image.paste(Image.fromarray(composite(panel,roi,gray)),(ox,oy));bar(draw,ox,oy)
    draw.text((margin,previewh-32),'Brighter display and larger mask area do not establish segmentation accuracy. No smoothing or spatial interpolation.',font=fonts[18],fill='#FFD37E')
    name='C1_three_method_input_previews.png';atomic_png(np.asarray(image),out/name);files.append(name)
    # Pairwise mask colors use saved masks only; area/overlap labels read saved measurements.
    dw=2*pw+2*margin+gap;dh=100+3*(ph+rowhead+rowgap)+52;image=Image.new('RGB',(dw,dh),'black');draw=ImageDraw.Draw(image)
    draw.text((margin,10),f"{c['sample_name']} | new low-only mask differences",font=fonts[23],fill='white')
    draw.text((margin,44),'Common = gray | added in low-only = green | removed from reference = magenta | original ROI outline = gray',font=fonts[18],fill='#DDDDDD')
    for k,title in enumerate(['Reference: original default','Reference: previous P1/P99 clipping']):draw.text((margin+k*(pw+gap),76),title,font=fonts[20],fill='white')
    for j,f in enumerate(starts):
        top=100+j*(ph+rowhead+rowgap)
        for k,ref in enumerate([0,1]):
            a=np.asarray(masks[ref][j,y0:y1,x0:x1],bool);b=np.asarray(masks[2][j,y0:y1,x0:x1],bool);rgb=np.zeros((*a.shape,3),np.uint8);rgb[a&b]=[92,92,92];rgb[b&~a]=green;rgb[a&~b]=magenta;rgb=np.repeat(np.repeat(rgb,scale,0),scale,1)
            ox=margin+k*(pw+gap);oy=top+rowhead
            row=next(r for r in pairs if r['scope']=='ROI' and int(r['original_frame_start'])==f and r['reference_method']==methods[ref] and r['test_method']==methods[2])
            draw.text((ox,top),f"Frame {f}: ROI IoU {float(row['iou']):.3f}; area delta {float(row['delta_percent']):+.2f}%",font=fonts[20],fill='white');image.paste(Image.fromarray(composite(rgb,roi,gray)),(ox,oy));bar(draw,ox,oy)
    draw.text((margin,dh-32),'Differences are full-mask foreground pixels before ROI clipping; displayed metrics are restricted to the original cell ROI.',font=fonts[16],fill='#FFD37E')
    name='C1_low_only_mask_pairwise_differences.png';atomic_png(np.asarray(image),out/name);files.append(name)
    # Quantitative panels plot only the saved science-stage table.
    plt.rcParams.update({'font.size':11,'axes.titlesize':13});fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    colors=['#687C91','#CF8D45','#32AE9C']
    for ax,scope in zip(axes,['ROI','full']):
        for k,method in enumerate(methods):
            selected=sorted([r for r in metrics if r['scope']==scope and r['method']==method],key=lambda r:int(r['original_frame_start']))
            if len(selected)!=3:raise ValueError('Missing area rows for scope/method.')
            ax.bar(np.arange(3)+(k-1)*.25,[float(r['area_um2']) for r in selected],.25,label=labels[k],color=colors[k])
        ax.set_xticks(np.arange(3),[f'{s}-{s+49}' for s in starts]);ax.set_xlabel('Original acquisition-frame block');ax.set_ylabel('Binary mask area (um^2)');ax.set_title('Inside original cell ROI' if scope=='ROI' else 'Full image before ROI clipping');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8,loc='upper left');ax.set_ylim(0,max(float(r['area_um2']) for r in metrics if r['scope']==scope)*1.28)
    fig.suptitle(f"{c['sample_name']} | three-method mask areas (saved measurements)",fontsize=15)
    name='C1_three_method_mask_area_comparison.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180);plt.close(fig);temp.replace(out/name);files.append(name)
    # Only the new input has sampled intermediate responses. No cross-method response comparison is claimed.
    responses={}
    for path in sorted((root/'masks/native_batches').glob('frames*/sampled_filter_response.npz')):
        with np.load(path) as z:
            for f,a in zip(z['original_frame_ids'],z['response']):
                if int(f) in starts:responses[int(f)]=a
    if set(responses)==set(starts):
        rh=ph+190;image=Image.new('RGB',(w,rh),'black');draw=ImageDraw.Draw(image)
        draw.text((margin,10),f"{c['sample_name']} | low-only input: saved native filter-response diagnostic",font=fonts[23],fill='white')
        draw.text((margin,45),'Response = max(Frangi response, positive LoG response). Not a raw image, probability, mask or accuracy measure.',font=fonts[18],fill='#DDDDDD')
        for j,f in enumerate(starts):
            a=np.asarray(responses[f],float);lo=float(np.nanmin(a));hi=float(np.nanmax(a))
            panel=np.repeat(gray_crop(a,crop,scale,[lo,hi if hi>lo else lo+1])[...,None],3,axis=2);ox=margin+j*(pw+gap);image.paste(Image.fromarray(composite(panel,roi,gray)),(ox,110));bar(draw,ox,110)
            draw.text((ox,77),f'Frame {f} | own display range {lo:.3g}-{hi:.3g}',font=fonts[18],fill='white')
        draw.text((margin,ph+129),'Each panel independently normalized for display only; amplitudes cannot be compared across these panels or against old-input responses.',font=fonts[18],fill='#FFD37E')
        draw.text((margin,ph+155),'Saved intermediate responses are shown without rerunning filters. They do not isolate which filter caused a particular hole.',font=fonts[18],fill='#DDDDDD')
        name='C1_low_only_saved_filter_response_diagnostic.png';atomic_png(np.asarray(image),out/name);files.append(name)
    # Bright-region coverage is an explicit saved diagnostic, not an accuracy score.
    bright=read_csv(root/'data/bright_pixel_diagnostic.csv')
    fig,ax=plt.subplots(figsize=(9.5,4.8),constrained_layout=True)
    for k,method in enumerate(methods):
        selected=sorted([r for r in bright if r['method']==method],key=lambda r:int(r['original_frame']))
        if len(selected)!=3:raise ValueError('Missing saved bright-coverage records.')
        vals=[float(r['coverage_percentage']) for r in selected]
        ax.bar(np.arange(3)+(k-1)*.25,vals,.25,label=labels[k],color=colors[k])
        for x,y in zip(np.arange(3)+(k-1)*.25,vals):ax.text(x,y+.8,f'{y:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(np.arange(3),[str(f) for f in starts]);ax.set_xlabel('Original frame');ax.set_ylabel('Saved mask coverage of bright-region pixels (%)');ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='lower left')
    ax.set_title(f"{c['sample_name']} | original bright-pixel coverage diagnostic",fontsize=14)
    fig.text(.5,-.015,'Raw C1 > original full-frame P99, restricted to ROI; fixed region across methods. Not a segmentation-accuracy measure.',ha='center',fontsize=9)
    name='C1_three_method_bright_region_coverage.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180,bbox_inches='tight');plt.close(fig);temp.replace(out/name);files.append(name)
    checks=[]
    with tifffile.TiffFile(out/tiffname) as saved:
        if len(saved.pages)!=150 or not saved.is_imagej or saved.series[0].shape!=(150,h,w,3):raise AssertionError('TIFF page count, ImageJ flag, or shape differs.')
        for page,expected in sorted(samplehash.items()):
            observed=hashlib.sha256(saved.pages[page].asarray().tobytes()).hexdigest()
            if observed!=expected:raise AssertionError('Lossless sample readback differs.')
            checks.append({'page_index':page,'original_frame':frames[page],'sha256':observed,'match':True})
        ij=saved.imagej_metadata
    titles={
        'C1_three_method_bright_region_coverage.png':'Saved diagnostic: mask coverage of bright original-C1 pixels',
        'C1_three_method_input_previews.png':'Original / double-clipped / low-only C1 inputs',
        'C1_three_method_mask_area_comparison.png':'Three-method ROI and full-image mask areas',
        'C1_low_only_mask_pairwise_differences.png':'New low-only mask vs both references: added / removed pixels',
        'C1_low_only_saved_filter_response_diagnostic.png':'Saved low-only native filter-response diagnostic',
        tiffname:'Fiji RGB TIFF: 150 pages from three disjoint segments'}
    for f in starts:titles[f'C1_three_method_masks_frame{f:04d}.png']=f'Three masks on identical original C1: frame {f}'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Low-only C1 preprocessing mask test</title><style>body{font:16px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#172b36}img{max-width:100%;height:auto}article{margin:32px 0}a{color:#086b88}.notice{background:#fff0cc;padding:16px}</style>',
      f'<h1>{html.escape(c["sample_name"])}: low-only C1 preprocessing test</h1>',
      '<p>Three saved methods: original default; per-frame MATLAB-style P1/P99 contrast clipping; per-frame low-P1 clipping with upper bound equal to the frame maximum. Nellie broad, support ≥10/50, post-min32, holes &lt;64, ROI and coordinates are unchanged.</p>',
      '<p class="notice"><strong>Disjoint segments:</strong> frames 0–49, 1200–1249 and 2450–2499. Actual frame/time are printed on every TIFF page. Page joins are time jumps; no continuous time interval is asserted.</p>',
      '<p>All three mask columns use identical original C1 pixels at fixed 407–651 ADU; red contours show full masks before ROI clipping. Difference panels: added green, removed magenta, common gray. Input previews have separately labelled display windows.</p>']
    for name,title in titles.items():
        if not (out/name).is_file():continue
        body.append(f'<article><h2><a href="{html.escape(name)}">{html.escape(title)}</a></h2>')
        if name.endswith('.png'):body.append(f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(title)}"></a>')
        else:body.append('<p>Open in Fiji. Page 1 = original frame 0; page 51 = 1200; page 101 = 2450.</p>')
        body.append('</article>')
    body.extend(['<p><a href="../data/method_summary.csv">Saved method areas</a> · <a href="../data/pairwise_mask_comparison.csv">Saved pairwise overlap statistics</a> · <a href="../data/frame_mapping.csv">Original-frame mapping</a> · <a href="../README_RENDERING.txt">Rendering notes</a></p>','<p>Mask area or brightness does not establish accuracy. No localization, assignment, TARDIS, filter computation, or statistical inference is performed by this publisher.</p></html>'])
    (out/'INDEX.html').write_text('\n'.join(body),encoding='utf-8')
    (root/'README_RENDERING.txt').write_text('''LOW-ONLY C1 MASK TEST: RENDERING NOTES
All PNGs, RGB TIFFs, graphs and indexes read saved numerical products only.
Scientific computation calls = 0. There is no Nellie, fitting, assignment, TARDIS,
filter-response recomputation, new inference or new mask cleanup in this publisher.
Binary pixel colors and boundaries are presentation of already saved masks.

TIFF viewing order combines disjoint original frames 0-49, 1200-1249 and 2450-2499.
Original frame/time and mask block are printed on every page. Page joins are time
jumps; do not treat the page sequence as one continuously acquired movie.
The original C1 pixel background, crop, display407-651 ADU and 2x nearest-neighbor
display replication are identical in all three main mask columns. Contours are
antialiased for presentation only; mask arrays remain unchanged. Scale bars: 2 um.
Adjusted input previews use0-65535 uint16 and are labelled separately.

The difference grid shows full-image mask membership before display cropping.
Common is gray; low-only additions green; low-only removals magenta. Numerical
labels read saved ROI measurements and do not describe full-image differences.

Optional response diagnostics show saved new-input responses only:
max(Frangi response, positive LoG response). Each panel uses its own min/max
display normalization, explicitly labelled. These do not compare filter amplitude
between methods or isolate which filter causes a particular segmentation hole.

Only ten agreed sampled TIFF pages are read back. This is not a full independent
verification of every original frame. Inputs, prior mask products, source results
and default pipeline settings are not modified. C1/C2 identity registration remains
independently unverified. Contrast sensitivity does not prove mask accuracy.
''',encoding='utf-8')
    manifest={'sample_id':c['sample_id'],'scientific_steps_invoked':0,'tiff':{'relative_path':'figures/'+tiffname,'pages':150,'shape_TYXS':[150,h,w,3],'dtype':'uint8','compression':'deflate','imagej':True,'disjoint_original_segments':[[s,s+50] for s in starts],'metadata':ij},'sample_page_readback':checks,'sample_pages_checked':len(checks),'files':[{'relative_path':'figures/'+name,'bytes':(out/name).stat().st_size} for name in files],'elapsed_seconds':time.perf_counter()-began}
    (root/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'render_complete':True,'scientific_steps_invoked':0,'sample_pages_checked':len(checks),'elapsed_seconds':manifest['elapsed_seconds'],'index':str(out/'INDEX.html')},indent=2),flush=True)
if __name__=='__main__':main()
