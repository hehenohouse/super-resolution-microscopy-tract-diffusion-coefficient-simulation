#!/usr/bin/env python3
"""Render saved default/strong-signal-union results, without scientific-stage calls."""
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
    methods=['A_default','B_strong_union']
    labels=['A: original default Nellie','B: default + raw strong-signal union']
    shortlabels=['Original default','Default + strong signal']
    masks=[np.load(root/'masks'/name/'clean_full_mask.npy',mmap_mode='r') for name in methods]
    if any(a.shape!=(3,*c['shape_TYX'][1:]) for a in masks):raise ValueError('Saved masks have incompatible shapes.')
    rows=read_csv(root/'data/frame_mapping.csv');frames=[int(r['original_frame']) for r in rows];starts=c['block_starts']
    if frames!=[f for s in [0,1200,2450] for f in range(s,s+50)] or [int(r['page_index']) for r in rows]!=list(range(150)):raise ValueError('Unexpected disjoint-frame mapping.')
    metrics=read_csv(root/'data/method_summary.csv');pairs=read_csv(root/'data/pairwise_mask_comparison.csv');bright=read_csv(root/'data/bright_pixel_diagnostic.csv')
    display=read_json(root/'data/display_settings.json') if (root/'data/display_settings.json').is_file() else read_json(root/'display_settings.json')
    crop=c['crop_xyxy_px'];scale=c['display_scale'];limits=c['display_original_C1_ADU'];x0,y0,x1,y1=crop
    pw=(x1-x0)*scale;ph=(y1-y0)*scale;margin=20;gap=20;header=112;footer=105;w=3*pw+2*margin+2*gap;h=header+ph+footer
    vertices=np.load(root/'masks/roi_vertices_xy.npy');roi=roi_alpha(vertices,crop,scale)
    gray=np.array([145,145,145],float);red=np.array([255,75,75],float);green=np.array([39,221,165],np.uint8);magenta=np.array([229,82,192],np.uint8)
    fontpath=next((p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('C:/Windows/Fonts/arial.ttf')] if p.is_file()),None)
    fonts={n:ImageFont.truetype(str(fontpath),n) if fontpath else ImageFont.load_default() for n in [16,18,20,23,27]}
    contours=[[mask_contour_alpha(m[j],crop,scale) for m in masks] for j in range(3)]
    diffpanels=[]
    for j in range(3):
        a=np.asarray(masks[0][j,y0:y1,x0:x1],bool);b=np.asarray(masks[1][j,y0:y1,x0:x1],bool);rgb=np.zeros((*a.shape,3),np.uint8);rgb[a&b]=[92,92,92];rgb[b&~a]=green;rgb[a&~b]=magenta
        diffpanels.append(composite(np.repeat(np.repeat(rgb,scale,0),scale,1),roi,gray))
    def atomic_png(a,path):
        t=path.with_name(path.name+'.tmp');Image.fromarray(a).save(t,format='PNG');t.replace(path)
    def bar(draw,ox,oy):
        yy=oy+ph-22;length=round(2/(c['pixel_size_nm']/1000)*scale);draw.line((ox+16,yy,ox+16+length,yy),fill='white',width=3);draw.text((ox+16,yy-23),'2 um',fill='white',font=fonts[16])
    def render(f,raw):
        j=starts.index((f//50)*50);g=gray_crop(raw,crop,scale,limits);rgb=np.repeat(g[...,None],3,axis=2)
        image=Image.new('RGB',(w,h),'black');draw=ImageDraw.Draw(image)
        draw.text((margin,12),f"{c['sample_name']} | original frame {f} | {f*c['frame_interval_s']:.2f} s | support block {f//50}",font=fonts[27],fill='white')
        draw.text((margin,49),'Original raw C1 | per-frame broad OR strong pixels -> support >=10/50 -> post-min32 -> holes <64',font=fonts[20],fill='#DDDDDD')
        for k,title in enumerate(labels+['Difference: hybrid versus default']):
            ox=margin+k*(pw+gap)
            panel=composite(composite(rgb,contours[j][k],red),roi,gray) if k<2 else diffpanels[j]
            image.paste(Image.fromarray(panel),(ox,header));draw.text((ox,82),title,font=fonts[20],fill='white');bar(draw,ox,header)
        fy=header+ph+12
        draw.text((margin,fy),f'A/B backgrounds: identical ORIGINAL C1 at fixed {limits[0]}-{limits[1]} ADU. Red: full-mask boundary. Gray outline: original ROI.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+27),'Difference: added green / removed magenta / common gray. Strong threshold: m + 6 x (MAD/0.67449); not calibrated noise.',font=fonts[18],fill='#DDDDDD')
        draw.text((margin,fy+54),'DISJOINT SEGMENTS: frames 0-49, 1200-1249, 2450-2499. TIFF page joins are acquisition-time jumps.',font=fonts[18],fill='#FFD37E')
        return np.asarray(image)
    rawfirst=[];samplehash={};samples=set(c['sample_frames'])
    with tifffile.TiffFile(c['source_C1']) as raw:
        if len(raw.pages)!=2500 or tuple(raw.pages[0].shape)!=(684,428):raise ValueError('Original C1 dimensions differ.')
        rawfirst=[raw.pages[f].asarray() for f in starts]
        def pages():
            for page,f in enumerate(frames):
                a=render(f,raw.pages[f].asarray())
                if f in samples:samplehash[page]=hashlib.sha256(a.tobytes()).hexdigest()
                if f in starts:
                    name=f'C1_strong_union_masks_frame{f:04d}.png';atomic_png(a,out/name);files.append(name)
                if (page+1)%50==0:print(f'Rendered {page+1}/150 pages; original frame {f}',flush=True)
                yield a
        tiffname='C1_strong_union_mask_comparison_150pages.tif';temp=out/(tiffname+'.tmp')
        tifffile.imwrite(temp,pages(),shape=(150,h,w,3),dtype=np.uint8,imagej=True,photometric='rgb',compression='deflate',metadata={'axes':'TYXS','mode':'color','loop':False,'Info':'150 RGB pages from DISJOINT original-frame segments 0-49, 1200-1249, 2450-2499. Actual original frame/time printed per page. Page joins are not consecutive acquisition frames. Within-segment frame interval0.02s. Backgrounds are identical original C1 at407-651ADU. Columns: original default Nellie; default broad OR raw strong-signal pixels before identical support and cleanup; mask differences. Strong-signal threshold is full-frame median +6x(MAD/0.6744897501960817), a robust intensity margin, not calibrated camera/background noise. No uniform joined-movie time interval is asserted.'})
        temp.replace(out/tiffname);files.append(tiffname)
    # A separate common display map reveals intensity variation hidden by the standard ADU window.
    low=float(display['low_ADU']);high=float(display['high_ADU']);gamma=float(display['gamma'])
    if not (high>low and gamma>0 and all(float(np.max(a))<high for a in rawfirst)):raise ValueError('Saved detail display does not avoid high clipping on the three original full frames.')
    rowhead=36;rowgap=22;dh=112+3*(ph+rowhead+rowgap)+78;image=Image.new('RGB',(w,dh),'black');draw=ImageDraw.Draw(image)
    draw.text((margin,10),f"{c['sample_name']} | original C1: shared non-saturating detail display",font=fonts[23],fill='white')
    draw.text((margin,44),f'DISPLAY ONLY: common {low:g}-{high:g} ADU; gamma = {gamma:g}. Upper limit exceeds all three full-frame maxima; no bright pixels clipped.',font=fonts[18],fill='#DDDDDD')
    for k,title in enumerate(['Original C1; no mask overlay',labels[0],labels[1]]):draw.text((margin+k*(pw+gap),80),title,font=fonts[20],fill='white')
    for j,f in enumerate(starts):
        top=112+j*(ph+rowhead+rowgap);draw.text((margin,top),f'Original frame {f} | {f*.02:.2f} s',font=fonts[20],fill='white')
        v=np.rint(np.clip((rawfirst[j][y0:y1,x0:x1].astype(float)-low)/(high-low),0,1)**gamma*255).astype(np.uint8)
        rgb=np.repeat(np.repeat(np.repeat(v[...,None],3,axis=2),scale,0),scale,1)
        for k in range(3):
            panel=composite(rgb,roi,gray) if k==0 else composite(composite(rgb,contours[j][k-1],red),roi,gray)
            ox=margin+k*(pw+gap);oy=top+rowhead;image.paste(Image.fromarray(panel),(ox,oy));bar(draw,ox,oy)
    draw.text((margin,dh-59),'Only the display window and gamma differ from the main figures. Source C1 pixels and scientific mask inputs are unchanged.',font=fonts[18],fill='#FFD37E')
    draw.text((margin,dh-33),'This view helps inspect bright-region structure; it is not a new preprocessing experiment or a segmentation-accuracy score.',font=fonts[18],fill='#DDDDDD')
    name='C1_strong_union_non_saturating_detail.png';atomic_png(np.asarray(image),out/name);files.append(name)
    # Existing broad/support/final products; first frame of first block only.
    sh=112+2*(ph+rowhead+rowgap)+66;image=Image.new('RGB',(w,sh),'black');draw=ImageDraw.Draw(image)
    draw.text((margin,10),f"{c['sample_name']} | saved-stage diagnostic, block 0-49",font=fonts[23],fill='white')
    draw.text((margin,44),'White = saved foreground | gray ROI outline | same native crop and spatial scale in every panel',font=fonts[18],fill='#DDDDDD')
    for k,title in enumerate(['Per-frame broad at original frame 0','Support >=10/50, before cleanup','Final mask after min32 / holes <64']):draw.text((margin+k*(pw+gap),79),title,font=fonts[20],fill='white')
    for m,mid in enumerate(methods):
        with np.load(root/'masks'/mid/'broad_packed/b000.npz') as z:
            shape=tuple(int(v) for v in z['shape']);broad=np.unpackbits(z['packed'][0],count=shape[1]*shape[2],bitorder='little').reshape(shape[1:]).astype(bool)
            if int(z['original_frame_ids'][0])!=0:raise ValueError('Saved broad frame mapping differs.')
        with np.load(root/'masks'/mid/'postprocess_stages.npz') as z:support=np.asarray(z['support_mask'][0],bool)
        top=112+m*(ph+rowhead+rowgap);draw.text((margin,top),labels[m],font=fonts[20],fill='white')
        for k,mask in enumerate([broad,support,masks[m][0]]):
            rgb=np.repeat(gray_crop(mask,crop,scale,[0,1])[...,None],3,axis=2);ox=margin+k*(pw+gap);oy=top+rowhead;image.paste(Image.fromarray(composite(rgb,roi,gray)),(ox,oy));bar(draw,ox,oy)
    draw.text((margin,sh-49),'Broad is one original frame; support and final masks summarize the full 50-frame block. They are distinct temporal quantities.',font=fonts[18],fill='#FFD37E')
    draw.text((margin,sh-24),'Saved masks only; no Nellie, thresholding, cleanup, localization, assignment or TARDIS is called by this publisher.',font=fonts[18],fill='#DDDDDD')
    name='C1_strong_union_saved_stages_block0000.png';atomic_png(np.asarray(image),out/name);files.append(name)
    plt.rcParams.update({'font.size':11,'axes.titlesize':13});fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);colors=['#687C91','#32AE9C']
    for ax,scope in zip(axes,['ROI','full']):
        for k,method in enumerate(methods):
            selected=sorted([r for r in metrics if r['scope']==scope and r['method']==method],key=lambda r:int(r['original_frame_start']))
            if len(selected)!=3:raise ValueError('Missing area rows for scope/method.')
            ax.bar(np.arange(3)+(k-.5)*.32,[float(r['area_um2']) for r in selected],.32,label=shortlabels[k],color=colors[k])
        ax.set_xticks(np.arange(3),[f'{s}-{s+49}' for s in starts]);ax.set_xlabel('Original acquisition-frame block');ax.set_ylabel('Binary mask area (um^2)');ax.set_title('Inside original cell ROI' if scope=='ROI' else 'Full image before ROI clipping');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='upper left');ax.set_ylim(0,max(float(r['area_um2']) for r in metrics if r['scope']==scope)*1.28)
    fig.suptitle(f"{c['sample_name']} | default versus strong-signal-union mask areas",fontsize=15)
    name='C1_strong_union_mask_area_comparison.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180);plt.close(fig);temp.replace(out/name);files.append(name)
    fig,ax=plt.subplots(figsize=(10,4.9),constrained_layout=True)
    for k,method in enumerate(methods):
        selected=sorted([r for r in bright if r['method']==method],key=lambda r:int(r['original_frame']))
        if len(selected)!=3:raise ValueError('Missing saved bright-coverage records.')
        vals=[float(r['coverage_percentage']) for r in selected]
        ax.bar(np.arange(3)+(k-.5)*.32,vals,.32,label=shortlabels[k],color=colors[k])
        for xx,yy in zip(np.arange(3)+(k-.5)*.32,vals):ax.text(xx,yy+.8,f'{yy:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(np.arange(3),[str(f) for f in starts]);ax.set_xlabel('Original frame');ax.set_ylabel('Saved mask coverage of bright-region pixels (%)');ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='lower left')
    ax.set_title(f"{c['sample_name']} | original bright-pixel coverage diagnostic",fontsize=14)
    fig.text(.5,-.015,'Raw C1 > original full-frame P99, restricted to ROI; same fixed region across methods. Not an accuracy measure.',ha='center',fontsize=9)
    name='C1_strong_union_bright_region_coverage.png';temp=out/(name+'.tmp');fig.savefig(temp,format='png',dpi=180,bbox_inches='tight');plt.close(fig);temp.replace(out/name);files.append(name)
    checks=[]
    with tifffile.TiffFile(out/tiffname) as saved:
        if len(saved.pages)!=150 or not saved.is_imagej or saved.series[0].shape!=(150,h,w,3):raise AssertionError('TIFF page count, ImageJ flag, or shape differs.')
        for page,expected in sorted(samplehash.items()):
            observed=hashlib.sha256(saved.pages[page].asarray().tobytes()).hexdigest()
            if observed!=expected:raise AssertionError('Lossless sample readback differs.')
            checks.append({'page_index':page,'original_frame':frames[page],'sha256':observed,'match':True})
        ij=saved.imagej_metadata
    titles={'C1_strong_union_non_saturating_detail.png':'Shared non-saturating raw-C1 detail: no overlay / default / hybrid','C1_strong_union_mask_area_comparison.png':'ROI and full-image areas: default versus hybrid','C1_strong_union_bright_region_coverage.png':'Coverage of fixed original-C1 bright regions','C1_strong_union_saved_stages_block0000.png':'Saved per-frame broad / 50-frame support / final masks',tiffname:'Fiji RGB TIFF: 150 pages from three disjoint segments'}
    for f in starts:titles[f'C1_strong_union_masks_frame{f:04d}.png']=f'Default / hybrid / differences on original C1: frame {f}'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Nellie plus raw strong-signal mask test</title><style>body{font:16px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#172b36}img{max-width:100%;height:auto}article{margin:32px 0}a{color:#086b88}.notice{background:#fff0cc;padding:16px}</style>',f'<h1>{html.escape(c["sample_name"])}: default Nellie plus raw strong-signal pixels</h1>','<p>A reuses the original default Nellie masks. B unions the saved per-frame default broad mask with raw C1 pixels exceeding the full-frame positive-pixel median + 6 × (MAD / 0.6744897501960817), before the identical 50-frame support and cleanup. This threshold is a robust intensity margin, not calibrated camera/background noise or a six-sigma confidence statement. No anchoring, dilation, blanket hole filling, or C2-based selection is used.</p>','<p class="notice"><strong>Disjoint segments:</strong> frames 0–49, 1200–1249 and 2450–2499. Actual acquisition frame/time are printed on every TIFF page. Page joins are time jumps; no continuous time interval is asserted.</p>','<p>Main A/B mask columns use identical original C1 pixels at fixed 407–651 ADU. Red contours show full masks before ROI clipping; gray outline is the original cell ROI. Difference panel: added green, removed magenta, common gray.</p>',f'<p>The separate detail image uses a common {low:g}–{high:g} ADU window and gamma {gamma:g}, with an upper display limit exceeding all three source-frame maxima. This display-only map reveals intensity variation hidden by the standard bright display and never changes segmentation inputs.</p>']
    for name,title in titles.items():
        if not (out/name).is_file():continue
        body.append(f'<article><h2><a href="{html.escape(name)}">{html.escape(title)}</a></h2>')
        if name.endswith('.png'):body.append(f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(title)}"></a>')
        else:body.append('<p>Open in Fiji. Page 1 = original frame 0; page 51 = 1200; page 101 = 2450.</p>')
        body.append('</article>')
    body+=['<p><a href="../data/method_summary.csv">Saved mask areas</a> · <a href="../data/pairwise_mask_comparison.csv">Saved overlap measurements</a> · <a href="../data/bright_pixel_diagnostic.csv">Saved bright-region diagnostic</a> · <a href="../data/frame_mapping.csv">Original-frame mapping</a> · <a href="../README_RENDERING.txt">Rendering notes</a></p>','<p>Area or bright-region coverage does not establish segmentation accuracy. No scientific stage, localization, assignment, TARDIS or statistical inference is performed by this publisher.</p></html>']
    (out/'INDEX.html').write_text('\n'.join(body),encoding='utf-8')
    (root/'README_RENDERING.txt').write_text(f'''NELLIE + RAW STRONG-SIGNAL UNION: RENDERING NOTES
All PNGs, RGB TIFF pages, plots and indexes read saved numerical products only.
Scientific-stage calls = 0. No Nellie, raw-signal thresholding, cleanup, inference,
localization, spatial assignment or TARDIS occurs in this publisher.

Raw C1 is unchanged. A reuses default Nellie results. B uses default per-frame
broad OR raw strong pixels, before the unchanged 50-frame support and cleanup.
The raw threshold is median + 6 x (MAD/0.6744897501960817), with the reference
pixels defined in the saved scientific configuration. This is a robust
intensity margin, not calibrated background/camera noise or a six-sigma claim.

TIFF order combines disjoint frames 0-49, 1200-1249 and 2450-2499.
Every page records actual frame, time and block. Page joins are time jumps.
No uniform acquisition interval is asserted for the joined sequence.
Main original-C1 display 407-651 ADU, crop and 2x nearest-neighbor replication
are fixed across A/B. Scale bars: 2 um. Antialiased boundaries are display only.
Added=hybrid green, removed=default magenta, common gray. Difference colors are
full-mask membership; source data tables separately report ROI statistics.

The non-saturating detail image uses shared {low:g}-{high:g} ADU, gamma {gamma:g}.
The high limit exceeds all three displayed source full-frame maxima; no bright
pixels are clipped by that map. This is display only. No raw input is rewritten.
The stage diagnostic shows frame-0 broad versus full-block support/final.
These are distinct temporal quantities. Bright-region coverage is read from
the saved table using identical original-C1 regions across methods; it is not
ground-truth accuracy.

Only ten specified TIFF pages are read back. No all-2500-frame replay or full
independent verification occurs. Original inputs, prior results and pipeline
defaults remain unchanged. C1/C2 identity registration is unverified.
''',encoding='utf-8')
    manifest={'sample_id':c['sample_id'],'scientific_steps_invoked':0,'tiff':{'relative_path':'figures/'+tiffname,'pages':150,'shape_TYXS':[150,h,w,3],'dtype':'uint8','compression':'deflate','imagej':True,'disjoint_original_segments':[[s,s+50] for s in starts],'metadata':ij},'sample_page_readback':checks,'sample_pages_checked':len(checks),'detail_display':display,'files':[{'relative_path':'figures/'+name,'bytes':(out/name).stat().st_size} for name in files],'elapsed_seconds':time.perf_counter()-began}
    (root/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'render_complete':True,'scientific_steps_invoked':0,'sample_pages_checked':len(checks),'elapsed_seconds':manifest['elapsed_seconds'],'index':str(out/'INDEX.html')},indent=2),flush=True)
if __name__=='__main__':main()
