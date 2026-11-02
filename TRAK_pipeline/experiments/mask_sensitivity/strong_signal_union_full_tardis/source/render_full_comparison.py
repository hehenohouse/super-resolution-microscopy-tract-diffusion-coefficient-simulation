#!/usr/bin/env python3
"""Read-only publication of saved full-movie hybrid-mask assignment/TARDIS products."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,html,json,os,sys,time
from pathlib import Path

def jsread(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def csvread(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def istrue(v):return str(v).strip().lower() in ("true","1")
def num(v):return float(v) if str(v).strip() else float("nan")
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path);a=p.parse_args()
    c=jsread(a.config);root=Path(c["run_directory"]).resolve()
    for marker in ["MASK_ASSIGNMENT_COMPLETE.json","TARDIS_COMPLETE.json"]:
        if not (root/marker).is_file():raise SystemExit("Required saved scientific products missing: "+marker)
    for key,rel in [("TMPDIR","tmp"),("TMP","tmp"),("TEMP","tmp"),("MPLCONFIGDIR","matplotlib"),("NUMBA_CACHE_DIR","numba"),("XDG_CACHE_HOME","xdg")]:
        d=root/"runtime"/rel;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
    os.environ["MPLBACKEND"]="Agg";sys.dont_write_bytecode=True
    import tempfile;tempfile.tempdir=str(root/"runtime/tmp")
    print("RENDER_IMPORT: NumPy",flush=True)
    import numpy as np
    # Bypass a blocked local bytecode cache read; load the same installed source.
    import importlib.machinery
    original_get_code=importlib.machinery.SourceFileLoader.get_code
    def get_code_without_bad_plot_cache(loader,fullname):
        if "site-packages" in loader.path:
            return compile(Path(loader.path).read_bytes(),loader.path,"exec",dont_inherit=True)
        return original_get_code(loader,fullname)
    importlib.machinery.SourceFileLoader.get_code=get_code_without_bad_plot_cache
    print("RENDER_IMPORT: plotting dependencies (unchanged source; cached-bytecode bypass)",flush=True)
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("RENDER_IMPORT: Pillow",flush=True)
    from PIL import Image,ImageDraw,ImageFont
    import PIL
    from point_stack_core import gray_crop,mask_contour_alpha,roi_alpha,composite,display_centers,marker_alpha,OFF
    print("RENDER_IMPORTS_COMPLETE",flush=True)
    began=time.perf_counter();out=root/"figures";out.mkdir(exist_ok=True);baseline=Path(c["source_output"])
    styles={"A_default":dict(label="Original default",color="#2E6E9E",linestyle="-"),"B_strong_union":dict(label="Default + strong signal",color="#E07B32",linestyle="--")}
    mids=list(styles);groups=["On","Not-on"];files=[];titles={};source_paths={str(a.config.resolve()),c["source_C1"],str(baseline/"masks/clean_full_mask.npy"),str(root/"masks/clean_full_mask.npy"),str(root/"masks/roi_vertices_xy.npy"),str(root/"data/assignment_labels.csv.gz"),str(baseline/"data/localizations.csv.gz")}
    def readrows(path):
        source_paths.add(str(Path(path).resolve()));return csvread(path)
    curves=readrows(root/"data/comparison_curves.csv")
    scalars=readrows(root/"data/comparison_curve_scalars.csv")
    summary=jsread(root/"classification_summary.json");source_paths.add(str(root/"classification_summary.json"))
    area=readrows(root/"data/mask_area_comparison.csv");counts=readrows(root/"data/per_frame_counts.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":14,"axes.labelsize":12,"legend.fontsize":10,"figure.facecolor":"white","savefig.facecolor":"white"})
    common_note="Original 1-SE ellipse assignment; true adjacent frames (20 ms). Negative bins retained; no smoothing."
    band_note="Shading: exploratory L25 pointwise 2.5-97.5% ranges (1000 resamples), conditional on fixed masks/labels; no mask-uncertainty propagation or significance test."
    limits={}
    fields={"signed":("signed_intra_density_um_inv","Signed Intra density (1/um)","Signed Intra"),
            "area":("area_normalized_density_um_inv","Signed-area-normalized density (1/um)","Signed-area-normalized Intra"),
            "peak":("positive_peak_normalized_intra","Positive-peak-normalized Intra","Positive-peak-normalized Intra")}
    bandcache={}
    def rowsfor(mid,group,scope="full"):
        r=sorted([r for r in curves if r["method"]==mid and r["group"]==group and r["scope"]==scope],key=lambda r:num(r["distance_mid_um"]))
        if len(r)!=300:raise ValueError(f"Expected 300 saved curve bins for {mid}/{group}/{scope}; found {len(r)}")
        return r
    def bands(mid,scope="full"):
        key=(mid,scope)
        if key not in bandcache:
            path=(baseline if mid=="A_default" else root)/"data/tardis/bootstrap"/f"paired_bands_{scope}_L25.csv"
            bandcache[key]=readrows(path) if path.is_file() else None
        return bandcache[key]
    def curvevals(mid,group,kind,scope="full"):
        rows=rowsfor(mid,group,scope);field=fields[kind][0]
        x=np.array([num(r["distance_mid_um"]) for r in rows]);y=np.array([num(r[field]) for r in rows])
        return x,y,rows
    def bandvals(mid,group,kind,scope="full"):
        b=bands(mid,scope)
        if b is None or kind=="peak":return None
        pre="On" if group=="On" else "Not_on";part="intra" if kind=="signed" else "shape"
        return (np.array([num(r["distance_mid_um"]) for r in b]),
                np.array([num(r[f"{pre}_{part}_low"]) for r in b]),
                np.array([num(r[f"{pre}_{part}_high"]) for r in b]))
    def ylim_for(kind,scopes=("full",),showbands=True):
        values=[0.]
        for scope in scopes:
            for group in groups:
                for mid in mids:
                    x,y,_=curvevals(mid,group,kind,scope);values.extend(y[np.isfinite(y)])
                    b=bandvals(mid,group,kind,scope) if showbands else None
                    if b:
                        for ar in b[1:]:values.extend(ar[np.isfinite(ar)])
        lo,hi=min(values),max(values)
        width=max(hi-lo,1e-3);return (lo-.07*width,hi+.12*width)
    def save(fig,name,title,extra_bottom=False):
        tmp=out/(name+".tmp");fig.savefig(tmp,format="png",dpi=180,bbox_inches="tight" if extra_bottom else None);plt.close(fig);tmp.replace(out/name);files.append(name);titles[name]=title
    def decorate(ax,kind,xmax,scope="full",legend=True):
        for mid in mids:
            st=styles[mid];x,y,rows=curvevals(mid,group=ax._group,kind=kind,scope=scope)
            ax.plot(x,y,color=st["color"],ls=st["linestyle"],lw=2,label=st["label"])
            b=bandvals(mid,ax._group,kind,scope)
            if b is not None:ax.fill_between(*b,color=st["color"],alpha=.11,lw=0)
            # Unavailable values stay NaN; never replace with zero.
            if not np.isfinite(y).any():ax.text(.5,.5,st["label"]+": unavailable",ha="center",transform=ax.transAxes,color=st["color"])
        ax.axhline(0,color="#777",lw=.8);ax.set_xlim(0,xmax);ax.set_ylim(limits[kind]);ax.grid(alpha=.18);ax.set_axisbelow(True)
        ax.set_xlabel("Distance (um)");ax.set_ylabel(fields[kind][1]);ax.set_title(ax._group+" mitochondria")
        if legend:ax.legend(frameon=False,loc="upper right")
    for kind in fields:
        limits[kind]=ylim_for(kind)
        for xmax,suffix in [(3.0,"full"),(.5,"zoom")]:
            fig,axs=plt.subplots(1,2,figsize=(14,5.6),sharey=True)
            for ax,group in zip(axs,groups):ax._group=group;decorate(ax,kind,xmax)
            fig.suptitle(c["sample_name"]+" | "+fields[kind][2],fontsize=17,y=.97)
            fig.subplots_adjust(top=.84,bottom=.20,wspace=.23,left=.075,right=.985)
            fig.text(.5,.10,common_note,ha="center",fontsize=10)
            fig.text(.5,.055,band_note if kind!="peak" else "Peak determined on the full 0-3 um signed curve; saved values only. No peak bootstrap bands.",ha="center",fontsize=9.5)
            name=f"TARDIS_{kind}_default_vs_strong_union_{suffix}.png"
            save(fig,name,fields[kind][2]+f": default versus hybrid, 0-{xmax:g} um")
    # Main full-and-zoom plot: identical y limits across the four panels.
    fig,axs=plt.subplots(2,2,figsize=(14,9.4),sharey=True)
    for i,xmax in enumerate([3.0,.5]):
        for j,group in enumerate(groups):
            ax=axs[i,j];ax._group=group;decorate(ax,"area",xmax);ax.set_title(group+" mitochondria | "+("full range" if i==0 else "0-0.5 um detail"))
    fig.suptitle(c["sample_name"]+" | mask-method sensitivity of Intra shape",fontsize=18,y=.98)
    fig.subplots_adjust(top=.91,bottom=.14,hspace=.36,wspace=.23,left=.075,right=.985)
    fig.text(.5,.075,common_note,ha="center",fontsize=10)
    fig.text(.5,.04,band_note,ha="center",fontsize=9.3)
    save(fig,"TARDIS_area_default_vs_strong_union_full_and_zoom.png","Main comparison: signed-area-normalized Intra, full range and zoom")
    # Decomposition reads the already saved total/background/inter/intra fields.
    fig,axs=plt.subplots(2,2,figsize=(14,9.2),sharex=True)
    decomp_values=[0.]
    compcols=[("total_density_um_inv","Total","#202b33"),("background_density_um_inv","Background","#6795ad"),("inter_density_um_inv","Inter","#969e54"),("intra_density_um_inv","Signed Intra","#9e4f9f")]
    for i,mid in enumerate(mids):
        for j,group in enumerate(groups):
            path=(baseline if mid=="A_default" else root)/"data/tardis/distribution"/group/"full.csv";rr=readrows(path);ax=axs[i,j]
            x=np.array([num(r["distance_mid_um"]) for r in rr])
            for col,label,color in compcols:
                yy=np.array([num(r[col]) for r in rr]);decomp_values.extend(yy[np.isfinite(yy)]);ax.plot(x,yy,label=label,color=color,lw=1.7)
            ax.axhline(0,color="#777",lw=.8);ax.grid(alpha=.18);ax.set_xlim(0,3);ax.set_title(styles[mid]["label"]+" | "+group);ax.set_xlabel("Distance (um)");ax.set_ylabel("Density (1/um)");ax.legend(frameon=False,fontsize=9)
    dl,dh=min(decomp_values),max(decomp_values);dw=max(dh-dl,1e-3)
    for ax in axs.flat:ax.set_ylim(dl-.07*dw,dh+.10*dw)
    fig.suptitle(c["sample_name"]+" | saved TARDIS decomposition",fontsize=17)
    fig.subplots_adjust(top=.90,bottom=.12,hspace=.34,wspace=.22,left=.075,right=.985)
    fig.text(.5,.045,"Each method refits its own background coefficient beta after spatial reassignment. No localization or Nellie rerun.",ha="center",fontsize=10)
    save(fig,"TARDIS_decomposition_default_vs_strong_union.png","Total / background / Inter / signed Intra decomposition")
    # Ten 250-frame time ranges; plot saved area-normalized curves, no new normalization.
    scopes=sorted({r["scope"] for r in curves if r["scope"]!="full"})
    if len(scopes)!=10:raise ValueError("Expected ten saved 250-frame scopes.")
    seglimits=ylim_for("area",scopes,showbands=False)
    fig,axs=plt.subplots(5,4,figsize=(18,16),sharex=True,sharey=True)
    for k,scope in enumerate(scopes):
        block=k//2;column=(k%2)*2
        for j,group in enumerate(groups):
            ax=axs[block,column+j]
            for mid in mids:
                x,y,_=curvevals(mid,group,"area",scope);st=styles[mid];ax.plot(x,y,color=st["color"],ls=st["linestyle"],lw=1.5)
                if not np.isfinite(y).any():ax.text(.04,.9-.12*mids.index(mid),st["label"]+": unavailable",fontsize=8,color=st["color"],transform=ax.transAxes)
            ax.axhline(0,color="#777",lw=.6);ax.grid(alpha=.15);ax.set_xlim(0,.5);ax.set_ylim(seglimits);ax.set_title(scope+" | "+group,fontsize=11);ax.set_xlabel("Distance (um)");ax.set_ylabel("Density (1/um)")
    fig.suptitle(c["sample_name"]+" | ten 250-frame Intra comparisons (zoom)",fontsize=19,y=.994)
    fig.subplots_adjust(top=.96,bottom=.07,hspace=.53,wspace=.25,left=.055,right=.99)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=styles[m]["color"],ls=styles[m]["linestyle"],lw=2,label=styles[m]["label"]) for m in mids],loc="lower center",bbox_to_anchor=(.5,.028),ncol=2,frameon=False)
    fig.text(.5,.009,"Saved signed-area-normalized curves; no smoothing. Same y range in every time segment; unavailable curves are not filled.",ha="center",fontsize=10)
    save(fig,"TARDIS_ten_segments_area_comparison_zoom.png","Ten acquisition segments: saved area-normalized Intra (0-0.5 um)")
    # Counts and On proportion use saved denominators, not new rules.
    old=summary["default_counts"];methods_counts=[old,summary];onpct=[summary["default_On_percentage"],summary["On_percentage"]]
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.4));x=np.arange(2);cols=[styles[m]["color"] for m in mids]
    bars=axs[0].bar(x,onpct,color=cols,width=.55)
    for b,v in zip(bars,onpct):axs[0].text(b.get_x()+b.get_width()/2,v+1,f"{v:.3f}%",ha="center",fontsize=12)
    axs[0].set_ylim(0,100);axs[0].set_ylabel("On / (On + Not-on) (%)");axs[0].set_xticks(x,[styles[m]["label"] for m in mids]);axs[0].set_title("Saved On percentage")
    width=.24
    for j,(key,label,color) in enumerate([("On","On","#27b18e"),("Not_on","Not-on","#ab78d2"),("unknown","Unknown","#8f979d")]):
        vals=[rr[key] for rr in methods_counts];bs=axs[1].bar(x+(j-1)*width,vals,width,label=label,color=color)
        for b,v in zip(bs,vals):axs[1].annotate(f"{v:,}",(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
    axs[1].set_xticks(x,[styles[m]["label"] for m in mids]);axs[1].set_ylabel("Selected localization observations");axs[1].set_title("Saved spatial-group counts");axs[1].legend(frameon=False)
    for ax in axs:ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(c["sample_name"]+" | assignment sensitivity",fontsize=17)
    fig.subplots_adjust(top=.84,bottom=.20,wspace=.23,left=.08,right=.985)
    fig.text(.5,.085,f"Original QC and ROI selection unchanged: {summary['selected']:,} selected; {summary['QC_excluded']:,} QC excluded. Unknown is outside On denominator.",ha="center",fontsize=10)
    fig.text(.5,.04,"Mask supplement is a spatial-method sensitivity test; proportions are not segmentation-accuracy scores.",ha="center",fontsize=10)
    save(fig,"Assignment_counts_and_On_percentage_comparison.png","Spatial counts and On percentage: original versus hybrid")
    fig,axs=plt.subplots(2,1,figsize=(12.5,7.2),sharex=True)
    t=np.array([num(r["time_s"]) for r in counts])
    for ax,group,key in zip(axs,groups,["On","Not_on"]):
        for mid,field in zip(mids,["default_"+key,key]):
            st=styles[mid];ax.plot(t,[int(r[field]) for r in counts],color=st["color"],ls=st["linestyle"],lw=.8,label=st["label"])
        ax.set_ylabel(group+" observations/frame");ax.grid(alpha=.18);ax.legend(frameon=False)
    axs[-1].set_xlabel("Acquisition time (s)");axs[-1].set_xlim(0,50)
    fig.suptitle(c["sample_name"]+" | saved per-frame localization counts",fontsize=17);fig.subplots_adjust(top=.92,bottom=.10,hspace=.22,left=.10,right=.985)
    save(fig,"Assignment_per_frame_counts_comparison.png","Per-frame On / Not-on observation counts")
    fig,axs=plt.subplots(2,1,figsize=(12.5,7.2),sharex=True)
    for ax,scope in zip(axs,["ROI","full"]):
        rr=sorted([r for r in area if r["scope"]==scope],key=lambda r:int(r["block_index"]));tx=np.array([int(r["original_frame_start"])*c["frame_interval_s"] for r in rr]);tx=np.r_[tx,50.]
        for mid,field in zip(mids,["default_area_um2","hybrid_area_um2"]):
            st=styles[mid];yy=np.array([num(r[field]) for r in rr]);ax.step(tx,np.r_[yy,yy[-1]],where="post",color=st["color"],ls=st["linestyle"],lw=1.9,label=st["label"])
        ax.set_title("Inside original cell ROI" if scope=="ROI" else "Full image before ROI clipping");ax.set_ylabel("Mask area (um^2)");ax.grid(alpha=.18);ax.legend(frameon=False)
    axs[-1].set_xlabel("Acquisition time (s)");axs[-1].set_xlim(0,50)
    fig.suptitle(c["sample_name"]+" | 50-frame support-mask area",fontsize=17);fig.subplots_adjust(top=.89,bottom=.14,hspace=.36,left=.09,right=.985)
    fig.text(.5,.045,"Same 50-frame mapping and 117 nm pixels. No error expansion; area does not measure segmentation accuracy.",ha="center",fontsize=10)
    save(fig,"Mask_area_default_vs_strong_union.png","Full-movie mask area: default versus hybrid")
    # Read-only raw-C1 representative overlays and same cached mean background.
    crop=c["crop_xyxy_px"];scale=int(c["display_scale"]);x0,y0,x1,y1=crop;pw=(x1-x0)*scale;ph=(y1-y0)*scale
    vertices=np.load(root/"masks/roi_vertices_xy.npy");ralpha=roi_alpha(vertices,crop,scale)
    oldmask=np.load(baseline/"masks/clean_full_mask.npy",mmap_mode="r");newmask=np.load(root/"masks/clean_full_mask.npy",mmap_mode="r")
    fontpath=next((str(p) for p in [Path("/System/Library/Fonts/Supplemental/Arial.ttf"),Path("C:/Windows/Fonts/arial.ttf")] if p.is_file()),None)
    fonts={n:ImageFont.truetype(fontpath,n) if fontpath else ImageFont.load_default() for n in [16,18,20,23]}
    def pngimg(canvas,name,title):
        tmp=out/(name+".tmp");canvas.save(tmp,format="PNG");tmp.replace(out/name);files.append(name);titles[name]=title
    def drawbar(d,ox,oy):
        length=round(2/(c["pixel_size_nm"]/1000)*scale);yy=oy+ph-24;d.line((ox+15,yy,ox+15+length,yy),fill="white",width=3);d.text((ox+15,yy-23),"2 um",font=fonts[16],fill="white")
    raw_display_samples=[]
    original_sample_hashes={int(v["original_frame"]):v["raw_pixels_sha256"] for v in jsread(Path(c["prior_pilot"])/"validation.json")["samples"]}
    with Image.open(c["source_C1"]) as raw:
        for frame in [0,1200,2450]:
            raw.seek(frame);arr=np.asarray(raw)
            if arr.shape!=tuple(c["shape_TYX"][1:]) or arr.dtype.kind!="u" or arr.dtype.itemsize!=2:raise ValueError("Representative C1 display grid/type differs")
            # Pillow preserves stored TIFF byte order; convert byte order without changing values.
            arr=np.asarray(arr,dtype=np.uint16)
            digest=hashlib.sha256(arr.tobytes()).hexdigest()
            if digest!=original_sample_hashes[frame]:raise ValueError("Pillow raw C1 values differ from original TIFF sample")
            raw_display_samples.append(dict(original_frame=frame,raw_values_match_original=True,native_uint16_SHA256=digest))
            rgb=np.repeat(gray_crop(arr,crop,scale,c["display_original_C1_ADU"])[...,None],3,axis=2);block=frame//50
            w=3*pw+80;h=ph+185;canvas=Image.new("RGB",(w,h),"black");d=ImageDraw.Draw(canvas)
            d.text((20,12),c["sample_name"]+f" | original frame {frame} ({frame*.02:.2f} s)",font=fonts[23],fill="white")
            d.text((20,45),"Same original C1 in both mask panels; final full masks before ROI clipping",font=fonts[18],fill="#dddddd")
            for j,(mask,title) in enumerate([(oldmask[block],"Original default"),(newmask[block],"Default + strong signal"),(None,"Difference: added green / removed magenta")]):
                ox=20+j*(pw+20);oy=100
                if mask is None:
                    a0=np.asarray(oldmask[block,y0:y1,x0:x1]);b0=np.asarray(newmask[block,y0:y1,x0:x1]);pix=np.zeros((*a0.shape,3),np.uint8);pix[a0&b0]=[90,90,90];pix[b0&~a0]=[39,221,165];pix[a0&~b0]=[229,82,192];panel=np.repeat(np.repeat(pix,scale,0),scale,1)
                else:panel=composite(rgb,mask_contour_alpha(mask,crop,scale),[255,75,75])
                panel=composite(panel,ralpha,[140,140,140]);canvas.paste(Image.fromarray(panel),(ox,oy));d.text((ox,73),title,font=fonts[18],fill="white");drawbar(d,ox,oy)
            d.text((20,h-69),"C1 display: 407-651 ADU. The strong-signal rule acts before 50-frame support; all original foreground is retained.",font=fonts[18],fill="#dddddd")
            d.text((20,h-41),"This display can saturate bright structures. The masks are a sensitivity comparison, not validated segmentation truth.",font=fonts[18],fill="#dddddd")
            pngimg(canvas,f"Mask_comparison_frame{frame:04d}.png",f"Original C1 and mask differences at frame {frame}")
    oldmask._mmap.close();newmask._mmap.close()
    # UID join only, no new selection/classification.
    labels={}
    with gzip.open(root/"data/assignment_labels.csv.gz","rt",newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid=r["source_detection_uid"]
            if uid in labels:raise ValueError("Duplicate UID in saved labels")
            labels[uid]=r
    xyold=[];xynew=[];seen=set()
    with gzip.open(baseline/"data/localizations.csv.gz","rt",newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid=r["source_detection_uid"]
            if uid not in labels:raise ValueError("Saved label join misses master UID")
            lr=labels[uid];seen.add(uid)
            if lr["default_label"]!=r["pointwise_label"]:raise ValueError("Default label changed")
            pos=[num(r["fit_x_px"]),num(r["fit_y_px"])]
            if r["pointwise_label"]=="non-mito-compatible":xyold.append(pos)
            if lr["pointwise_label"]=="non-mito-compatible":xynew.append(pos)
    if len(seen)!=len(labels) or [len(xyold),len(xynew)]!=[old["Not_on"],summary["Not_on"]]:raise ValueError("Not-on overlay counts do not match saved summary")
    meanpath=baseline/"data/render_sources/C1_mean_frames0000_2499.npy";source_paths.add(str(meanpath))
    if not meanpath.is_file():raise ValueError("Saved C1 arithmetic mean missing; publisher will not recompute it.")
    mean=np.load(meanpath,mmap_mode="r")
    if mean.shape!=tuple(c["shape_TYX"][1:]):raise ValueError("Cached C1 mean grid differs")
    rgb=np.repeat(gray_crop(mean,crop,scale,c["display_original_C1_ADU"])[...,None],3,axis=2);base=composite(rgb,ralpha,[140,140,140])
    w=2*pw+60;h=ph+185;canvas=Image.new("RGB",(w,h),"black");d=ImageDraw.Draw(canvas);d.text((20,12),c["sample_name"]+" | Not-on observations on cached mean C1",font=fonts[23],fill="white");d.text((20,45),"Gray: arithmetic mean of all 2500 original C1 frames (50 s). Purple: saved Not-on centers.",font=fonts[18],fill="#dddddd")
    for j,(xy,mid) in enumerate(zip([xyold,xynew],mids)):
        alpha=marker_alpha((ph,pw),display_centers(xy,crop,scale),.40*scale);panel=composite(base,alpha,OFF);ox=20+j*(pw+20);canvas.paste(Image.fromarray(panel),(ox,100));d.text((ox,73),styles[mid]["label"]+f" | n={len(xy):,}",font=fonts[18],fill="white");drawbar(d,ox,100)
    d.text((20,h-69),"Same original 407-651 ADU display, ROI, crop and dot size. Repeated observations retained; dots are not error sizes.",font=fonts[18],fill="#dddddd")
    d.text((20,h-41),"Assignments use the corresponding 1 s mask block; 50 s mean projection is qualitative. Registration remains unverified.",font=fonts[18],fill="#dddddd")
    pngimg(canvas,"Not_on_on_mean_C1_default_vs_strong_union.png","Not-on centers on the saved 2500-frame mean C1")
    mean._mmap.close()
    # Final index and bounded file structural inspection.
    descriptions={
        "TARDIS_area_default_vs_strong_union_full_and_zoom.png":"The main old/new comparison. Both groups, full range and zoom use a common y range; blue solid=default, orange dashed=hybrid.",
        "Assignment_counts_and_On_percentage_comparison.png":"On% is On/(On+Not-on); unknown and QC excluded are outside that denominator.",
        "Not_on_on_mean_C1_default_vs_strong_union.png":"Only saved Not-on labels and floating centers are shown; no trajectories or molecular identities are inferred."
    }
    order=["TARDIS_area_default_vs_strong_union_full_and_zoom.png","Assignment_counts_and_On_percentage_comparison.png","Mask_area_default_vs_strong_union.png"]+[f for f in files if f not in ["TARDIS_area_default_vs_strong_union_full_and_zoom.png","Assignment_counts_and_On_percentage_comparison.png","Mask_area_default_vs_strong_union.png"]]
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-movie hybrid-mask comparison</title><style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#17303d}img{max-width:100%;height:auto}article{margin:35px 0}a{color:#086b88}.notice{padding:16px;background:#fff0cc}</style>',f'<h1>{html.escape(c["sample_name"])}: full-movie default versus strong-signal union</h1>',
        '<p>Original frames 0–2499 (50 s). The new method adds raw-C1 strong pixels to the saved default per-frame broad mask before the unchanged support≥10/50, post-min32 and holes&lt;64. Strong threshold = full-frame positive-pixel median + 6 × (MAD/0.6744897501960817); this robust intensity margin is not a camera/background noise calibration.</p>',
        '<p>All methods retain the original QC, ROI and per-localization 1-SE ellipse assignment. TARDIS is recomputed for the resulting groups; original localization fitting and Nellie broad segmentation are reused. Original inputs, baseline results and batch defaults remain unchanged.</p>',
        '<p class="notice">The publisher only reads saved scientific products. Curves, signed-area normalization and positive-peak normalization are precomputed scientific fields; no smoothing or implicit scientific recalculation occurs. Missing saved products stop publication. Shading is a within-method exploratory pointwise range, not a paired new-versus-old interval, simultaneous band or significance test. No p values are calculated.</p>',
        '<p>Blue solid: original default. Orange dashed: default plus raw strong-signal union. Full/zoom curve figures share the same y limits within each normalization and across On/Not-on. Peak curves have no saved bootstrap bands. Negative bins are retained.</p>']
    for name in order:
        body.append('<article><h2><a href="'+html.escape(name)+'">'+html.escape(titles[name])+'</a></h2>')
        if name in descriptions:body.append('<p>'+html.escape(descriptions[name])+'</p>')
        body.append('<a href="'+html.escape(name)+'"><img src="'+html.escape(name)+'" alt="'+html.escape(titles[name])+'"></a></article>')
    body.append('<p><a href="../data/comparison_curves.csv">Complete-precision comparison curves</a> · <a href="../data/comparison_curve_scalars.csv">TARDIS scalars</a> · <a href="../classification_summary.json">Spatial-group counts</a> · <a href="../data/mask_area_comparison.csv">Mask areas</a></p><p>Manufacturer-nominal calibration; C1/C2 identity registration independently unverified. Bootstrap ranges are conditional on the fixed masks and labels and do not propagate segmentation/registration uncertainty. Single-cell sensitivity test, not a biological replicate comparison.</p></html>')
    (out/"INDEX.html").write_text("\n".join(body),encoding="utf-8")
    checks=[]
    for name in files:
        with Image.open(out/name) as im:
            if im.format!="PNG" or min(im.size)<500:raise ValueError("Unexpected rendered image format/size")
            checks.append(dict(relative_path="figures/"+name,width=im.width,height=im.height,bytes=(out/name).stat().st_size))
    manifest=dict(sample_id=c["sample_id"],run_id=c["run_id"],scientific_steps_invoked=0,read_only_publisher=True,plotting_environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,matplotlib=matplotlib.__version__,Pillow=PIL.__version__),raw_C1_display_reader="Pillow uint16 TIFF original pages; stored byte order converted losslessly to native uint16; no resampling except display replication",raw_display_samples=raw_display_samples,bytecode_cache_bypass="Pure-Python plotting dependency sources loaded unchanged to avoid blocked local .pyc reads",figures=checks,index="figures/INDEX.html",source_paths=sorted(source_paths),shared_full_zoom_ylimits=limits,normalization_values="Read precomputed saved scientific columns; no normalization performed",bootstrap_note=band_note+" Not a paired new-old comparison interval.",mean_C1_recomputed=False,elapsed_seconds=time.perf_counter()-began)
    (root/"render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(render_complete=True,scientific_steps_invoked=0,figures=len(files),index=str(out/"INDEX.html"),elapsed_seconds=manifest["elapsed_seconds"]),indent=2),flush=True)
if __name__=="__main__":main()
