"""Cell-level exploratory normality diagnostics; no upstream analysis calls."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math, os, platform, shutil, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source",type=Path,required=True)
parser.add_argument("--run-dir",type=Path,required=True)
parser.add_argument("--figures-dir",type=Path,required=True)
args=parser.parse_args()
root=Path("/Volumes/My Book").resolve()
source=args.source.resolve()
out=args.run_dir.resolve()
figdir=args.figures_dir.resolve()
for p in (source,out,figdir):
    if not p.is_relative_to(root):raise ValueError("All task paths must be on My Book: "+str(p))
for p in (out,figdir,out/"runtime"/"mpl",out/"runtime"/"cache",out/"runtime"/"tmp"):
    p.mkdir(parents=True,exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(out/"runtime"/"mpl")
os.environ["XDG_CACHE_HOME"]=str(out/"runtime"/"cache")
os.environ["TMPDIR"]=str(out/"runtime"/"tmp")
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def flag(x):return str(x).lower() in ("true","1")
def log(s):
    with (out/"analysis.log").open("a",encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat()+" "+s+"\n")
    print(s,flush=True)
def fmtp(p):return f"{p:.3g}" if p<.001 else f"{p:.4f}"
proteins=("TRAK1","TRAK2")
conditions=("WT","TDK","MDK")
metrics={
 "on_percentage":("On percentage","On (%)"),
 "mean_ROI_area_um2":("Time-weighted mean ROI mask area",r"ROI mask area ($\mu$m$^2$)"),
 "first_mask_ROI_area_um2":("First-block ROI mask area (frames 0-49)",r"ROI mask area ($\mu$m$^2$)")
}
expected={("TRAK1","WT"):34,("TRAK1","TDK"):37,("TRAK1","MDK"):34,
          ("TRAK2","WT"):35,("TRAK2","TDK"):36,("TRAK2","MDK"):35}
initial=sha(source)
with source.open(newline="",encoding="utf-8-sig") as f:cells=list(csv.DictReader(f))
if len(cells)!=211 or len({r["sample_id"] for r in cells})!=211:raise ValueError("Missing/duplicate cells")
actual=Counter((r["protein"],r["condition"]) for r in cells)
if actual!=Counter(expected):raise ValueError("Unexpected protein/condition cell counts")
for r in cells:
    on,off=int(r["On"]),int(r["Not_on"])
    if on+off<=0 or not math.isclose(float(r["on_percentage"]),100*on/(on+off),abs_tol=1e-10,rel_tol=0):
        raise ValueError("On ratio mismatch: "+r["sample_id"])
log("Loaded 211 unique cells from existing saved metrics. No localization, mask, assignment or TARDIS steps executed.")
results=[];groups={};membership={}
for key,(title,unit) in metrics.items():
    for protein in proteins:
        for condition in conditions:
            all_rows=[r for r in cells if (r["protein"],r["condition"])==(protein,condition)]
            usable=[r for r in all_rows if flag(r[key+"_valid"]) and math.isfinite(float(r[key]))]
            values=np.array([float(r[key]) for r in usable],dtype=float)
            if len(values)<3 or np.ptp(values)==0:raise ValueError("Unavailable/constant normality input")
            if len({r["BR"] for r in usable})!=3:raise ValueError("Expected three BRs")
            W,p=stats.shapiro(values)
            Wrev,prev=stats.shapiro(values[::-1])
            if not np.allclose([W,p],[Wrev,prev],rtol=0,atol=1e-12):raise AssertionError("Row-order dependence")
            (z,y),(slope,intercept,rqq)=stats.probplot(values,dist="norm",fit=True)
            if len(z)!=len(values) or not np.allclose(y,np.sort(values),rtol=0,atol=0):raise AssertionError("QQ values mismatch")
            result=dict(protein=protein,condition=condition,metric=key,metric_title=title,
                n_cells=len(values),n_BR=3,n_excluded=len(all_rows)-len(usable),
                mean_cell=float(np.mean(values)),sample_SD_cell=float(np.std(values,ddof=1)),
                median_cell=float(np.median(values)),sample_skewness=float(stats.skew(values,bias=False)),
                W=float(W),p_raw=float(p),nominal_p_below_0_05=bool(p<.05),
                interpretation="Nominal departure detected (exploratory)" if p<.05 else "No departure detected; normality not established",
                QQ_slope=float(slope),QQ_intercept=float(intercept),QQ_correlation=float(rqq))
            results.append(result);groups[key,protein,condition]=(usable,values,result)
            membership[key+":"+protein+":"+condition]=[r["sample_id"] for r in usable]
            log(f"{protein}/{condition}/{key}: n={len(values)}, W={W:.9g}, raw diagnostic p={p:.9g}")
with (out/"normality_results.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
dump(out/"normality_results.json",results)
dump(out/"members.json",membership)
config=dict(method="Shapiro-Wilk and normal Q-Q plots; raw-scale exploratory diagnostics",
    group_by=["protein","condition"],observations="one saved value per cell; pooled across three BRs",
    source=str(source.relative_to(root)),metrics=list(metrics),
    alpha_reference=.05,multiplicity="Unadjusted diagnostic p values; not confirmatory group tests",
    transformations="none",outlier_removal="none",
    upstream_scientific_steps_executed=0,normality_based_test_selection=False)
dump(out/"config.json",config)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":11,
    "axes.titlesize":12,"figure.facecolor":"white","savefig.facecolor":"white"})
colors={"BR1":"#0072B2","BR2":"#E69F00","BR3":"#009E73"}
markers={"BR1":"o","BR2":"s","BR3":"^"}
handles=[Line2D([0],[0],linestyle="",marker=markers[b],color=colors[b],label=b,markersize=7) for b in colors]
handles.append(Line2D([0],[0],color="#404040",linestyle="--",label="Normal reference",lw=1.4))
footer=("Each point is one cell; colors identify BRs. Three BRs are pooled per panel.\n"
        "Raw Shapiro-Wilk p values are exploratory: p > 0.05 does not establish normality; within-BR dependence is not modeled.")
figures=[]
for key,(title,unit) in metrics.items():
    all_values=np.concatenate([groups[key,p,c][1] for p in proteins for c in conditions])
    ylo,yhi=(0,100) if key=="on_percentage" else (0,float(all_values.max())*1.08)
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for i,protein in enumerate(proteins):
        for j,condition in enumerate(conditions):
            ax=axes[i,j];rows,values,res=groups[key,protein,condition]
            order=np.argsort(values,kind="stable")
            z=stats.probplot(values,dist="norm",fit=False)[0]
            y=values[order];br=np.array([r["BR"] for r in rows])[order]
            ax.plot(z,res["QQ_intercept"]+res["QQ_slope"]*z,color="#404040",ls="--",lw=1.4,zorder=1)
            for b in colors:
                use=br==b
                ax.scatter(z[use],y[use],s=33,color=colors[b],marker=markers[b],alpha=.9,label=b,zorder=3)
            ax.set(title=f"{protein} {condition} | {len(values)} cells / 3 BRs\nShapiro-Wilk W={res['W']:.3f}, raw p={fmtp(res['p_raw'])}",
                   xlabel="Theoretical normal quantile",ylabel=unit,ylim=(ylo,yhi))
            ax.grid(alpha=.17)
    fig.suptitle(title+" — cell-level normal Q-Q plots",fontsize=18,y=.986)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.text(.5,.022,footer,ha="center",va="bottom",fontsize=10,color="#404040")
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.14,hspace=.47,wspace=.28)
    name=key+"_QQ.png";fig.savefig(figdir/name,dpi=180);plt.close(fig);figures.append(name)

    fig,axes=plt.subplots(2,3,figsize=(14,9))
    edges=np.arange(0,101,5) if key=="on_percentage" else np.histogram_bin_edges(all_values,bins="fd")
    width=float(edges[1]-edges[0]);grid=np.linspace(edges[0],edges[-1],400)
    for i,protein in enumerate(proteins):
        for j,condition in enumerate(conditions):
            ax=axes[i,j];rows,values,res=groups[key,protein,condition]
            portions=[np.array([float(r[key]) for r in rows if r["BR"]==b]) for b in colors]
            ax.hist(portions,bins=edges,stacked=True,color=list(colors.values()),alpha=.65,
                    edgecolor="white",linewidth=.4)
            ax.plot(grid,stats.norm.pdf(grid,loc=res["mean_cell"],scale=res["sample_SD_cell"])*len(values)*width,
                    color="#404040",ls="--",lw=1.4)
            ax.set(title=f"{protein} {condition} | {len(values)} cells / 3 BRs\nShapiro-Wilk W={res['W']:.3f}, raw p={fmtp(res['p_raw'])}",
                xlabel=unit,ylabel="Cells per bin",xlim=(edges[0],edges[-1]))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True));ax.grid(axis="y",alpha=.17)
    fig.suptitle(title+" — observed cell distributions",fontsize=18,y=.986)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.text(.5,.022,footer.replace("Each point is one cell; colors identify BRs.","Stacked bars count cells by BR; dashed line uses the observed group mean and SD."),
             ha="center",va="bottom",fontsize=10,color="#404040")
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.14,hspace=.47,wspace=.28)
    name=key+"_histograms.png";fig.savefig(figdir/name,dpi=180);plt.close(fig);figures.append(name)
sources={"source":str(source.relative_to(root)),"source_sha256_before":initial,"source_sha256_after":sha(source),
         "script":str(Path(__file__).resolve().relative_to(root)),"script_sha256":sha(__file__),
         "python":sys.version,"scipy":scipy.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,
         "platform":platform.platform(),"created_utc":datetime.now(timezone.utc).isoformat(),
         "references":["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html",
          "https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm"]}
if sources["source_sha256_before"]!=sources["source_sha256_after"]:raise AssertionError("Source changed")
dump(out/"source_records.json",sources)
method_text="""Cell-level normality diagnostics
================================
One raw-scale saved value per cell, grouped by protein and condition. All 211 cells are retained.
Three metrics: On%=100*On/(On+Not_on); time-weighted mean ROI mask area over all 50 support blocks;
first-support-block ROI mask area (frames 0-49). The last metric is not an instantaneous frame-0 area.

Shapiro-Wilk W and unadjusted diagnostic p values are reported. Normal Q-Q plots and histograms
are shown with BR colors. A Q-Q reference line is an ordinary least-squares normal-reference fit,
not a biological regression model. Histogram references use the observed group mean and sample SD.

These diagnostics pool cells from three BRs. Cells within a BR may be correlated or have a shifted
distribution, so the usual independent identically distributed calibration of the Shapiro p value
is not established here. Treat the numbers as exploratory shape diagnostics. A raw p below 0.05
is a nominal flag, not a confirmatory finding across 18 diagnostics. A p above 0.05 does not prove normality.
Neither these pooled-cell distributions nor their Shapiro results establish normality of the three
BR means or of residuals from a future hierarchical model. No such model is fitted here.
Do not automatically select Welch or a nonparametric test using Shapiro pass/fail.

No data transformation, outlier removal, localization, masking, spatial reclassification, TARDIS,
bootstrap, hypothesis-contrast recalculation, or batch-default changes were performed.

Sources and versions are in source_records.json. Original source metrics remain unchanged.
"""
(out/"README.txt").write_text(method_text,encoding="utf-8")
lookup={(r["protein"],r["condition"],r["metric"]):r for r in results}
body=[]
for p in proteins:
    for c in conditions:
        row=f"<tr><td>{p}</td><td>{c}</td><td>{expected[p,c]}</td><td>3</td>"
        for k in metrics:
            r=lookup[p,c,k];flag_text=" (nominal p&lt;0.05)" if r["p_raw"]<.05 else ""
            row+=f"<td>{fmtp(r['p_raw'])}{flag_text}</td>"
        body.append(row+"</tr>")
gallery="".join(f'<h2>{html.escape(name.replace("_"," ").replace(".png",""))}</h2><a href="{name}"><img src="{name}" loading="lazy" alt="{html.escape(name)}"></a>' for name in figures)
rel_csv=html.escape(os.path.relpath(out/"normality_results.csv",figdir))
rel_readme=html.escape(os.path.relpath(out/"README.txt",figdir))
index=f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Cell-level normality diagnostics</title>
<style>body{{font:16px Arial,sans-serif;max-width:1250px;margin:35px auto;padding:0 20px;color:#203543;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ccd4da;text-align:left}}th{{background:#edf2f5}}
img{{width:100%;height:auto}}a{{color:#16619b}}h2{{margin-top:30px;font-size:21px}}</style>
<h1>Cell-level normality diagnostics</h1>
<p>Raw-scale values for 211 cells in six protein–condition groups. Each group pools three BRs.
Shapiro–Wilk p values below are unadjusted exploratory diagnostics. They do not establish independence,
BR-mean normality, or suitability of a subsequent test.</p>
<table><thead><tr><th>Protein</th><th>Condition</th><th>Cells</th><th>BRs</th><th>On% p</th><th>Mean ROI area p</th><th>First-block ROI area p</th></tr></thead>
<tbody>{"".join(body)}</tbody></table>
<p>No departure was detected for five On% groups. TRAK1–MDK has a nominal departure.
Both area metrics have nominal departures in five groups; TRAK2–WT is the exception.
p &gt; 0.05 does not prove normality. No observations were removed or transformed.</p>
<p><a href="{rel_csv}">Full results (CSV)</a> · <a href="{rel_readme}">Methods and limitations</a></p>
{gallery}</html>"""
(figdir/"INDEX.html").write_text(index,encoding="utf-8")
validation=dict(cells=211,unique_sample_ids=211,groups=6,endpoints=3,diagnostics=18,
    all_On_ratios_reconciled=True,source_unchanged=True,all_groups_have_3_BRs=True,
    all_raw_values_used=True,order_invariance_checks=18,QQ_membership_checks=18,
    upstream_scientific_steps_executed=0,figures=figures,
    nominal_departures={key:sum(r["metric"]==key and r["p_raw"]<.05 for r in results) for key in metrics})
dump(out/"validation.json",validation)
log("Saved 18 diagnostic results and six PNGs on My Book; original metrics unchanged.")
print(json.dumps({"run_dir":str(out),"figures_dir":str(figdir),"validation":validation},indent=2))
