#!/usr/bin/env python3
"""Publish saved metric products; never aggregate or run statistical tests."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math,os,uuid
from pathlib import Path
from urllib.parse import quote
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
_runtime=Path(__file__).resolve().parent/'runtime'
for _key,_relative in {'MPLCONFIGDIR':'matplotlib','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TEMP':'tmp','TMP':'tmp'}.items():
    _folder=_runtime/_relative;_folder.mkdir(parents=True,exist_ok=True);os.environ[_key]=str(_folder)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
METRICS={
'on_percentage':('On_percentage_protein_comparison.png','On mitochondria percentage across proteins and conditions','On mitochondria (%)','Per cell: 100 × On / (On + Not-on); unknown and QC-excluded points are outside the denominator.'),
'mean_ROI_area_um2':('ROI_mean_area_protein_comparison.png','Mean mitochondrial mask area across proteins and conditions','Mean ROI mask area (µm²)','Per cell: time-weighted mean over all 50 support-mask blocks, restricted to the original cell ROI.'),
'first_mask_ROI_area_um2':('ROI_first_block_area_protein_comparison.png','First-block mitochondrial mask area across proteins and conditions','First-block ROI mask area (µm²)','Per cell: support mask for original frames 0–49 (0–1 s), inside the ROI; not an instantaneous frame-0 mask.')}
MARKERS={'BR1':'o','BR2':'s','BR3':'^'}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(v):
    try:r=float(v)
    except (TypeError,ValueError):return None
    return r if math.isfinite(r) else None
def valid(row,metric):return str(row.get(metric+'_valid','true')).lower() not in {'false','0','no'} and number(row.get(metric)) is not None
def single(rows,**filters):
    found=[r for r in rows if all(str(r.get(k))==str(v) for k,v in filters.items())]
    if len(found)!=1:raise ValueError('Expected one saved record for %s; found %d'%(filters,len(found)))
    return found[0]
def p_text(v):
    v=number(v)
    return 'unavailable' if v is None else ('< 0.001' if v<.001 else '= %.3f'%v)
def jitter(uid,span=.052):
    token=int(hashlib.sha256(str(uid).encode()).hexdigest()[:12],16)
    return ((token/float(16**12-1))-.5)*2*span

def comparison(rows,endpoint,pa,ca,pb,cb):
    wanted={(pa,ca,pb,cb),(pb,cb,pa,ca)}
    found=[r for r in rows if r['endpoint']==endpoint and (r['protein_a'],r['condition_a'],r['protein_b'],r['condition_b']) in wanted]
    if len(found)!=1:raise ValueError('Missing or ambiguous saved comparison: %s'%((endpoint,pa,ca,pb,cb),))
    return found[0]
def save(fig,path):
    tmp=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.rendering.png')
    fig.savefig(tmp,dpi=180,facecolor='white');plt.close(fig);os.replace(tmp,path)

def plot_metric(metric,config,data,output):
    name,title,ylabel,definition=METRICS[metric]
    show_effects=bool(config.get('show_effect_sizes'))
    if show_effects:name=name.replace('.png','_effect_sizes.png')
    proteins,conditions=config['proteins'],config['conditions']
    colors=config.get('colors',{'TRAK1':'#2774B8','TRAK2':'#E7892A'})
    fig=plt.figure(figsize=(16,10.7))
    fig.text(.065,.958,title,fontsize=22,weight='bold',color='#183143')
    fig.text(.065,.919,definition,fontsize=12.7,color='#36454F')
    fig.text(.065,.889,'Cells are descriptive observations; independent BR means are statistical units (n = 3 per protein × condition).',fontsize=12.6,color='#36454F')
    ax=fig.add_axes([.073,.358,.89,.495])
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    ax.grid(axis='y',color='#DCE3E8',alpha=.8,linewidth=.7)
    ax.set_ylabel(ylabel,fontsize=16,labelpad=12)
    ax.set_xticks(range(len(conditions)),conditions,fontsize=16)
    ax.set_xlim(-.55,len(conditions)-.45);ax.tick_params(axis='y',labelsize=13)
    tops={};br_offsets={'BR1':-.034,'BR2':0.,'BR3':.034}
    for ci,condition in enumerate(conditions):
        top_values=[]
        for pi,protein in enumerate(proteins):
            center=ci+(-.195 if pi==0 else .195)
            cells=[r for r in data['cells'] if r['protein']==protein and r['condition']==condition and valid(r,metric)]
            ax.scatter([center+jitter(r['sample_id']) for r in cells],[number(r[metric]) for r in cells],s=21,color=colors[protein],alpha=.25,edgecolors='none',zorder=2)
            top_values.extend(number(r[metric]) for r in cells)
            for br,marker in MARKERS.items():
                row=single(data['br'],protein=protein,condition=condition,BR=br,endpoint=metric)
                value=number(row['mean'])
                if value is None:continue
                ax.scatter([center+br_offsets[br]],[value],s=108,marker=marker,facecolors=colors[protein],edgecolors='white',linewidths=1.25,zorder=4)
                top_values.append(value)
            row=single(data['condition'],protein=protein,condition=condition,endpoint=metric)
            value,sd=number(row['mean']),number(row['sample_sd'])
            if value is not None:
                ax.errorbar([center+.094],[value],yerr=([sd] if sd is not None else None),fmt='D',ms=7.4,color=colors[protein],mec='#1F2933',mew=1.05,mfc='white',capsize=5.2,elinewidth=2.,capthick=1.8,zorder=5)
                top_values.append(value+(sd or 0.))
            ax.text(center,-.087,'%s\n%d cells / %s BRs'%(protein,len(cells),row['n_BR']),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10.8,color=colors[protein])
        tops[condition]=max(top_values) if top_values else 1.
    maximum=max(tops.values());axis_top=max(100.,maximum*1.25) if metric=='on_percentage' else maximum*1.26
    ax.set_ylim(0,axis_top)
    for ci,condition in enumerate(conditions):
        row=comparison(data['comparisons'],metric,proteins[0],condition,proteins[1],condition)
        y=tops[condition]+axis_top*.04;h=axis_top*.018;left,right=ci-.235,ci+.295
        ax.plot([left,left,right,right],[y,y+h,y+h,y],color='#33434D',lw=1.1,clip_on=False)
        if show_effects:
            effect=single(data['effects'],endpoint=metric,condition=condition)
            unit='pp' if metric=='on_percentage' else 'µm²'
            label='Δ = %+.2f %s (%+.1f%% vs TRAK2)\nuncorrected p %s'%(number(effect['difference_a_minus_b']),unit,number(effect['relative_percent_vs_TRAK2']),p_text(row['p_raw']))
        else:label='Uncorrected p '+p_text(row['p_raw'])
        ax.text((left+right)/2,y+h+axis_top*.012,label,ha='center',va='bottom',fontsize=11.3 if show_effects else 12.5,linespacing=1.4)
    handles=[Line2D([],[],linestyle='none',marker='o',markersize=7,markerfacecolor=colors[p],markeredgecolor='none',label=p) for p in proteins]
    handles += [Line2D([],[],linestyle='none',marker=m,markersize=8,markerfacecolor='#617480',markeredgecolor='white',label=br) for br,m in MARKERS.items()]
    handles += [Line2D([],[],color='#617480',marker='D',markerfacecolor='white',markersize=7,label='Equal-BR mean ± SD')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.515,.245),ncol=6,frameon=False,fontsize=12,handletextpad=.5,columnspacing=1.5)
    fig.text(.073,.233,'Observed differences — equal-BR means; relative change uses TRAK2 as baseline' if show_effects else 'Within-protein condition contrasts — uncorrected p',fontsize=13.5,color='#183143',weight='bold')
    table_ax=fig.add_axes([.073,.118,.89,.097]);table_ax.axis('off')
    if show_effects:
        unit='pp' if metric=='on_percentage' else 'µm²'
        meanunit='%' if metric=='on_percentage' else 'µm²'
        rows=[]
        for condition in conditions:
            effect=single(data['effects'],endpoint=metric,condition=condition)
            rows.append([condition,'%.2f'%number(effect['mean_a']),'%.2f'%number(effect['mean_b']),'%+.2f'%number(effect['difference_a_minus_b']),'%+.1f%%'%number(effect['relative_percent_vs_TRAK2']),'%.3f'%number(effect['p_raw'])])
        headers=['Condition','TRAK1 ('+meanunit+')','TRAK2 ('+meanunit+')','Δ ('+unit+')','Relative vs TRAK2','Uncorrected p']
        widths=[.13,.16,.16,.16,.23,.16]
    else:
        pairs=[('WT','TDK'),('WT','MDK'),('TDK','MDK')]
        rows=[[p]+[p_text(comparison(data['comparisons'],metric,p,a,p,b)['p_raw']) for a,b in pairs] for p in proteins]
        headers=['Protein']+['%s vs %s'%pair for pair in pairs];widths=[.22,.26,.26,.26]
    table=table_ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='center',colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(11.2 if show_effects else 12.8);table.scale(1,1.35 if show_effects else 1.55)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.5);cell.set_edgecolor('#D5DEE4')
        cell.set_facecolor('#ECF1F4' if r==0 else ('#FAFCFD' if r%2 else 'white'))
        if r==0:cell.get_text().set_weight('bold');cell.get_text().set_color('#183143')
        elif c==0:
            if not show_effects:cell.get_text().set_color(colors[proteins[r-1]])
            cell.get_text().set_weight('bold')
    fig.text(.073,.075,'Exact two-sided independent-BR permutation tests. Displayed p values are uncorrected.',fontsize=11.5,color='#3F4D56')
    fig.text(.073,.049,'With 3 vs 3 BRs, the smallest attainable raw p is 0.10. Limited BR numbers constrain inference; non-significance does not establish equivalence.',fontsize=11.5,color='#3F4D56')
    fig.text(.073,.024,'Small points: cells. Large colored circles/squares/triangles: BR1/BR2/BR3. No BR pairing assumed. Raw p and effects are saved in the data tables.',fontsize=11.,color='#56636C')
    save(fig,output/name);return name

def plot_averaging(config,data,output):
    rows=[r for r in data['audit'] if r['BR']=='ALL']
    if not rows:return None
    colors=config.get('colors',{'TRAK1':'#2774B8','TRAK2':'#E7892A'})
    fig,axes=plt.subplots(2,3,figsize=(16,9.3),sharey=True)
    fig.subplots_adjust(left=.065,right=.97,bottom=.19,top=.80,wspace=.15,hspace=.53)
    fig.text(.065,.951,'Three averaging rules give different On percentages',fontsize=22,weight='bold',color='#183143')
    fig.text(.065,.906,'The primary condition estimate averages cell percentages within each BR, then averages the three BR means equally.',fontsize=13,color='#36454F')
    fig.text(.065,.872,'Dividing mean On by mean (On + Not-on) instead gives a localization-count-weighted, pooled percentage.',fontsize=13,color='#36454F')
    keys=['pooled_On_counts_ratio','mean_of_cell_ratios','equal_BR_mean_of_cell_ratios']
    labels=['Pooled\nlocalization counts','Equal\ncell mean','Equal BR mean\n(primary)']
    for pi,protein in enumerate(config['proteins']):
        for ci,condition in enumerate(config['conditions']):
            ax=axes[pi,ci];row=single(rows,protein=protein,condition=condition);ys=[number(row[k]) for k in keys]
            ax.scatter(range(3),ys,s=100,color=colors[protein],edgecolors='white',linewidth=1,zorder=3)
            ax.plot(range(3),ys,color=colors[protein],alpha=.45,linewidth=1,zorder=2)
            for x,y in enumerate(ys):ax.annotate('%.2f%%'%y,(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=12.5,weight='bold')
            ax.set_title('%s · %s (%s cells, %s BRs)'%(protein,condition,row['n_cells'],row['n_BR']),fontsize=14.2,color=colors[protein],pad=10)
            ax.set_xticks(range(3),labels,fontsize=11);ax.set_xlim(-.5,2.5);ax.set_ylim(0,100)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E0E5E9',linewidth=.6);ax.set_axisbelow(True)
            if ci==0:ax.set_ylabel('On mitochondria (%)',fontsize=13)
    fig.text(.065,.095,'These are different estimands, not a numerical inconsistency. The original per-cell On / (On + Not-on) values are unchanged.',fontsize=12.7,color='#36454F')
    fig.text(.065,.060,'This diagnostic compares averaging rules only; it does not provide additional independent replicates or additional hypothesis tests.',fontsize=12.7,color='#36454F')
    name='On_percentage_averaging_audit.png';save(fig,output/name);return name

def rel_link(target,base):return quote(os.path.relpath(target,base).replace(os.sep,'/'),safe='/.')
def index(config,files,output,run):
    descriptions={METRICS[m][0]:METRICS[m][3] for m in METRICS}
    descriptions['On_percentage_averaging_audit.png']='The distinction between pooled-count, equal-cell and equal-BR percentages.'
    titles={METRICS[m][0]:METRICS[m][1] for m in METRICS}
    titles['On_percentage_averaging_audit.png']='On percentage: averaging audit'
    cards='\n'.join('<section><h2>'+html.escape(titles[k])+'</h2><p>'+html.escape(descriptions[k])+'</p><a href="'+quote(k)+'"><img src="'+quote(k)+'" alt="'+html.escape(descriptions[k])+'"></a></section>' for k in files)
    links=[(output/'Protein_Comparison.xlsx','Reader-friendly Excel workbook'),(run/'data/cells.csv','Per-cell values'),(run/'data/br_metrics.csv','BR means, SD and SEM'),(run/'data/condition_metrics.csv','Condition means, SD and SEM'),(run/'data/comparisons.csv','Full saved statistical results, effects and sample sizes (unchanged)'),(run/'data/aggregation_audit.csv','On percentage averaging audit'),(run/'config.json','Analysis configuration')]
    linked=''.join('<li><a href="'+rel_link(path,output)+'">'+html.escape(label)+'</a></li>' for path,label in links)
    head='''<!doctype html><html lang="en"><meta charset="utf-8"><title>TRAK1 / TRAK2 comparison</title><style>body{font:17px/1.5 -apple-system,BlinkMacSystemFont,Arial,sans-serif;max-width:1400px;margin:35px auto;padding:0 22px;color:#20323f;background:#f7f9fa}h1{font-size:32px}h2{font-size:22px}section{background:white;border:1px solid #dce3e8;border-radius:10px;padding:22px;margin:25px 0}img{display:block;width:100%;height:auto}a{color:#176795}.note{background:#eef3f6;padding:16px 20px;border-left:4px solid #2774b8}</style><h1>TRAK1 / TRAK2: On percentage and mitochondrial mask area</h1><p class="note">Independent biological replicates (BRs), with pairing unconfirmed. Cells receive equal weight within each BR; BRs receive equal weight within each condition. All 211 existing cells are reused. No localization, segmentation or TARDIS was rerun.</p><p>Small points show individual cells; large markers distinguish BR1, BR2 and BR3. Diamonds and error bars show the equal-BR condition mean ± 1 sample SD. Statistical tests use BR means, not cell points: exact two-sided independent permutation tests. Displayed p values are uncorrected. With 3 versus 3 BRs the smallest attainable raw p is 0.10. An absence of significance does not demonstrate equivalence.</p><h2>Data and methods</h2><ul>'''
    (output/'INDEX.html').write_text(head+linked+'</ul>'+cards+'</html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'));run=Path(config['run_dir']);output=Path(config['figures_dir']);output.mkdir(parents=True,exist_ok=True)
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.unicode_minus':True})
    paths={'cells':run/'data/cells.csv','br':run/'data/br_metrics.csv','condition':run/'data/condition_metrics.csv','comparisons':run/'data/comparisons.csv','audit':run/'data/aggregation_audit.csv'}
    data={k:read_csv(p) for k,p in paths.items()};files=[plot_metric(m,config,data,output) for m in config['endpoints']]
    name=plot_averaging(config,data,output)
    if name:files.append(name)
    index(config,files,output,run)
    report={'publisher':'render_metrics.py','scientific_step_calls':0,'inputs':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},'outputs':[str(output/name) for name in files]+[str(output/'INDEX.html')],'source_values':'Saved scientific products only; no re-aggregation or hypothesis tests in the publisher.'}
    (run/'render_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures_dir':str(output),'PNG_count':len(files),'scientific_step_calls':0}))
if __name__=='__main__':main()
