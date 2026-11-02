#!/usr/bin/env python3
"""Publish prepared descriptive effects; no statistics or scientific aggregation."""
from pathlib import Path
import argparse,hashlib,json,os
from urllib.parse import quote
import render_metrics as render

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);a=parser.parse_args()
    config=json.loads(a.config.read_text());config['show_effect_sizes']=True
    run=Path(config['run_dir']);out=Path(config['figures_dir'])
    sources={'cells':run/'data/cells.csv','br':run/'data/br_metrics.csv','condition':run/'data/condition_metrics.csv','comparisons':run/'data/comparisons.csv','effects':run/'data/effect_sizes_proteins.csv'}
    data={key:render.read_csv(path)for key,path in sources.items()}
    hashes={key:hashlib.sha256(path.read_bytes()).hexdigest()for key,path in sources.items()}
    render.matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.unicode_minus':True})
    files=[render.plot_metric(metric,config,data,out)for metric in config['endpoints']]
    html=out/'INDEX.html';text=html.read_text()
    start='<!-- DESCRIPTIVE_EFFECTS_START -->';end='<!-- DESCRIPTIVE_EFFECTS_END -->'
    if start in text:text=text.split(start)[0]+text.split(end,1)[1]
    block=start+'<section><h2>Quantified protein differences</h2><p>Δ = TRAK1 − TRAK2. On% differences use percentage points (pp); area differences use µm². Relative change = Δ / TRAK2 mean × 100%. Means retain equal-cell weighting within BR and equal-BR weighting within condition. These are observed mean differences, not new hypothesis tests.</p><p><a href="Protein_Differences.csv">Reader table of quantified differences</a></p>'
    for name in files:block+='<a href="'+quote(name)+'"><img src="'+quote(name)+'" alt="Observed protein differences"></a>'
    block+='</section>'+end
    if '<h2>Data and methods</h2>'in text:text=text.replace('<h2>Data and methods</h2>',block+'<h2>Data and methods</h2>')
    else:text+=block
    html.write_text(text)
    assert hashes=={key:hashlib.sha256(path.read_bytes()).hexdigest()for key,path in sources.items()}
    report=dict(PNGs=files,source_sha256=hashes,scientific_step_calls=0,statistical_tests_recomputed=0,definition='Read prepared observed effects only')
    (run/'validation_effects_render.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
