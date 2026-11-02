#!/usr/bin/env python3
"""Explicit scientific statistics stage; reads saved cell metrics only.
No localization, segmentation, assignment or single-cell TARDIS is called.
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,os,statistics,tempfile
from datetime import datetime,timezone
from pathlib import Path

METHOD_VERSION='independent_BR_exact_permutation_v1'
COUNT_FIELDS=('On','Not_on','unknown','QC_excluded','selected','ROI_candidates')
ENDPOINTS=('on_percentage','mean_ROI_area_um2','first_mask_ROI_area_um2')

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:f.write(content)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
def write_json(path,obj):atomic(path,json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows,fields):
    import io
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows);atomic(path,stream.getvalue())
def descriptives(values):
    n=len(values)
    if not n:return {'mean':None,'sample_sd':None,'sem':None}
    sd=statistics.stdev(values) if n>1 else None
    return dict(mean=statistics.mean(values),sample_sd=sd,sem=sd/math.sqrt(n) if sd is not None else None)
def exact_permutation(a,b):
    """All independent label partitions; two-sided = 2 * min(exact tails).
    No Monte Carlo correction; tolerance follows floating relative scale.
    """
    if not a or not b:raise ValueError('Both groups require BR values')
    observed=statistics.mean(a)-statistics.mean(b);v=tuple(a)+tuple(b);n=len(a)
    null=[]
    for inds in itertools.combinations(range(len(v)),n):
        chosen=set(inds)
        left=[v[i] for i in range(len(v)) if i in chosen]
        right=[v[i] for i in range(len(v)) if i not in chosen]
        null.append(statistics.mean(left)-statistics.mean(right))
    tol=100*2.220446049250313e-16*max(abs(observed),1.0)
    lower=sum(x<=observed+tol for x in null)/len(null)
    upper=sum(x>=observed-tol for x in null)/len(null)
    return observed,min(1.0,2*min(lower,upper)),len(null)
def holm(pvalues):
    n=len(pvalues);order=sorted(range(n),key=lambda i:pvalues[i]);out=[None]*n;prev=0.0
    for k,i in enumerate(order):
        prev=max(prev,min(1.0,(n-k)*pvalues[i]));out[i]=prev
    return out

def read_sources(config):
    cells=[];records=[];seen=set();checks=[]
    for protein in config['proteins']:
        p=Path(config['sources'][protein]);digest=sha(p);rows=list(csv.DictReader(p.open(newline='',encoding='utf-8-sig')))
        records.append(dict(protein=protein,path=str(p),bytes=p.stat().st_size,sha256_before=digest,n_cells=len(rows)))
        for source in rows:
            sid=source['sample_id']
            if sid in seen:raise ValueError(f'Duplicate sample_id {sid}')
            seen.add(sid)
            if source['molecule']!=protein:raise ValueError(f'Protein mismatch: {sid}')
            if source['genotype'] not in config['conditions']:raise ValueError(f'Condition mismatch: {sid}')
            r={k:source[k] for k in ('sample_id','BR','cell','acquisition_date')}
            r.update(protein=protein,condition=source['genotype'])
            r.update({k:int(source[k]) for k in COUNT_FIELDS})
            if r['On']+r['Not_on']+r['unknown']!=r['selected']:raise ValueError(f'Selected counts do not conserve: {sid}')
            if r['selected']+r['QC_excluded']!=r['ROI_candidates']:raise ValueError(f'ROI counts do not conserve: {sid}')
            denominator=r['On']+r['Not_on'];r['on_denominator']=denominator
            computed=100*r['On']/denominator if denominator else None
            saved=float(source['on_percentage']) if source['on_percentage'] else None
            if computed is None or saved is None or not math.isclose(computed,saved,rel_tol=1e-12,abs_tol=1e-10):raise ValueError(f'On percentage mismatch: {sid}')
            r['on_percentage']=computed
            for name in ENDPOINTS[1:]:r[name]=float(source[name]) if source[name] else None
            valid=json.loads(source['valid_metrics']);reasons=json.loads(source['reasons'])
            for name in ENDPOINTS:
                value=r[name];ok=bool(valid.get(name,False)) and value is not None and math.isfinite(value)
                if name!='on_percentage':ok=ok and value>=0
                r[name+'_valid']=ok
                r[name+'_exclusion_reason']='' if ok else '; '.join(reasons.get(name,[])) or 'Saved metric unavailable or invalid'
            r['first_mask_original_frame_start']=int(source['first_mask_original_frame_start'])
            r['first_mask_original_frame_stop_exclusive']=int(source['first_mask_original_frame_stop_exclusive'])
            if (r['first_mask_original_frame_start'],r['first_mask_original_frame_stop_exclusive'])!=(0,50):raise ValueError(f'First mask mapping differs: {sid}')
            r['registration_status']=source['registration_status'];r['calibration_status']=source['calibration_status']
            cells.append(r)
        records[-1]['sha256_after']=sha(p)
        if records[-1]['sha256_after']!=digest:raise ValueError('Source changed while reading '+str(p))
    return cells,records

def aggregate(config,cells):
    brs=[];conditions=[];audit=[]
    for protein in config['proteins']:
        for condition in config['conditions']:
            subset=[r for r in cells if r['protein']==protein and r['condition']==condition]
            replicas=sorted({r['BR'] for r in subset})
            for replica in replicas:
                members=[r for r in subset if r['BR']==replica]
                for endpoint in config['endpoints']:
                    usable=[r for r in members if r[endpoint+'_valid']]
                    brs.append(dict(protein=protein,condition=condition,BR=replica,endpoint=endpoint,n_cells=len(usable),n_cells_available=len(members),**descriptives([r[endpoint] for r in usable])))
                audit.append(audit_row(protein,condition,replica,members,None))
            for endpoint in config['endpoints']:
                nodes=[r for r in brs if r['protein']==protein and r['condition']==condition and r['endpoint']==endpoint and r['mean'] is not None]
                usable=[r for r in subset if r[endpoint+'_valid']]
                conditions.append(dict(protein=protein,condition=condition,endpoint=endpoint,n_BR=len(nodes),n_cells=len(usable),n_cells_available=len(subset),**descriptives([r['mean'] for r in nodes]),pooled_cell_mean=statistics.mean([r[endpoint] for r in usable]) if usable else None))
            br_ratios=[r['mean'] for r in brs if r['protein']==protein and r['condition']==condition and r['endpoint']=='on_percentage' and r['mean'] is not None]
            audit.append(audit_row(protein,condition,'ALL',subset,br_ratios))
    return brs,conditions,audit

def audit_row(protein,condition,BR,members,br_ratios):
    ratios=[r['on_percentage'] for r in members if r['on_percentage_valid']]
    on=sum(r['On'] for r in members);off=sum(r['Not_on'] for r in members)
    out=dict(protein=protein,condition=condition,BR=BR,n_cells=len(members),n_BR=len({r['BR'] for r in members}),
        mean_of_cell_ratios=statistics.mean(ratios) if ratios else None,
        pooled_On_counts_ratio=100*on/(on+off) if on+off else None,
        equal_BR_mean_of_cell_ratios=statistics.mean(br_ratios) if br_ratios else statistics.mean(ratios) if ratios else None)
    out.update({k+'_per_cell_mean':statistics.mean([r[k] for r in members]) for k in COUNT_FIELDS})
    return out

def comparisons(config,brs):
    rows=[]
    def add(endpoint,kind,protein_a,condition_a,protein_b,condition_b):
        aa=[r for r in brs if (r['protein'],r['condition'],r['endpoint'])==(protein_a,condition_a,endpoint) and r['mean'] is not None]
        bb=[r for r in brs if (r['protein'],r['condition'],r['endpoint'])==(protein_b,condition_b,endpoint) and r['mean'] is not None]
        if len(aa)!=3 or len(bb)!=3:raise ValueError(f'Expected 3 independent BRs each: {protein_a}/{condition_a} vs {protein_b}/{condition_b}')
        a=[r['mean'] for r in aa];b=[r['mean'] for r in bb]
        effect,p,n=exact_permutation(a,b)
        rows.append(dict(comparison_id=f'{endpoint}:{protein_a}_{condition_a}_minus_{protein_b}_{condition_b}',endpoint=endpoint,comparison_type=kind,
            protein_a=protein_a,condition_a=condition_a,protein_b=protein_b,condition_b=condition_b,
            n_BR_a=len(a),n_BR_b=len(b),n_cells_a=sum(r['n_cells'] for r in aa),n_cells_b=sum(r['n_cells'] for r in bb),
            mean_a=statistics.mean(a),mean_b=statistics.mean(b),difference_a_minus_b=effect,
            effect_unit='percentage_points' if endpoint=='on_percentage' else 'um2',p_raw=p,p_holm=None,n_partitions=n,
            significant_holm=False,alpha=config['tests']['alpha'],statistical_unit='independent_BR',pairing='unpaired',method=METHOD_VERSION))
    for endpoint in config['endpoints']:
        for condition in config['conditions']:add(endpoint,'proteins_within_condition','TRAK1',condition,'TRAK2',condition)
        for protein in config['proteins']:
            for c1,c2 in itertools.combinations(config['conditions'],2):add(endpoint,'conditions_within_protein',protein,c1,protein,c2)
    for r,p in zip(rows,holm([r['p_raw'] for r in rows])):r['p_holm']=p;r['significant_holm']=p<=r['alpha']
    if len(rows)!=27:raise ValueError('Expected exactly 27 prespecified comparisons')
    return rows

def validate(config,cells,brs,conditions,comparisons,records):
    import numpy as np
    from scipy.stats import permutation_test
    for r in comparisons:
        a=[v['mean'] for v in brs if (v['protein'],v['condition'],v['endpoint'])==(r['protein_a'],r['condition_a'],r['endpoint'])]
        b=[v['mean'] for v in brs if (v['protein'],v['condition'],v['endpoint'])==(r['protein_b'],r['condition_b'],r['endpoint'])]
        ref=permutation_test((np.array(a),np.array(b)),lambda x,y:np.mean(x)-np.mean(y),permutation_type='independent',vectorized=False,n_resamples=np.inf,alternative='two-sided')
        if not math.isclose(r['p_raw'],float(ref.pvalue),abs_tol=1e-12):raise AssertionError('SciPy permutation mismatch '+r['comparison_id'])
        if len(ref.null_distribution)!=r['n_partitions']:raise AssertionError('Partition count mismatch')
    fixture=holm([.01,.04,.03,.2]);expected=[.04,.09,.09,.2]
    if not all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(fixture,expected)):raise AssertionError('Holm fixture failed')
    unequal=[{'BR':'BR1','values':[0,0,0]},{'BR':'BR2','values':[100]},{'BR':'BR3','values':[50,50]}]
    equal_BR=statistics.mean([statistics.mean(r['values']) for r in unequal]);pooled=statistics.mean(sum([r['values'] for r in unequal],[]))
    if equal_BR!=50 or math.isclose(equal_BR,pooled):raise AssertionError('Equal BR weighting fixture failed')
    checks=dict(n_cells=len(cells),n_BR=len({(r['protein'],r['condition'],r['BR']) for r in cells}),n_comparisons=len(comparisons),
        all_cell_On_ratios_match_saved=True,all_counts_conserve=True,all_sources_unchanged=all(r['sha256_before']==sha(r['path']) for r in records),
        scipy_exact_permutation_tests_matched=len(comparisons),holm_known_fixture_passed=True,unequal_cell_number_equal_BR_fixture_passed=True,
        scientific_upstream_calls=0,expected_cells_by_protein={p:sum(r['protein']==p for r in cells) for p in config['proteins']},
        all_metrics_valid=all(r[e+'_valid'] for r in cells for e in ENDPOINTS),min_exact_p=min(r['p_raw'] for r in comparisons),
        n_holm_rejections=sum(r['significant_holm'] for r in comparisons))
    if len(cells)!=211 or checks['n_BR']!=18 or not checks['all_sources_unchanged']:raise AssertionError('Final data cardinality/source check failed')
    return checks

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(config['run_dir']);data=out/'data'
    cells,records=read_sources(config);brs,conditions,audit=aggregate(config,cells);tests=comparisons(config,brs)
    checks=validate(config,cells,brs,conditions,tests,records)
    datasets={'cells':cells,'br_metrics':brs,'condition_metrics':conditions,'comparisons':tests,'aggregation_audit':audit}
    for name,rows in datasets.items():write_csv(data/(name+'.csv'),rows,list(rows[0]))
    for r in records:r['sha256_after']=sha(r['path'])
    write_json(data/'source_records.json',records)
    metadata=dict(method_version=METHOD_VERSION,created_utc=utc(),config=config,
        files={name:str(data/(name+'.csv')) for name in datasets},validation=checks,
        interpretation={'primary_condition_estimate':'equal mean of three BR means; BR means use equal cell values',
          'On_ratio':'100 * On / (On + Not_on), per cell; unknown and QC excluded are not in denominator',
          'areas':'ROI-contained binary support-mask area, first block frames 0-49 and time-weighted mean of 50 blocks',
          'inference':'Independent BR label permutation, two-sided, no confirmed pairing; BR is the experimental unit',
          'multiplicity':'Holm FWER across all 27 planned endpoint comparisons',
          'resolution':'3 vs 3 BRs yields 20 exact partitions and a minimum attainable two-sided p of 0.1; no test can reach 0.05 with this design',
          'cell_points':'Cells are shown descriptively; cells are not treated as independent BRs for hypothesis tests',
          'scope':'No localization, segmentation, spatial assignment or TARDIS recomputation'},
        br_metrics=brs,condition_metrics=conditions,comparisons=tests,aggregation_audit=audit)
    write_json(out/'summary.json',metadata);write_json(out/'validation_stats.json',checks)
    print(json.dumps(checks,indent=2));print('Statistics saved to '+str(out))
if __name__=='__main__':main()
