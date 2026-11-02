"""Compare saved additive-scale mask artifacts only; no new filtering or classification."""
import argparse,csv,hashlib,json,os
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
cfg=json.loads(Path(args.config).read_text());root=Path(cfg['run_directory'])
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYDANTIC_DISABLE_PLUGINS='__all__')
import numpy as np,tifffile
if not (root/'NUMERICAL_COMPLETE.json').is_file():raise SystemExit('Scientific masks missing; comparison was not started')
roi=np.load(root/'masks/roi_mask.npy')
methods={m['id']:np.load(root/'masks'/m['id']/'clean_full_mask.npy') for m in cfg['methods']}
stages={}
for method in methods:
    with np.load(root/'masks'/method/'postprocess_stages.npz') as z:stages[method]={k:np.array(z[k],copy=True) for k in z.files}
with (Path(cfg['prior_reference_test'])/'data/preprocessing_limits.csv').open(newline='') as f:limits={int(row['original_frame']):float(row['effective_high_ADU']) for row in csv.DictReader(f)}
def write(name,rows):
    with (root/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summaries=[];pairwise=[];bright=[];stage_rows=[];stage_bright=[]
for i,start in enumerate(cfg['block_starts']):
    for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
        for method,all_masks in methods.items():
            n=int((all_masks[i]&gate).sum());summaries.append(dict(scope=scope,block_index=start//50,original_frame_start=start,original_frame_stop_exclusive=start+50,method=method,pixels=n,area_um2=n*.117**2))
        for reference,test in [('A_default','B_add_to_1p5um'),('A_default','C_add_to_2p0um'),('B_add_to_1p5um','C_add_to_2p0um')]:
            a=methods[reference][i]&gate;b=methods[test][i]&gate
            n0=int(a.sum());n1=int(b.sum());inter=int((a&b).sum());union=int((a|b).sum());added=int((b&~a).sum());removed=int((a&~b).sum())
            assert n0+added-removed==n1
            pairwise.append(dict(scope=scope,block_index=start//50,original_frame_start=start,original_frame_stop_exclusive=start+50,reference_method=reference,test_method=test,old_pixels=n0,new_pixels=n1,area_old_um2=n0*.117**2,area_new_um2=n1*.117**2,delta_pixels=n1-n0,delta_area_um2=(n1-n0)*.117**2,delta_percent=100*(n1/n0-1) if n0 else None,intersection_pixels=inter,union_pixels=union,added_pixels=added,removed_pixels=removed,iou=inter/union if union else 1.,dice=2*inter/(n0+n1) if n0+n1 else 1.))
    with tifffile.TiffFile(root/f'inputs/C1_original_frames{start:04d}_{start+49:04d}.ome.tif') as movie:raw=movie.pages[0].asarray()
    region=(raw>limits[start])&roi;n=int(region.sum())
    for method,all_masks in methods.items():
        hit=int((region&all_masks[i]).sum())
        bright.append(dict(original_frame=start,block_index=start//50,method=method,raw_brightness_cutoff_ADU=limits[start],region_definition='raw C1 > original full-frame P99 then restricted to ROI; same region across methods',bright_region_pixels=n,covered_bright_pixels=hit,uncovered_bright_pixels=n-hit,coverage_percentage=100*hit/n if n else None))
        packed=root/'masks'/method/'broad_packed'/f'b{start//50:03d}.npz'
        with np.load(packed) as z:first_broad=np.unpackbits(z['packed'][0:1],axis=1,count=684*428,bitorder='little').reshape(684,428).astype(bool)
        products={'block_first_single_frame_broad':first_broad,'block_support_ge10':stages[method]['support_mask'][i],'block_postmin32':stages[method]['post_min32_mask'][i],'block_final_holes_lt64':all_masks[i]}
        for stage,mask in products.items():
            for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
                pixels=int((mask&gate).sum());stage_rows.append(dict(original_frame_start=start,block_index=start//50,method=method,stage=stage,scope=scope,pixels=pixels,area_um2=pixels*.117**2,temporal_scope='one original frame' if stage=='block_first_single_frame_broad' else '50-frame support block'))
            hit=int((region&mask).sum());stage_bright.append(dict(original_frame=start,block_index=start//50,method=method,stage=stage,bright_region_pixels=n,covered_bright_pixels=hit,coverage_percentage=100*hit/n if n else None,temporal_scope='one original frame' if stage=='block_first_single_frame_broad' else '50-frame support block'))
write('method_summary.csv',summaries);write('pairwise_mask_comparison.csv',pairwise);write('bright_pixel_diagnostic.csv',bright);write('stage_areas.csv',stage_rows);write('bright_pixel_stage_coverage.csv',stage_bright)
dump={'method_version':'saved_mask_scale_comparison_v1','method_order':list(methods),'bright_diagnostic':'Raw original block-first frame pixels > previously computed full-frame P99, then intersect unchanged ROI; a signal diagnostic, not ground-truth accuracy.','stage_comparison_note':'Single-frame broad and 50-frame block masks represent different temporal scopes.','upstream_scientific_calls':dict(nellie=0,localization=0,assignment=0,TARDIS=0),'source_mask_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in methods.items()}}
(root/'comparison_config.json').write_text(json.dumps(dump,indent=2)+'\n')
summary={'sample_id':cfg['sample_id'],'frames':150,'methods':cfg['methods'],'ROI_areas_um2':[r for r in summaries if r['scope']=='ROI'],'bright_pixel_coverage':bright,'pairwise_ROI':[r for r in pairwise if r['scope']=='ROI'],'default_pipeline_changed':False,'localization_calls':0,'assignment_calls':0,'TARDIS_calls':0}
(root/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('ROI areas (um2):')
for row in summaries:
    if row['scope']=='ROI':print(row['original_frame_start'],row['method'],round(row['area_um2'],4))
print('Bright-pixel coverage diagnostic (%), not accuracy:')
for row in bright:print(row['original_frame'],row['method'],round(row['coverage_percentage'],2))
