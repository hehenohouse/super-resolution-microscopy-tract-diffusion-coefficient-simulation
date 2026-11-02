"""Compare three saved C1 mask methods; no segmentation, fitting or classification."""
import argparse,csv,json,hashlib,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());root=Path(cfg['run_directory'])
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYDANTIC_DISABLE_PLUGINS='__all__')
import numpy as np,tifffile
prior=Path(cfg['prior_double_clip_test'])
assert (root/'NUMERICAL_COMPLETE.json').is_file()
roi=np.load(root/'masks/roi_mask.npy')
methods={'original_default':np.load(root/'masks/baseline_clean_full_mask.npy'),'double_clip_p01_p99':np.load(root/'masks/prior_double_clip_full_mask.npy'),'low_only_p01_max':np.load(root/'masks/clean_full_mask.npy')}
def write(name,records):
    with (root/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary=[];pairs=[];bright=[]
for i,start in enumerate(cfg['block_starts']):
    for scope,gate in [('ROI',roi),('full',np.ones_like(roi))]:
        for method,masks in methods.items():
            n=int((masks[i]&gate).sum())
            summary.append(dict(scope=scope,block_index=start//50,original_frame_start=start,original_frame_stop_exclusive=start+50,method=method,pixels=n,area_um2=n*.117**2))
        for reference,test in [('original_default','double_clip_p01_p99'),('original_default','low_only_p01_max'),('double_clip_p01_p99','low_only_p01_max')]:
            x=methods[reference][i]&gate;y=methods[test][i]&gate
            n0=int(x.sum());n1=int(y.sum());inter=int((x&y).sum());union=int((x|y).sum());added=int((y&~x).sum());removed=int((x&~y).sum())
            assert n0+added-removed==n1
            pairs.append(dict(scope=scope,block_index=start//50,original_frame_start=start,original_frame_stop_exclusive=start+50,reference_method=reference,test_method=test,old_pixels=n0,new_pixels=n1,area_old_um2=n0*.117**2,area_new_um2=n1*.117**2,delta_pixels=n1-n0,delta_area_um2=(n1-n0)*.117**2,delta_percent=100*(n1/n0-1) if n0 else None,intersection_pixels=inter,union_pixels=union,added_pixels=added,removed_pixels=removed,iou=inter/union if union else 1.,dice=2*inter/(n0+n1) if n0+n1 else 1.))
    # Fixed raw-image bright-pixel diagnostic: same region for all three methods.
    with tifffile.TiffFile(root/f'inputs/C1_original_frames{start:04d}_{start+49:04d}.ome.tif') as movie:raw=movie.pages[0].asarray()
    with (prior/'data/preprocessing_limits.csv').open(newline='') as f:
        limit=next(row for row in csv.DictReader(f) if int(row['original_frame'])==start)
    cutoff=float(limit['effective_high_ADU'])
    region=(raw>cutoff)&roi
    n=int(region.sum())
    for method,masks in methods.items():
        hit=int((region&masks[i]).sum())
        bright.append(dict(original_frame=start,block_index=start//50,method=method,raw_brightness_cutoff_ADU=cutoff,region_definition='original frame > original estimated P99 AND ROI; fixed across methods',bright_region_pixels=n,covered_bright_pixels=hit,uncovered_bright_pixels=n-hit,coverage_percentage=100*hit/n if n else None))
write('method_summary.csv',summary);write('pairwise_mask_comparison.csv',pairs);write('bright_pixel_diagnostic.csv',bright)
(root/'comparison_config.json').write_text(json.dumps({'method_order':list(methods),'pixel_size_nm':117,'scopes':['ROI','full'],'normalization':'none; binary mask pixel counts and geometric areas','bright_diagnostic':'three block-first original frames; raw pixels above the previously computed P99 and inside unchanged ROI. This region is a signal diagnostic, not ground truth.','scientific_upstream_calls':{'nellie':0,'localization':0,'assignment':0,'TARDIS':0},'source_mask_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in methods.items()}},indent=2)+'\n')
print('ROI mask areas (um2):')
for r in summary:
    if r['scope']=='ROI': print(r['original_frame_start'],r['method'],round(r['area_um2'],6))
print('Raw bright-pixel diagnostic coverage (%), not accuracy:')
for r in bright:print(r['original_frame'],r['method'],round(r['coverage_percentage'],2),r['bright_region_pixels'])
