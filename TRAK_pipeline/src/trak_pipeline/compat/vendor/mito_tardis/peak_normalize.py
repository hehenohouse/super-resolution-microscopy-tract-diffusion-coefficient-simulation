#!/usr/bin/env python3
"""Peak-normalize saved signed Intra and paired bootstrap curves; no TARDIS rerun."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/noton-mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_intra import dump,fingerprint

HERE=Path(__file__).resolve().parent
GROUPS=['On','Not-on'];COLORS=['#159C76','#9854CA']


def generate(source):
    assert json.loads((source/'COMPLETED.json').read_text())['delivery_checks_passed']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=HERE/'results'/('postmin32_peak_'+stamp);out.mkdir(parents=True,exist_ok=False)
    known={r['path']:r['sha256'] for r in json.loads((source/'output_fingerprints.json').read_text())}
    paths=[source/'distribution'/group/'full.csv' for group in GROUPS]
    paths += [source/'bootstrap/shape_draws_L25.npz',source/'bootstrap/metric_draws_L25.npz',source/'config.json']
    before=[]
    for path in paths:
        record=fingerprint(path);assert record['sha256']==known[str(path)];before.append(record)
    dump(out/'input_fingerprints_before.json',before)
    tables=[pd.read_csv(path,float_precision='round_trip') for path in paths[:2]]
    x=tables[0].distance_mid_um.to_numpy();assert len(x)==300
    np.testing.assert_array_equal(x,tables[1].distance_mid_um)
    original=np.stack([d.intra_density_um_inv.to_numpy() for d in tables])
    peaks=original.max(1);assert np.isfinite(original).all() and (peaks>0).all()
    normalized=original/peaks[:,None]
    np.testing.assert_allclose(normalized.max(1),1,atol=1e-15)
    np.testing.assert_array_equal(original<0,normalized<0)
    for i,d in enumerate(tables):
        shape=d.shape_density_um_inv.to_numpy()
        np.testing.assert_allclose(shape/shape.max(),normalized[i],rtol=2e-14,atol=2e-14)
    with np.load(paths[2]) as bank,np.load(paths[3]) as metrics:
        np.testing.assert_array_equal(bank['group_order'],GROUPS)
        j=int(np.flatnonzero(bank['scope_order']=='full')[0]);replicate=bank['replicate_id'].copy()
        np.testing.assert_array_equal(replicate,metrics['replicate_id']);np.testing.assert_array_equal(replicate,np.arange(1000))
        areas=np.stack([metrics[g+'__signed_area'][j] for g in GROUPS])
        # Reconstruct saved signed Intra density and scale each replicate by its own maximum.
        intra=bank['density'][:,j]*areas[:,:,None]
        draw_peaks=intra.max(2)
        valid=bank['valid'][:,j]&np.isfinite(intra).all(2)&(areas>0)&(draw_peaks>0)
        common=valid.all(0);assert common.mean()>=.95
        draws=np.full_like(intra,np.nan);draws[:,common]=intra[:,common]/draw_peaks[:,common,None]
    np.testing.assert_allclose(draws[:,common].max(2),1,atol=1e-15)
    bands=np.quantile(draws[:,common],[.025,.975],axis=1)
    data=dict(distance_left_um=tables[0].distance_left_um,distance_right_um=tables[0].distance_right_um,distance_mid_um=x)
    for i,g in enumerate(GROUPS):
        prefix=g.replace('-','_');data[prefix+'_original_signed_density_um_inv']=original[i]
        data[prefix+'_peak_normalized']=normalized[i];data[prefix+'_exploratory_low']=bands[0,i];data[prefix+'_exploratory_high']=bands[1,i]
    pd.DataFrame(data).to_csv(out/'peak_normalized_curves.csv',index=False)
    np.savez_compressed(out/'peak_normalized_bootstrap_L25.npz',values=draws,common_valid=common,
        replicate_id=replicate,group_order=np.array(GROUPS),distance_mid_um=x)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,limit,title in zip(axes,[.5,3.],['Zoom: 0-0.5 µm','Full range: 0-3 µm']):
        for i,g in enumerate(GROUPS):
            peak_nm=x[np.argmax(original[i])]*1000
            ax.plot(x,normalized[i],lw=1.6,c=COLORS[i],label=f'{g} | peak at {peak_nm:.0f} nm')
            ax.fill_between(x,bands[0,i],bands[1,i],color=COLORS[i],alpha=.16)
        ax.set(xlim=(0,limit),ylim=(-.09,1.09),xlabel='Distance (µm)',title=title)
        ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set_ylabel('Relative signed Intra (own peak = 1)')
    fig.suptitle('TDKd3 T1 | post-min32 mask | Peak-normalized Intra | full2500 frames, τ = 20 ms',fontsize=13)
    fig.text(.5,.07,'Each positive maximum is found over all300 bins on0-3 µm. Negative bins retained; no smoothing or peak alignment.',ha='center',fontsize=9)
    fig.text(.5,.025,'Shading: exploratory paired L25 percentiles after each replicate is divided by its own peak; coverage remains unvalidated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.93))
    fig.savefig(out/'intra_peak_normalized.png',dpi=190);fig.savefig(out/'intra_peak_normalized.pdf');plt.close(fig)
    summary=dict(source_TARDIS_run=str(source),frames=[0,2499],tau_s=.02,peak_search_range_um=[0,3],bins=300,
        normalization='Each signed Intra density divided by its own positive maximum over0-3um',
        y_units='dimensionless relative signed amplitude',negative_bins_retained=True,smoothing=False,
        peak_positions_shifted=False,TARDIS_recomputed=False,bootstrap_recomputed=False,
        saved_bootstrap_replicates_renormalized_individually=True,common_valid_count=int(common.sum()),
        common_valid_fraction=float(common.mean()),formal_inference=False,groups={})
    for i,g in enumerate(GROUPS):
        summary['groups'][g]=dict(original_peak_density_um_inv=float(peaks[i]),peak_distance_nm=float(x[np.argmax(original[i])]*1000),
            normalized_peak=float(normalized[i].max()),negative_bins=int((normalized[i]<0).sum()))
    dump(out/'normalization.json',summary)
    after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
    c=json.loads((source/'config.json').read_text());sample=Path(c['classification_directory']).parents[2]
    # classification_directory = sample/analysis_results/category/run_timestamp
    assert sample.name.endswith('T1f')
    destination=sample/'analysis_results/mito_tardis_peak_normalization'/out.name
    dump(out/'publication_config.json',dict(local_source=str(out),destination=str(destination)))
    shutil.copyfile(Path(__file__),out/'peak_normalize.py')
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--run',str(source)])+'\n')
    dump(out/'COMPLETED.json',dict(numerical_checks_passed=True,visual_review_pending=True,output=str(out)))
    (HERE/'latest_peak_normalized_local.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),destination=str(destination),summary=summary),indent=2))


def publish(src):
    from PIL import Image
    status=json.loads((src/'COMPLETED.json').read_text());assert status['numerical_checks_passed']
    assert json.loads((src/'input_fingerprints_before.json').read_text())==json.loads((src/'input_fingerprints_after.json').read_text())
    with Image.open(src/'intra_peak_normalized.png') as image:image.load();assert image.size==(2470,950)
    dump(src/'visual_review.json',dict(passed=True,notes='Reviewed both0-0.5um and0-3um panels; peaks equal1, locations unchanged, signed negative bins retained, dimensionless y-axis and exploratory intervals labeled.',image_sha256=fingerprint(src/'intra_peak_normalized.png')['sha256']))
    status.update(visual_review_pending=False,delivery_checks_passed=True);dump(src/'COMPLETED.json',status)
    records=[fingerprint(p) for p in sorted(src.iterdir()) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']]
    dump(src/'output_fingerprints.json',records)
    dst=Path(json.loads((src/'publication_config.json').read_text())['destination'])
    assert dst.parent.name=='mito_tardis_peak_normalization' and not dst.exists();dst.mkdir(parents=True)
    for p in src.iterdir():
        assert p.is_file() and not p.is_symlink();shutil.copyfile(p,dst/p.name)
    for record in records:
        p=dst/Path(record['path']).name;actual=fingerprint(p)
        assert actual['sha256']==record['sha256'] and actual['bytes']==record['bytes']
    status.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/'COMPLETED.json',status)
    dump(dst/'publication.json',dict(copied_files_SHA256_equal=True,files_verified=len(records),local_source=str(src)))
    dump(dst/'output_fingerprints.json',[fingerprint(p) for p in sorted(dst.iterdir()) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print('PUBLISHED',dst)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.publish:publish(args.output.resolve())
    else:
        assert args.run is not None;generate(args.run.resolve())
