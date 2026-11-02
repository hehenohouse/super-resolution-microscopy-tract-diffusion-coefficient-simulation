"""Descriptive moments/peaks of signed Intra, without clipping negative bins."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(left,right,density):
    left,right,density=map(lambda v:np.asarray(v,float),(left,right,density))
    assert np.isfinite([left,right,density]).all() and (right>left).all()
    mid=(left+right)/2;mass=density*(right-left);area=mass.sum()
    assert area>0
    cdf=mass.cumsum();previous=np.r_[0,cdf[:-1]];target=area/2
    roots=[];plateau=False
    for lo,hi,m,p,q in zip(left,right,mass,previous,cdf):
        if m==0 and p==target:plateau=True
        if m!=0 and min(p,q)<=target<=max(p,q):
            roots.append(float(lo+(target-p)/m*(hi-lo)))
    roots=np.unique(np.round(roots,14))
    half=float(roots[0]) if len(roots)==1 and not plateau else None
    peak=np.flatnonzero(density==density.max())
    return dict(signed_first_moment_nm=float(mid@mass/area*1000),
        signed_half_area_location_nm=half*1000 if half is not None else None,
        half_area_crossing_count=len(roots),half_area_plateau=plateau,
        mode_bin_center_nm=float(mid[peak[0]]*1000) if len(peak)==1 else None,
        mode_bin_left_nm=float(left[peak[0]]*1000) if len(peak)==1 else None,
        mode_bin_right_nm=float(right[peak[0]]*1000) if len(peak)==1 else None,
        tied_peak_bin_count=len(peak),negative_bins=int((mass<0).sum()),
        negative_area_abs=float(-mass[mass<0].sum()),signed_area=float(area),
        ordinary_probability_moments_applicable=bool((mass>=0).all()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();complete=json.loads((run/'COMPLETED.json').read_text())
    assert complete['delivery_checks_passed']
    expected={r['path']:r['sha256'] for r in json.loads((run/'output_fingerprints.json').read_text())}
    # Ordinary nonnegative reference and a signed curve with ambiguous 50% crossings.
    test=summarize([0,1,2],[1,2,3],[.25,.5,.25])
    assert test['signed_first_moment_nm']==test['signed_half_area_location_nm']==test['mode_bin_center_nm']==1500
    test=summarize([0,1,2],[1,2,3],[1,-1,1])
    assert test['half_area_crossing_count']==3 and test['signed_half_area_location_nm'] is None
    out=run.parents[1]/'mito_tardis_summary'/('T1_C2_On_NotOn_descriptive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    rows=[];fingerprints={}
    for group in ['On','Not-on']:
        path=run/'distribution'/group/'full.csv'
        fingerprints[str(path)]=digest(path);assert fingerprints[str(path)]==expected[str(path)]
        d=pd.read_csv(path,float_precision='round_trip')
        result=summarize(d.distance_left_um,d.distance_right_um,d.shape_density_um_inv)
        assert result['half_area_crossing_count']==1 and result['tied_peak_bin_count']==1
        # Positive constant normalization leaves the signed moment, half-area crossing and peak unchanged.
        raw=summarize(d.distance_left_um,d.distance_right_um,d.intra_density_um_inv)
        for field in ['signed_first_moment_nm','signed_half_area_location_nm','mode_bin_center_nm']:
            np.testing.assert_allclose(result[field],raw[field],atol=1e-10,rtol=1e-12)
        rows.append(dict(group=group,range_left_um=0.,range_right_um=3.,**result))
        shutil.copyfile(path,out/(group+'_source_distribution.csv'))
    pd.DataFrame(rows).to_csv(out/'descriptive_statistics.csv',index=False)
    description=(
        'Source: '+str(run)+'\n'
        'Full2500 frames, tau20ms, fixed0-3um range, all300 bins retained; no smoothing or negative clipping.\n'
        'Mean: signed first moment=sum(midpoint*bin_mass)/sum(bin_mass).\n'
        'Median surrogate: unique50% signed-area crossing, piecewise-uniform density within original bins; NOT an ordinary probability median when negative bins exist.\n'
        'Mode: center of the uniquely highest native10nm density bin; peak-bin interval saved. No sub-bin peak fit.\n'
        'Signed curves are not nonnegative PDFs. These are descriptive quantities, not calibrated molecular-motion population moments.\n'
        'Far-distance signed residuals can materially affect the first moment. Display zoom0-0.5um does not change this0-3um calculation.\n'
        'Synthetic reference/ambiguous-crossing checks and raw-vs-normalized invariance checks passed.\n')
    (out/'README.txt').write_text(description)
    (out/'run_command.txt').write_text(f'{sys.executable!r} {str(Path(__file__).resolve())!r} --run {str(run)!r}\n')
    shutil.copyfile(__file__,out/Path(__file__).name)
    assert {p:digest(p) for p in fingerprints}==fingerprints
    (out/'provenance.json').write_text(json.dumps(dict(source_directory=str(run),source_sha256=fingerprints,
        inputs_unchanged=True,negative_bins_preserved=True,bootstrap_not_recomputed=True,results=rows),indent=2)+'\n')
    (Path(__file__).parent/'latest_statistics_output.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),statistics=rows),indent=2))


if __name__=='__main__':main()
