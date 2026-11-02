"""Plot a display-range histogram from saved X/Y errors; do not refit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--min-nm', type=float, default=0)
    ap.add_argument('--max-nm', type=float, default=200)
    ap.add_argument('--bins', type=int, default=40)
    args = ap.parse_args()
    assert args.max_nm > args.min_nm and args.bins > 0
    source = args.source.resolve()
    inputs = [source/'all_ROI_target_fits.csv', source/'summary.json']
    before = [fingerprint(path) for path in inputs]
    table = pd.read_csv(inputs[0], usecols=[
        'histogram_eligible','qc_unflagged_model_fit','x_se_nm','y_se_nm'])
    summary = json.loads(inputs[1].read_text())
    shown = table[table.histogram_eligible]
    clean = shown[shown.qc_unflagged_model_fit]
    review = shown[~shown.qc_unflagged_model_fit]
    assert len(shown) == summary['finite_interior_ROI']
    edges = np.linspace(args.min_nm, args.max_nm, args.bins+1)
    out = source/f'error_histograms_{args.min_nm:g}_{args.max_nm:g}nm_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    out.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
    counts = {}
    bin_rows = []
    for ax, field, label in zip(axes,['x_se_nm','y_se_nm'],['X','Y']):
        values = shown[field].to_numpy()
        assert np.isfinite(values).all()
        above = int((values > args.max_nm).sum())
        below = int((values < args.min_nm).sum())
        total_in_range = int(np.histogram(values,bins=edges)[0].sum())
        assert total_in_range + above + below == len(shown)
        counts[field] = dict(total=len(shown), shown=total_in_range,
            above_range=above, below_range=below,
            unflagged_median_all_values_nm=float(clean[field].median()))
        for group, color, name in [(clean,'#16818e','Unflagged'),(review,'#b350a7','Overlap review')]:
            hist = np.histogram(group[field].to_numpy(),bins=edges)[0]
            ax.hist(group[field],bins=edges,histtype='step',linewidth=1.8 if name=='Unflagged' else 1.4,
                color=color,label=f'{name}: {int(hist.sum()):,} shown')
            bin_rows.extend(dict(axis=label,group=name,bin_left_nm=float(a),bin_right_nm=float(b),count=int(n))
                for a,b,n in zip(edges[:-1],edges[1:],hist))
        ax.axvline(clean[field].median(),color='#16818e',ls='--',
            label=f'Median: {clean[field].median():.1f} nm')
        ax.set(xlim=(args.min_nm,args.max_nm),xscale='linear',yscale='linear',
            xlabel=f'{label} local model standard error (nm)',ylabel='Localizations',
            title=f'{label}: {args.min_nm:g}–{args.max_nm:g} nm | {above:,} above range')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    first,last = summary['frames_inclusive']
    fig.suptitle(f'T1 C2 | Original frames {first}–{last} | TrackMate 0.28 um → 7x7 annular-mean constrained fit\n'
        f'Linear histograms, {args.bins} bins | ±1 model SE; uncertainty coverage NOT validated',fontsize=11)
    image_path = out/'localization_uncertainty_histogram_0_200nm.png'
    fig.savefig(image_path,dpi=175)
    plt.close(fig)
    pd.DataFrame(bin_rows).to_csv(out/'histogram_bin_counts.csv',index=False)
    after = [fingerprint(path) for path in inputs]
    assert before == after
    metadata = dict(source=str(source),range_nm=[args.min_nm,args.max_nm],bins=args.bins,
        bin_width_nm=float(edges[1]-edges[0]),xscale='linear',yscale='linear',refitting=False,
        full_range_results_preserved=True,cache_fingerprints_unchanged=True,
        displayed_group_selection='same histogram_eligible and QC flags as original plots',
        medians_use_all_unflagged_values=True,counts=counts,input_fingerprints=before)
    (out/'render_config_and_validation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),
        '--source',str(source),'--min-nm',str(args.min_nm),'--max-nm',str(args.max_nm),'--bins',str(args.bins)])+'\n')
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    print(json.dumps(dict(output=str(out),image=str(image_path),counts=counts),indent=2),flush=True)


if __name__ == '__main__':
    main()
