"""Joint sx/sy count histogram for all cached ROI candidates, including QC exclusions."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
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

from run_preview import dump, fingerprint

plt.style.use('default')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--classification-output',type=Path,required=True)
    args=parser.parse_args();parent=args.classification_output.resolve()
    source=json.loads((parent/'source_verification.json').read_text())
    path=Path(source['localization_directory'])/'all_ROI_target_fits.csv'
    before=fingerprint(path);assert before['sha256']==source['expected_sha256'][str(path)]
    rows=pd.read_csv(path,usecols=['candidate_index','sx_px','sy_px'],low_memory=False)
    assert len(rows)==127059 and rows.candidate_index.is_unique
    good=np.isfinite(rows.sx_px)&np.isfinite(rows.sy_px)
    xy=rows.loc[good,['sx_px','sy_px']].to_numpy()
    edges=np.arange(82)*.05-.025
    counts,ex,ey=np.histogram2d(xy[:,0],xy[:,1],bins=[edges,edges])
    assert int(counts.sum())==len(xy)
    # Verify axis orientation and count conservation against 1D marginals.
    np.testing.assert_array_equal(counts.sum(axis=1),np.histogram(xy[:,0],edges)[0])
    np.testing.assert_array_equal(counts.sum(axis=0),np.histogram(xy[:,1],edges)[0])
    out=parent/('fit_widths_2d_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    fig,ax=plt.subplots(figsize=(8.6,7.3),layout='constrained')
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('white')
    heat=ax.pcolormesh(ex,ey,np.ma.masked_equal(counts.T,0),cmap=cmap,
        vmin=0,vmax=float(counts.max()),shading='flat',rasterized=True)
    cb=fig.colorbar(heat,ax=ax,pad=.025,shrink=.88)
    cb.set_label('Localizations per 0.05 × 0.05 pixel bin (linear)')
    ax.axvline(3.5,color='#D63131',ls='--',lw=1.7,label='3.5 px upper fitting bounds')
    ax.axhline(3.5,color='#D63131',ls='--',lw=1.7)
    ax.set(xlim=(0,4),ylim=(0,4),aspect='equal',xlabel='sx: Gaussian PSF standard deviation (pixels)',
        ylabel='sy: Gaussian PSF standard deviation (pixels)',
        title=f'T1 C2 | sx vs sy | {len(xy):,} fitted points\nAll ROI candidates; QC exclusions included; {int((~good).sum())} unavailable')
    ax.legend(loc='upper left',fontsize=8.5,framealpha=.95)
    fig.savefig(out/'sx_sy_2d_histogram.png',dpi=180,facecolor='white');plt.close(fig)
    ii,jj=np.indices(counts.shape)
    pd.DataFrame(dict(sx_bin_low_px=ex[ii.ravel()],sx_bin_high_px=ex[ii.ravel()+1],
        sy_bin_low_px=ey[jj.ravel()],sy_bin_high_px=ey[jj.ravel()+1],
        count=counts.ravel().astype(int))).to_csv(out/'histogram_2d_bin_counts.csv',index=False)
    np.savez_compressed(out/'histogram_2d_counts.npz',sx_edges_px=ex,sy_edges_px=ey,counts_sx_sy=counts.astype(np.int64))
    after=fingerprint(path);assert before==after
    bx=np.isclose(xy[:,0],3.5,atol=1e-5,rtol=0);by=np.isclose(xy[:,1],3.5,atol=1e-5,rtol=0)
    dump(out/'parameters_and_validation.json',dict(all_passed=True,source=before,
        total_ROI_candidates=len(rows),finite_width_pairs=len(xy),unavailable_pairs=int((~good).sum()),
        selection='all original ROI candidates, no QC filtering',sx_upper_bound_count=int(bx.sum()),
        sy_upper_bound_count=int(by.sum()),both_widths_at_upper_bound=int((bx&by).sum()),
        upper_bound_matching_tolerance=1e-5,marked_fitting_bounds_px=[3.5,3.5],
        bin_width_px=.05,bin_edges_px=edges.tolist(),axes_display_range_px=[0,4],
        color_scale='linear counts; zero-count bins white',count_matrix_axis_order='sx,sy; transpose for plotting',
        counts_conserved=True,marginals_match_independent_1d_histograms=True,source_unchanged=True,no_refitting=True))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--classification-output',str(parent)])+'\n')
    (Path(__file__).parent/'latest_width_histogram_2d.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),n=len(xy),missing=int((~good).sum()),both_at_upper_bound=int((bx&by).sum()))))


if __name__=='__main__':main()
