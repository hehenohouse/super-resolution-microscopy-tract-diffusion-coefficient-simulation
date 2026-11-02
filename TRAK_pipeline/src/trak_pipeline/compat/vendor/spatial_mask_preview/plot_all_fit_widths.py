"""All ROI candidates: fitted Gaussian sx/sy histograms including QC exclusions."""
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
    loc=Path(source['localization_directory'])
    assert json.loads((loc/'COMPLETED.json').read_text())['all_checks_passed']
    rules=json.loads((loc/'rules_before_run.json').read_text())
    low,upper=rules['fit_rules']['width_bounds_px'];assert upper==3.5
    path=loc/'all_ROI_target_fits.csv';before=fingerprint(path)
    assert before['sha256']==source['expected_sha256'][str(path)]
    table=pd.read_csv(path,usecols=['candidate_index','frame','sx_px','sy_px','audit_status'],low_memory=False)
    assert len(table)==127059 and table.candidate_index.is_unique
    out=parent/('all_fit_width_histograms_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    # Put exact constraint values at bin centers, avoiding floating-roundoff
    # splitting a boundary pile-up across two adjacent histogram bins.
    edges=np.arange(82)*.05-.025
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True,layout='constrained')
    summaries={};bin_counts={}
    for ax,field,color,label in zip(axes,['sx_px','sy_px'],['#228A92','#8761B2'],['sx','sy']):
        values=table[field].to_numpy(float);finite=np.isfinite(values);v=values[finite]
        assert (v>=low-1e-10).all() and (v<=upper+1e-10).all()
        n,unused=np.histogram(v,bins=edges);assert int(n.sum())==int(finite.sum())
        bin_counts[label+'_count']=n
        summaries[field]=dict(total_candidates=len(table),finite_fitted_widths=int(finite.sum()),
            missing_widths=int((~finite).sum()),median_px=float(np.median(v)),
            at_upper_bound_within_1e_minus_5=int(np.isclose(v,upper,atol=1e-5,rtol=0).sum()),
            at_lower_bound_within_1e_minus_5=int(np.isclose(v,low,atol=1e-5,rtol=0).sum()))
        ax.stairs(n,edges,fill=True,color=color,alpha=.78,lw=.7)
        ax.axvline(upper,color='#D63131',ls='--',lw=1.8,label='Upper fitting bound: 3.5 px')
        ax.set(xlim=(0,4),xlabel=f'{label}: Gaussian PSF standard deviation (pixels)',
            title=f'{label} | n = {len(v):,} fitted values; {int((~finite).sum())} unavailable')
        ax.grid(axis='y',color='#DFE3E8',lw=.6);ax.set_axisbelow(True)
        ax.legend(fontsize=8.5,loc='upper right')
    axes[0].set_ylabel('Number of localizations')
    fig.suptitle('T1 C2 | All 127,059 ROI candidates | QC-excluded fits included',fontsize=12)
    fig.savefig(out/'sx_sy_all_candidates_histograms.png',dpi=180,facecolor='white');plt.close(fig)
    pd.DataFrame(dict(bin_low_px=edges[:-1],bin_high_px=edges[1:],**bin_counts)).to_csv(out/'histogram_bin_counts.csv',index=False)
    dump(out/'parameters_and_validation.json',dict(all_passed=True,source=before,
        classification_output=str(parent),selection='all original ROI candidates, no QC filtering',
        original_frames=[0,2499],fitting_box_px=7,units='Gaussian PSF standard deviation in native camera pixels',
        pixel_size_nm=117,width_bounds_px=[low,upper],marked_threshold_px=upper,bin_width_px=.05,
        bin_edges_px=edges.tolist(),axis_display_range_px=[0,4],y_axis='counts, linear',
        missing_width_policy='not imputed or entered as zero; separately reported',
        summaries=summaries,source_unchanged=fingerprint(path)==before,no_refitting=True,
        bound_interpretation='Constrained fitted widths; pile-up at the bound is not an unconstrained width distribution'))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--classification-output',str(parent)])+'\n')
    (Path(__file__).parent/'latest_width_histograms.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),summaries=summaries),ensure_ascii=False))


if __name__=='__main__':main()
