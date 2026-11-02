"""Plot three unsmoothed count curves from completed spatial classification."""
import argparse
from datetime import datetime, timezone
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
    args=parser.parse_args();source=args.classification_output.resolve()
    completed=json.loads((source/'COMPLETED.json').read_text());assert completed['all_checks_passed']
    path=source/'per_frame_summary.csv';before=fingerprint(path)
    frame=pd.read_csv(path);np.testing.assert_array_equal(frame.frame,np.arange(2500))
    assert (frame['mito-compatible']+frame['non-mito-compatible']+frame['unknown']==frame.selected).all()
    assert (frame.selected+frame.excluded==frame.ROI_candidates).all()
    out=source/('counts_by_frame_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    table=frame[['frame','time_s','mito-compatible','non-mito-compatible','excluded','unknown','selected','ROI_candidates']].rename(columns={'excluded':'QC_excluded'})
    table.to_csv(out/'counts_by_frame.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,5.2),layout='constrained')
    for field,color,label in [('mito-compatible','#159C76','mito-compatible'),
            ('non-mito-compatible','#9854CA','non-mito-compatible'),('QC_excluded','#E88C1C','QC excluded')]:
        ax.plot(table.frame,table[field],color=color,lw=.85,label=label)
    ax.set(xlim=(0,2499),ylim=(0,None),xlabel='Original frame (0-based)',
        ylabel='Number of localizations per frame',title='T1 C2 | Pointwise k=1 spatial classification and QC')
    ax.set_xticks([0,250,500,750,1000,1250,1500,1750,2000,2250,2499])
    ax.grid(axis='y',color='#DCE0E5',lw=.6);ax.set_axisbelow(True)
    ax.legend(loc='upper right',frameon=True)
    fig.suptitle('Unsmoothed per-frame counts; C1/C2 identity registration unverified',fontsize=10,color='#555555')
    fig.savefig(out/'counts_by_frame.png',dpi=180,facecolor='white');plt.close(fig)
    after=fingerprint(path);assert before==after
    dump(out/'parameters_and_validation.json',dict(all_passed=True,source=before,
        classification_output=str(source),original_frames=[0,2499],frame_count=2500,
        frame_interval_s=.02,plotted_fields=['mito-compatible','non-mito-compatible','QC_excluded'],
        smoothing=False,source_unchanged=True,unknown_total=int(table.unknown.sum()),
        totals={field:int(table[field].sum()) for field in ['mito-compatible','non-mito-compatible','QC_excluded']},
        registration_status=completed['registration_status'],no_refitting_or_reclassification=True))
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--classification-output',str(source)])+'\n')
    (Path(__file__).parent/'latest_counts_graph.txt').write_text(str(out)+'\n')
    print(out)


if __name__=='__main__':main()
