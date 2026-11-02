#!/usr/bin/env python3
"""Supplement the completed cached-fit review; no detection or fitting."""
import argparse
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

from pilot_qc import key_name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    run=Path(parser.parse_args().run).resolve()
    assert (run/'COMPLETED.json').exists()
    summary=pd.read_csv(run/'tardis_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.9),layout='constrained')
    columns={};metrics=[]
    for row,color in zip(summary.itertuples(),['#3b86bc','#ee8b22','#489b6e','#ad67ab','#666666']):
        name=key_name(int(row.box),row.threshold_factor)
        data=pd.read_csv(run/'tardis'/name/'distributions.csv')
        x=data.distance_mid_m.to_numpy()*1e6
        width=(data.distance_right_m-data.distance_left_m).to_numpy()*1e6
        signed=data.intra_contribution_density_um_inv.to_numpy()
        net_area=float(np.sum(signed*width));peak=float(np.max(signed))
        np.testing.assert_allclose(net_area,1-row.beta,rtol=1e-12)
        assert peak>0 and net_area>0
        peak_scaled=signed/peak;area_scaled=signed/net_area
        np.testing.assert_allclose(np.max(peak_scaled),1)
        np.testing.assert_allclose(np.sum(area_scaled*width),1)
        np.testing.assert_allclose(area_scaled,data.intra_conditional_density_um_inv,rtol=1e-10,atol=1e-12)
        columns['distance_um']=x
        columns[name+'_peak_one']=peak_scaled
        columns[name+'_signed_area_one_um_inv']=area_scaled
        label=f'Box {int(row.box)}, {row.threshold_factor:g}x'
        for ax,curve in zip(axes,[peak_scaled,area_scaled]):ax.plot(x,curve,color=color,lw=1,label=label)
        metrics.append({'box':int(row.box),'threshold_factor':row.threshold_factor,
                        'signed_net_area':net_area,'original_peak_um_inv':peak,
                        'peak_bin_center_nm':float(x[np.argmax(signed)]*1000),
                        'negative_bins_preserved':True,'normalization_range_um':[0,3]})
    for ax,title,ylabel in zip(axes,['Peak = 1','Signed net area = 1 over 0–3 um'],
                               ['Relative signed contribution','Signed shape (1/um)']):
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlim=(0,.6),xlabel='Distance (um)',ylabel=ylabel,title=title)
        ax.legend(fontsize=8)
    fig.suptitle('Continuous frames 0–49: compare shapes with common normalization\nNegative bins retained; signed estimates are not nonnegative probability PDFs',fontsize=12)
    fig.savefig(run/'qc/09_normalized_tardis_shapes.png',dpi=165,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(columns).to_csv(run/'qc/normalized_tardis_shapes.csv',index=False)
    (run/'qc/normalization_checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    notes={
        'kind':'qualitative visual inspection of selected examples; no ground truth labels',
        'source_pixels_changed':False,
        'observations':[
            'The unflagged start-segment example (frame 32) is broad and weak. Passing heuristic checks does not establish a single emitter.',
            'The selected negative-amplitude example resembles background fluctuations; the negative-background example fits across an exact-zero source edge.',
            'Low-threshold additions include localized weak peaks and broad/background-like structures. The selected high-threshold removal retains a visible localized peak.',
            'The sign-recovered example at frame 2473 has a fixed-zero row just outside its fit patch but inside the finite-difference detection support. It is flagged in the main input.',
            'The sign-recovered example at frame 3 has a localized raw peak without zero-containing detector support; this is compatible with a width-sign artifact, but is not experimentally validated.',
            'Adding the detector-support check changes the reference 0–49 input from 2421 to 2244 points and beta from 0.76178 to 0.76321; its displayed signed curve changes little.',
            'All five configurations have a short-distance peak; changing threshold and box changes membership and beta. No best parameter or physical diffusion estimate is selected.'
        ]}
    (run/'qc/visual_inspection_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    script=Path(__file__).resolve()
    shutil.copy2(script,run/'reproducibility'/script.name)
    (run/'supplement_command.txt').write_text(shlex.join(sys.orig_argv)+'\n')
    report=run/'REVIEW.md';text=report.read_text();marker='## Visual inspection and normalized-shape supplement'
    if marker not in text:
        text+='\n'+marker+'\n\n'
        text+='\n'.join('- '+observation for observation in notes['observations'])+'\n\n'
        text+='`qc/09_normalized_tardis_shapes.png` compares peak=1 and signed net area=1. All 0–3 um bins are used in normalization; negative bins are retained. Net area is 1−beta before scaling. Therefore changes in the original signed peak height alone do not establish changes in distribution width or localization precision. The normalized curves remain exploratory, noisy signed estimates. CSV, normalization checks, qualitative notes and exact supplementary command are saved alongside the images.\n'
        report.write_text(text)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
