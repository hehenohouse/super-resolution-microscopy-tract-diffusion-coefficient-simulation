"""Inspect cached large CRLB examples without refitting or changing any QC."""
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
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
from picasso.fitting import precision
import yaml

from picasso_refinement import model_images


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source):
    table_path=source/'ROI_points_with_refinement.csv'
    audit_path=source/'fit_backend_audit.npz'
    config=yaml.safe_load((source/'config.yaml').read_text())
    movie_path=next((Path(config['trackmate_preview'])/'native_detection').glob('*/input_frames.npy'))
    protected=[table_path,audit_path,movie_path]
    before={str(p):sha(p) for p in protected}
    t=pd.read_csv(table_path,float_precision='round_trip')
    a=np.load(audit_path);movie=np.load(movie_path,mmap_mode='r')
    camera=yaml.safe_load((source/'nominal_camera_source.yaml').read_text())['camera']
    out=source/('large_uncertainty_review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--source',str(source)])+'\n')
    # Prespecified inspection examples from the cached table: largest axis
    # bound, a very broad fit, largest unflagged bound, median-signal control.
    selections=[(9183,'Narrow-fit extreme'),(5751,'Broad-fit extreme'),
                (7109,'Unflagged but large bound'),(6330,'Unflagged reference')]
    fig,axes=plt.subplots(len(selections),4,figsize=(15,12),constrained_layout=True)
    records=[]
    for row,(spot,label) in enumerate(selections):
        record=t[t.spot_id==spot].iloc[0]
        j=int(np.flatnonzero(a['spot_id']==spot)[0]);theta=a['canonical_theta'][j]
        raw=a['original_ADU_patches'][j]
        data=(raw-camera['baseline_adu'])*camera['sensitivity_e_per_adu']/camera['em_gain']
        model=model_images(theta[None],7)[0]
        yy,xx=np.mgrid[:7,:7];N,x,y,sx,sy,bg=theta
        s=model-bg
        g=np.stack([s/N,s*(xx-x)/sx**2,s*(yy-y)/sy**2,
                    s*((xx-x)**2/sx**3-1/sx),s*((yy-y)**2/sy**3-1/sy),np.ones_like(s)])
        variance=camera['read_noise_e_rms']**2
        fisher=np.einsum('aij,bij,ij->ab',g,g,1/(np.maximum(model,1e-3)+variance))
        diag=np.sqrt(np.diag(fisher));scaled=fisher/np.outer(diag,diag)
        # Dimensionless conditioning avoids comparing parameter units.
        eig=np.linalg.eigvalsh(scaled)
        native=precision._gauss_crlb(theta[None],7,False,
                    variance=np.full((1,7,7),variance,np.float32))[0]
        np.testing.assert_allclose(np.sqrt(native[1:3])*117.,[record.lpx_nm,record.lpy_nm],rtol=2e-7)
        anchor_x=int(record.fit_anchor_x_px);anchor_y=int(record.fit_anchor_y_px)
        context=movie[int(record.frame)-config['frames'][0],anchor_y-10:anchor_y+11,anchor_x-10:anchor_x+11]
        np.testing.assert_array_equal(context[7:14,7:14],raw)
        ax=axes[row,0]
        ax.imshow(context,origin='upper',cmap='gray',vmin=406,vmax=554)
        ax.add_patch(Rectangle((6.5,6.5),7,7,fill=False,edgecolor='orange',lw=1.5))
        ax.add_patch(Circle((record.x_px-anchor_x+10,record.y_px-anchor_y+10),3,fill=False,edgecolor='cyan',lw=1))
        ax.plot(record.fit_x_px-anchor_x+10,record.fit_y_px-anchor_y+10,'+',color='magenta',ms=8)
        ax.set_title(f'{label}\nframe {int(record.frame)}, spot {spot}',fontsize=11)
        ax.set_ylabel(f'CRLB X/Y: {record.lpx_nm:.1f} / {record.lpy_nm:.1f} nm\nsx/sy: {sx:.2f} / {sy:.2f} px',fontsize=10)
        ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
        low,high=float(data.min()),float(data.max())
        for col,array in enumerate([data,model,data-model],start=1):
            ax=axes[row,col]
            if col<3:
                im=ax.imshow(array,cmap='gray',vmin=low,vmax=high,origin='upper')
            else:
                vmax=max(float(np.max(np.abs(array))),1)
                im=ax.imshow(array,cmap='coolwarm',vmin=-vmax,vmax=vmax,origin='upper')
            ax.set_title(['Raw 7x7, nominal e-/px','Fitted Gaussian (same scale)','Residual, nominal e-/px'][col-1],fontsize=11)
            ax.set_xticks([0,3,6]);ax.set_yticks([0,3,6]);fig.colorbar(im,ax=ax,shrink=.75)
        records.append(dict(label=label,spot_id=spot,frame=int(record.frame),
            nominal_lpx_nm=float(record.lpx_nm),nominal_lpy_nm=float(record.lpy_nm),
            sx_px=float(sx),sy_px=float(sy),fitted_signal_e_equivalent=float(N),
            fitted_background_e_equivalent_per_pixel=float(bg),
            fitted_gaussian_signal_sum_in_49_pixels=float(s.sum()),
            fraction_of_fitted_infinite_plane_signal_sampled_in_box=float(s.sum()/N),
            minimum_scaled_information_eigenvalue=float(eig[0]),
            maximum_scaled_information_eigenvalue=float(eig[-1]),
            scaled_information_condition_number=float(eig[-1]/eig[0]),
            qc_unflagged_model_fit=bool(record.qc_unflagged_model_fit),
            fit_review_flags=record.fit_review_flags if pd.notna(record.fit_review_flags) else '',
            refinement_shift_px=float(record.refinement_shift_px)))
    fig.suptitle('Cached 7x7 MLE: extreme uncertainty examples and a reference\n21x21 raw context: orange=fit box, cyan=original TrackMate marker, magenta=MLE center',fontsize=14)
    fig.savefig(out/'large_uncertainty_examples.png',dpi=160);plt.close(fig)
    (out/'examples.json').write_text(json.dumps(records,indent=2)+'\n')
    max_axis=t[['lpx_nm','lpy_nm']].max(axis=1)
    counts=[]
    for cut in [100,200,500,1000]:
        selected=t[t.nominal_precision_valid&(max_axis>cut)]
        counts.append(dict(max_axis_above_nm=cut,admissible_finite_rows=len(selected),
                           unflagged_rows=int(selected.qc_unflagged_model_fit.sum())))
    (out/'threshold_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    after={str(p):sha(p) for p in protected};assert before==after
    (out/'validation.json').write_text(json.dumps(dict(sources_unchanged=True,source_sha256=before,
        raw_fit_boxes_match_source_movie=True,nominal_CRLB_reproduced=True,
        no_refitting_or_QC_changes=True,selection_is_targeted_diagnostic_not_random_sample=True),indent=2)+'\n')
    print('OUTPUT_DIRECTORY:',out)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    run(parser.parse_args().source)
