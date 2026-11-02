"""All 50 truth emitters: report integrated fitted electrons with failure flags."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--source-run',required=True);args=ap.parse_args()
source=Path(args.source_run);files=[source/'all_trackmate_picasso_candidates.csv',source/'all_50_truth_match_status.csv']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
d=pd.read_csv(files[0]);truth=pd.read_csv(files[1]);matched=d[d.truth_matched_from_TrackMate].set_index('truth_id')
assert matched.index.is_unique
t=truth[['spot_id','expected_total_electrons','realized_total_electrons','x_px','y_px','TM_matched']].rename(columns={
    'spot_id':'truth_id','expected_total_electrons':'true_expected_N_e','realized_total_electrons':'true_realized_N_e'})
fields=matched[['spot_id','photons','bg','sx','sy','nominal_precision_valid','qc_unflagged_model_fit','fit_parameter_issues','fit_review_flags']].rename(columns={
    'spot_id':'TrackMate_candidate_id','photons':'raw_fitted_integrated_N_e','bg':'fitted_bg_e_per_pixel'})
t=t.join(fields,on='truth_id')
t['status']=np.select([~t.TM_matched,~t.nominal_precision_valid.eq(True),~t.qc_unflagged_model_fit.eq(True)],
    ['TrackMate_missed','matched_invalid_fit','matched_fit_requires_review'],default='matched_fit_original_QC_unflagged')
t['fit_to_expected_ratio']=t.raw_fitted_integrated_N_e/t.true_expected_N_e
t['relative_error_pct']=(t.fit_to_expected_ratio-1)*100
out=source/('electron_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
t.to_csv(out/'all_50_true_spots_electron_recovery.csv',index=False)
assert len(t)==50 and t.truth_id.is_unique
good=t[t.status=='matched_fit_original_QC_unflagged']
summary=dict(n_true=50,status_counts=t.status.value_counts().to_dict(),
    original_QC_matched=dict(n=len(good),true_expected_N_median_e=float(good.true_expected_N_e.median()),
        fitted_N_median_e=float(good.raw_fitted_integrated_N_e.median()),fitted_N_mean_e=float(good.raw_fitted_integrated_N_e.mean()),
        median_abs_relative_error_pct=float(np.median(np.abs(good.relative_error_pct))),
        within_20pct_count=int((np.abs(good.relative_error_pct)<=20).sum())),
    fit_N_definition='N = 2*pi*backend_peak_height*abs(sx)*abs(sy), whole-plane Gaussian integral; not peak intensity or box sum',
    units='known simulated electrons, with recorded uint16 ADU quantization; no real-data calibration inference')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(15,6))
fig.subplots_adjust(top=.77,bottom=.15,left=.065,right=.98,wspace=.23)
a=axs[0];a.scatter(t.truth_id,t.true_expected_N_e,facecolors='none',edgecolors='black',s=32,label='True expected N')
for status,color,label in [('matched_fit_original_QC_unflagged','#248A61','Fit: original QC unflagged'),
    ('matched_fit_requires_review','#D98928','Fit: requires review'),('matched_invalid_fit','#CE3341','Fit: invalid')]:
    q=t[t.status==status];a.scatter(q.truth_id,q.raw_fitted_integrated_N_e,color=color,s=20,label=label)
a.set(yscale='log',xlabel='True emitter ID (0-49)',ylabel='Integrated total electrons (log scale)',
    title='All 50 truth emitters; all reported matched fits\nExtreme invalid values retained, not hidden',xlim=(-1,50));a.legend(fontsize=8)
a=axs[1];a.scatter(good.true_expected_N_e,good.raw_fitted_integrated_N_e,color='#248A61',s=30)
a.plot([0,2500],[0,2500],'k--',label='Fitted = true')
a.set(xlabel='True expected total electrons',ylabel='Fitted integrated total electrons',xlim=(0,1700),ylim=(0,2500),
    title=f'{len(good)} truth-matched original-QC fits only\nMedian absolute relative error = {summary["original_QC_matched"]["median_abs_relative_error_pct"]:.1f}%');a.legend()
fig.suptitle('Per-emitter electron recovery: TrackMate 0.28 um / threshold 6.5 -> Picasso 7x7 MLE\n6 emitters missed; 5 matched fits invalid; current noise model mismatches the simulation',fontsize=13,y=.97)
fig.savefig(out/'per_spot_electron_recovery.png',dpi=130);plt.close(fig)
assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
shutil.copyfile(__file__,out/Path(__file__).name)
(out/'validation.json').write_text(json.dumps(dict(all_50_truth_rows_retained=True,missed_and_invalid_not_discarded=True,source_files_unchanged=True,source_sha256=hashes),indent=2))
print('OUTPUT:',out);print(json.dumps(summary,indent=2))
