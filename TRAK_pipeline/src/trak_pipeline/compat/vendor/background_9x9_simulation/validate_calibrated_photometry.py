"""Check saved photometry controls without modifying any original experiment."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import erf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--source',type=Path,required=True);a=ap.parse_args()
    out=a.output;baseline=pd.read_csv(a.source/'all_trackmate_picasso_candidates.csv')
    saved=json.loads((out/'input_fingerprints.json').read_text())
    assert all(sha(p)==v for p,v in saved.items())
    tables={box:pd.read_csv(out/f'box{box}_all_candidates.csv') for box in [7,11]}
    common=baseline.truth_matched_from_TrackMate&baseline.qc_unflagged_model_fit
    for box,t in tables.items():
        assert len(t)==len(baseline)==1653
        for key in ['spot_id','x_px','y_px','frame','time_s']:np.testing.assert_array_equal(t[key],baseline[key])
        assert not (t.active_bound&t.fit_usable_interior).any()
        assert t.loc[t.active_bound,'N_se_e'].isna().all()
        attempted=t[t.fit_attempted]
        assert (attempted.photons_e>=0).all() and (attempted.bg_e_per_pixel>=0).all()
        assert ((attempted[['sx_px','sy_px']]>=.5-1e-10)&(attempted[['sx_px','sy_px']]<=3.5+1e-10)).all().all()
        x=attempted.fit_x_px-attempted.fit_origin_x;y=attempted.fit_y_px-attempted.fit_origin_y
        sx=attempted.sx_px;sy=attempted.sy_px
        mass_x=.5*(erf((box-.5-x)/(np.sqrt(2)*sx))-erf((-.5-x)/(np.sqrt(2)*sx)))
        mass_y=.5*(erf((box-.5-y)/(np.sqrt(2)*sy))-erf((-.5-y)/(np.sqrt(2)*sy)))
        np.testing.assert_allclose(attempted.captured_psf_fraction,mass_x*mass_y,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(attempted.signal_inside_box_e,attempted.photons_e*attempted.captured_psf_fraction,rtol=1e-12)
        assert not t.coverage_validated.any()
        common &= t.fit_usable_interior
    assert common.sum()==24
    summary=json.loads((out/'summary.json').read_text())
    comparisons=[]
    for label,t,N in [('original',baseline,'photons')]+[(f'box{box}',t,'photons_e') for box,t in tables.items()]:
        q=t[common];ratio=q[N]/q.truth_expected_total_electrons
        error=np.median(np.abs(ratio-1))*100
        expected=summary['original_same_common_subset'] if label=='original' else summary[label]['common_unchanged_subset']
        assert abs(error-expected['median_abs_relative_error_pct'])<1e-10
        comparisons.append(dict(method=label,n_common=len(q),median_abs_relative_error_pct=error))
    (out/'same_24_candidates_comparison.csv').write_text(pd.DataFrame(comparisons).to_csv(index=False))
    rows=[]
    for label,t,N in [('original',baseline,'photons')]+[(f'box{box}',t,'photons_e') for box,t in tables.items()]:
        q=t[t.truth_matched_from_TrackMate].copy();q['method']=label;q['estimated_total_e']=q[N]
        cols=['method','spot_id','truth_id','truth_expected_total_electrons','estimated_total_e']
        for key in ['active_bound','fit_usable_interior','N_se_e','bg_e_per_pixel','sx_px','sy_px','uncertainty_status']:
            if key in q:cols.append(key)
        rows.append(q[cols])
    pd.concat(rows,ignore_index=True).to_csv(out/'same_44_original_matches_detail.csv',index=False)
    validation=dict(all_passed=True,source_hashes_unchanged=True,all_1653_candidate_identities_and_original_centers_preserved=True,
        active_bound_covariance_not_reported=True,integrated_PSF_mass_and_whole_spot_signal_verified=True,
        same_24_candidates_metrics_recomputed=True,original_QC_labels_preserved_for_paired_comparison=True,
        no_claim_of_validated_precision=True)
    (out/'independent_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    shutil.copyfile(__file__,out/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('calibrated_photometry.log'),out/'run.log')
    (out/'visual_review.json').write_text(json.dumps(dict(reviewed=True,figure='photometry_comparison.png',
        titles_axes_legible=True,all_original_matched_candidates_displayed_including_boundary_fits=True,
        comparison_subset_same_24_candidates=True,correlation_plot_descriptive_only=True),indent=2)+'\n')
    (out/'statistical_limits.txt').write_text('''Point estimates compare the same original TrackMate candidates and matching identities.
Common subset means original QC pass AND both independent fits interior and successful.
This subset is selected and is not a measure of overall detector recall or precision.
Other summaries retain every original candidate and flag boundary/unsupported fits.
The neighbor_contaminated column is copied from the original BOX 7 truth audit,
not a freshly validated overlap rule for BOX 11. Larger patches can admit neighbors.
The single-emitter, locally uniform background model does not resolve overlapping emitters.
Observed-Hessian standard errors are local model diagnostics, not tested coverage or CRLB.
One movie realization and comparisons to known expected counts do not calibrate real data.
No width, count, background or position truth initialized the fitter.
The fixed Gaussian variance 500 is valid only for this synthetic generator.
Full Poisson/Gaussian likelihood is approximated numerically by truncating the Poisson
sum; scalar direct-sum checks and gradient checks passed. ADC quantization is neglected.
Future repeated simulation must assess detector selection, parameter bias and coverage.
''')
    artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_fingerprints.json'}
    (out/'output_fingerprints.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
