"""Compare cached 5x5 and 7x7 fits at identical TrackMate candidates."""
import argparse
from datetime import datetime, timezone
import io
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
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
import yaml

from run_preview import dump,fingerprint,video_and_checks


def q(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return {label:float(np.quantile(values,p)) if len(values) else None
            for label,p in [('q25',.25),('median',.5),('q75',.75),('q90',.9),('q99',.99),('max',1)]}


def shared_qc(table):
    # Same fixed width interval for both boxes, also retaining each original
    # fit validity/support rule. This is a comparison diagnostic, not a change
    # to either saved run's original QC flags or statistical membership.
    return (table.nominal_precision_valid & table.sx.between(.5,2.5)
        & table.sy.between(.5,2.5) & ~table.suspect_overlap
        & table.source_zero_pixels_fit_box.eq(0))


def run(box5,box7):
    config5=yaml.safe_load((box5/'config.yaml').read_text())
    config7=yaml.safe_load((box7/'config.yaml').read_text())
    assert config5['box']==5 and config7['box']==7
    assert {k:v for k,v in config5.items() if k!='box'}=={k:v for k,v in config7.items() if k!='box'}
    for path in [box5,box7]:assert json.loads((path/'COMPLETED.json').read_text())['all_checks_passed']
    protected=[]
    for path in [box5,box7]:
        protected.extend([path/name for name in ['config.yaml','ROI_points_with_refinement.csv',
            'all_detections_with_refinement.csv','summary.json','fit_backend_audit.npz',
            'nominal_camera_source.yaml','source_snapshot/gaussfit.py','source_snapshot/precision.py',
            'software_versions.json','COMPLETED.json']])
    protected.extend(sorted((box5/'frames').glob('frame_*.png')))
    protected.extend(sorted((box7/'frames').glob('frame_*.png')))
    before=[fingerprint(p) for p in protected]
    for name in ['nominal_camera_source.yaml','source_snapshot/gaussfit.py','source_snapshot/precision.py']:
        assert fingerprint(box5/name)['sha256']==fingerprint(box7/name)['sha256']
    a=pd.read_csv(io.BytesIO((box5/'ROI_points_with_refinement.csv').read_bytes()),float_precision='round_trip')
    b=pd.read_csv(io.BytesIO((box7/'ROI_points_with_refinement.csv').read_bytes()),float_precision='round_trip')
    fields=['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response',
            'radius_setting_um','roi_inside','roi_status','radius_setting_px','fit_anchor_x_px','fit_anchor_y_px']
    pd.testing.assert_frame_equal(a[fields],b[fields],check_exact=True)
    assert len(a)==len(b)==1173 and a.spot_id.is_unique
    audit5=np.load(box5/'fit_backend_audit.npz');audit7=np.load(box7/'fit_backend_audit.npz')
    np.testing.assert_array_equal(audit5['spot_id'],audit7['spot_id'])
    np.testing.assert_array_equal(audit5['original_ADU_patches'],audit7['original_ADU_patches'][:,1:6,1:6])
    out=box5/('comparison_box5_vs_box7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--box5',str(box5),'--box7',str(box7)])+'\n')
    dump(out/'rules.json',{'box5':str(box5),'box7':str(box7),'shared_TrackMate_candidates':1173,
        'radius_um':.28,'frames':config5['frames'],'nominal_calibration_only':True,
        'native_width_review_intervals_px':{'box5':[.5,2.5],'box7':[.5,3.5]},
        'comparison_shared_QC':'both fits physically admissible with finite positive precision, both sigma axes in [0.5,2.5] px, no original fit-box zeros or residual-overlap flags',
        'native_QC_preserved':True,'no_error_cutoff_or_parameter_constraints_added':True,
        'interpretation':'Smaller reported CRLB alone does not establish improved accuracy; subset selection and fit stability must be examined.'})
    pair=a.merge(b,on='spot_id',suffixes=('_5','_7'),validate='one_to_one',sort=False)
    common_valid=a.nominal_precision_valid.to_numpy()&b.nominal_precision_valid.to_numpy()
    common_clean=shared_qc(a).to_numpy()&shared_qc(b).to_numpy()
    pair['both_nominal_precision_valid']=common_valid
    pair['both_pass_shared_width_QC']=common_clean
    pair['fitted_center_difference_nm']=117*np.hypot(pair.fit_x_px_5-pair.fit_x_px_7,pair.fit_y_px_5-pair.fit_y_px_7)
    pair.to_csv(out/'all_1173_paired_fits.csv',index=False,float_format='%.17g')
    rows=[]
    for box,t in [(5,a),(7,b)]:
        maximum=t[['lpx_nm','lpy_nm']].max(axis=1)
        valid=t.nominal_precision_valid
        clean=t.qc_unflagged_model_fit
        rows.append({'box':box,'candidates':len(t),'converged':int(t.fit_converged.sum()),
            'admissible_finite_precision':int(valid.sum()),'native_QC_unflagged':int(clean.sum()),
            'valid_requires_review':int((valid&~clean).sum()),'invalid_or_unavailable':int((~valid).sum()),
            'pass_shared_width_QC':int(shared_qc(t).sum()),
            'native_unflagged_lpx_nm':q(t.loc[clean,'lpx_nm']),
            'native_unflagged_lpy_nm':q(t.loc[clean,'lpy_nm']),
            'both_valid_lpx_nm':q(t.loc[common_valid,'lpx_nm']),
            'both_valid_lpy_nm':q(t.loc[common_valid,'lpy_nm']),
            'both_shared_QC_lpx_nm':q(t.loc[common_clean,'lpx_nm']),
            'both_shared_QC_lpy_nm':q(t.loc[common_clean,'lpy_nm']),
            'valid_max_axis_above_200nm':int((valid&(maximum>200)).sum()),
            'valid_max_axis_above_500nm':int((valid&(maximum>500)).sum()),
            'valid_max_axis_above_1000nm':int((valid&(maximum>1000)).sum()),
            'fitted_center_outside_ROI':int(t.fit_center_roi_inside.eq(0).sum())})
    summary={'groups':rows,'both_valid':int(common_valid.sum()),'both_shared_QC':int(common_clean.sum()),
        'native_QC_membership':{'both':int((a.qc_unflagged_model_fit&b.qc_unflagged_model_fit).sum()),
            'only_box5':int((a.qc_unflagged_model_fit&~b.qc_unflagged_model_fit).sum()),
            'only_box7':int((~a.qc_unflagged_model_fit&b.qc_unflagged_model_fit).sum())},
        'paired_center_difference_nm_both_valid':q(pair.loc[common_valid,'fitted_center_difference_nm']),
        'paired_center_difference_nm_both_shared_QC':q(pair.loc[common_clean,'fitted_center_difference_nm'])}
    dump(out/'summary.json',summary)
    prior_examples=[9183,5751,7109,6330]
    pair[pair.spot_id.isin(prior_examples)].to_csv(out/'previous_large_error_examples_paired.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0];bottom=np.zeros(2)
    for name,color in [('native_QC_unflagged','#35A97A'),('valid_requires_review','#BD62BA'),('invalid_or_unavailable','#DD8B34')]:
        values=[r[name] for r in rows];ax.bar(['5x5','7x7'],values,bottom=bottom,label=name,color=color);bottom+=values
    ax.set(ylabel='Same ROI candidates',title='Native QC upper sigma limit: 2.5 / 3.5 px');ax.legend(fontsize=8)
    ax=axes[0,1]
    for t,name,color in [(a,'5x5','#E07C24'),(b,'7x7','#1F77B4')]:
        for field,style in [('lpx_nm','-'),('lpy_nm','--')]:
            v=np.sort(t.loc[common_clean,field].to_numpy());ax.plot(v,np.arange(1,len(v)+1)/len(v),style,color=color,label=f'{name} {field}')
    ax.set(xscale='log',xlabel='Nominal per-axis CRLB SD (nm)',ylabel='Cumulative fraction',title=f'Identical shared-QC subset: n={common_clean.sum()}');ax.legend(fontsize=9);ax.grid(alpha=.2)
    ax=axes[1,0]
    max5=a.loc[common_valid,['lpx_nm','lpy_nm']].max(axis=1);max7=b.loc[common_valid,['lpx_nm','lpy_nm']].max(axis=1)
    ax.scatter(max7,max5,s=8,alpha=.45)
    low=min(max5.min(),max7.min());high=max(max5.max(),max7.max())
    ax.plot([low,high],[low,high],color='gray',lw=1)
    ax.set(xscale='log',yscale='log',xlabel='7x7 max-axis nominal CRLB SD (nm)',ylabel='5x5 max-axis nominal CRLB SD (nm)',title=f'Paired admissible fits: n={common_valid.sum()}')
    ax=axes[1,1]
    v=np.sort(pair.loc[common_valid,'fitted_center_difference_nm'].to_numpy());ax.plot(v,np.arange(1,len(v)+1)/len(v),label='Both admissible')
    v=np.sort(pair.loc[common_clean,'fitted_center_difference_nm'].to_numpy());ax.plot(v,np.arange(1,len(v)+1)/len(v),label='Both shared QC')
    ax.set(xlabel='Difference between refined centers (nm)',ylabel='Cumulative fraction',title='Position sensitivity to fitting window');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('TrackMate radius 0.28 um | Same 1,173 candidates | 5x5 versus 7x7 MLE\nManufacturer-nominal calibration; no Picasso detection threshold',fontsize=14)
    fig.savefig(out/'box5_vs_box7_QC.png',dpi=170);plt.close(fig)
    # Four-panel video reuses the already validated raw/overlay panels; no
    # re-rendering, coordinate transformation, contrast change or refitting.
    preview=yaml.safe_load((Path(config5['trackmate_preview'])/'resolved_config.yaml').read_text())
    w=795;h=594;header=104;width=4*w+70;width+=width%2;height=804
    big=ImageFont.truetype(preview['font_path'],27);small=ImageFont.truetype(preview['font_path'],23);detail=ImageFont.truetype(preview['font_path'],19)
    frames=out/'frames';frames.mkdir();manifest=[]
    start,end=config5['frames']
    for frame in range(start,end+1):
        with Image.open(box5/'frames'/f'frame_{frame:04d}.png') as i5,Image.open(box7/'frames'/f'frame_{frame:04d}.png') as i7:
            first5=np.asarray(i5.crop((14,header,14+w,header+h)));first7=np.asarray(i7.crop((14,header,14+w,header+h)))
            np.testing.assert_array_equal(first5,first7)
            image=Image.new('RGB',(width,height),'#101317');draw=ImageDraw.Draw(image)
            draw.text((14,12),f'T1 C2 | Original frame {frame:04d} | {frame*.02:.2f} s | Same TrackMate radius 0.28 um candidates',font=big,fill='white')
            g5=a[a.frame==frame];g7=b[b.frame==frame]
            labels=['Raw single frame',f'TrackMate | {len(g5)} ROI candidates',f'7x7 | {g7.nominal_precision_valid.sum()} admissible',f'5x5 | {g5.nominal_precision_valid.sum()} admissible']
            panels=[i7.crop((14,header,14+w,header+h)),i7.crop((14+w+14,header,14+w+14+w,header+h)),
                i7.crop((14+2*(w+14),header,14+2*(w+14)+w,header+h)),i5.crop((14+2*(w+14),header,14+2*(w+14)+w,header+h))]
            for j,(panel,label) in enumerate(zip(panels,labels)):
                x=14+j*(w+14);image.paste(panel,(x,header));draw.text((x,60),label,font=small,fill='white')
            draw.text((14,710),'Green: native QC unflagged | Magenta: requires review | Orange X: invalid/unavailable bound at original TrackMate center',font=detail,fill='#DCE3EB')
            draw.text((14,740),'Crosses: uncapped +/- nominal per-axis CRLB. Native upper width QC is box/2 (2.5 px for 5x5, 3.5 px for 7x7).',font=detail,fill='#DCE3EB')
            draw.text((14,770),'Same 406-554 ADU display, C2 grid and ROI. No candidates deleted. Nominal CRLB does not establish actual accuracy.',font=detail,fill='#DCE3EB')
            meta=PngImagePlugin.PngInfo();meta.add_text('original_frame',str(frame));meta.add_text('ROI_candidates',str(len(g5)))
            image.save(frames/f'frame_{frame:04d}.png',pnginfo=meta,compress_level=5)
            manifest.append(dict(frame=frame,time_s=frame*.02,ROI_candidates=len(g5),box5_admissible=int(g5.nominal_precision_valid.sum()),box7_admissible=int(g7.nominal_precision_valid.sum())))
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False)
    video,checks=video_and_checks(out,preview)
    after=[fingerprint(p) for p in protected];assert before==after
    dump(out/'source_fingerprints.json',before)
    dump(out/'validation.json',{'all_checks_passed':True,'same_original_candidates_exact':True,
        'same_original_coordinates_anchors_ROI_and_frames':True,'box5_is_exact_center_crop_of_box7_raw_patches':True,
        'same_camera_and_Picasso_backend_sources':True,'source_results_unchanged':True,'raw_video_panels_pixel_exact':True,'video_checks':checks})
    lines=['TrackMate radius 0.28 um, same 1173 ROI candidates, original frames 2000-2050.',
        'Fitting box is the only changed run parameter. The shared candidate identities, original fields, anchors, calibration and fitting backend are exact matches.',
        'Native width QC is sigma in [0.5,box/2] px: [0.5,2.5] for 5x5 and [0.5,3.5] for 7x7. Native flags are never rewritten.',
        'Paired comparisons also use an identical subset passing both fit validity checks and fixed sigma [0.5,2.5] px on both axes, with no fit-box zero or residual-overlap flags.',
        f'Both admissible: {common_valid.sum()}; both shared QC: {common_clean.sum()}.',
        'All original 1173 candidates retained in all_1173_paired_fits.csv; all FOV detections retained in each source run. No new detection, Picasso gradient gate, trajectories, masking or TARDIS.',
        'CRLB values are manufacturer-nominal model lower bounds, not measured accuracy. Comparing different unflagged subsets or smaller CRLB alone cannot establish improved accuracy.',
        'The video shows 51 original frames at 10 fps, 5.1 seconds; error extents are unchanged and uncapped.']
    for r in rows:
        lines.append(f'Box {r["box"]}: converged {r["converged"]}, admissible {r["admissible_finite_precision"]}, native QC unflagged {r["native_QC_unflagged"]}; shared-subset X/Y medians {r["both_shared_QC_lpx_nm"]["median"]} / {r["both_shared_QC_lpy_nm"]["median"]} nm; max-axis >500 nm: {r["valid_max_axis_above_500nm"]}.')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    dump(out/'COMPLETED.json',dict(output_directory=str(out),video=str(video),all_checks_passed=True,**summary))
    print('OUTPUT_DIRECTORY:',out,flush=True);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--box5',type=Path,required=True);parser.add_argument('--box7',type=Path,required=True)
    args=parser.parse_args();run(args.box5,args.box7)
