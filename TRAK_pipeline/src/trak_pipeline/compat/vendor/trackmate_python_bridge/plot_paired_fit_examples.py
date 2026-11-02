"""Plot the same cached spots with 4x4/5x5/7x7 raw/model/residual panels."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
import numpy as np
import pandas as pd
from picasso.fitting import precision
import yaml

from picasso_refinement import model_images


EXAMPLES=[(9183,'Previously narrow 7x7 fit'),(5751,'Previously broad 7x7 fit'),
          (7109,'Previously unflagged 7x7, large bound'),(6330,'Previously unflagged 7x7 reference')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(box5,box7,box4=None):
    sources={5:box5,7:box7}
    if box4 is not None:sources[4]=box4
    boxes=sorted(sources)
    configs={k:yaml.safe_load((p/'config.yaml').read_text()) for k,p in sources.items()}
    for box in boxes:
        assert configs[box]['box']==box
        assert {k:v for k,v in configs[box].items() if k!='box'}=={k:v for k,v in configs[7].items() if k!='box'}
    movie_path=next((Path(configs[5]['trackmate_preview'])/'native_detection').glob('*/input_frames.npy'))
    protected=[movie_path]
    for path in sources.values():
        protected.extend(path/name for name in ['config.yaml','ROI_points_with_refinement.csv',
            'fit_backend_audit.npz','nominal_camera_source.yaml','COMPLETED.json','output_fingerprints.json'])
    before={str(p):sha(p) for p in protected}
    tables={};audits={}
    for box,path in sources.items():
        assert json.loads((path/'COMPLETED.json').read_text())['all_checks_passed']
        assert sha(path/'ROI_points_with_refinement.csv')==json.loads((path/'output_fingerprints.json').read_text())['ROI_points_with_refinement.csv']
        tables[box]=pd.read_csv(path/'ROI_points_with_refinement.csv',float_precision='round_trip')
        audits[box]=np.load(path/'fit_backend_audit.npz')
    same=['spot_id','frame','x_px','y_px','fit_anchor_x_px','fit_anchor_y_px','roi_inside']
    for box in boxes:
        pd.testing.assert_frame_equal(tables[box][same],tables[7][same],check_exact=True)
        np.testing.assert_array_equal(audits[box]['spot_id'],audits[7]['spot_id'])
        margin=3-box//2
        np.testing.assert_array_equal(audits[box]['original_ADU_patches'],audits[7]['original_ADU_patches'][:,margin:margin+box,margin:margin+box])
        assert sha(sources[box]/'nominal_camera_source.yaml')==sha(box7/'nominal_camera_source.yaml')
    camera=yaml.safe_load((box5/'nominal_camera_source.yaml').read_text())['camera']
    movie=np.load(movie_path,mmap_mode='r')
    tag='_'.join(map(str,boxes))
    out=sources[boxes[0]]/(f'fit_examples_box{tag}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--box5',str(box5),'--box7',str(box7)]
    if box4 is not None:command+=['--box4',str(box4)]
    (out/'run_command.txt').write_text(shlex.join(command)+'\n')
    rows=[];cases={}
    for spot,label in EXAMPLES:
        case={}
        for box in boxes:
            t=tables[box][tables[box].spot_id==spot].iloc[0]
            j=int(np.flatnonzero(audits[box]['spot_id']==spot)[0])
            theta=audits[box]['canonical_theta'][j]
            raw=audits[box]['original_ADU_patches'][j]
            frame=int(t.frame);ax=int(t.fit_anchor_x_px);ay=int(t.fit_anchor_y_px)
            context=movie[frame-configs[box]['frames'][0],ay-10:ay+11,ax-10:ax+11]
            half=box//2
            np.testing.assert_array_equal(raw,context[10-half:10-half+box,10-half:10-half+box])
            data=(raw-camera['baseline_adu'])*camera['sensitivity_e_per_adu']/camera['em_gain']
            model=model_images(theta[None],box)[0]
            residual=data-model
            bound=precision._gauss_crlb(theta[None],box,False,
                variance=np.full((1,box,box),camera['read_noise_e_rms']**2,np.float32))[0]
            np.testing.assert_allclose(np.sqrt(bound[1:3])*117.,[t.lpx_nm,t.lpy_nm],rtol=2e-7)
            case[box]={'record':t,'context':context,'data':data,'model':model,'residual':residual}
            rows.append({'spot_id':spot,'frame':frame,'box':box,
                'lpx_nm':float(t.lpx_nm),'lpy_nm':float(t.lpy_nm),'sx_px':float(t.sx),'sy_px':float(t.sy),
                'signal_nominal_e_equivalent':float(t.photons),'background_nominal_e_per_pixel':float(t.bg),
                'fit_x_px':float(t.fit_x_px),'fit_y_px':float(t.fit_y_px),
                'fit_review_flags':t.fit_review_flags if pd.notna(t.fit_review_flags) else '',
                'fit_parameter_issues':t.fit_parameter_issues if pd.notna(t.fit_parameter_issues) else '',
                'nominal_precision_valid':bool(t.nominal_precision_valid),
                'qc_unflagged_model_fit':bool(t.qc_unflagged_model_fit)})
        for box in boxes:np.testing.assert_array_equal(case[box]['context'],case[7]['context'])
        case['gray_limits']=(float(case[7]['data'].min()),float(case[7]['data'].max()))
        case['residual_limit']=max([1.]+[float(np.max(np.abs(case[box]['residual']))) for box in boxes])
        cases[spot]=case

    def figure(selected,path):
        nrows=len(boxes)*len(selected)
        fig,axes=plt.subplots(nrows,4,figsize=(15,3.2*nrows+.45),constrained_layout=True)
        for example,(spot,label) in enumerate(selected):
            case=cases[spot]
            for offset,box in enumerate(boxes):
                row=len(boxes)*example+offset;v=case[box];t=v['record'];axesrow=axes[row]
                ax=axesrow[0]
                ax.imshow(v['context'],origin='upper',interpolation='nearest',cmap='gray',vmin=406,vmax=554)
                half=box//2
                ax.add_patch(Rectangle((10-half-.5,10-half-.5),box,box,fill=False,edgecolor='orange',lw=1.6))
                cx=t.x_px-t.fit_anchor_x_px+10;cy=t.y_px-t.fit_anchor_y_px+10
                ax.add_patch(Circle((cx,cy),3,fill=False,edgecolor='cyan',lw=1))
                ax.plot(t.fit_x_px-t.fit_anchor_x_px+10,t.fit_y_px-t.fit_anchor_y_px+10,'+',color='magenta',ms=9,mew=1.6)
                status='invalid' if not t.nominal_precision_valid else ('review' if not t.qc_unflagged_model_fit else 'unflagged')
                ax.set_title(f'Frame {int(t.frame)}, spot {spot} | {box}x{box}: {status}',fontsize=11)
                ax.set_ylabel(f'{box}x{box}\nCRLB X/Y: {t.lpx_nm:.1f} / {t.lpy_nm:.1f} nm\nsx/sy: {t.sx:.2f} / {t.sy:.2f} px',fontsize=10)
                ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
                for col,field in enumerate(['data','model','residual'],start=1):
                    ax=axesrow[col]
                    left=-half-.5;right=left+box
                    if col<3:
                        im=ax.imshow(v[field],origin='upper',extent=(left,right,right,left),interpolation='nearest',cmap='gray',vmin=case['gray_limits'][0],vmax=case['gray_limits'][1])
                    else:
                        limit=case['residual_limit']
                        im=ax.imshow(v[field],origin='upper',extent=(left,right,right,left),interpolation='nearest',cmap='coolwarm',vmin=-limit,vmax=limit)
                    ax.set_title([f'Raw {box}x{box}, nominal e-/px','Fitted Gaussian, shared scale','Residual, shared scale'][col-1],fontsize=11)
                    ax.set_xlim(-3.5,3.5);ax.set_ylim(3.5,-3.5)
                    ax.set_facecolor('#fff2da')
                    ax.set_xticks([-3,0,3]);ax.set_yticks([-3,0,3])
                    fig.colorbar(im,ax=ax,shrink=.75)
        order=' / '.join(f'{box}x{box}' for box in boxes)
        fig.suptitle(f'Same TrackMate 0.28 um spots: {order} rows from top to bottom per spot\nShared intensity, residual and pixel scales; beige border=outside fit box; axes=px from integer anchor\nOrange=fit box, cyan=original position marker, magenta=MLE center; nominal CRLB only',fontsize=13)
        fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)

    for spot,label in EXAMPLES:figure([(spot,label)],out/f'spot_{spot}_boxes_{tag}.png')
    figure(EXAMPLES[:2],out/'paired_extreme_examples.png')
    figure(EXAMPLES[2:],out/'paired_reference_examples.png')
    figure(EXAMPLES,out/f'all_four_spots_boxes_{tag}.png')
    pd.DataFrame(rows).to_csv(out/'paired_example_values.csv',index=False,float_format='%.17g')
    counts=[]
    common=np.ones(len(tables[7]),dtype=bool)
    common_upper=min(boxes)/2
    for box in boxes:
        table=tables[box];valid=table.nominal_precision_valid
        common &= (valid & table.sx.between(.5,common_upper)
            & table.sy.between(.5,common_upper) & ~table.suspect_overlap
            & table.source_zero_pixels_fit_box.eq(0)).to_numpy()
        counts.append({'box':box,'ROI_candidates':len(table),
            'backend_converged':int(table.fit_converged.sum()),
            'admissible_finite_CRLB':int(valid.sum()),'invalid_or_unavailable':int((~valid).sum()),
            'per_run_QC_unflagged':int(table.qc_unflagged_model_fit.sum()),
            'per_run_width_QC_upper_sigma_px':box/2,
            'admissible_max_axis_above_500nm':int((valid & (table[['lpx_nm','lpy_nm']].max(axis=1)>500)).sum())})
    for record in counts:
        table=tables[record['box']]
        record['all_boxes_shared_QC_count']=int(common.sum())
        record['shared_QC_lpx_median_nm']=float(table.loc[common,'lpx_nm'].median()) if common.any() else None
        record['shared_QC_lpy_median_nm']=float(table.loc[common,'lpy_nm'].median()) if common.any() else None
    pd.DataFrame(counts).to_csv(out/'all_candidate_box_summary.csv',index=False)
    dump={'sources':{box:str(p) for box,p in sources.items()},'spot_ids':[s for s,_ in EXAMPLES],
        'order':boxes,
        'selection':'Same four spots from previous 7x7 example figure; targeted examples, not random sample',
        'shared_scale':'Raw and model use original 7x7 data min/max for every box, separately per spot. Residual uses symmetric maximum absolute residual across every box, separately per spot.',
        'pixel_scale':'All fit/data/residual panels span -3.5 to 3.5 original pixels relative to the shared integer anchor. A box uses offsets -floor(box/2) through box-floor(box/2)-1. Beige outer margin was not fitted and is not zero-valued data. For 4x4, the original pixel centers are -2,-1,0,1, with geometric center -0.5 relative to anchor.',
        'context_display_range_adu':[406,554],'native_pixel_values_unchanged':True,
        'nominal_calibration_only':True,'no_refitting_no_QC_changes':True}
    dump['all_candidate_shared_QC']='Each run admissible finite positive bounds, sigma on both axes in [0.5,min(boxes)/2] px, no fitting-box zero or residual-overlap flags; use the intersection by unchanged row/spot ID.'
    (out/'plot_rules.json').write_text(json.dumps(dump,indent=2)+'\n')
    after={str(p):sha(p) for p in protected};assert before==after
    (out/'validation.json').write_text(json.dumps({'sources_unchanged':True,'source_sha256':before,
        'same_source_frames_and_candidate_ids':True,'original_raw_patches_verified':True,
        'CRLB_reproduced_from_cached_theta':True,'all_raw_boxes_match_original_pixel_offsets':True},indent=2)+'\n')
    (out/'output_fingerprints.json').write_text(json.dumps({p.name:sha(p) for p in out.glob('*.png')},indent=2)+'\n')
    print('OUTPUT_DIRECTORY:',out,flush=True)
    print(pd.DataFrame(rows)[['spot_id','frame','box','lpx_nm','lpy_nm','sx_px','sy_px','fit_review_flags']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--box5',type=Path,required=True);parser.add_argument('--box7',type=Path,required=True)
    parser.add_argument('--box4',type=Path)
    args=parser.parse_args();run(args.box5,args.box7,args.box4)
