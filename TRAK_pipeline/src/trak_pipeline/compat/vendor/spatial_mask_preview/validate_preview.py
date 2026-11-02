"""Independent cached-output validation using GEOS distance to pixel unions."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import shapely
from shapely import affinity
import tifffile

from geometry import pixel_square_union
from run_preview import dump,fingerprint,read_selected_records


def b(value):return value=='True'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output.resolve()
    c=json.loads((out/'resolved_config.json').read_text())
    source=json.loads((out/'source_verification.json').read_text())
    frames=c['frames'];loc=Path(source['localization_directory']);mask_dir=Path(source['mask_directory'])
    fields,original=read_selected_records(loc/'all_ROI_target_fits.csv',frames)
    with (out/'all_three_frame_ROI_candidates_with_spatial_preview.csv').open(newline='') as stream:
        audit=list(csv.DictReader(stream))
    assert [{k:r[k] for k in fields} for r in audit]==original
    before=json.loads((out/'input_fingerprints_before.json').read_text())
    after=json.loads((out/'input_fingerprints_after.json').read_text());assert before==after
    selected=np.load(out/'selected_frame_inputs.npz')
    np.testing.assert_array_equal(selected['original_frames'],frames)
    full=np.load(mask_dir/'clean_full_mask.npy',mmap_mode='r')
    summaries=json.loads((out/'summary.json').read_text())
    results=[];geos_checks=0
    with tifffile.TiffFile(source['C1_file']) as t1,tifffile.TiffFile(source['C2_file']) as t2:
        for j,frame in enumerate(frames):
            C1=t1.pages[frame].asarray();C2=t2.pages[frame].asarray();mask=full[frame]
            np.testing.assert_array_equal(C1,selected['C1_raw'][j])
            np.testing.assert_array_equal(C2,selected['C2_raw'][j])
            np.testing.assert_array_equal(mask,selected['clean_full_mask'][j])
            np.testing.assert_array_equal(selected['analysis_ROI_mask'][j],mask&selected['roi_mask'])
            rows=[r for r in audit if int(r['frame'])==frame]
            chosen=[r for r in rows if b(r['spatial_selected'])]
            assert len(chosen)==c['expected_selected_counts'][str(frame)]
            assert len(rows)-len(chosen)==c['expected_excluded_counts'][str(frame)]
            mean_a=sum(float(r['x_se_nm']) for r in chosen)/len(chosen)/117
            mean_b=sum(float(r['y_se_nm']) for r in chosen)/len(chosen)/117
            np.testing.assert_allclose([mean_a,mean_b],
                [summaries[j]['mean_axis_x_px'],summaries[j]['mean_axis_y_px']],rtol=0,atol=2e-14)
            geometry=pixel_square_union(mask)
            common_scaled=affinity.scale(geometry,xfact=1/mean_a,yfact=1/mean_b,origin=(0,0))
            zero_y,zero_x=np.nonzero(C1==0)
            inside=0;added_p=0;added_m=0;unknown_counts={'pointwise':0,'frame_mean':0}
            pairs={}
            for row in rows:
                assert row['validated_spatial_classification']=='unavailable_registration_unverified'
                assert row['spatial_registration_status']=='exploratory_identity_registration_unverified'
                if not b(row['spatial_selected']):
                    assert row['pointwise_label']==row['frame_mean_label']=='excluded_qc'
                    assert row['pointwise_axis_x_px']==row['pointwise_axis_y_px']==''
                    continue
                assert b(row['qc_unflagged_model_fit']) and b(row['fitted_center_inside_ROI'])
                x,y=float(row['fit_x_px']),float(row['fit_y_px'])
                center=geometry.covers(shapely.Point(x,y))
                assert center==b(row['conditional_center_inside_original_mask']);inside+=center
                for name,a,bb in [('pointwise',float(row['x_se_nm'])/117,float(row['y_se_nm'])/117),
                                  ('frame_mean',mean_a,mean_b)]:
                    g=common_scaled if name=='frame_mean' else affinity.scale(geometry,xfact=1/a,yfact=1/bb,origin=(0,0))
                    # Independent computational route: GEOS distance to the
                    # affine-scaled union, without production rectangle search.
                    distance=g.distance(shapely.Point(x/a,y/bb))
                    compatible=distance**2<=1+c['intersection_q_tolerance']
                    zero=bool(np.any((np.maximum(np.abs(zero_x-x)-.5,0)/a)**2+
                        (np.maximum(np.abs(zero_y-y)-.5,0)/bb)**2<=1+c['intersection_q_tolerance']))
                    outside=(x-a<-.5-1e-12 or x+a>427.5+1e-12 or y-bb<-.5-1e-12 or y+bb>683.5+1e-12)
                    label='unknown' if zero or outside else ('mito-compatible' if compatible else 'non-mito-compatible')
                    assert label==row[name+'_label'],(frame,row['candidate_index'],name,distance,label,row[name+'_label'])
                    unknown_counts[name]+=label=='unknown';geos_checks+=1
                    if label=='mito-compatible' and not center:
                        if name=='pointwise':added_p+=1
                        else:added_m+=1
                key=(row['pointwise_label'],row['frame_mean_label']);pairs[key]=pairs.get(key,0)+1
            png=out/f'frame_{frame:04d}_two_methods.png'
            with Image.open(png) as im:
                assert int(im.info['original_frame'])==frame
                assert int(im.info['selected_points'])==len(chosen)
                assert im.info['registration_status']=='exploratory_identity_registration_unverified'
                assert im.size==(2450,1890)
                im.verify()
            results.append(dict(frame=frame,selected=len(chosen),original_center_inside=int(inside),
                added_by_individual_ellipse=added_p,added_by_mean_ellipse=added_m,
                unknown=unknown_counts,paired_membership=[dict(pointwise=k[0],frame_mean=k[1],n=v) for k,v in pairs.items()]))
    result=dict(all_passed=True,source_fields_preserved_verbatim=True,source_SHA256_before_after_equal=True,
        source_pages_and_mask_snapshots_equal=True,independent_GEOS_distance_checks=geos_checks,
        original_frame_mean_keys_and_values_correct=True,PNG_metadata_and_decode_valid=True,
        no_source_writes_no_refitting=True,registration_validated=False,results=results)
    dump(out/'independent_validation.json',result)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
