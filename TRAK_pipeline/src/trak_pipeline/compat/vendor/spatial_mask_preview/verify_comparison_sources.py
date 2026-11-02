"""Read-only two-recipe, cache-identity and ROI/MAT verification in science env."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from verify_sources import verify_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();c=yaml.safe_load(args.config.read_text())
    assert c['mask_recipe']=='min16' and c['new_mask_recipe']=='W5_min32'
    old=verify_config(c,args.config)
    new=verify_config(dict(c,mask_version=c['new_mask_version'],mask_recipe=c['new_mask_recipe']),args.config)
    assert old['localization_directory']==new['localization_directory']
    np.testing.assert_array_equal(old['roi_vertices'],new['roi_vertices'])
    expected=dict(old['expected_sha256'])
    for path,digest in new['expected_sha256'].items():
        if path in expected:assert expected[path]==digest
        expected[path]=digest
    directory=Path(c['old_classification_directory'])
    complete=json.loads((directory/'COMPLETED.json').read_text())
    assert complete['all_checks_passed'] and complete['completed_frames']==2500
    assert complete['counts']['selected']==c['expected_selected']
    assert complete['counts']['ROI_candidates']==c['expected_ROI_candidates']
    assert complete['counts']['excluded']==c['expected_excluded']
    for label,n in c['expected_old_counts'].items():assert complete['counts'][label]==n
    prior=json.loads((directory/'resolved_config.json').read_text())
    for key in ['mask_version','registration_status','covariance_assumption','pixel_size_nm','frame_interval_s','k',
                'intersection_q_tolerance','ROI_distance_tolerance_px']:
        assert prior[key]==c[key],key
    manifest=json.loads((directory/'output_fingerprints.json').read_text())
    for filename in ['all_ROI_candidates_with_spatial_classification.csv',
            'selected_localizations_with_spatial_classification.csv','per_frame_summary.csv']:
        p=directory/filename
        match=[r for r in manifest if (Path(r['path']) if 'path' in r else directory/r['relative_path'])==p]
        assert len(match)==1;expected[str(p)]=match[0]['sha256']
    par=json.loads((Path(new['mask_directory'])/'parameters.json').read_text())
    assert Path(par['source_broad_labels'])==Path(old['mask_directory'])/'full_component_labels.npy'
    # Canonical input cache and mask identities from the completed old analysis.
    prior_sources=json.loads((directory/'source_verification.json').read_text())
    assert prior_sources['localization_directory']==old['localization_directory']
    assert prior_sources['mask_directory']==old['mask_directory']
    for path,digest in prior_sources['expected_sha256'].items():
        if path in expected:assert expected[path]==digest
    c['verification_python']=old['config']['verification_python']
    result=dict(config=c,old=old,new=new,expected_sha256=expected,
        old_classification_directory=str(directory),roi_vertices=old['roi_vertices'],
        checks=dict(original_localization_cache_identity_verified=True,
            original_selection_and_geometry_rules_identical=True,MAT_and_both_ROI_vertex_sets_identical=True,
            W5_uses_original_precleanup_broad_labels=True,both_recipes_explicitly_validated=True))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
