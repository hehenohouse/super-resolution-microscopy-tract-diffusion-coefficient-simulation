"""Read-only source and MAT/ROI checks in the existing scientific environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from contracts import resolve_frames
from mask_metadata import check_recipe

ROOT=Path(__file__).resolve().parents[1]


def verify_config(config,config_path):
    c=dict(config)
    c['localization_cache_pointer']=str((config_path.parent/c['localization_cache_pointer']).resolve())
    c['verification_python']=str((config_path.parent/c['verification_python']).resolve())
    loc=Path(Path(c['localization_cache_pointer']).read_text().strip())
    base=Path(c['data_directory']);mask=base/'mitochondria_masks'/c['mask_version']
    complete=json.loads((mask/'COMPLETE.json').read_text())
    state=json.loads((mask/'run_status.json').read_text())
    val=json.loads((mask/'validation.json').read_text())
    assert json.loads((loc/'COMPLETED.json').read_text())['all_checks_passed']
    par=json.loads((mask/'parameters.json').read_text())
    recipe=check_recipe(complete,state,val,par,c)
    rules=json.loads((loc/'rules_before_run.json').read_text())
    assert rules['fit_rules']['box_px']==7 and rules['native_TrackMate']['radius_um']==.28
    assert rules['fit_rules']['pixel_size_nm']==117 and rules['fit_rules']['frame_interval_s']==.02
    assert c['registration_status']=='exploratory_identity_registration_unverified'
    assert c['k']==1
    c['resolved_frames']=resolve_frames(c)
    assert not any(c[k] for k in ['tracking','tardis','refitting','resegmenting'])
    spec=importlib.util.spec_from_file_location('spatial_preview_ROI_reader',ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py')
    roi_lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=roi_lib;spec.loader.exec_module(roi_lib)
    roi_path=base/c['roi_filename'];vertices=roi_lib.load_imagej_roi(roi_path)
    np.testing.assert_array_equal(vertices,np.load(mask/'roi_vertices_xy.npy',mmap_mode='r'))
    np.testing.assert_array_equal(vertices,pd.read_csv(loc/'roi_vertices.csv').to_numpy())
    with h5py.File(c['mat_file'],'r') as h:
        matches=[]
        for ref in h['data'][()].ravel(order='F'):
            if not ref:continue
            record=h[ref]
            if not isinstance(record,h5py.Group) or 'name' not in record:continue
            name=''.join(chr(int(v)) for v in record['name'][()].ravel(order='F') if int(v))
            if name==c['mat_record']:matches.append(record['roiData/roiCoordinates'][()].T)
        assert len(matches)==1
        np.testing.assert_array_equal(vertices,matches[0])
    assert len(vertices)==c['expected_ROI_vertices']
    yy,xx=np.mgrid[:684,:428]
    raster=roi_lib.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428)
    np.testing.assert_array_equal(raster,np.load(mask/'roi_mask.npy',mmap_mode='r'))
    assert raster.sum()==c['expected_ROI_pixels']
    native=json.loads((mask/'nellie_parameters.json').read_text());module_checks={}
    for name,digest in native['source_modules_sha256'].items():
        module_checks[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    assert all(module_checks.values())
    expected={}
    for row in json.loads((loc/'output_fingerprints.json').read_text()):
        if row['relative_path'] in ['all_detections.csv','all_ROI_target_fits.csv','roi_vertices.csv','frame_counts.csv']:
            expected[str(loc/row['relative_path'])]=row['sha256']
    for row in json.loads((loc/'input_fingerprints_after.json').read_text()):
        if row['path'] in [str(base/'C2-101625_TDKd3_TRAK1_T1.tiff'),str(roi_path),c['mat_file']]:
            expected[row['path']]=row['sha256']
    input_manifest='input_fingerprints.json' if recipe=='min16' else 'input_fingerprints_before.json'
    original_mask_inputs=json.loads((mask/input_manifest).read_text())
    c1=base/'C1-101625_TDKd3_TRAK1_T1.tiff';expected[str(c1)]=original_mask_inputs[str(c1)]
    mask_manifest=json.loads((mask/'output_fingerprints.json').read_text())
    mask_names=['clean_full_mask.npy','analysis_ROI_mask.npy','roi_mask.npy',
                 'roi_vertices_xy.npy','original_frame_numbers.npy','frame_completed.npy',
                 'page_frame_mapping.csv']
    mask_names+=['per_frame_summary.csv'] if recipe=='min16' else ['per_frame_statistics.csv','frame_windows_and_status.csv','parameters.json']
    for name in mask_names:
        expected[str(mask/name)]=mask_manifest[name]
    if 'three_frame_reference_directory' in c:
        reference=Path(c['three_frame_reference_directory'])
        assert json.loads((reference/'COMPLETED.json').read_text())['all_checks_passed']
        reference_table=reference/'all_three_frame_ROI_candidates_with_spatial_preview.csv'
        reference_manifest=json.loads((reference/'output_fingerprints.json').read_text())
        match=[r for r in reference_manifest if
            (Path(r['path']) if 'path' in r else reference/r['relative_path'])==reference_table]
        assert len(match)==1
        expected[str(reference_table)]=match[0]['sha256']
    result=dict(config=c,localization_directory=str(loc),mask_directory=str(mask),
        C1_file=str(c1),C2_file=str(base/'C2-101625_TDKd3_TRAK1_T1.tiff'),
        roi_vertices=vertices.tolist(),expected_sha256=expected,
        checks=dict(MAT_ROI_and_both_cache_vertex_sets_identical=True,
            inclusive_ROI_raster_matches_current_polygon_function=True,
            mask_completion_and_fixed_parameters_valid=True,current_pipeline_is_cached_annular_box7=True,
            installed_Nellie_source_hashes_match_recorded_versions=module_checks),
        registration_evidence=par.get('registration',{'independent_registration_evidence':False,
            'note':'W5 changes temporal masks, not C1/C2 registration'}),registration_transform='identity_assumption_only',
        verification_environment=dict(python=sys.version,executable=sys.executable,
            numpy=np.__version__,pandas=pd.__version__,h5py=h5py.__version__,PyYAML=yaml.__version__))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args();result=verify_config(yaml.safe_load(args.config.read_text()),args.config)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
