"""Meaningful cache/contract boundaries; no production movie is recalculated."""
import copy
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from trak_pipeline.identity import scientific_key, IdentityError
from trak_pipeline.cache import CacheStore, source_identity, atomic_json
from trak_pipeline.graph import default_specs, stage_parameters, expected_identity, build_plan
from trak_pipeline.types import Sample, Artifact, StageResult
from trak_pipeline.contracts import validate_mask_contract, unknown_reason_for_frame, validate_localization_table, ContractError

TEST_ROOT = Path(__file__).resolve().parents[1] / 'validation'

class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='foundation_', dir=TEST_ROOT)
        self.root = Path(self.tmp.name)
        self.cell = self.root / 'WT/BR1/T5'; self.cell.mkdir(parents=True)
        for name in ('C1.tif','C2.tif','cell.roi'):
            (self.cell / name).write_bytes(name.encode())
        self.sample = Sample('WT_PROTEIN_BR1_T5', self.cell, {}, self.cell/'C1.tif', self.cell/'C2.tif', self.cell/'cell.roi')
        self.config = {'data_root': str(self.root), 'package_root': str(TEST_ROOT.parent), 'methods': {}, 'style': {}}
        self.sources = {k:Artifact(source_identity(self.cell/f), k, {'image':str(self.cell/f)}) for k,f in [('C1','C1.tif'),('C2','C2.tif'),('roi','cell.roi')]}
    def tearDown(self):
        self.tmp.cleanup()
    def all_saved(self, config=None):
        config = config or self.config
        saved = dict(self.sources)
        for spec in default_specs(config):
            p=stage_parameters(config,spec); saved[spec.stage]=Artifact(expected_identity(spec,p,saved),spec.stage,{},parameters=p)
        return saved
    def test_scientific_identity_canonical_and_finite(self):
        self.assertEqual(scientific_key('m','1',{'b':[1.,2],'a':0.},{'x':'a'}),scientific_key('m','1',{'a':0,'b':(1,2.)},{'x':'a'}))
        with self.assertRaises(IdentityError):scientific_key('m','1',{'a':float('nan')},{})
        with self.assertRaises(IdentityError):scientific_key('m','1',{'a':Path('not-a-scientific-value')},{})
    def test_source_identity_independent_of_path_and_mtime(self):
        a=self.cell/'C1.tif';b=self.cell/'renamed.tif';b.write_bytes(a.read_bytes());os.utime(b,(1,1))
        self.assertEqual(source_identity(a),source_identity(b));b.write_bytes(b'changed');self.assertNotEqual(source_identity(a),source_identity(b))
    def test_mask_threshold_reuses_counts_and_unrelated_fit(self):
        saved=self.all_saved();config=copy.deepcopy(self.config);config['methods']['mask_threshold']={'parameters':{'minimum_support':12}}
        plan={r['stage']:r['action'] for r in build_plan(config,self.sample,saved)}
        for stage in ['detection','fit','broad','support_counts','c1_mean']:self.assertEqual(plan[stage],'reuse')
        for stage in ['mask_threshold','mask_cleanup','mask_roi','assignment','distances','decomposition','bootstrap','statistics']:self.assertEqual(plan[stage],'compute')
    def test_window_changes_support_but_roi_does_not_change_nellie(self):
        saved=self.all_saved();config=copy.deepcopy(self.config);config['methods']['support_counts']={'parameters':{'window_frames':25}}
        plan={r['stage']:r['action']for r in build_plan(config,self.sample,saved)}
        self.assertEqual(plan['broad'],'reuse');self.assertEqual(plan['support_counts'],'compute')
        saved['roi']=Artifact('new-roi','roi',{})
        plan={r['stage']:r['action']for r in build_plan(self.config,self.sample,saved)}
        self.assertEqual(plan['broad'],'reuse');self.assertEqual(plan['support_counts'],'reuse');self.assertEqual(plan['fit'],'compute');self.assertEqual(plan['mask_roi'],'compute')
    def test_style_only_publishes_and_normalization_recomputes(self):
        saved=self.all_saved();config=copy.deepcopy(self.config);config['style']['on_color']='red'
        plan=build_plan(config,self.sample,saved)
        self.assertTrue(all(r['action']=='reuse'for r in plan if r['scientific']))
        self.assertEqual(plan[-1]['action'],'publish')
        config=copy.deepcopy(self.config);config['methods']['decomposition']={'parameters':{'normalization_rule':'different'}}
        plan={r['stage']:r['action']for r in build_plan(config,self.sample,saved)}
        self.assertEqual(plan['distances'],'reuse');self.assertEqual(plan['decomposition'],'compute')
    def test_changed_labels_edges_and_selected_set_invalidate_counts(self):
        saved=self.all_saved();saved['assignment']=Artifact('labels-changed','assignment',{})
        # Direct ID dependency check avoids pretending an incompatible labeling
        # artifact satisfies this config's assignment stage.
        spec=next(s for s in default_specs()if s.stage=='distances')
        baseline=self.all_saved();old=expected_identity(spec,{},baseline)
        self.assertNotEqual(old,expected_identity(spec,{},saved))
        self.assertNotEqual(old,expected_identity(spec,{'edges_um':[0,1,3]},baseline))
        baseline['fit']=Artifact('selected-and-coordinate-set-changed','fit',{})
        self.assertNotEqual(old,expected_identity(spec,{},baseline))
        with self.assertRaises(ValueError):stage_parameters({'methods':{'distances':{'parameters':{'tau_frames':2}}}},spec)
    def test_commit_is_atomic_and_content_verified_not_mtime(self):
        spec=default_specs()[0];cache=CacheStore(self.config,self.sample);out=cache.output_dir(spec,{},self.sources);out.mkdir(parents=True)
        file=out/'detections.csv';file.write_text('uid,x\na,1\n')
        artifact=cache.commit(spec,{},self.sources,StageResult({'table':file},{'ignored_nan':float('nan')}))
        self.assertEqual(cache.lookup(spec,{},self.sources).id,artifact.id)
        os.utime(file,(1,1));self.assertIsNotNone(cache.lookup(spec,{},self.sources))
        file.write_text('uid,x\na,2\n');self.assertIsNone(cache.lookup(spec,{},self.sources))
    def test_parallel_state_writes_have_unique_temporaries(self):
        path=self.root/'state.json'
        with ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(lambda n:atomic_json(path,{'writer':n,'value':'x'*1000}),range(50)))
        self.assertIn(json.loads(path.read_text())['writer'],range(50));self.assertFalse(list(self.root.glob('*.tmp')))
    def test_legacy_source_ids_bind_saved_content_not_changed_sources(self):
        import hashlib
        from trak_pipeline.legacy import LegacyReader, source_artifacts
        config=copy.deepcopy(self.config)
        config['control_root']=str(self.root/'control')
        config['acquisition']={'axes':'TYX','shape':[2500,684,428],'pixel_size_nm':117.,'frame_interval_s':.02,'pixel_center':'integer_xy_centers','crop_offset_xy_px':[0,0]}
        records={}
        for stage,name in [('C1','C1.tif'),('C2','C2.tif'),('ROI','cell.roi')]:
            value=(self.cell/name).read_bytes()
            records[stage+'_file']={'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
        output=self.cell/'old';output.mkdir()
        (output/'sampled_input_fingerprints.json').write_text('{}')
        reader=LegacyReader(config);reader._locations={};reader._registry_entries={self.sample.id:{'source_files':records}}
        job={'headers':{'C1':{'axes':'TYX','shape':[2500,684,428]},'C2':{'axes':'TYX','shape':[2500,684,428]}},'pixel_size_nm':117.,'frame_interval_s':.02}
        original=source_artifacts(config,self.sample)
        old=reader._legacy_source_ids(job,output,self.sample,original)
        self.assertTrue(all(old[k]==original[k].id for k in old))
        self.sample.C1_file.write_bytes(b'new-C1')
        current=source_artifacts(config,self.sample)
        bound=reader._legacy_source_ids(job,output,self.sample,current)
        self.assertNotEqual(bound['C1'],current['C1'].id)
        self.assertEqual(bound['C2'],current['C2'].id)
        self.sample.ROI_file.write_bytes(b'changed-ROI')
        current=source_artifacts(config,self.sample)
        bound=reader._legacy_source_ids(job,output,self.sample,current)
        self.assertNotEqual(bound['roi'],current['roi'].id)
        config['acquisition']['shape']=[2500,700,428]
        self.assertNotEqual(source_artifacts(config,self.sample)['roi'].id,current['roi'].id)
    @staticmethod
    def mask_contract():
        return dict(axes='TYX',shape=[2,5,7],pixel_size_nm=117,frame_interval_s=.02,crop_offset_xy_px=[0,0],pixel_center_convention='integer_centers_xy;closed_unit_squares',registration={'transform':'identity','status':'unverified'},original_frame_ids=[0,1,2,3],frame_to_block={'0':0,'1':0,'2':1,'3':1},block_frame_ids=[[0,1],[2]],expected_block_frame_ids=[[0,1],[2,3]],block_valid=[True,True],block_reasons=['',''])
    def test_missing_participant_marks_entire_block_unknown(self):
        original=self.mask_contract();c=validate_mask_contract(original)
        self.assertEqual(unknown_reason_for_frame(c,0),'');self.assertEqual(unknown_reason_for_frame(c,2),'incomplete_support_block');self.assertEqual(unknown_reason_for_frame(c,3),'incomplete_support_block')
        self.assertEqual(unknown_reason_for_frame(c,9),'missing_original_frame');self.assertTrue(original['block_valid'][1])
        c=self.mask_contract();c['coordinate_transform']='translation';c['registration']['transform']='translation'
        with self.assertRaises(ContractError):validate_mask_contract(c)
        c=self.mask_contract();c['block_frame_ids'][1]=[1,2]
        with self.assertRaises(ContractError):validate_mask_contract(c)
    @staticmethod
    def row(uid='source:chunk:7'):
        return dict(source_detection_uid=uid,candidate_uid=uid,spot_id='7',frame='0',time_s='0',fit_x_px='1.2',fit_y_px='2.3',x_px='1',y_px='2',x_se_nm='25',y_se_nm='26',uncertainty_status='full_observed_Hessian_with_nuisances',qc_unflagged_model_fit='True',spatial_selected='True')
    def test_uid_is_persisted_unique_and_error_meaning_required(self):
        row=self.row();det=dict(row)
        self.assertEqual(validate_localization_table([row],[det])['ten_frame_detection_correspondences'],1)
        with self.assertRaises(ContractError):validate_localization_table([row,row])
        row=self.row();row['uncertainty_status']='some-precision'
        with self.assertRaises(ContractError):validate_localization_table([row])
        row=self.row('row-number-7')
        with self.assertRaises(ContractError):validate_localization_table([row])
        row=self.row();del row['x_se_nm']
        with self.assertRaises(ContractError):validate_localization_table([row])

if __name__=='__main__':unittest.main()
