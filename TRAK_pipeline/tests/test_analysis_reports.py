"""Small fixtures only: no existing cell science or historical output changes."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from trak_pipeline.types import Sample,Artifact,StageContext
from trak_pipeline.stages import distances,tardis,statistics,presentation,summaries

ROOT=Path(__file__).resolve().parents[1]

class AnalysisReportsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='analysis_fixture_',dir=ROOT/'validation');self.root=Path(self.temp.name);self.cell=self.root/'WT/PROTEIN_WT_BR1/T1';self.cell.mkdir(parents=True)
        self.sample=Sample('WT_PROTEIN_BR1_T1',self.cell,dict(molecule='PROTEIN',genotype='WT',BR='BR1',cell='T1'),self.cell/'C1.tif',self.cell/'C2.tif',self.cell/'cell.roi')
        self.config=dict(data_root=str(self.root),control_root=str(self.root/'_batch_pipeline'),acquisition=dict(frames=50,pixel_size_nm=117,frame_interval_s=.02),methods=dict(distances=dict(parameters=dict(frames=50,pixel_size_nm=117,frame_interval_s=.02,tau_frames=1,edges_um=np.linspace(0,3,301).tolist())),decomposition=dict(parameters=dict(frames=50,segment_frames=50,minimum_total_tail_pairs=0,minimum_background_tail_ordered_pairs=0)),bootstrap=dict(parameters=dict(replicates=3,seed=4,block_lengths_frames=[10,25,50],primary_block_frames=25,strata_frames=50)),support_counts=dict(parameters=dict(window_frames=50))))
        rows=[];labels=[]
        for f in range(50):
            for g,(x,y) in enumerate(((3.,3.),(8.,8.))):
                uid=f'detection_source_A:spot_{f*2+g}'
                rows.append(dict(source_detection_uid=uid,frame=f,time_s=f*.02,fit_x_px=x,fit_y_px=y,x_se_nm=25.,y_se_nm=30.,photons_e=100+f,bg_e_per_pixel=3,sx_px=1,sy_px=1,qc_unflagged_model_fit=True,fitted_center_inside_ROI=True,x_px=x,y_px=y))
                labels.append(dict(source_detection_uid=uid,pointwise_label='On' if g==0 else 'Not-on'))
        table=self.root/'fits.csv';pd.DataFrame(rows).to_csv(table,index=False);label=self.root/'labels.csv';pd.DataFrame(labels).to_csv(label,index=False)
        roi=np.ones((12,12),bool);full=np.ones((1,12,12),bool);v=np.array([[0,0],[11,0],[11,11],[0,11],[0,0]],float)
        np.save(self.root/'roi.npy',roi);np.save(self.root/'full.npy',full);np.save(self.root/'vertices.npy',v)
        mapping=self.root/'mapping.json';mapping.write_text(json.dumps(dict(frame_to_block={str(f):0 for f in range(50)})))
        contract=self.root/'contract.json';contract.write_text(json.dumps(dict(frame_to_block={str(f):0 for f in range(50)},block_valid=[True])))
        self.inputs=dict(fit=Artifact('fit1','fit',dict(table=str(table))),assignment=Artifact('labels1','assignment',dict(labels=str(label))),roi=Artifact('roi1','roi',dict(mask=str(self.root/'roi.npy'),vertices=str(self.root/'vertices.npy'))),mask_cleanup=Artifact('mask1','mask_cleanup',dict(full_mask=str(self.root/'full.npy'),contract=str(contract))),support_counts=Artifact('support1','support_counts',dict(mapping=str(mapping))))
    def tearDown(self):self.temp.cleanup()
    def ctx(self,stage,params=None,inputs=None):return StageContext(self.config,self.sample,params or {},inputs or self.inputs,self.root/stage,ROOT)
    def result(self,stage,result):self.inputs[stage]=Artifact(stage+'1',stage,{k:str(v) for k,v in result.files.items()},result.metadata)
    def test_explicit_uid_join_and_order_independence(self):
        a,selected,_=distances.joined_table(self.ctx('distances'));self.assertEqual(len(selected),100)
        p=Path(self.inputs['assignment'].files['labels']);df=pd.read_csv(p).sample(frac=1,random_state=7);df.to_csv(p,index=False)
        _,again,_=distances.joined_table(self.ctx('distances'));self.assertEqual(selected.tardis_group.tolist(),again.tardis_group.tolist())
    def test_duplicate_uid_rejected(self):
        p=Path(self.inputs['assignment'].files['labels']);a=pd.read_csv(p);a.loc[1,'source_detection_uid']=a.loc[0,'source_detection_uid'];a.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'unique'):distances.joined_table(self.ctx('distances'))
    def test_tau_other_than_one_rejected(self):
        with self.assertRaisesRegex(ValueError,'tau_frames=1'):distances.execute(self.ctx('distances',dict(tau_frames=2)))
    def test_exact_zero_and_original_frame_counts(self):
        r=distances.execute(self.ctx('distances'));h=np.load(r.files['counts']);self.assertEqual(h['total'].sum(),98);self.assertEqual(h['background'].sum(),0);self.assertEqual(h['background_zero_removed'].sum(),100);self.assertFalse(h['endpoint_valid'][-1]);self.assertEqual(h['source_frame'][0],0);self.assertEqual(h['target_frame'][0],1)
    def test_changed_edges_are_used(self):
        r=distances.execute(self.ctx('distances',dict(edges_um=np.linspace(0,2,21).tolist())));h=np.load(r.files['counts']);self.assertEqual(h['total'].shape,(2,50,20));np.testing.assert_array_equal(h['edges_um'],np.linspace(0,2,21))
    def test_mask_mapping_required(self):
        self.inputs['support_counts'].files={};self.inputs['mask_cleanup'].files.pop('contract')
        with self.assertRaisesRegex(FileNotFoundError,'Explicit'):statistics.execute(self.ctx('statistics'))
    def test_statistics_parameters_and_conservation(self):
        r=statistics.execute(self.ctx('statistics',dict(SE_edges_nm=[0,20,40,60],signal_bins=7)));h=np.load(r.files['histograms']);self.assertEqual(len(h['photons_edges_e']),8);self.assertEqual(h['SE_2D__All'].shape,(3,3));self.assertEqual(r.metadata['metrics']['counts']['selected'],100);self.assertEqual(r.metadata['metrics']['on_percentage'],50)
    def test_incomplete_mask_block_is_unknown_area(self):
        p=Path(self.inputs['mask_cleanup'].files['contract']);contract=json.loads(p.read_text());contract['block_valid']=[False];p.write_text(json.dumps(contract))
        r=statistics.execute(self.ctx('statistics'));a=pd.read_csv(Path(r.files['directory'])/'mask_area_by_frame.csv');self.assertTrue(a.ROI_mask_area_um2.isna().all())
    def test_bootstrap_keeps_original_internal_pairs(self):
        self.result('distances',distances.execute(self.ctx('distances')));self.result('decomposition',tardis.execute_decomposition(self.ctx('decomposition')));r=tardis.execute_bootstrap(self.ctx('bootstrap'));units=np.load(r.files['units']);self.assertFalse(np.isin(units['L10__source_frame'],[9,19,29,39,49]).any());np.testing.assert_array_equal(units['L10__target_frame'],units['L10__source_frame']+1);weights=np.load(r.files['weights']);self.assertEqual(weights['L10'].shape[1],3);self.assertTrue((Path(r.files['directory'])/'metric_draws_L10.npz').is_file())
    def test_refresh_missing_science_runs_zero_steps(self):
        with patch.object(distances,'execute',side_effect=AssertionError('science called')),patch.object(tardis,'execute_decomposition',side_effect=AssertionError('science called')),patch.object(statistics,'execute',side_effect=AssertionError('science called')):
            with self.assertRaisesRegex(FileNotFoundError,'refresh does not run'):presentation.execute(self.ctx('presentation'))
        with self.assertRaisesRegex(FileNotFoundError,'performs no science'):summaries.refresh(self.config)
    def test_old_publication_not_silently_adopted_after_style_change(self):
        figures=self.root/'existing_figures';figures.mkdir()
        for name in presentation.required_cell_pngs():(figures/name).write_bytes(b'preserved')
        self.sample.legacy_figures=figures;before={p.name:p.read_bytes() for p in figures.iterdir()};self.config['style']={'on_color':'red'}
        with self.assertRaisesRegex(FileNotFoundError,'refresh does not run'):presentation.execute(self.ctx('presentation'))
        self.assertEqual(before,{p.name:p.read_bytes() for p in figures.iterdir()})
    def test_c1_mean_is_raw_and_independent_of_masks(self):
        import tifffile
        raw=np.arange(50,dtype=np.uint16)[:,None,None]*np.ones((50,12,12),np.uint16);raw[:,0,0]=0
        tifffile.imwrite(self.sample.C1_file,raw,photometric='minisblack')
        result=statistics.execute_c1_mean(self.ctx('c1_mean',inputs={}))
        mean=np.load(result.files['mean']);self.assertEqual(mean[0,0],0);self.assertEqual(mean[1,1],24.5);self.assertTrue(result.metadata['zeros_included'])
    def test_hierarchy_saved_reader_jobs_and_pure_refresh_missing(self):
        samples=[]
        for br,counts in [('BR1',(20,80)),('BR2',(60,40))]:
            directory=self.root/f'WT/PROTEIN_WT_{br}/T1';directory.mkdir(parents=True,exist_ok=True);old=directory/'legacy';(old/'masks').mkdir(parents=True);(old/'data/tardis/distribution/On').mkdir(parents=True);(old/'data/tardis/distribution/Not-on').mkdir(parents=True)
            n_on,n_not=counts;(old/'classification_summary.json').write_text(json.dumps(dict(On=n_on,Not_on=n_not,unknown=0,QC_excluded=2,selected=100,ROI_candidates=102)))
            pd.DataFrame([dict(original_frame_start=0,original_frame_stop_exclusive=50,mask_ROI_area_px2=100),dict(original_frame_start=50,original_frame_stop_exclusive=100,mask_ROI_area_px2=200)]).to_csv(old/'masks/per_block_statistics.csv',index=False)
            edges=np.linspace(0,3,301);x=(edges[:-1]+edges[1:])/2;density=np.exp(-x*10)
            for group in ('On','Not-on'):pd.DataFrame(dict(distance_left_um=edges[:-1],distance_right_um=edges[1:],distance_mid_um=x,intra_density_um_inv=density)).to_csv(old/'data/tardis/distribution'/group/'full.csv',index=False)
            pd.DataFrame([dict(group=g,scope='full',valid=True) for g in ('On','Not-on')]).to_csv(old/'data/tardis/per_scope_summary.csv',index=False)
            (old/'data/tardis/metadata.json').write_text(json.dumps(dict(config_rules={'tau_frames':1},scientific_fingerprint='same_method')))
            sample=Sample(f'WT_PROTEIN_{br}_T1',directory,dict(molecule='PROTEIN',genotype='WT',BR=br,cell='T1'),directory/'C1.tif',directory/'C2.tif',directory/'cell.roi',old);samples.append(sample)
        result=summaries.summarize(self.config,samples);manifest=json.loads(Path(result['manifest']).read_text());self.assertEqual(len(manifest['nodes']),6)
        reordered=summaries.summarize(self.config,list(reversed(samples)));self.assertEqual(result['summary_id'],reordered['summary_id']);again=json.loads(Path(reordered['manifest']).read_text());self.assertEqual({n['key']:n['numeric_signature'] for n in manifest['nodes']},{n['key']:n['numeric_signature'] for n in again['nodes']});self.assertTrue(all(not Path(n['data']).is_absolute() for n in manifest['nodes']))
        top=next(n for n in manifest['nodes'] if n['level']=='comparison');self.assertTrue(top['intra']);job=json.loads((self.root/top['workbook_job']).read_text());names={s['name'] for s in job['sheets']};self.assertIn('Localization Means',names)
        condition=next(n for n in manifest['nodes'] if n['level']=='condition');means=pd.read_csv(self.root/condition['data']/'localization_count_means.csv');self.assertEqual(float(means.loc[means.averaging_code=='equal_replica_mean','on_count_mean'].iloc[0]),40.)
        (self.root/top['workbook_job']).unlink()
        with patch.object(distances,'execute',side_effect=AssertionError('science called')),patch.object(statistics,'execute',side_effect=AssertionError('science called')):
            with self.assertRaisesRegex(FileNotFoundError,'no science'):summaries.refresh(self.config,result['manifest'])
    def test_full_saved_data_cell_publisher_has_zero_science_calls(self):
        import tifffile
        raw=np.full((50,12,12),500,np.uint16)
        tifffile.imwrite(self.sample.C1_file,raw,photometric='minisblack');tifffile.imwrite(self.sample.C2_file,raw,photometric='minisblack')
        self.result('distances',distances.execute(self.ctx('distances')));self.result('decomposition',tardis.execute_decomposition(self.ctx('decomposition')));self.result('bootstrap',tardis.execute_bootstrap(self.ctx('bootstrap')));self.result('statistics',statistics.execute(self.ctx('statistics')));self.result('c1_mean',statistics.execute_c1_mean(self.ctx('c1_mean')))
        with patch.object(distances,'execute',side_effect=AssertionError('science called')),patch.object(tardis,'execute_decomposition',side_effect=AssertionError('science called')),patch.object(tardis,'execute_bootstrap',side_effect=AssertionError('science called')),patch.object(statistics,'execute',side_effect=AssertionError('science called')),patch.object(statistics,'execute_c1_mean',side_effect=AssertionError('science called')),patch.object(np,'histogram',side_effect=AssertionError('histogram science called')),patch.object(np,'histogram2d',side_effect=AssertionError('histogram science called')):
            result=presentation.execute(self.ctx('presentation'))
        self.assertEqual(result.metadata['scientific_stage_calls'],0);self.assertEqual(result.metadata['missing'],[]);self.assertTrue(presentation.required_cell_pngs(50,50)<={p.name for p in Path(result.files['directory']).glob('*.png')})
    def test_expected_inventory_preserves_incomplete_progress(self):
        root=self.root
        expected=[dict(sample_id='WT_PROTEIN_BR1_T1',genotype='WT',BR='BR1',cell='T1',source_directory_relative='WT/PROTEIN_WT_BR1/T1'),dict(sample_id='WT_PROTEIN_BR1_T2',genotype='WT',BR='BR1',cell='T2',source_directory_relative='WT/PROTEIN_WT_BR1/T2'),dict(sample_id='WT_PROTEIN_BR2_T1',genotype='WT',BR='BR2',cell='T1',source_directory_relative='WT/PROTEIN_WT_BR2/T1')]
        self.config['_expected_samples']=expected;curve=np.exp(-(np.arange(300)+.5)*.1)
        record=dict(self.sample.metadata,sample_id=self.sample.id,source_directory_relative='WT/PROTEIN_WT_BR1/T1',status='complete',On=50,Not_on=50,unknown=0,QC_excluded=0,selected=100,ROI_candidates=100,on_percentage=50.,first_mask_ROI_area_um2=5.,mean_ROI_area_um2=4.)
        cell=dict(sample_id=self.sample.id,status='complete',eligible=True,reasons=[],curves={'On':curve,'Not-on':curve},edges=np.linspace(0,3,301),method_definition={'same':'method'},result_digest='curve')
        with patch.object(summaries,'_load_cell',return_value=(record,cell)):
            result=summaries.summarize(self.config,[self.sample])
        manifest=json.loads(Path(result['manifest']).read_text());br=next(n for n in manifest['nodes'] if n['key']=='WT/BR1');condition=next(n for n in manifest['nodes'] if n['level']=='condition');top=next(n for n in manifest['nodes'] if n['level']=='comparison')
        brnode=json.loads((root/br['data']/'intra_node.json').read_text());cnode=json.loads((root/condition['data']/'intra_node.json').read_text());tnode=json.loads((root/top['data']/'intra_node.json').read_text())
        self.assertEqual(brnode['progress']['expected'],2);self.assertEqual(cnode['progress']['expected'],3);self.assertEqual(tnode['progress']['expected'],3)
        self.assertEqual(brnode['progress']['completed'],1);self.assertEqual(cnode['progress']['status'],'PROGRESS');self.assertEqual(tnode['progress']['status'],'PROGRESS');self.assertEqual(cnode['progress']['expected_BRs'],2)
    def test_identical_summary_inputs_skip_loading_and_writes(self):
        curve=np.exp(-(np.arange(300)+.5)*.1)
        record=dict(self.sample.metadata,sample_id=self.sample.id,source_directory_relative='WT/PROTEIN_WT_BR1/T1',status='complete',On=50,Not_on=50,unknown=0,QC_excluded=0,selected=100,ROI_candidates=100,on_percentage=50.,first_mask_ROI_area_um2=5.,mean_ROI_area_um2=4.)
        cell=dict(sample_id=self.sample.id,status='complete',eligible=True,reasons=[],curves={'On':curve,'Not-on':curve},edges=np.linspace(0,3,301),method_definition={'same':'method'},result_digest='curve')
        artifacts={'statistics':Artifact('statistics_scientific1','statistics',{}),'decomposition':Artifact('decomposition_scientific1','decomposition',{})}
        entries=[{'sample':self.sample,'artifacts':artifacts}]
        with patch.object(summaries,'_load_cell',return_value=(record,cell)):
            first=summaries.summarize(self.config,entries)
        with patch.object(summaries,'_load_cell',side_effect=AssertionError('reloaded unchanged cell')),patch.object(summaries,'_save_csv',side_effect=AssertionError('rewrote unchanged CSV')):
            second=summaries.summarize(self.config,entries)
        self.assertTrue(second['reused']);self.assertFalse(second['scientific_statistics_computed']);self.assertEqual(first['summary_id'],second['summary_id'])
        artifacts['statistics']=Artifact('changed_scientific_input','statistics',{})
        with patch.object(summaries,'_load_cell',return_value=(record,cell)),patch.object(summaries,'_save_csv',side_effect=AssertionError('rewrote unchanged node CSV')):
            changed=summaries.summarize(self.config,entries)
        self.assertFalse(changed['reused']);self.assertEqual(first['summary_id'],changed['summary_id'])
    def test_unsupported_summary_science_rejected(self):
        for key,value in [('cell_weighting','localization_count'),('condition_weighting','pooled_cells'),('signed_area_min_abs',.02),('require_positive_peak',False),('require_both_groups',False),('normalization','absolute_area')]:
            config=dict(self.config,summary={key:value})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Unsupported hierarchy summary'):
                summaries.summarize(config,[])
    def test_cell_and_replica_equal_weighting(self):
        rows=[]
        for br,values in [('BR1',[20.,40.]),('BR2',[80.])]:
            for i,v in enumerate(values):rows.append(dict(sample_id=br+str(i),genotype='WT',BR=br,source_directory_relative=f'WT/P_WT_{br}/T{i}',on_percentage=v,first_mask_ROI_area_um2=v,mean_ROI_area_um2=v,selected=100,On=int(v),Not_on=100-int(v),unknown=0,QC_excluded=0,ROI_candidates=100))
        nodes=summaries._metric_nodes(rows,self.root);condition=next(n for n in nodes if n['level']=='condition');stat=next(r for r in condition['summary_records'] if r['metric']=='on_percentage');self.assertEqual(stat['mean'],55.);self.assertEqual(stat['n_cells_valid'],3);self.assertEqual(stat['n_BRs_valid'],2)


    def _area_aggregation_fixture(self, include_invalid=False):
        """Distinct signed shapes and unequal areas distinguish the two averaging orders."""
        rows, cells, records, samples = [], {}, {}, []
        unit_shapes = []
        for values in ((150., -50., 0.), (0., 150., -50.), (-25., 0., 125.)):
            curve = np.zeros(300, dtype=float)
            curve[:3] = values
            unit_shapes.append(curve)
        spec = [('BR1', 'T1', .2, unit_shapes[0]),
                ('BR1', 'T2', .8, unit_shapes[1]),
                ('BR2', 'T1', .6, unit_shapes[2])]
        if include_invalid:
            spec.append(('BR1', 'T3', .4, unit_shapes[0]))
        for br, cell_name, area, shape in spec:
            sid = f'WT_PROTEIN_{br}_{cell_name}'
            directory = self.root / f'WT/PROTEIN_WT_{br}/{cell_name}'
            directory.mkdir(parents=True, exist_ok=True)
            metadata = dict(molecule='PROTEIN', genotype='WT', BR=br, cell=cell_name)
            row = dict(metadata, sample_id=sid,
                       source_directory_relative=directory.relative_to(self.root).as_posix())
            record = dict(row, status='complete', On=10, Not_on=20, unknown=0,
                          QC_excluded=2, selected=30, ROI_candidates=32,
                          on_percentage=100./3., first_mask_ROI_area_um2=1.,
                          mean_ROI_area_um2=2.)
            signed = area * shape
            off = np.roll(signed, 3)
            if include_invalid and cell_name == 'T3':
                off = np.zeros(300)
            rows.append(row)
            records[sid] = record
            cells[sid] = dict(sample_id=sid, status='complete', eligible=True,
                              reasons=[], curves={'On': signed, 'Not-on': off},
                              edges=np.linspace(0, 3, 301),
                              method_definition={'fixture': 'same_cell_method'},
                              result_digest='fixture_curve_' + sid)
            samples.append(Sample(sid, directory, metadata, directory/'C1.tif',
                                  directory/'C2.tif', directory/'cell.roi'))
        return rows, cells, records, samples

    def test_BR_area_mean_uses_equal_cell_normalized_shapes(self):
        from trak_pipeline.compat import intra_summary_core as core
        rows, cells, _, _ = self._area_aggregation_fixture()
        nodes = core.summarize_cells(rows, cells, {'data_root': str(self.root)})
        br = next(n for n in nodes if n['level'] == 'BR' and n['key'] == 'WT/BR1')
        expected = np.zeros(300); expected[:3] = [75., 50., -25.]
        actual = br['groups']['On']['area']['mean']
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(actual * core.WIDTHS)), 1., places=12)
        self.assertLess(actual[2], 0., 'Signed negative bins must survive aggregation')
        # The previous order would instead give [30, 110, -40] in these bins.
        raw = np.stack([cells[r['sample_id']]['curves']['On'] for r in rows[:2]])
        old = raw.mean(0) / np.sum(raw.mean(0) * core.WIDTHS)
        self.assertFalse(np.allclose(actual, old))
        # Correcting the area view must not change the signed or peak conventions.
        np.testing.assert_allclose(br['groups']['On']['signed']['mean'], raw.mean(0),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(br['groups']['On']['peak']['mean'],
                                   (raw / raw.max(axis=1)[:, None]).mean(0),
                                   rtol=0, atol=1e-12)

    def test_condition_area_mean_keeps_equal_BR_weights(self):
        from trak_pipeline.compat import intra_summary_core as core
        rows, cells, _, _ = self._area_aggregation_fixture()
        nodes = core.summarize_cells(rows, cells, {'data_root': str(self.root)})
        condition = next(n for n in nodes if n['level'] == 'condition' and n['key'] == 'WT')
        brs = {n['BR']: n for n in nodes if n['level'] == 'BR' and n['condition'] == 'WT'}
        expected = np.zeros(300); expected[:3] = [25., 25., 50.]
        actual = condition['groups']['On']['area']
        np.testing.assert_allclose(actual['mean'], expected, rtol=0, atol=1e-12)
        self.assertEqual(actual['n'], 2)
        for br_name in ('BR1', 'BR2'):
            np.testing.assert_allclose(actual['series'][br_name],
                                       brs[br_name]['groups']['On']['area']['mean'],
                                       rtol=0, atol=1e-12)
        # BR1 has two cells and BR2 has one: pooled-cell averaging is a different estimand.
        pooled = np.mean([c['curves']['On'] / np.sum(c['curves']['On'] * core.WIDTHS)
                          for c in cells.values()], axis=0)
        self.assertFalse(np.allclose(actual['mean'], pooled))
        self.assertAlmostEqual(float(np.sum(actual['mean'] * core.WIDTHS)), 1., places=12)
        delta = np.zeros(300); delta[:3] = [100., 50., -150.]
        np.testing.assert_allclose(actual['sd'], abs(delta) / np.sqrt(2.),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual['sem'], abs(delta) / 2.,
                                   rtol=0, atol=1e-12)

    def test_area_correction_keeps_one_common_cell_roster(self):
        from trak_pipeline.compat import intra_summary_core as core
        rows, cells, _, _ = self._area_aggregation_fixture(include_invalid=True)
        nodes = core.summarize_cells(rows, cells, {'data_root': str(self.root)})
        br = next(n for n in nodes if n['level'] == 'BR' and n['key'] == 'WT/BR1')
        expected = {'WT_PROTEIN_BR1_T1', 'WT_PROTEIN_BR1_T2'}
        self.assertEqual({m['sample_id'] for m in br['members']}, expected)
        self.assertIn('WT_PROTEIN_BR1_T3', {m['sample_id'] for m in br['exclusions']})
        for group in ('On', 'Not-on'):
            for view in ('signed', 'area', 'peak'):
                self.assertEqual(set(br['groups'][group][view]['series']), expected)

    def test_completed_summary_area_correction_runs_no_cell_upstream(self):
        from contextlib import ExitStack
        from trak_pipeline import runner
        _, cells, records, samples = self._area_aggregation_fixture()
        def saved_cell(sample, artifacts, config):
            return dict(records[sample.id]), dict(cells[sample.id])
        with ExitStack() as stack:
            stack.enter_context(patch.object(summaries, '_load_cell', side_effect=saved_cell))
            for module, name in [(distances, 'execute'),
                                 (tardis, 'execute_decomposition'),
                                 (tardis, 'execute_bootstrap'),
                                 (statistics, 'execute'),
                                 (statistics, 'execute_c1_mean'),
                                 (runner, 'run_sample'), (runner, '_execute')]:
                stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError('Completed summary called cell upstream: ' + name)))
            result = summaries.summarize(self.config, samples)
        self.assertEqual(result['upstream_stages_run'], [])
        self.assertEqual(result['new_cells_started'], 0)
        manifest = json.loads(Path(result['manifest']).read_text())
        condition = next(n for n in manifest['nodes']
                         if n['level'] == 'condition' and n['key'] == 'WT')
        saved = json.loads((self.root / condition['data'] / 'intra_node.json').read_text())
        expected = np.zeros(300); expected[:3] = [25., 25., 50.]
        np.testing.assert_allclose(saved['groups']['On']['area']['mean'], expected,
                                   rtol=0, atol=1e-12)


    def test_area_aggregation_version_and_old_order_are_not_interchangeable(self):
        expected = dict(method_id='hierarchy_summary', method_version='2',
                        area_order='normalize_cells_then_equal_cell_mean',
                        condition_area_order='equal_BR_mean_of_corrected_area_means')
        configured = summaries._summary_parameters(dict(self.config, summary=expected))
        for key, value in expected.items():
            self.assertEqual(configured[key], value)
        for field, old_value in [('area_order', 'normalize_BR_signed_mean'),
                                 ('method_version', '1')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Unsupported hierarchy summary'):
                summaries._summary_parameters(dict(self.config, summary={field: old_value}))
        # Otherwise unchanged scientific inputs must not select a v1 summary cache.
        _, _, _, samples = self._area_aggregation_fixture()
        entries = [(sample, {'statistics': Artifact('stats_'+sample.id, 'statistics', {}),
                             'decomposition': Artifact('curve_'+sample.id, 'decomposition', {})})
                   for sample in samples]
        old_parameters = dict(configured, method_version='1',
                              area_order='normalize_BR_signed_mean')
        old_parameters.pop('condition_area_order')
        old_signature = summaries._summary_input_signature(
            self.config, entries, old_parameters, self.root)
        new_signature = summaries._summary_input_signature(
            self.config, entries, configured, self.root)
        self.assertIsNotNone(old_signature)
        self.assertNotEqual(new_signature, old_signature)
        for altered in (dict(configured, method_version='1'),
                        dict(configured, area_order='normalize_BR_signed_mean')):
            self.assertNotEqual(new_signature, summaries._summary_input_signature(
                self.config, entries, altered, self.root))



    def test_refresh_rejects_old_or_missing_area_method_without_execution(self):
        import sys, types
        from contextlib import ExitStack
        from unittest.mock import Mock
        expected = summaries._summary_parameters(self.config)
        manifest = self.root / 'refresh_guard_manifest.json'
        plotter = types.ModuleType('intra_summary_plots')
        plotter.render_node = Mock(side_effect=AssertionError('Rendered incompatible summary'))
        writer = types.ModuleType('export_reader_tables')
        writer.write_workbook = Mock(side_effect=AssertionError('Wrote incompatible summary'))
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.use = Mock()
        forbidden = AssertionError('Refresh invoked a scientific or cell-loading step')
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                'intra_summary_plots': plotter,
                'export_reader_tables': writer,
                'matplotlib': matplotlib}))
            metric_plot = stack.enter_context(patch.object(
                summaries, '_metric_plot', side_effect=AssertionError('Rendered incompatible metrics')))
            for module, name in [(summaries, 'summarize'), (summaries, '_load_cell'),
                                 (distances, 'execute'), (tardis, 'execute_decomposition'),
                                 (tardis, 'execute_bootstrap'), (statistics, 'execute'),
                                 (statistics, 'execute_c1_mean')]:
                stack.enter_context(patch.object(module, name, side_effect=forbidden))
            legacy = dict(expected, method_version='1',
                          area_order='normalize_BR_signed_mean')
            legacy.pop('condition_area_order')
            for method in (None, legacy,
                           dict(expected, area_order='normalize_BR_signed_mean')):
                saved = dict(summary_id='fixture_v1', nodes=[])
                if method is not None:
                    saved['method'] = method
                manifest.write_text(json.dumps(saved))
                with self.subTest(method=method), self.assertRaisesRegex(
                        ValueError, 'scientific method is missing or incompatible'):
                    summaries.refresh(self.config, manifest)
                self.assertFalse((manifest.parent / 'publication_report.json').exists())
            plotter.render_node.assert_not_called()
            writer.write_workbook.assert_not_called()
            metric_plot.assert_not_called()
            matplotlib.use.assert_not_called()
            # A valid v2 manifest remains a legal pure refresh even with no nodes.
            manifest.write_text(json.dumps(dict(
                summary_id='fixture_v2', nodes=[], method=expected)))
            result = summaries.refresh(self.config, manifest)
            self.assertEqual(result['summary_id'], 'fixture_v2')
            self.assertEqual(result['nodes'], [])
            self.assertEqual(result['scientific_stage_calls'], 0)
            self.assertEqual(result['new_cells_started'], 0)
            plotter.render_node.assert_not_called()
            writer.write_workbook.assert_not_called()
            metric_plot.assert_not_called()


if __name__=='__main__':unittest.main()
