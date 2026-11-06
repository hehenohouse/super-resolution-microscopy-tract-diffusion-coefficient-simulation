"""Saved WT-only control summaries; no cell-level scientific computation."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from trak_pipeline.compat import intra_summary_core as core
from trak_pipeline.identity import digest
from trak_pipeline.stages import summaries
from trak_pipeline.types import Sample, Artifact

ROOT=Path(__file__).resolve().parents[1]


class ConditionSubsetSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='condition_subset_',dir=ROOT/'validation')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def config(self):
        return dict(data_root=str(self.root),control_root=str(self.root/'control'),molecule='HaloTag_only',
                    conditions=['WT'],replicas=['BR1','BR2'],condition_directories={'WT':'NegativeControl/WT'},
                    acquisition=dict(pixel_size_nm=117))

    def fixture(self):
        rows=[];samples=[];artifacts={};edges=np.linspace(0,3,301)
        raw=np.zeros(300);raw[:3]=[150.,-50.,0.]
        other=np.zeros(300);other[:3]=[0.,150.,-50.]
        for br,count in [('BR1',2),('BR2',10)]:
            for i in range(1,count+1):
                sid=f'WT_HaloTagOnly_{br}_T{i}';directory=self.root/f'NegativeControl/WT/HaloOnly_{br}/T{i}'
                meta=dict(genotype='WT',BR=br,cell=f'T{i}',molecule='HaloTag_only',acquisition_date='2026-09-30')
                row=dict(meta,sample_id=sid,source_directory_relative=directory.relative_to(self.root).as_posix());rows.append(row)
                if br=='BR2':continue
                directory.mkdir(parents=True)
                sample=Sample(sid,directory,meta,directory/'C1.tif',directory/'C2.tif',directory/'roi.roi');samples.append(sample)
                decomp=directory/'saved/decomposition';decomp.mkdir(parents=True)
                for group in ('On','Not-on'):
                    target=decomp/'distribution'/group;target.mkdir(parents=True)
                    curve=(raw*.2 if i==1 else other*.8)
                    if group=='Not-on':curve=np.roll(curve,4)
                    pd.DataFrame(dict(distance_left_um=edges[:-1],distance_right_um=edges[1:],intra_density_um_inv=curve)).to_csv(target/'full.csv',index=False)
                pd.DataFrame([dict(group=g,scope='full',valid=True) for g in ('On','Not-on')]).to_csv(decomp/'per_scope_summary.csv',index=False)
                (decomp/'metadata.json').write_text(json.dumps(dict(config_rules=dict(tau_frames=1))))
                on=40 if i==1 else 80
                metrics=dict(counts=dict(On=on,Not_on=100-on,unknown=0,QC_excluded=5,selected=100,ROI_candidates=105),
                             first_mask_ROI_area_um2=100. if i==1 else 300.,mean_ROI_area_um2=150. if i==1 else 450.)
                artifacts[sid]=dict(statistics=Artifact('stats_'+sid,'statistics',{},dict(metrics=metrics)),
                    decomposition=Artifact('decomp_'+sid,'decomposition',dict(directory=str(decomp),metadata=str(decomp/'metadata.json'))))
        config=self.config();config['_expected_samples']=rows
        return config,rows,samples,artifacts

    def test_explicit_subset_identity_and_historical_full_roster(self):
        original=dict(method_id='hierarchy_summary',method_version='2',cell_weighting='equal',condition_weighting='equal_replica',BR_weighting='equal',
            count_main_summary='pooled_per_cell',count_additional_mean='equal_replica',on_percentage_denominator='On+Not_on',
            intra_views=['signed','signed_area','peak'],intra_groups=['On','Not-on'],signed_area_min_abs=.01,minimum_signed_area_abs=.01,
            require_positive_peak=True,require_both_groups=True,area_order='normalize_cells_then_equal_cell_mean',
            condition_area_order='equal_BR_mean_of_corrected_area_means',peak_within_BR='mean_cell_own_positive_peak_normalized',negative_bins_retained=True,SD_ddof=1)
        self.assertEqual(summaries._summary_parameters({}),original)
        self.assertEqual(summaries._summary_parameters(dict(conditions=['WT','TDK','MDK'])),original)
        wt=summaries._summary_parameters(dict(conditions=['WT']))
        self.assertEqual(wt,dict(original,condition_order=['WT']))
        self.assertNotEqual(digest(wt),digest(original))
        pair=summaries._summary_parameters(dict(conditions=['MDK','WT']))
        self.assertEqual(pair['condition_order'],['MDK','WT'])
        with self.assertRaises(ValueError):summaries._summary_parameters(dict(conditions=['MDK','TDK','WT']))

    def test_invalid_condition_rosters_rejected(self):
        for value in ([],['WT','WT'],['WT','KO'],['WT',[]],'WT',None):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'conditions'):
                summaries._summary_parameters(dict(conditions=value))

    def test_expected_condition_and_control_molecule_are_not_mixed(self):
        config,rows,_,_=self.fixture()
        for change in (dict(genotype='TDK'),dict(molecule='TRAK1')):
            invalid=dict(config,_expected_samples=[dict(rows[0],**change)])
            with self.subTest(change=change),self.assertRaises(ValueError):
                summaries._expected_inventory(invalid,[],self.root)
        with self.assertRaisesRegex(ValueError,'Completed cell molecule'):
            summaries._expected_inventory(config,[dict(rows[0],molecule='TRAK1')],self.root)

    def test_explicit_output_location_is_portable_and_not_scientific_identity(self):
        config=self.config();method=summaries._summary_parameters(config)
        alternate=dict(config,condition_directories={'WT':'another/WT'})
        self.assertEqual(summaries._summary_parameters(alternate),method)
        a=summaries._summary_input_signature(config,[],method,self.root)
        b=summaries._summary_input_signature(alternate,[],method,self.root)
        self.assertNotEqual(a,b,'Output relocation must update the manifest, without invalidating scientific method IDs')
        for value in ({'WT':'../escape'},{'TDK':'TDK'},{'WT':''},[]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                summaries._condition_directories(dict(config,condition_directories=value),self.root)

    def test_full_roster_kernel_output_is_unchanged(self):
        config=dict(data_root=str(self.root));rows=[];cells={}
        for condition in core.CONDITIONS:
            sid=condition+'_P_BR1_T1';rows.append(dict(sample_id=sid,genotype=condition,BR='BR1',cell='T1',source_directory_relative=f'{condition}/P_BR1/T1'))
            curve=np.zeros(300);curve[:2]=[150.,-50.]
            cells[sid]=dict(status='complete',eligible=True,reasons=[],curves={g:curve.copy() for g in core.GROUPS})
        expected=core.summarize_cells(rows,cells,config)
        actual=summaries._condition_summary_nodes(core,rows,cells,config)
        self.assertEqual(digest(summaries._plain(expected)),digest(summaries._plain(actual)))

    def test_wt_only_saved_fixture_publishes_pending_BR_and_reader_workbooks(self):
        config,rows,samples,artifacts=self.fixture()
        result=summaries.summarize(config,samples,artifacts)
        self.assertEqual(result['upstream_stages_run'],[]);self.assertEqual(result['new_cells_started'],0)
        manifest=json.loads(Path(result['manifest']).read_text());nodes=manifest['nodes']
        self.assertEqual(len(nodes),6)
        self.assertEqual({n['key'] for n in nodes if n['level']=='condition'},{'WT'})
        self.assertEqual(next(n for n in nodes if n['level']=='comparison')['key'],'WT')
        condition=next(n for n in nodes if n['level']=='condition')
        self.assertEqual(condition['directory'],'NegativeControl/WT')
        condition_data=self.root/condition['data']
        node=json.loads((condition_data/'intra_node.json').read_text())
        self.assertEqual(node['molecule'],'HaloTag_only')
        self.assertEqual(node['progress']['expected'],12);self.assertEqual(node['progress']['completed'],2)
        self.assertEqual(node['progress']['expected_BRs'],2);self.assertEqual(node['progress']['valid_BRs'],1)
        self.assertEqual(node['progress']['status'],'PROGRESS')
        br2=next(n for n in nodes if n['key']=='WT/BR2')
        pending=json.loads((self.root/br2['data']/'intra_node.json').read_text())
        self.assertEqual(pending['progress']['expected'],10);self.assertFalse(pending['available'])
        self.assertIsNone(pending['groups']['On']['signed']['mean'])
        for group in ('On','Not-on'):
            for view in ('signed','area','peak'):
                self.assertEqual(list(node['groups'][group][view]['series']),['BR1'])
        expected_area=np.zeros(300);expected_area[:3]=[75.,50.,-25.]
        np.testing.assert_allclose(node['groups']['On']['area']['mean'],expected_area,rtol=0,atol=1e-12)
        scalar=pd.read_csv(condition_data/'summary_statistics.csv').set_index('metric')
        self.assertEqual(scalar.loc['on_percentage','mean'],60.)
        self.assertEqual(scalar.loc['on_percentage','n'],1)
        self.assertTrue(np.isnan(scalar.loc['on_percentage','sd']))
        br_scalar=pd.read_csv(self.root/next(n for n in nodes if n['key']=='WT/BR1')['data']/'BR_statistics.csv').set_index('metric')
        self.assertAlmostEqual(br_scalar.loc['on_percentage','sd'],np.std([40.,80.],ddof=1))
        self.assertEqual(scalar.loc['first_mask_ROI_area_um2','mean'],200.)
        self.assertEqual(scalar.loc['mean_ROI_area_um2','mean'],300.)
        self.assertEqual(scalar.loc['on_percentage','n_cells_valid'],2)
        job=json.loads((condition_data/'reader_workbook_job.json').read_text())
        self.assertTrue(all('HaloTag_only' in sheet['title'] for sheet in job['sheets']))
        cell_sheet=next(sheet for sheet in job['sheets'] if sheet['name']=='Cells')
        self.assertEqual(len(cell_sheet['rows']),2)
        self.assertIn('On count',cell_sheet['headers']);self.assertIn('Not-on count',cell_sheet['headers'])
        self.assertTrue(all(row[5]==100 and row[6]+row[7]==100 for row in cell_sheet['rows']))
        with ExitStack() as stack:
            for name in ('_load_cell','_metric_nodes','_condition_summary_nodes'):
                stack.enter_context(patch.object(summaries,name,side_effect=AssertionError('refresh ran scientific summary')))
            report=summaries.refresh(config,result['manifest'])
        self.assertEqual(report['scientific_stage_calls'],0);self.assertEqual(report['new_cells_started'],0)
        self.assertFalse(report['missing']);self.assertFalse((self.root/'WT').exists())
        figures=self.root/'NegativeControl/WT/Analysis_Figures_Intra'
        for view in ('signed','area','peak'):
            for extent in ('full','zoom'):
                self.assertTrue((figures/f'Intra_{view}_{extent}.png').is_file())
                self.assertTrue((figures/'On_NotOn_Overlay'/f'Intra_On_NotOn_{view}_{extent}.png').is_file())
        self.assertTrue((figures/'On_NotOn_Overlay/INDEX.html').is_file())
        for row in nodes:
            workbook=self.root/row['directory']/'Analysis_Figures_Cell_Metrics/Cell_Metrics.xlsx'
            with zipfile.ZipFile(workbook) as archive:
                xml=archive.read('xl/workbook.xml').decode()
                self.assertIn('Cells' if row['level']!='cell' else 'Cell Summary',xml)
        self.assertFalse(any('TDK' in p.parts or 'MDK' in p.parts for p in self.root.rglob('*')))


if __name__=='__main__':unittest.main()
