"""Five-replica summary fixtures only; no biological cell science is run."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np
from trak_pipeline.compat import intra_summary_core as core
from trak_pipeline.compat import intra_summary_plots as plots
from trak_pipeline.stages import summaries

ROOT = Path(__file__).resolve().parents[1]
REPLICAS = ['BR1', 'BR2', 'BR3', 'BR4', 'BR5']

class FiveReplicaSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='five_br_', dir=ROOT/'validation')
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def fixture(self):
        a=np.zeros(300);a[:3]=[150.,-50.,0.]
        b=np.zeros(300);b[:3]=[0.,150.,-50.]
        c=np.zeros(300);c[:3]=[-25.,0.,125.]
        definitions=[[(a,.2),(b,.8)],[(c,.4)],[(a,.9)],[(b,.25)],[(c,.6),(a,.2),(b,.1)]]
        percentages=[[20.,80.],[30.],[40.],[50.],[10.,20.,90.]]
        rows=[];cells={};metrics=[]
        for condition in core.CONDITIONS:
            for i,br in enumerate(REPLICAS):
                for j,(shape,scale) in enumerate(definitions[i],1):
                    sid=f'{condition}_P_{br}_T{j}';rel=f'{condition}/P_{condition}_{br}/T{j}'
                    row=dict(sample_id=sid,genotype=condition,BR=br,cell=f'T{j}',molecule='P',source_directory_relative=rel)
                    rows.append(row);raw=scale*shape
                    cells[sid]=dict(sample_id=sid,status='complete',eligible=True,reasons=[],curves={'On':raw,'Not-on':np.roll(raw,5)},source_refs=[])
                    value=percentages[i][j-1]
                    metrics.append(dict(row,on_percentage=value,first_mask_ROI_area_um2=2*value,mean_ROI_area_um2=3*value))
        return rows,cells,metrics
    def nodes(self,rows,cells):
        return core.summarize_cells(rows,cells,dict(data_root=str(self.root),replicas=REPLICAS))
    def test_fifteen_BR_nodes_and_complete_roster(self):
        rows,cells,_=self.fixture();nodes=self.nodes(rows,cells)
        self.assertEqual(len(nodes),19)
        self.assertEqual(sum(n['level']=='BR' for n in nodes),15)
        top=next(n for n in nodes if n['level']=='comparison')
        self.assertEqual(top['progress']['expected_BRs'],15)
        self.assertEqual(top['progress']['valid_BRs'],15)
        self.assertEqual(top['progress']['status'],'COMPLETE')
        for condition in core.CONDITIONS:
            n=next(n for n in nodes if n['level']=='condition' and n['key']==condition)
            self.assertEqual(n['progress']['expected_BRs'],5)
            for group in core.GROUPS:
                for view in core.VIEWS:self.assertEqual(list(n['groups'][group][view]['series']),REPLICAS)
    def test_cell_first_area_then_five_equal_BR_means(self):
        rows,cells,_=self.fixture();nodes=self.nodes(rows,cells)
        br=next(n for n in nodes if n['key']=='WT/BR1')
        expected=np.zeros(300);expected[:3]=[75.,50.,-25.]
        np.testing.assert_allclose(br['groups']['On']['area']['mean'],expected,rtol=0,atol=1e-12)
        self.assertLess(br['groups']['On']['area']['mean'][2],0.)
        n=next(n for n in nodes if n['level']=='condition' and n['key']=='WT')
        expected=np.zeros(300);expected[:3]=[145./3.,110./3.,15.]
        p=n['groups']['On']['area']
        np.testing.assert_allclose(p['mean'],expected,rtol=0,atol=1e-12)
        self.assertAlmostEqual(float(p['mean']@core.WIDTHS),1.,places=12)
        self.assertEqual(p['n'],5)
        refs=np.stack([p['series'][br] for br in REPLICAS])
        np.testing.assert_allclose(p['sd'],np.std(refs,axis=0,ddof=1),rtol=0,atol=1e-12)
        np.testing.assert_allclose(p['sem'],p['sd']/np.sqrt(5),rtol=0,atol=1e-12)
        raw=[cells[r['sample_id']]['curves']['On'] for r in rows if r['genotype']=='WT']
        pooled=np.mean([v/(v@core.WIDTHS) for v in raw],axis=0)
        self.assertFalse(np.allclose(p['mean'],pooled),'BR cell counts must not weight condition means')
        raw_br=np.stack(raw[:2])
        np.testing.assert_allclose(br['groups']['On']['signed']['mean'],raw_br.mean(0))
        np.testing.assert_allclose(br['groups']['On']['peak']['mean'],(raw_br/raw_br.max(axis=1)[:,None]).mean(0))
    def test_pending_and_failed_cells_never_become_zero_curves(self):
        rows,cells,_=self.fixture();cells['WT_P_BR4_T1']['status']='pending';cells['WT_P_BR5_T3']['status']='failed'
        nodes=self.nodes(rows,cells);n=next(n for n in nodes if n['level']=='condition' and n['key']=='WT')
        self.assertEqual(n['progress']['status'],'PROGRESS');self.assertEqual(n['progress']['expected_BRs'],5)
        self.assertEqual(n['progress']['valid_BRs'],4);self.assertEqual(n['progress']['failed'],1)
        for group in core.GROUPS:
            for view in core.VIEWS:self.assertNotIn('BR4',n['groups'][group][view]['series'])
        cells['WT_P_BR4_T1']['status']='complete'
        n=next(n for n in self.nodes(rows,cells) if n['level']=='comparison')
        self.assertEqual(n['progress']['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
    def test_declared_roster_is_validated_and_bound(self):
        implicit=summaries._summary_parameters({});self.assertNotIn('replica_order',implicit)
        explicit=summaries._summary_parameters(dict(replicas=REPLICAS))
        self.assertEqual(explicit['replica_order'],REPLICAS)
        for bad in [[],['BR1','BR1'],['BR1',[]],['replica1']]:
            with self.assertRaises(ValueError):summaries._summary_parameters(dict(replicas=bad))
        rows,cells,_=self.fixture()
        with self.assertRaisesRegex(core.SummarySourceError,'roster differs'):
            core.summarize_cells(rows,cells,dict(data_root=str(self.root),replicas=REPLICAS[:3]))
        config=dict(_expected_samples=rows,replicas=REPLICAS)
        a=summaries._summary_input_signature(config,[],explicit,self.root)
        other=dict(replicas=list(reversed(REPLICAS)),_expected_samples=rows)
        b=summaries._summary_input_signature(other,[],summaries._summary_parameters(other),self.root)
        self.assertNotEqual(a,b)
    def test_scalar_condition_mean_is_equal_BR_not_pooled_cells(self):
        _,_,records=self.fixture();nodes=summaries._metric_nodes(records,self.root)
        node=next(n for n in nodes if n['level']=='condition' and n['key']=='WT')
        row=next(r for r in node['summary_records'] if r['metric']=='on_percentage')
        self.assertEqual(row['mean'],42.);self.assertEqual(row['n'],5);self.assertEqual(row['n_cells_valid'],8)
        self.assertNotEqual(row['mean'],42.5)
        br=next(n for n in nodes if n['level']=='BR' and n['key']=='WT/BR1')
        self.assertEqual(next(r for r in br['summary_records'] if r['metric']=='on_percentage')['mean'],50.)
    def test_final_failure_override_is_terminal_but_committed_science_wins(self):
        rows,cells,_=self.fixture();sid='WT_P_BR4_T1';kept=cells.pop(sid)
        updated=summaries._apply_failed_samples(dict(summary_failed_sample_ids=[sid]),rows,cells)
        self.assertEqual(updated[sid]['status'],'failed');self.assertFalse(updated[sid]['eligible'])
        n=next(n for n in self.nodes(rows,updated) if n['level']=='comparison')
        self.assertEqual(n['progress']['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
        cells[sid]=kept
        self.assertEqual(summaries._apply_failed_samples(dict(summary_failed_sample_ids=[sid]),rows,cells)[sid]['status'],'complete')
        with self.assertRaisesRegex(ValueError,'outside'):
            summaries._apply_failed_samples(dict(summary_failed_sample_ids=['not_expected']),rows,cells)
    def test_refresh_missing_science_calls_zero_upstream_steps(self):
        from trak_pipeline import runner
        from trak_pipeline.stages import distances,tardis,statistics
        config=dict(data_root=str(self.root),control_root=str(self.root/'control'),replicas=REPLICAS)
        manifest=self.root/'saved/summary_manifest.json';manifest.parent.mkdir()
        manifest.write_text(json.dumps(dict(summary_id='five_fixture',method=summaries._summary_parameters(config),nodes=[dict(key='WT',data='missing_data',workbook_job='missing_workbook.json',intra=True)])))
        with ExitStack() as stack:
            for module,name in [(summaries,'_load_cell'),(runner,'run_sample'),(distances,'execute'),(tardis,'execute_decomposition'),(tardis,'execute_bootstrap'),(statistics,'execute')]:
                stack.enter_context(patch.object(module,name,side_effect=AssertionError('refresh launched science')))
            with self.assertRaisesRegex(FileNotFoundError,'no science was run'):summaries.refresh(config,manifest)
        report=json.loads((manifest.parent/'publication_report.json').read_text())
        self.assertEqual(report['scientific_stage_calls'],0);self.assertEqual(report['new_cells_started'],0)
    def test_five_BR_plots_keep_full_zoom_and_group_y_limits(self):
        rows,cells,_=self.fixture();n=next(n for n in self.nodes(rows,cells) if n['level']=='condition' and n['key']=='WT')
        captured=[]
        def capture(fig,path):
            captured.append((Path(path).name,[ax.get_ylim() for ax in fig.axes],fig.axes[0].get_legend_handles_labels()[1]))
        with patch.object(plots,'_atomic_figure',side_effect=capture):manifest=plots.render_node(n,self.root/'figures')
        self.assertEqual(len(captured),6)
        for view in core.VIEWS:
            pair=[r for r in manifest['plots'] if r['view']==view]
            self.assertEqual(pair[0]['ylim'],pair[1]['ylim'])
            self.assertEqual(pair[0]['series_ids']['On'],REPLICAS)
            self.assertEqual(pair[0]['series_ids']['Not-on'],REPLICAS)
        for _,limits,labels in captured:
            self.assertEqual(limits[0],limits[1]);self.assertTrue(set(REPLICAS)<=set(labels))

if __name__=='__main__':unittest.main()
