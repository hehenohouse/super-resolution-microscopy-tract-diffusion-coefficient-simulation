"""Top-level reader pages are rebuilt from the committed modular summary, with no science.

Regression for 2026-11-09: INTRA_RESULTS.html, CELL_METRICS_RESULTS.html and CELL_METRICS_TABLES.html
still showed the BR1-BR3 summaries after the five-replica summary was published.
"""
import json,tempfile,unittest
from pathlib import Path
from trak_pipeline.stages import overview_pages

PACKAGE=Path(__file__).resolve().parents[1]
SUMMARY='abc123'


class OverviewPagesTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PACKAGE/'validation');self.root=Path(self.temp.name).resolve()/'Xinran_TEST'
        nodes=[self.node('comparison','WT_TDK_MDK','.',15,186)]
        for condition in ('MDK','WT'):
            nodes.append(self.node('condition',condition,condition,5,60))
            for br in (5,1,4,2,3):nodes.append(self.node('BR',condition+'/BR'+str(br),condition+'/BR'+str(br),1,12))
        nodes.append(self.node('cell','WT_TEST_BR5_T1','WT/BR5/T1',None,None))
        control=self.root/'_batch_pipeline/batches/b';(control/'modular_summaries').mkdir(parents=True)
        (control/'modular_summaries/summary_manifest.json').write_text(json.dumps(dict(summary_id=SUMMARY,nodes=nodes)))
        self.config=dict(data_root=str(self.root),control_root=str(control),molecule='TRAKX')
        # old overlay manifest, a current one, and macOS AppleDouble files listed in an INDEX page
        overlay=self.root/'WT/Analysis_Figures_Intra/On_NotOn_Overlay';overlay.mkdir(parents=True,exist_ok=True)
        (overlay/'overlay_manifest.json').write_text(json.dumps(dict(summary_id='v2_old')))
        current=self.root/'MDK/Analysis_Figures_Intra/On_NotOn_Overlay';current.mkdir(parents=True,exist_ok=True)
        (current/'overlay_manifest.json').write_text(json.dumps(dict(summary_id=SUMMARY)))
        figures=self.root/'Analysis_Figures_Intra';figures.mkdir(parents=True,exist_ok=True)
        (figures/'._Intra_peak_full.png').write_bytes(b'\0'*16);(figures/'Intra_peak_full.png').write_bytes(b'png')
        (figures/'INDEX.html').write_text('<ul><li><a href="._Intra_peak_full.png">._Intra_peak_full.png</a><br><img src="._Intra_peak_full.png" width="700"></li>'
                                          '<li><a href="Intra_peak_full.png">Intra_peak_full.png</a><br><img src="Intra_peak_full.png" width="700"></li></ul>')
    def tearDown(self):self.temp.cleanup()

    def node(self,level,key,directory,brs,cells):
        data=self.root/directory/('analysis_results/modular_summary_'+SUMMARY)
        data.mkdir(parents=True,exist_ok=True)
        intra=level!='cell'
        if intra:
            (data/'intra_node.json').write_text(json.dumps(dict(progress=dict(completed=cells,expected=cells,included=cells,status='COMPLETE',valid_BRs=brs,expected_BRs=brs))))
            (data/'intra_curves_long.csv.gz').write_bytes(b'x')
        if level=='condition':
            (data/'summary_statistics.csv').write_text('condition,metric,n,mean,sd,sem,n_cells_valid,n_BRs_valid\n'+key+',on_percentage,5,55.8235,5.79,2.59,64,5\n')
        return dict(level=level,key=key,directory=directory,data=(Path(directory)/('analysis_results/modular_summary_'+SUMMARY)).as_posix(),intra=intra)

    def test_pages_list_five_replicas_and_link_current_curves(self):
        report=overview_pages.write_overview_pages(self.config)
        self.assertEqual(report['scientific_stage_calls'],0);self.assertEqual(report['summary_groups'],13);self.assertEqual(report['cells'],1)
        intra=(self.root/'INTRA_RESULTS.html').read_text(encoding='utf-8')
        for br in range(1,6):self.assertIn('WT/BR'+str(br),intra)
        self.assertLess(intra.index('WT/BR1'),intra.index('WT/BR2'));self.assertLess(intra.index('WT/BR4'),intra.index('WT/BR5'))
        self.assertLess(intra.index('WT_TDK_MDK'),intra.index('>MDK<'));self.assertLess(intra.index('>WT<'),intra.index('>MDK<'))
        self.assertIn('<td>5/5</td>',intra);self.assertIn('<td>15/15</td>',intra)
        self.assertIn('modular_summary_'+SUMMARY+'/intra_curves_long.csv.gz',intra);self.assertNotIn('intra_summary_v2',intra.split('Older folders')[0])
        metrics=(self.root/'CELL_METRICS_RESULTS.html').read_text(encoding='utf-8')
        self.assertIn('55.82',metrics);self.assertIn('On percentage (%)',metrics)
        tables=(self.root/'CELL_METRICS_TABLES.html').read_text(encoding='utf-8')
        self.assertIn('WT_TEST_BR5_T1',tables);self.assertIn('Excel workbook (missing)',tables)

    def test_stale_overlay_manifest_and_appledouble_files_are_removed(self):
        overview_pages.write_overview_pages(self.config)
        self.assertFalse((self.root/'WT/Analysis_Figures_Intra/On_NotOn_Overlay/overlay_manifest.json').exists())
        self.assertTrue((self.root/'MDK/Analysis_Figures_Intra/On_NotOn_Overlay/overlay_manifest.json').exists())
        figures=self.root/'Analysis_Figures_Intra'
        self.assertFalse((figures/'._Intra_peak_full.png').exists());self.assertTrue((figures/'Intra_peak_full.png').exists())
        index=(figures/'INDEX.html').read_text();self.assertNotIn('._',index);self.assertIn('Intra_peak_full.png',index)

    def test_missing_manifest_raises(self):
        (self.root/'_batch_pipeline/batches/b/modular_summaries/summary_manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):overview_pages.write_overview_pages(self.config)


if __name__=='__main__':unittest.main()
