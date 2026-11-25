"""Manual cells-in-ROI counts in the summary: per-cell values, workbook columns, input signature.
Small fixtures only; the ROI-level metrics and all science stay unchanged."""
from __future__ import annotations
import json,os,re,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from trak_pipeline.types import Sample
from trak_pipeline.stages import summaries

ROOT=Path(__file__).resolve().parents[1]


class CellCountTableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cell_counts_',dir=ROOT/'validation');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def write(self,text):
        path=self.root/'counts.csv';path.write_text(text,encoding='utf-8');return path
    def test_blank_is_unavailable_and_halves_are_kept(self):
        counts=summaries._load_cell_counts(self.write('sample_id,cells_in_ROI,source\nA,1,x\nB,0.5,x\nC,,x\nD,2,x\n'))
        self.assertEqual(counts,{'A':1.0,'B':0.5,'D':2.0})
    def test_duplicates_and_nonpositive_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):summaries._load_cell_counts(self.write('sample_id,cells_in_ROI\nA,1\nA,2\n'))
        with self.assertRaisesRegex(ValueError,'positive'):summaries._load_cell_counts(self.write('sample_id,cells_in_ROI\nA,0\n'))
        with self.assertRaisesRegex(ValueError,'columns'):summaries._load_cell_counts(self.write('sample_id,n\nA,1\n'))
    def test_missing_file_means_no_counts(self):
        self.assertEqual(summaries._load_cell_counts(self.root/'absent.csv'),{})
    def test_environment_override(self):
        path=self.write('sample_id,cells_in_ROI\nA,3\n')
        with patch.dict(os.environ,{summaries.CELL_COUNTS_ENV:str(path)}):self.assertEqual(summaries._load_cell_counts(),{'A':3.0})
    def test_per_cell_values_divide_roi_values_only(self):
        record=dict(sample_id='A',first_mask_ROI_area_um2=200.,mean_ROI_area_um2=100.,total_localizations=1000,on_percentage=40.)
        summaries._attach_cell_counts(record,{'A':2.0})
        self.assertEqual((record['cells_in_ROI'],record['first_mask_ROI_area_per_cell_um2'],record['mean_ROI_area_per_cell_um2'],record['localizations_per_cell']),(2.0,100.,50.,500.))
        self.assertEqual((record['first_mask_ROI_area_um2'],record['on_percentage']),(200.,40.))
        half=summaries._attach_cell_counts(dict(sample_id='B',first_mask_ROI_area_um2=200.,mean_ROI_area_um2=None,total_localizations=10),{'B':.5})
        # a partial cell counts as one: the ROI value is kept, never doubled
        self.assertEqual((half['cells_in_ROI'],half['cells_for_per_cell'],half['first_mask_ROI_area_per_cell_um2'],half['mean_ROI_area_per_cell_um2'],half['localizations_per_cell']),(.5,1,200.,None,10.))
        for n,divisor in ((1.,1),(1.5,2),(2.,2),(2.5,3),(3.,3)):
            r=summaries._attach_cell_counts(dict(sample_id='X',first_mask_ROI_area_um2=600.,mean_ROI_area_um2=600.,total_localizations=600),{'X':n})
            self.assertEqual((r['cells_for_per_cell'],r['first_mask_ROI_area_per_cell_um2']),(divisor,600./divisor))
        missing=summaries._attach_cell_counts(dict(sample_id='C',first_mask_ROI_area_um2=1.,mean_ROI_area_um2=1.,total_localizations=1),{})
        self.assertTrue(all(missing[k] is None for k in summaries.CELL_COUNT_FIELDS))


class CellCountSummaryTests(unittest.TestCase):
    """Two WT replicas x two cells from legacy-style saved products; counts 1, 2 (BR1) and 0.5, blank (BR2)."""
    COUNTS={'WT_PROTEIN_BR1_T1':'1','WT_PROTEIN_BR1_T2':'2','WT_PROTEIN_BR2_T1':'0.5','WT_PROTEIN_BR2_T2':''}
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cell_count_summary_',dir=ROOT/'validation');self.root=Path(self.temp.name)
        self.config=dict(data_root=str(self.root),control_root=str(self.root/'_batch_pipeline'),acquisition=dict(frames=50,pixel_size_nm=117,frame_interval_s=.02),methods={})
        self.samples=[];edges=np.linspace(0,3,301);x=(edges[:-1]+edges[1:])/2
        for br in ('BR1','BR2'):
            for cell,px in (('T1',100),('T2',300)):
                directory=self.root/f'WT/PROTEIN_WT_{br}/{cell}';old=directory/'legacy'
                for sub in ('masks','data/tardis/distribution/On','data/tardis/distribution/Not-on'):(old/sub).mkdir(parents=True)
                (old/'classification_summary.json').write_text(json.dumps(dict(On=40,Not_on=60,unknown=0,QC_excluded=0,selected=100,ROI_candidates=100)))
                pd.DataFrame([dict(original_frame_start=0,original_frame_stop_exclusive=50,mask_ROI_area_px2=px)]).to_csv(old/'masks/per_block_statistics.csv',index=False)
                for group in ('On','Not-on'):pd.DataFrame(dict(distance_left_um=edges[:-1],distance_right_um=edges[1:],distance_mid_um=x,intra_density_um_inv=np.exp(-x*10))).to_csv(old/'data/tardis/distribution'/group/'full.csv',index=False)
                pd.DataFrame([dict(group=g,scope='full',valid=True) for g in ('On','Not-on')]).to_csv(old/'data/tardis/per_scope_summary.csv',index=False)
                (old/'data/tardis/metadata.json').write_text(json.dumps(dict(config_rules={'tau_frames':1},scientific_fingerprint='same_method')))
                self.samples.append(Sample(f'WT_PROTEIN_{br}_{cell}',directory,dict(molecule='PROTEIN',genotype='WT',BR=br,cell=cell),directory/'C1.tif',directory/'C2.tif',directory/'cell.roi',old))
        self.counts=self.root/'cell_counts.csv';self.write_counts(self.COUNTS)
        self.env=patch.dict(os.environ,{summaries.CELL_COUNTS_ENV:str(self.counts)});self.env.start()
    def tearDown(self):self.env.stop();self.temp.cleanup()
    def write_counts(self,counts):
        self.counts.write_text('sample_id,cells_in_ROI\n'+''.join(f'{k},{v}\n' for k,v in counts.items()),encoding='utf-8')
    def node(self,result,level):
        manifest=json.loads(Path(result['manifest']).read_text());return next(n for n in manifest['nodes'] if n['level']==level)

    def test_cell_metrics_and_condition_statistics(self):
        result=summaries.summarize(self.config,self.samples);top=self.node(result,'comparison')
        cells=pd.read_csv(self.root/top['data']/'cell_metrics.csv').set_index('sample_id')
        area=117**2/1e6  # one pixel in um^2
        self.assertAlmostEqual(cells.loc['WT_PROTEIN_BR1_T2','first_mask_ROI_area_per_cell_um2'],300*area/2)
        self.assertAlmostEqual(cells.loc['WT_PROTEIN_BR2_T1','first_mask_ROI_area_per_cell_um2'],100*area)  # 0.5 cell -> divided by 1
        self.assertEqual(cells.loc['WT_PROTEIN_BR2_T1','cells_for_per_cell'],1)
        self.assertAlmostEqual(cells.loc['WT_PROTEIN_BR1_T2','localizations_per_cell'],50.)
        self.assertTrue(np.isnan(cells.loc['WT_PROTEIN_BR2_T2','cells_in_ROI']) and np.isnan(cells.loc['WT_PROTEIN_BR2_T2','first_mask_ROI_area_per_cell_um2']))
        self.assertAlmostEqual(cells.loc['WT_PROTEIN_BR1_T2','first_mask_ROI_area_um2'],300*area)  # ROI value unchanged
        stats=pd.read_csv(self.root/top['data']/'summary_statistics.csv')
        row=stats[(stats.metric=='first_mask_ROI_area_per_cell_um2')&(stats.condition=='WT')].iloc[0]
        br1=(100*area+150*area)/2;br2=100*area  # BR2 has one counted cell (0.5 -> 1)
        self.assertAlmostEqual(row['mean'],(br1+br2)/2);self.assertEqual(int(row['n_cells_valid']),3)
        base=stats[(stats.metric=='first_mask_ROI_area_um2')&(stats.condition=='WT')].iloc[0];self.assertEqual(int(base['n_cells_valid']),4)

    def test_workbook_columns_formats_and_tables(self):
        from export_reader_tables import verify_workbook
        result=summaries.summarize(self.config,self.samples);summaries.refresh(self.config,result['manifest'])
        for level,sheet_name,expected in (('comparison','Condition Summary',['ROIs with cell count','ROIs with >1 cell','First-mask area per cell mean (µm²)']),('comparison','Cells',['Cells in ROI (manual count)','Cells used for per-cell values','Localizations per counted cell']),('cell','Cell Summary',['Cells in ROI (manual count)'])):
            node=self.node(result,level) if level!='cell' else next(n for n in json.loads(Path(result['manifest']).read_text())['nodes'] if n['key']=='WT_PROTEIN_BR1_T2')
            job=json.loads((self.root/node['workbook_job']).read_text());sheet=next(s for s in job['sheets'] if s['name']==sheet_name)
            for header in expected:self.assertIn(header,sheet['headers'])
            workbook=self.root/node['directory']/'Analysis_Figures_Cell_Metrics'/'Cell_Metrics.xlsx';verify_workbook(workbook,job)
            with zipfile.ZipFile(workbook) as archive:
                for name in archive.namelist():
                    if name.startswith('xl/tables/'):
                        xml=archive.read(name).decode();count=int(re.search(r'tableColumns count="(\d+)"',xml).group(1))
                        self.assertEqual(count,len(re.findall(r'<(?:\w+:)?tableColumn ',xml)))
                        widths={len(s['headers']) for s in job['sheets']};self.assertIn(count,widths)
        cond=next(s for s in json.loads((self.root/self.node(result,'comparison')['workbook_job']).read_text())['sheets'] if s['name']=='Condition Summary')
        row=dict(zip(cond['headers'],cond['rows'][0]));self.assertEqual((row['ROIs with cell count'],row['ROIs with >1 cell']),(3,1))

    def test_changed_counts_change_the_saved_summary(self):
        first=summaries.summarize(self.config,self.samples);before=self.node(first,'comparison')['numeric_signature']  # manifest file is rewritten below
        self.write_counts(dict(self.COUNTS,WT_PROTEIN_BR2_T2='3'));changed=summaries.summarize(self.config,self.samples)
        cells=pd.read_csv(self.root/self.node(changed,'comparison')['data']/'cell_metrics.csv').set_index('sample_id');self.assertEqual(cells.loc['WT_PROTEIN_BR2_T2','cells_in_ROI'],3)
        self.assertEqual(first['summary_id'],changed['summary_id'])  # same summary method; only the numeric files change
        self.assertNotEqual(before,self.node(changed,'comparison')['numeric_signature'])

    def test_fast_path_signature_includes_counts(self):
        from trak_pipeline.types import Artifact
        entries=[(s,dict(statistics=Artifact('st_'+s.id,'statistics',{}),decomposition=Artifact('de_'+s.id,'decomposition',{}))) for s in self.samples]
        with patch.object(summaries,'_expected_inventory',return_value=[]):
            sign=lambda counts:summaries._summary_input_signature(self.config,entries,{},self.root.resolve(),counts)
            base=sign({'WT_PROTEIN_BR1_T1':1.0})
            self.assertEqual(base,sign({'WT_PROTEIN_BR1_T1':1.0,'OTHER_DATASET_T1':2.0}))  # other datasets' counts do not matter
            self.assertNotEqual(base,sign({'WT_PROTEIN_BR1_T1':2.0}));self.assertNotEqual(base,sign({}))
            with patch.object(summaries,'_summary_code_digest',return_value='other summary code'):
                self.assertNotEqual(base,sign({'WT_PROTEIN_BR1_T1':1.0}))  # records made by older summary code are not reused

    def test_without_count_file_columns_are_blank(self):
        self.counts.unlink();result=summaries.summarize(self.config,self.samples)
        cells=pd.read_csv(self.root/self.node(result,'comparison')['data']/'cell_metrics.csv')
        self.assertTrue(cells.cells_in_ROI.isna().all() and cells.first_mask_ROI_area_per_cell_um2.isna().all());self.assertTrue(cells.first_mask_ROI_area_um2.notna().all())


if __name__=='__main__':unittest.main()
