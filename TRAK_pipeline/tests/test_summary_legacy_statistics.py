"""Saved scalar-schema regressions; no detection, segmentation or TARDIS."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from trak_pipeline.types import Sample,Artifact
from trak_pipeline.stages import summaries

ROOT=Path(__file__).resolve().parents[1]

class SavedScalarSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='scalar_schema_',dir=ROOT/'validation');self.root=Path(self.tmp.name);self.cell=self.root/'WT/P_BR1/T1';self.cell.mkdir(parents=True)
        self.sample=Sample('WT_P_BR1_T1',self.cell,dict(genotype='WT',BR='BR1',cell='T1',molecule='P'),self.cell/'C1.tif',self.cell/'C2.tif',self.cell/'roi.roi')
        self.config=dict(data_root=str(self.root),acquisition=dict(pixel_size_nm=117))
        self.counts=dict(On=20,Not_on=30,unknown=2,selected=52,QC_excluded=3,ROI_candidates=55)
        self.areas=dict(first_mask_ROI_area_um2=1.5,mean_ROI_area_um2=1.75)
    def tearDown(self):self.tmp.cleanup()
    def stats(self,metric,on_disk=False):
        if on_disk:
            p=self.root/'stats.json';p.write_text(json.dumps(dict(metrics=metric)));return Artifact('s','statistics',dict(metadata=str(p)))
        return Artifact('s','statistics',{},dict(metrics=metric))
    def load(self,metric,assignment=None,extra=None,on_disk=False):
        artifacts=dict(statistics=self.stats(metric,on_disk));artifacts.update(extra or {})
        if assignment is not None:artifacts['assignment']=assignment
        return summaries._load_scalar_metrics(self.sample,artifacts,self.config)
    def test_legacy_flat_adapter_metrics(self):
        metric=dict(self.counts,on_percentage=40.,**self.areas,source_table='old/cell_metrics.csv')
        value=self.load(metric)
        self.assertEqual({k:value[k] for k in self.counts},self.counts);self.assertEqual(value['on_percentage'],40.);self.assertEqual(value['mean_ROI_area_um2'],1.75)
    def test_modern_nested_metrics_on_disk(self):
        value=self.load(dict(counts=self.counts,on_percentage=40.,**self.areas),on_disk=True)
        self.assertEqual(value,dict(self.counts,on_percentage=40.,**self.areas))
    def test_counts_and_area_sources_are_independent(self):
        p=self.root/'classification.json';p.write_text(json.dumps(dict(self.counts)))
        art=Artifact('a','assignment',dict(summary=str(p)))
        value=self.load(self.areas,art);self.assertEqual(value['on_percentage'],40.)
        self.assertEqual(value['unknown'],2);self.assertEqual(value['QC_excluded'],3)
    def test_assignment_nested_summary_and_metadata_agree(self):
        p=self.root/'classification.json';p.write_text(json.dumps(dict(counts=self.counts)))
        art=Artifact('a','assignment',dict(summary=str(p)),dict(counts=dict(self.counts)))
        self.assertEqual(self.load(self.areas,art)['selected'],52)
    def test_conflicting_counts_are_rejected(self):
        other=dict(self.counts,On=25,Not_on=25)
        art=Artifact('a','assignment',{},dict(counts=other))
        with self.assertRaisesRegex(ValueError,'Conflicting saved localization counts'):
            self.load(dict(self.counts,**self.areas),art)
    def test_nonconserving_or_fractional_counts_are_rejected(self):
        for change in (dict(selected=51),dict(QC_excluded=4),dict(On=20.5),dict(On=-1)):
            with self.assertRaisesRegex(ValueError,'count'):
                self.load(dict(counts=dict(self.counts,**change),**self.areas))
    def test_partial_declared_nested_counts_do_not_get_guessed(self):
        with self.assertRaisesRegex(ValueError,'Missing saved count selected'):
            self.load(dict(counts={k:v for k,v in self.counts.items() if k!='selected'},**self.areas))
    def test_missing_saved_counts_is_informative(self):
        with self.assertRaisesRegex(FileNotFoundError,'Complete saved localization counts missing'):
            self.load(self.areas)
    def test_on_percentage_is_count_based_and_stale_values_rejected(self):
        value=self.load(dict(self.counts,**self.areas));self.assertEqual(value['on_percentage'],40.)
        with self.assertRaisesRegex(ValueError,'On percentage disagrees'):
            self.load(dict(self.counts,on_percentage=50.,**self.areas))
    def test_saved_block_areas_use_explicit_time_weights_and_first_frame(self):
        p=self.root/'blocks.csv'
        pd.DataFrame([dict(original_frame_start=50,original_frame_stop_exclusive=200,mask_ROI_area_um2=200.),dict(original_frame_start=0,original_frame_stop_exclusive=50,mask_ROI_area_um2=100.)]).to_csv(p,index=False)
        art=Artifact('m','mask_roi',dict(statistics=str(p)))
        value=self.load(self.counts,extra=dict(mask_roi=art))
        self.assertEqual(value['first_mask_ROI_area_um2'],100.);self.assertEqual(value['mean_ROI_area_um2'],175.)
    def test_block_durations_are_not_inferred(self):
        p=self.root/'blocks.csv';pd.DataFrame([dict(mask_ROI_area_um2=100.)]).to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Block durations are missing'):
            self.load(self.counts,extra=dict(mask_roi=Artifact('m','mask_roi',dict(statistics=str(p)))))
    def test_saved_unavailable_area_stays_unavailable(self):
        value=self.load(dict(counts=self.counts,first_mask_ROI_area_um2=None,mean_ROI_area_um2=None))
        self.assertIsNone(value['mean_ROI_area_um2'])
    def test_full_cell_loader_uses_flat_counts_and_retains_signed_curves(self):
        d=self.root/'decomp';d.mkdir();edges=np.linspace(0,3,301);mids=(edges[:-1]+edges[1:])/2;curve=np.exp(-10*mids)-.0001
        for group in ('On','Not-on'):
            q=d/'distribution'/group;q.mkdir(parents=True)
            pd.DataFrame(dict(distance_left_um=edges[:-1],distance_right_um=edges[1:],intra_density_um_inv=curve)).to_csv(q/'full.csv',index=False)
        pd.DataFrame([dict(group=g,scope='full',valid=True) for g in ('On','Not-on')]).to_csv(d/'per_scope_summary.csv',index=False)
        (d/'metadata.json').write_text(json.dumps(dict(config_rules=dict(tau_frames=1))))
        artifacts=dict(statistics=self.stats(dict(self.counts,**self.areas)),decomposition=Artifact('d','decomposition',dict(directory=str(d),metadata=str(d/'metadata.json'))))
        record,cell=summaries._load_cell(self.sample,artifacts,self.config)
        self.assertEqual(record['on_percentage'],40.);self.assertTrue(cell['eligible']);self.assertLess(cell['curves']['On'][-1],0.)
        np.testing.assert_allclose(cell['curves']['On'],curve,rtol=1e-13,atol=1e-15)

if __name__=='__main__':unittest.main()
