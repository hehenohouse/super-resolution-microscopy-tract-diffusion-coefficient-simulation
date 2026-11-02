"""Small deterministic contract/geometry fixtures; no production pipeline run."""
from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tifffile
from trak_pipeline.types import Artifact,Sample,StageContext
from trak_pipeline.contracts import validate_mask_contract,unknown_reason_for_frame,ContractError
from trak_pipeline.stages import mask,assignment

PACKAGE=Path(__file__).resolve().parents[1]

class MaskAssignmentTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PACKAGE/'validation');self.directory=Path(self.temp.name)
        self.config=dict(acquisition=dict(shape_yx=[9,9],n_frames=4,pixel_size_nm=117,frame_interval_s=.02),sample_frames=[0,1,2,3])
        self.sample=Sample('test_cell',self.directory,{},self.directory/'C1.tif',self.directory/'C2.tif',self.directory/'cell.roi')
        self.roi=np.ones((9,9),bool);np.save(self.directory/'roi.npy',self.roi)
        self.roi_art=Artifact('roi1','roi',dict(mask=str(self.directory/'roi.npy')))
    def tearDown(self):self.temp.cleanup()
    def context(self,stage,p,inputs):
        return StageContext(self.config,self.sample,p,inputs,self.directory/stage,PACKAGE,{})
    def artifact(self,stage,result,parameters=None):
        return Artifact(stage+'1',stage,{k:str(v) for k,v in result.files.items()},result.metadata,parameters or {})
    def broad(self,ids=(0,1,2,3)):
        values=np.zeros((len(ids),9,9),bool)
        for i,f in enumerate(ids):values[i,3,3]=True;values[i,4,4]=f%2==0
        folder=self.directory/'packed';folder.mkdir(exist_ok=True)
        mask.save_packed(folder/'b.npz',values,list(ids))
        return Artifact('broad1','broad',dict(packed_dir=str(folder))),values
    def count(self,ids=(0,1,2,3)):
        broad,_=self.broad(ids);parameters=dict(window_frames=2,stride_frames=2,frame_ids=list(range(4)),count_rule='sum_bool_per_pixel',partial_window_policy='unknown')
        return self.artifact('support_counts',mask.execute_support_counts(self.context('support_counts',parameters,dict(broad=broad))),parameters)
    def pipeline_masks(self):
        count=self.count()
        threshold=self.artifact('mask_threshold',mask.execute_threshold(self.context('mask_threshold',dict(minimum_support=1),dict(support_counts=count))))
        cleanup=self.artifact('mask_cleanup',mask.execute_cleanup(self.context('mask_cleanup',dict(min_foreground_pixels=1,foreground_connectivity=8,max_hole_area_exclusive=1,background_connectivity=4),dict(mask_threshold=threshold))))
        return count,threshold,cleanup
    def test_native_worker_exit_status_boundary(self):
        import contextlib,io
        from unittest.mock import patch
        for error,expected in [(ModuleNotFoundError('nellie'),74),(ImportError('shared dependency'),74),
                               (OSError(5,'external drive failure'),74),(FileNotFoundError(2,'individual input'),1),
                               (ValueError('individual native mask'),1),(KeyboardInterrupt(),130)]:
            with patch.object(mask,'_native_worker',side_effect=error),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(mask._native_main(['--native-worker','fixture.json']),expected)
        with patch.object(mask,'_native_worker',return_value=None):
            self.assertEqual(mask._native_main(['--native-worker','fixture.json']),0)
    def test_packed_roundtrip_and_padding(self):
        _,values=self.broad();packed=mask.pack_broad(values)
        np.testing.assert_array_equal(mask.unpack_broad(packed,values.shape),values)
        corrupt=packed.copy();corrupt[:,-1]|=128
        with self.assertRaises(ValueError):mask.unpack_broad(corrupt,values.shape)
    def test_legacy_boolean_array_streaming(self):
        _,values=self.broad()
        path=self.directory/'broad.npy';frame_path=self.directory/'frames.npy'
        np.save(path,values);np.save(frame_path,np.arange(4,dtype=np.int32))
        artifact=Artifact('old-array','broad',dict(array=str(path),frame_ids=str(frame_path)))
        params=dict(window_frames=2,stride_frames=2,frame_ids=list(range(4)),count_rule='sum_bool_per_pixel',partial_window_policy='unknown')
        result=mask.execute_support_counts(self.context('array_count',params,dict(broad=artifact)))
        expected=np.stack([values[:2].sum(axis=0),values[2:].sum(axis=0)])
        np.testing.assert_array_equal(np.load(result.files['counts']),expected)
    def test_counts_complete_and_explicit_mapping(self):
        result=self.count();counts=np.load(result.file('counts'));contract=result.metadata['contract']
        self.assertEqual(counts.shape,(2,9,9));self.assertEqual(counts[0,3,3],2)
        self.assertEqual(contract['frame_to_block'],{'0':0,'1':0,'2':1,'3':1})
        self.assertEqual(contract['block_frame_ids'],[[0,1],[2,3]])
        self.assertEqual(contract['block_valid'],[True,True])
    def test_missing_participant_is_unknown_not_empty_negative(self):
        result=self.count((0,2,3));contract=validate_mask_contract(result.metadata['contract'])
        self.assertEqual(contract['block_valid'],[False,True])
        self.assertEqual(unknown_reason_for_frame(contract,0),'incomplete_support_block')
        threshold=mask.execute_threshold(self.context('threshold',dict(minimum_support=1),dict(support_counts=result)))
        values=np.load(threshold.files['threshold_mask']);self.assertFalse(values[0].any());self.assertTrue(values[1].any())
    def test_threshold_only_does_not_mutate_counts(self):
        counts=self.count();before=counts.file('counts').read_bytes()
        first=mask.execute_threshold(self.context('one',dict(minimum_support=1),dict(support_counts=counts)))
        second=mask.execute_threshold(self.context('two',dict(minimum_support=2),dict(support_counts=counts)))
        self.assertEqual(before,counts.file('counts').read_bytes())
        self.assertGreater(np.load(first.files['threshold_mask']).sum(),np.load(second.files['threshold_mask']).sum())
    def test_fg8_cleanup_threshold_and_bg4_hole_strict_limit(self):
        image=np.zeros((9,9),bool);image[1,1]=True;image[2,2]=True
        final,_=mask.cleanup_mask(image,2,1,PACKAGE);self.assertTrue(final[1,1]);self.assertTrue(final[2,2])
        image=np.ones((9,9),bool);image[4,4]=False
        final,_=mask.cleanup_mask(image,1,1,PACKAGE);self.assertFalse(final[4,4])
        final,_=mask.cleanup_mask(image,1,2,PACKAGE);self.assertTrue(final[4,4])
        image[0,4]=False;image[1:5,4]=False
        final,_=mask.cleanup_mask(image,1,10,PACKAGE);self.assertFalse(final[4,4])
    def test_roi_intersection_separate(self):
        _,_,cleanup=self.pipeline_masks();before=cleanup.file('full_mask').read_bytes()
        self.roi[3,3]=False;np.save(self.directory/'roi.npy',self.roi)
        result=mask.execute_roi(self.context('mask_roi',{},dict(mask_cleanup=cleanup,roi=self.roi_art)))
        np.testing.assert_array_equal(np.load(result.files['roi_mask']),np.load(cleanup.file('full_mask'))&self.roi)
        self.assertEqual(before,cleanup.file('full_mask').read_bytes())
    def test_contract_unit_and_crop_conflict_rejected(self):
        contract=self.count().metadata['contract']
        with self.assertRaises(ContractError):validate_mask_contract(contract,expected_pixel_size_nm=100)
        bad=dict(contract,coordinate_transform='translate',registration=dict(transform='translate',status='unverified'))
        with self.assertRaises(ContractError):validate_mask_contract(bad)
    def row(self,uid='source:1',frame=0,x=3,y=3,se=23.4,qc=True):
        return dict(source_detection_uid=uid,frame=str(frame),fit_x_px=str(x),fit_y_px=str(y),x_se_nm=str(se),y_se_nm=str(se),
            qc_unflagged_model_fit=str(qc),fitted_center_inside_ROI='True',uncertainty_status='full_observed_Hessian_nominal_camera')
    def test_tangent_diagonal_and_fov_geometry(self):
        geom=assignment._geometry(PACKAGE);region=np.zeros((9,9),bool);region[3,3]=True;zero=np.zeros_like(region)
        self.assertEqual(geom.classify(region,zero,3.7,3,.2,.2,1e-12)['label'],'mito-compatible')
        self.assertEqual(geom.classify(region,zero,3.7,3.7,.2,.2,1e-12)['label'],'non-mito-compatible')
        self.assertEqual(geom.classify(region,zero,-.4,3,.2,.2,1e-12)['label'],'unknown')
        zero[3,3]=True;self.assertEqual(geom.classify(region,zero,3,3,.2,.2,1e-12)['label'],'unknown')
    def test_missing_block_and_qc_exclusion(self):
        geom=assignment._geometry(PACKAGE);contract=validate_mask_contract(self.count((0,2,3)).metadata['contract'])
        result=assignment.classify_one(self.row(),None,None,contract,117,1e-12,geom,'incomplete_support_block')
        self.assertEqual(result['pointwise_label'],'unknown')
        result=assignment.classify_one(self.row(qc=False),None,None,contract,117,1e-12,geom,'incomplete_support_block')
        self.assertEqual(result['pointwise_label'],'excluded_qc')
    def test_uid_unique_and_standard_error_meaning(self):
        path=self.directory/'fit.csv';rows=[self.row(),self.row()]
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.assertRaises(ValueError):assignment._input_rows(path,set(range(4)),.02)
        geom=assignment._geometry(PACKAGE);row=self.row();row['uncertainty_status']='arbitrary_precision'
        with self.assertRaises(ValueError):assignment.classify_one(row,None,None,{'frame_to_block':{'0':0}},117,1e-12,geom)
    def test_full_assignment_readonly_master_and_empty_frames(self):
        _,_,cleanup=self.pipeline_masks();movie=np.full((4,9,9),500,np.uint16)
        tifffile.imwrite(self.sample.C1_file,movie,ome=True,metadata=dict(axes='TYX'))
        rows=[self.row('s:1',0),self.row('s:2',0,7,7),self.row('s:3',1,qc=False),self.row('s:4',3)]
        fits=self.directory/'fits.csv.gz'
        with gzip.open(fits,'wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        before=fits.read_bytes();fit=Artifact('fit1','fit',dict(table=str(fits)))
        c1=Artifact('C11','C1',dict(image=str(self.sample.C1_file)))
        ctx=self.context('assignment',dict(method='error_ellipse_k1',k=1,tolerance=1e-12,closed_pixel_squares=True,diagonal=True),dict(fit=fit,C1=c1,roi=self.roi_art,mask_cleanup=cleanup))
        result=assignment.execute_assignment(ctx);summary=json.loads(Path(result.files['summary']).read_text())
        self.assertEqual(summary['counts'],dict(On=2,Not_on=1,unknown=0,selected=3,QC_excluded=1,ROI_candidates=4))
        self.assertEqual(before,fits.read_bytes())
        with Path(result.files['per_frame']).open() as stream:counts=list(csv.DictReader(stream))
        self.assertEqual(len(counts),4);self.assertEqual(counts[2]['selected'],'0')
        with gzip.open(result.files['labels'],'rt') as stream:labels=list(csv.DictReader(stream))
        self.assertEqual({r['source_detection_uid'] for r in labels},{r['source_detection_uid'] for r in rows})
        repeated=assignment.execute_assignment(ctx)
        self.assertEqual(Path(result.files['summary']).read_bytes(),Path(repeated.files['summary']).read_bytes())

if __name__=='__main__':unittest.main()
