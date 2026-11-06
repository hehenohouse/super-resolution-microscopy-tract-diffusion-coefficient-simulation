"""The assignment selection check uses the same continuous ROI polygon as fit QC.

Regression for WT_TRAK1_BR5_T5: a selected refined center inside the ROI polygon lay in a
boundary pixel whose center is outside the polygon (raster value 0), and the old raster-based
check stopped the whole cell.
"""
import csv,gzip,json,unittest
from pathlib import Path
import numpy as np
import tifffile
from trak_pipeline.types import Artifact
from trak_pipeline.stages import assignment
import test_mask_assignment as base  # module import only, so its TestCase is not collected twice
PACKAGE=base.PACKAGE

# Sloped top edge y = 5 - x/2 between (8,1) and (0,5).
VERTICES=np.array([[0.,0.],[8.,0.],[8.,1.],[0.,5.]])
BOUNDARY_POINT=(4.6,2.6)   # inside the polygon (edge y=2.7 at x=4.6); its pixel (5,3) has center y=3 > 2.5, outside
OUTSIDE_POINT=(7.4,7.4)    # outside the polygon


class AssignmentRoiPolygonCheckTest(unittest.TestCase):
    setUp=base.MaskAssignmentTest.setUp;tearDown=base.MaskAssignmentTest.tearDown;context=base.MaskAssignmentTest.context
    artifact=base.MaskAssignmentTest.artifact;broad=base.MaskAssignmentTest.broad;count=base.MaskAssignmentTest.count
    pipeline_masks=base.MaskAssignmentTest.pipeline_masks;row=base.MaskAssignmentTest.row

    def polygon_roi(self):
        polygon=assignment._roi_polygon(PACKAGE)
        yy,xx=np.mgrid[:9,:9]
        raster=polygon.points_in_polygon(np.column_stack([xx.ravel(),yy.ravel()]).astype(float),VERTICES).reshape(9,9)
        np.save(self.directory/'roi_poly.npy',raster);np.save(self.directory/'vertices.npy',VERTICES)
        return raster,Artifact('roi2','roi',dict(mask=str(self.directory/'roi_poly.npy'),vertices=str(self.directory/'vertices.npy')))

    def run_assignment(self,points,roi_artifact):
        _,_,cleanup=self.pipeline_masks()
        tifffile.imwrite(self.sample.C1_file,np.full((4,9,9),500,np.uint16),ome=True,metadata=dict(axes='TYX'))
        rows=[self.row('s:%d'%i,0,x,y) for i,(x,y) in enumerate(points)]
        fits=self.directory/'fits_poly.csv.gz'
        with gzip.open(fits,'wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        ctx=self.context('assignment_poly',dict(method='error_ellipse_k1',k=1,tolerance=1e-12,closed_pixel_squares=True,diagonal=True),
            dict(fit=Artifact('fit2','fit',dict(table=str(fits))),C1=Artifact('C12','C1',dict(image=str(self.sample.C1_file))),roi=roi_artifact,mask_cleanup=cleanup))
        return assignment.execute_assignment(ctx)

    def test_fixture_reproduces_the_boundary_case(self):
        raster,_=self.polygon_roi();polygon=assignment._roi_polygon(PACKAGE);geom=assignment._geometry(PACKAGE)
        x,y=BOUNDARY_POINT
        self.assertTrue(polygon.points_in_polygon(np.array([[x,y]]),VERTICES)[0])
        self.assertFalse(raster[3,5]);self.assertFalse(geom.point_inside_mask(raster,x,y,1e-12))

    def test_selected_center_inside_polygon_on_boundary_pixel_is_accepted(self):
        _,roi=self.polygon_roi()
        result=self.run_assignment([BOUNDARY_POINT,(2,2)],roi)
        summary=json.loads(Path(result.files['summary']).read_text())
        self.assertEqual(summary['counts']['selected'],2)

    def test_selected_center_outside_polygon_still_stops_the_cell(self):
        _,roi=self.polygon_roi()
        with self.assertRaisesRegex(ValueError,'Selected refined center lies outside supplied ROI'):
            self.run_assignment([OUTSIDE_POINT],roi)

    def test_raster_check_remains_when_no_polygon_is_available(self):
        raster,_=self.polygon_roi()
        legacy=Artifact('roi3','roi',dict(mask=str(self.directory/'roi_poly.npy')))
        with self.assertRaisesRegex(ValueError,'Selected refined center lies outside supplied ROI'):
            self.run_assignment([BOUNDARY_POINT],legacy)


if __name__=='__main__':unittest.main()
