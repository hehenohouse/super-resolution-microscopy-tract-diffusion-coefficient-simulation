"""ROI geometry accepts ImageJ sub-pixel ROIs; integer ROIs keep the vendored TARDIS reader.

Regression for TDK_TRAK1_BR5_T11, TDK_TRAK1_BR5_T14 and TDK_TRAK2_BR4_T4: their freehand ROIs
were saved by ImageJ with sub-pixel coordinates (option bit 128), which the TARDIS reader rejects,
so each cell failed before any science stage ran.
"""
import struct,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from trak_pipeline.kernels import paths
from trak_pipeline.stages import localization

PACKAGE=Path(__file__).resolve().parents[1]
paths('TARDIS/path_irrelevant_python/tardis_tau1')
from locbio import load_imagej_roi,points_in_polygon  # noqa: E402


def imagej_roi(x,y,roi_type=7,subpixel=False,int_x=None,int_y=None):
    """Minimal ImageJ ROI (version 228): header, relative integer vertices, optional absolute floats."""
    xi=np.rint(x).astype(int) if int_x is None else np.asarray(int_x);yi=np.rint(y).astype(int) if int_y is None else np.asarray(int_y)
    top,left,bottom,right=int(yi.min()),int(xi.min()),int(yi.max()),int(xi.max())
    header=bytearray(64);header[:4]=b'Iout';struct.pack_into('>h',header,4,228);header[6]=roi_type
    struct.pack_into('>4h',header,8,top,left,bottom,right);struct.pack_into('>H',header,16,len(x))
    struct.pack_into('>H',header,50,128 if subpixel else 0)
    body=struct.pack('>%dH'%len(x),*(xi-left))+struct.pack('>%dH'%len(x),*(yi-top))
    if subpixel:body+=struct.pack('>%df'%len(x),*x)+struct.pack('>%df'%len(x),*y)
    return bytes(header)+body


class RoiSubpixelReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PACKAGE/'validation');self.directory=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def write(self,name,raw):
        path=self.directory/name;path.write_bytes(raw);return path

    def geometry(self,roi_path):
        ctx=SimpleNamespace(config=dict(acquisition=dict(shape=[4,12,14])),sample=SimpleNamespace(ROI_file=roi_path),output_dir=self.directory/('out_'+roi_path.stem))
        return localization.execute_roi_geometry(ctx)

    def test_integer_roi_is_read_by_the_tardis_reader_unchanged(self):
        path=self.write('int.roi',imagej_roi(np.array([1.,9.,9.,1.]),np.array([2.,2.,8.,8.])))
        vertices,precision=localization.load_roi_vertices(path,load_imagej_roi)
        self.assertEqual(precision,'integer');np.testing.assert_array_equal(vertices,load_imagej_roi(path))
        result=self.geometry(path)
        self.assertEqual(set(result.metadata),{'vertices','pixels','crop_xyxy_px','coordinate_system'})
        self.assertEqual(result.metadata['pixels'],9*7)

    def test_subpixel_roi_uses_imagej_float_vertices(self):
        x=np.array([1.5,9.,9.,1.5]);y=np.array([2.,2.,8.5,8.5])
        path=self.write('sub.roi',imagej_roi(x,y,subpixel=True))
        with self.assertRaisesRegex(ValueError,'Subpixel ROI coordinates are not supported'):load_imagej_roi(path)
        vertices,precision=localization.load_roi_vertices(path,load_imagej_roi)
        self.assertEqual(precision,'imagej_subpixel_float');np.testing.assert_array_equal(vertices,np.c_[x,y])
        result=self.geometry(path)
        mask=np.load(result.files['mask']);saved=np.load(result.files['vertices'])
        np.testing.assert_array_equal(saved,np.c_[x,y])
        yy,xx=np.mgrid[:12,:14]
        np.testing.assert_array_equal(mask,points_in_polygon(np.c_[xx.ravel(),yy.ravel()].astype(float),np.c_[x,y]).reshape(12,14))
        self.assertEqual(int(mask.sum()),8*7)  # columns 2..9, rows 2..8: the half-pixel edges exclude column 1
        self.assertEqual(result.metadata['roi_coordinate_precision'],'imagej_subpixel_float')

    def test_subpixel_layout_mismatch_is_rejected(self):
        x=np.array([1.5,9.,9.,1.5]);y=np.array([2.,2.,8.5,8.5])
        path=self.write('bad.roi',imagej_roi(x,y,subpixel=True,int_x=np.array([5,9,9,5]),int_y=np.rint(y).astype(int)))
        with self.assertRaisesRegex(ValueError,'disagree'):localization.load_roi_vertices(path,load_imagej_roi)

    def test_truncated_subpixel_roi_is_rejected(self):
        raw=imagej_roi(np.array([1.5,9.,9.,1.5]),np.array([2.,2.,8.5,8.5]),subpixel=True)
        with self.assertRaisesRegex(ValueError,'Truncated'):localization.load_roi_vertices(self.write('cut.roi',raw[:-4]),load_imagej_roi)

    def test_unsupported_roi_type_is_rejected(self):
        raw=imagej_roi(np.array([1.5,9.,9.,1.5]),np.array([2.,2.,8.5,8.5]),roi_type=1,subpixel=True)
        with self.assertRaisesRegex(ValueError,'Only polygon'):localization.load_roi_vertices(self.write('rect.roi',raw),load_imagej_roi)


if __name__=='__main__':unittest.main()
