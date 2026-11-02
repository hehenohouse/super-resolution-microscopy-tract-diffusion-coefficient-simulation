"""Scientific adapter checks: asymmetric PSF, original frames, noise, failures."""
import copy
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from picasso import localize, io
import yaml

from pilot_core import (cut_raw_spots, fit_candidates, identify_candidates,
                        model_images, sample_ids, save_locs)


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((Path(__file__).parent/'config_T1_C2.yaml').read_text())
        cls.ids = np.array([0,1250,2499],dtype=np.uint32)
        yy,xx=np.mgrid[:40,:48]
        cls.truth=dict(x=23.3,y=17.6,photons=6000.,sx=1.15,sy=1.65,bg=20.)
        t=cls.truth
        mu=t['bg']+t['photons']/(2*np.pi*t['sx']*t['sy'])*np.exp(-.5*((xx-t['x'])**2/t['sx']**2+(yy-t['y'])**2/t['sy']**2))
        rng=np.random.default_rng(2107)
        electrons=rng.poisson(mu,size=(3,*mu.shape))+rng.normal(0,1.6,size=(3,*mu.shape))
        cls.movie=np.round(100+electrons/.46).astype(np.uint16)
        cls.candidates=pd.DataFrame(dict(frame=cls.ids,x=[23]*3,y=[18]*3,net_gradient=[1000.]*3,n_id=np.arange(3,dtype=np.uint32)))
        cls.locs,cls.theta,cls.timing=fit_candidates(cls.candidates,cls.ids,cls.movie,11,cls.config)

    def test_asymmetric_noisy_gaussian_coordinates_and_signal(self):
        locs=self.locs;t=self.truth
        self.assertTrue(locs.fit_converged.all())
        self.assertTrue(locs.usable_model_fit.all())
        for key,tolerance in [('x',.12),('y',.12),('sx',.14),('sy',.16)]:
            np.testing.assert_allclose(locs[key],t[key],atol=tolerance,rtol=0)
        np.testing.assert_allclose(locs.photons,t['photons'],rtol=.12)
        np.testing.assert_allclose(locs.bg,t['bg'],atol=4.,rtol=0)
        self.assertTrue(locs.precision_model_finite.all())
        self.assertTrue(locs.qc_unflagged_model_fit.all())
        self.assertTrue((locs.precision_status=='nominal_model_only').all())
        self.assertFalse(locs.suspect_zero_pixel_support.any())
        self.assertTrue((locs.lpx>0).all() and (locs.lpy>locs.lpx).all())
        self.assertAlmostEqual(self.timing['nominal_variance_e2'],2.56)

    def test_original_frames_units_and_model(self):
        np.testing.assert_array_equal(self.locs.frame,self.ids)
        np.testing.assert_allclose(self.locs.time_s,[0,25,49.98])
        for key in ['x','y','sx','sy','lpx','lpy']:
            np.testing.assert_allclose(self.locs[key+'_nm'],self.locs[key]*117.,rtol=0,atol=0)
        self.assertEqual(model_images(self.theta,11).shape,(3,11,11))
        self.assertTrue((self.locs.calibration_status=='manufacturer_nominal').all())
        self.assertFalse(self.locs.quantitatively_calibrated.any())

    def test_raw_patch_and_conversion(self):
        raw=cut_raw_spots(self.candidates,self.ids,self.movie,11)
        np.testing.assert_array_equal(raw[1],self.movie[1,13:24,18:29])
        actual=localize._to_photons(raw,dict(Baseline=100.,Sensitivity=.46,Gain=1.))
        np.testing.assert_allclose(actual,(raw-100)*np.float32(.46),rtol=0,atol=0)

    def test_failed_fits_kept_and_serialized(self):
        conf=copy.deepcopy(self.config)
        conf['fit'].update(max_iterations=1,tolerance=1e-12)
        locs,_,_=fit_candidates(self.candidates,self.ids,self.movie,11,conf)
        self.assertEqual(len(locs),3)
        self.assertTrue((locs.fit_state==1).all())
        self.assertFalse(locs.fit_converged.any())
        self.assertTrue((locs.precision_status=='unavailable_or_invalid_fit').all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.hdf5'
            save_locs(path,locs,dict(test=True))
            with h5py.File(path) as f:
                np.testing.assert_array_equal(f['locs']['frame'],self.ids)
                np.testing.assert_array_equal(f['locs']['fit_state'],[1]*3)
                self.assertFalse(f.attrs['quantitatively_calibrated'])
            self.assertEqual(len(pd.read_csv(path.with_suffix('.csv'))),3)

    def test_empty_and_boundary_frames(self):
        # Thresholding a genuinely blank frame should yield no fits.
        blank=np.full((1,40,48),100,np.uint16)
        all_ids,border=identify_candidates(self.ids[:1],blank,9)
        selected=all_ids.loc[all_ids.net_gradient>100]
        self.assertEqual(len(selected),0)
        locs,_,_=fit_candidates(selected,self.ids[:1],blank,9,self.config)
        self.assertEqual(len(locs),0)
        with tempfile.TemporaryDirectory() as folder:
            save_locs(Path(folder)/'blank.hdf5',locs,dict(blank_test=True))
        edge=blank.copy();edge[0,4,4]=60000
        all_ids,border=identify_candidates(self.ids[:1],edge,9)
        self.assertFalse(((all_ids.x==4)&(all_ids.y==4)).any())
        self.assertGreater(int(border.boundary_candidates_excluded.sum()),0)

    def test_sample_stop_guard(self):
        ids=sample_ids(self.config)
        self.assertEqual(len(ids),150)
        np.testing.assert_array_equal(ids[:50],np.arange(50))
        np.testing.assert_array_equal(ids[50:100],np.arange(1225,1275))
        np.testing.assert_array_equal(ids[100:],np.arange(2450,2500))
        conf=copy.deepcopy(self.config);conf['sample_segments']=[[0,2499]]
        with self.assertRaises(ValueError): sample_ids(conf)

    def test_zero_support_flag_keeps_source_and_candidate(self):
        image=self.movie[:1].copy();image[0,13,18]=0
        locs,_,_=fit_candidates(self.candidates.iloc[:1],self.ids[:1],image,11,self.config)
        self.assertEqual(len(locs),1)
        self.assertEqual(int(locs.source_zero_pixels.iloc[0]),1)
        self.assertTrue(locs.suspect_zero_pixel_support.iloc[0])
        self.assertFalse(locs.qc_unflagged_model_fit.iloc[0])
        self.assertEqual(int(image[0,13,18]),0)

    def test_picasso_metadata_and_native_import(self):
        info={'Height':40,'Width':48,'Frames':2500,'calibration_status':'manufacturer_nominal'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'native.hdf5'
            save_locs(path,self.locs,info)
            # Embedded /metadata alone must be valid, with no YAML present.
            imported,metadata=io.load_locs(str(path))
            self.assertEqual(metadata,[info])
            np.testing.assert_array_equal(imported.frame,self.ids)
            path.with_suffix('.yaml').write_text(yaml.safe_dump_all([info]),encoding='utf-8')
            self.assertEqual(io.load_info(str(path)),[info])


if __name__=='__main__':
    unittest.main(verbosity=2)
