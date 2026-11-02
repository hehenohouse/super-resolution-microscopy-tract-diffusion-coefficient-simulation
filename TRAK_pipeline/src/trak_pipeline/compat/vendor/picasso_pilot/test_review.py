import unittest

import numpy as np
from picasso.fitting import precision

from pilot_core import canonical_gaussian_parameters,model_images,detection_zero_counts
from pilot_review import brute_pair_histograms
from tardis_tau1.core import pair_histogram
from tardis_tau1 import Tau1Config


class SignAndFrameTests(unittest.TestCase):
    def test_all_width_signs_leave_model_and_xy_bound_unchanged(self):
        base=np.array([230.,4.2,3.7,1.15,1.65,20.])
        for sx,sy in [(1,1),(-1,1),(1,-1),(-1,-1)]:
            backend=base[None].copy();backend[:,3]*=sx;backend[:,4]*=sy
            raw=backend.copy();raw[:,0]*=2*np.pi*raw[:,3]*raw[:,4]
            canonical=canonical_gaussian_parameters(backend)
            np.testing.assert_allclose(model_images(raw,9),model_images(canonical,9),rtol=1e-14,atol=1e-12)
            self.assertGreater(canonical[0,0],0)
            np.testing.assert_array_equal(canonical[0,1:3],base[1:3])
            variance=np.full((1,9,9),2.56)
            before=precision._gauss_crlb(raw,9,False,variance=variance)
            after=precision._gauss_crlb(canonical,9,False,variance=variance)
            np.testing.assert_allclose(before[:,1:3],after[:,1:3],rtol=1e-9,atol=1e-12)

    def test_negative_amplitude_is_not_rescued_by_absolute_signal(self):
        backend=np.array([[-230.,4.,4.,-1.2,1.3,20.]])
        result=canonical_gaussian_parameters(backend)
        self.assertLess(result[0,0],0)
        self.assertGreater(result[0,3],0);self.assertGreater(result[0,4],0)

    def test_gaps_and_frame_order_do_not_create_false_tau1_pairs(self):
        frames=np.array([0,1,49,1225]);groups=[np.array([[0.,0.],[.12e-6,0.]]) for _ in frames]
        positions=np.vstack([np.column_stack([np.repeat(f,2),xy]) for f,xy in zip(frames,groups)])
        edges=np.linspace(0,3e-6,301)
        brute,background,pairs=brute_pair_histograms(positions,edges)
        self.assertEqual(pairs,[(0,1)])
        stats=pair_histogram(frames,groups,edges,'total',Tau1Config())
        self.assertEqual(stats.source_frame_pairs,1)
        np.testing.assert_array_equal(stats.histogram,brute)

    def test_detector_halo_sees_zero_outside_fit_but_not_unused_corner(self):
        import pandas as pd
        ids=pd.DataFrame({'frame':[0],'x':[7],'y':[7]})
        movie=np.full((1,15,15),400,np.uint16)
        # Box 9 is rows/cols 3..11, gradient halo is 2..12.
        movie[0,7,2]=0
        self.assertFalse((movie[0,3:12,3:12]==0).any())
        self.assertEqual(int(detection_zero_counts(ids,np.array([0]),movie,9)[0]),1)
        movie[0,7,2]=400;movie[0,2,2]=0
        self.assertEqual(int(detection_zero_counts(ids,np.array([0]),movie,9)[0]),0)


if __name__=='__main__':unittest.main(verbosity=2)
