"""Small validation fixtures for the explicit saved-data statistics stage."""
import math
import unittest
from compute_metrics import exact_permutation,holm,descriptives,audit_row

class StatisticalContractTests(unittest.TestCase):
    def test_exact_twosided_minimum_and_partition_count(self):
        effect,p,n=exact_permutation([1.,2.,3.],[4.,5.,6.])
        self.assertEqual(n,20);self.assertEqual(p,.1);self.assertEqual(effect,-3.)
    def test_complete_ties(self):
        effect,p,n=exact_permutation([1.,1.,1.],[1.,1.,1.])
        self.assertEqual((effect,p,n),(0.,1.,20))
    def test_group_swap_preserves_p_and_reverses_effect(self):
        a=exact_permutation([3.,7.,2.],[10.,8.,3.]);b=exact_permutation([10.,8.,3.],[3.,7.,2.])
        self.assertEqual(a[1:],b[1:]);self.assertAlmostEqual(a[0],-b[0])
    def test_holm_stepdown(self):
        for observed,expected in zip(holm([.01,.04,.03,.2]),[.04,.09,.09,.2]):self.assertAlmostEqual(observed,expected)
    def test_sample_sd_and_sem(self):
        result=descriptives([1.,2.,3.]);self.assertEqual(result['sample_sd'],1.);self.assertAlmostEqual(result['sem'],1/math.sqrt(3))
    def test_single_value_uncertainty_unavailable(self):
        self.assertEqual(descriptives([1.]),dict(mean=1.,sample_sd=None,sem=None))
    def test_distinct_ratio_estimators(self):
        rows=[dict(BR='BR1',On=1,Not_on=1,unknown=0,QC_excluded=0,selected=2,ROI_candidates=2,on_percentage=50.,on_percentage_valid=True),
              dict(BR='BR1',On=0,Not_on=100,unknown=0,QC_excluded=0,selected=100,ROI_candidates=100,on_percentage=0.,on_percentage_valid=True)]
        result=audit_row('P','WT','BR1',rows,None)
        self.assertEqual(result['mean_of_cell_ratios'],25.)
        self.assertAlmostEqual(result['pooled_On_counts_ratio'],100/102)
        self.assertNotEqual(result['mean_of_cell_ratios'],result['pooled_On_counts_ratio'])

if __name__=='__main__':unittest.main()
