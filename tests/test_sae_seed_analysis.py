import unittest
import numpy as np
from length_budget_distill.sae_seed_analysis import aligned_test_activations, confirmed_subset

class SeedAnalysisTests(unittest.TestCase):
    def row(self,tid,value):
        return {'trace_id':tid,'problem_id':tid,'question_split':'test','corpus_index':tid,
                'analysis_length_label':'short','solution_token_count':'10','feature_2_token_mean_activation':str(value)}

    def test_trace_order_does_not_define_alignment(self):
        rows=[self.row('b',2),self.row('a',1)]
        a,identity=aligned_test_activations(rows,[{'feature_id':2}])
        b,_=aligned_test_activations(rows[::-1],[{'feature_id':2}],identity)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,0],[1,2])

    def test_missing_duplicate_and_changed_trace_are_rejected(self):
        rows=[self.row('a',1),self.row('b',2)]
        _,identity=aligned_test_activations(rows,[{'feature_id':2}])
        for changed in (rows[:1],rows+[rows[0]],[rows[0],{**rows[1],'solution_token_count':'11'}]):
            with self.assertRaises(ValueError):aligned_test_activations(changed,[{'feature_id':2}],identity)

    def test_unconfirmed_features_cannot_fill_supplement(self):
        rows=[{'feature_id':i,'direction':'short','discovery_rank':i,'confirmed':i>1} for i in range(1,4)]
        self.assertEqual(confirmed_subset(rows,2),[2,3])
        with self.assertRaises(ValueError):confirmed_subset(rows,3)

if __name__=='__main__':unittest.main()
