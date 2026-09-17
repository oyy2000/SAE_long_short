import unittest
from length_budget_distill.sae_seed_features import feature_selection

class SeedFeatureSelectionTests(unittest.TestCase):
    def test_unconfirmed_top_rank_is_not_used_to_fill_missing_features(self):
        rows=[{'feature_id':10,'discovery_rank':1,'confirmed':False,'direction':'short'},
              {'feature_id':20,'discovery_rank':2,'confirmed':True,'direction':'short'},
              {'feature_id':30,'discovery_rank':1,'confirmed':True,'direction':'long'}]
        result=feature_selection(rows,2)
        self.assertFalse(result['short_direction_feasible']);self.assertFalse(result['both_directions_feasible'])
        self.assertEqual(result['features']['short'],[20])

    def test_original_rank_and_direction_are_preserved(self):
        rows=[{'feature_id':i,'discovery_rank':rank,'confirmed':True,'direction':direction}
              for i,rank,direction in [(1,2,'short'),(2,1,'short'),(3,1,'long')]]
        result=feature_selection(rows,1)
        self.assertTrue(result['both_directions_feasible']);self.assertEqual(result['features'],{'short':[2],'long':[3]})

if __name__=='__main__':unittest.main()
