import unittest
from copy import deepcopy
from length_budget_distill.sae_seed_generation import specifications

class SeedGenerationTests(unittest.TestCase):
    def test_dictionary_ids_and_reference_readback_are_distinct(self):
        cfg={'seeds':[17,42,73,101],'feature_count':3,'rho':.3,'sae':{'short_features':list(range(8))}}
        rows=[{'seed':s,'supplement_short_ids':[s*10+i for i in range(3)],'confirmed_short':3} for s in cfg['seeds']]
        specs=specifications(cfg,rows)
        self.assertEqual(len(specs),8)
        chosen=next(r for r in specs if r['name']=='seed_42_top_3')
        self.assertEqual(chosen['injected_feature_ids'],[420,421,422])
        self.assertEqual(chosen['measured_short_feature_ids'],[0,1,2])
        self.assertEqual(specs[0]['rho'],0.)
        self.assertEqual(next(r for r in specs if r['name']=='seed_17_top_8')['count'],8)
        for mutated in (rows[:3],rows+[rows[0]],[{**rows[0],'confirmed_short':2},*rows[1:]]):
            with self.assertRaises(ValueError):specifications(cfg,mutated)

if __name__=='__main__':unittest.main()
