import unittest
from length_budget_distill.legacy_sae_calibration import specifications, seed_for


class CalibrationSpecificationTests(unittest.TestCase):
    def config(self, directions):
        return {'features': {'test': {'available_directions': directions}},
                'parent_intervention': {'short_strength_grid': [1.,2.,4.], 'long_strength_grid': [.25,.5,1.]}}

    def test_both_directions_have_matched_random_enhancement_grid(self):
        specs = specifications(self.config(['short','long']), 'test')
        self.assertEqual(len(specs), 10)
        self.assertEqual([r.strength for r in specs if r.mode=='short_enhance'],
                         [r.strength for r in specs if r.mode=='random_enhance'])

    def test_absent_features_are_not_silently_replaced(self):
        self.assertEqual([r.mode for r in specifications(self.config([]),'test')], ['none'])
        self.assertEqual({r.mode for r in specifications(self.config(['long']),'test')}, {'none','long_suppress'})

    def test_seeds_are_stable_and_candidate_specific(self):
        self.assertEqual(seed_for(17,'q',0), seed_for(17,'q',0))
        self.assertNotEqual(seed_for(17,'q',0), seed_for(17,'q',1))


if __name__=='__main__':unittest.main()
