import unittest
from length_budget_distill.math_cohort_freeze import components, take_components, assign_roles


class MathCohortTests(unittest.TestCase):
    def test_transitive_duplicate_components(self):
        self.assertEqual(components(['a','b','c','d'], [('b','c'),('a','b')]), [['a','b','c'],['d']])
        with self.assertRaises(ValueError): components(['a','a'], [])
        with self.assertRaises(ValueError): components(['a'], [('a','missing')])

    def test_holdout_counts_preserve_component_and_are_deterministic(self):
        groups = [['a','b'],['c','d'],['e'],['f'],['g','h','i']]
        chosen, remaining = take_components(groups, 4, 17, 'dev')
        self.assertEqual(sum(map(len, chosen)), 4)
        self.assertEqual(sorted(chosen+remaining), sorted(groups))
        self.assertEqual((chosen, remaining), take_components(groups[::-1], 4, 17, 'dev'))
        with self.assertRaises(ValueError): take_components([['a','b']], 1, 17, 'dev')

    def test_dap_neighbors_never_reach_other_roles(self):
        groups = [['a','b'],['c'],['d'],['e'],['f'],['g']]
        roles = assign_roles(groups, ['a'], {'calibration':2,'development':1}, 17)
        self.assertEqual(roles['a'], 'dap_development_reserved')
        self.assertEqual(roles['b'], roles['a'])
        self.assertEqual(list(roles.values()).count('calibration'), 2)
        self.assertEqual(list(roles.values()).count('development'), 1)
        self.assertEqual(list(roles.values()).count('student_pool'), 2)


if __name__ == '__main__': unittest.main()
