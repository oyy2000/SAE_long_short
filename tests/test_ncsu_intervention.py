"""Causal-control integrity checks for the NCSU intervention adapter."""
import unittest
from unittest.mock import patch
import torch
from length_budget_distill.ncsu_intervention import MatchedController, specs
from length_budget_distill.sae_norm_intervention import NormMatchedController


class TestMatchedDirections(unittest.TestCase):
    def test_random_joint_uses_disjoint_signed_groups(self):
        ctrl = object.__new__(MatchedController)
        ctrl.decoder = torch.eye(6)
        ctrl.random_ids_list = [0, 1, 2, 3]
        ctrl.direction = None
        spec = {'count': 2, 'mode': 'joint', 'random_control': True}
        with patch.object(NormMatchedController, 'begin'):
            ctrl.begin(spec, 1)
        torch.testing.assert_close(ctrl.direction, torch.tensor([.5, .5, -.5, -.5, 0., 0.]))

    def test_random_short_uses_only_one_group(self):
        ctrl = object.__new__(MatchedController)
        ctrl.decoder = torch.eye(6)
        ctrl.random_ids_list = [0, 1, 2, 3]
        with patch.object(NormMatchedController, 'begin'):
            ctrl.begin({'count': 2, 'mode': 'short', 'random_control': True}, 1)
        torch.testing.assert_close(ctrl.direction, torch.tensor([2**-.5, 2**-.5, 0., 0., 0., 0.]))

    def test_each_target_has_same_strength_count_timing_control(self):
        cfg={'intervention': {'feature_count_per_side':8, 'start':0, 'modes':['short','joint'], 'strengths':[.05,.15,.3]}}
        rows=specs(cfg)
        self.assertEqual(len(rows),13)
        by_name={r['name']:r for r in rows}
        for row in rows[1:]:
            if row.get('random_control'):continue
            control=by_name['random_'+row['name']]
            for key in ('count','start','rho','mode'):
                self.assertEqual(row[key], control[key])


if __name__ == '__main__':
    unittest.main()
