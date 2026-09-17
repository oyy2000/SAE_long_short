import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from length_budget_distill.sae_paired_intervention import MeasuredSAEController
from length_budget_distill.sae_mechanism_readback import (
    independent_replay, last_state_hook, probe_positions, response_regions, sparse_change, unit)


class ReadbackTests(unittest.TestCase):
    def test_real_topk_codes_preserve_bfloat16_and_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sae.safetensors'
            save_file({'activation_scale': torch.tensor(2.), 'activation_mean': torch.zeros(4),
                'encoder_weight': torch.eye(4), 'encoder_bias': torch.zeros(4),
                'decoder_weight': torch.eye(4), 'decoder_bias': torch.zeros(4)}, str(path))
            controller = MeasuredSAEController(torch_module=torch, checkpoint_path=path,
                layer_module=torch.nn.Identity(), short_feature_ids=[0], long_feature_ids=[1],
                random_feature_ids=[2], measured_feature_ids=[0, 1, 2, 3], k=2,
                maximum_delta_fraction=.3, device='cpu', dtype=torch.bfloat16)
            hidden = torch.tensor([[3., 1., -2., 2.], [-3., -1., -2., -4.]], dtype=torch.bfloat16)
            indices, values = controller.sparse_codes(hidden)
            expected = torch.zeros_like(hidden).scatter(-1, indices, values)
            self.assertEqual(values.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(controller.feature_values(hidden), expected))
            self.assertEqual(expected[0].tolist(), [6., 0., 0., 4.])
            self.assertEqual(expected[1].sum(), 0.)

    def test_sparse_effects_exclude_zero_topk_slots(self):
        left = (torch.tensor([0, 1, 2]), torch.tensor([2., 3., 0.]))
        right = (torch.tensor([0, 2, 3]), torch.tensor([4., 5., 0.]))
        result = sparse_change(left, right, [0])
        self.assertEqual(result['active_entered'], 1)
        self.assertEqual(result['active_exited'], 1)
        self.assertEqual(result['target_l1_change'], 2.)
        self.assertEqual(result['nontarget_l1_change'], 8.)
        self.assertAlmostEqual(result['active_jaccard'], 1/3)

    def test_region_and_next_token_position_contract(self):
        text = 'Compute 2.\nAnswer: 2'
        offsets = [(0, 7), (7, 10), (10, 11), (11, 17), (17, 18), (18, 19), (19, 20)]
        regions, found = response_regions(text, offsets, 'reference', r'(?m)^(?P<marker>Answer):')
        self.assertTrue(found)
        self.assertEqual(regions.tolist(), [0, 0, 0, 1, 1, 2, 2])
        positions = probe_positions(regions)
        self.assertEqual(positions['first_response_predictor'], -1)
        self.assertEqual(positions['before_marker'], 2)
        self.assertEqual(positions['after_marker'], 4)
        self.assertEqual(positions['response_end'], 6)
        no_marker, _ = response_regions('work\n2', [(0, 4), (4, 5), (5, 6)], 'no_marker', r'(?P<marker>Answer):')
        self.assertEqual(no_marker.tolist(), [0, 0, 2])
        self.assertNotIn('after_marker', probe_positions(no_marker))

    def test_cached_local_intervention_matches_full_prefix_and_is_independent(self):
        torch.manual_seed(19)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=3, num_attention_heads=2,
            num_key_value_heads=2)).eval()
        prefix = [1, 4, 5, 8, 2]
        direction = unit(torch.randn(16))
        with torch.inference_mode():
            rows = list(independent_replay(model, model.model.layers[1], prefix,
                [('zero', 0., direction), ('change', .3, direction), ('zero_again', 0., direction)]))
            full = model(torch.tensor([prefix])).logits[0, -1].float()
        self.assertTrue(torch.allclose(rows[0][2], full, atol=2e-7, rtol=1e-5))
        self.assertTrue(torch.equal(rows[0][2], rows[2][2]))
        self.assertFalse(torch.allclose(rows[0][2], rows[1][2]))
        self.assertTrue(torch.equal(rows[0][3]['before'], rows[1][3]['before']))
        change = rows[1][3]
        self.assertAlmostEqual(float((change['after']-change['before']).norm()/change['before'].norm()), .3, places=6)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_hook_rejects_full_prefill_and_measures_bfloat_rounding(self):
        hook = last_state_hook(torch.tensor([1., 0.]), .1, {})
        with self.assertRaisesRegex(RuntimeError, 'one independently'):
            hook(None, None, torch.ones(1, 2, 2))
        captured = {}; original = torch.tensor([[[3., 4.]]], dtype=torch.bfloat16)
        changed = last_state_hook(torch.tensor([1., 0.]), .1, captured)(None, None, original)
        self.assertTrue(torch.equal(original, torch.tensor([[[3., 4.]]], dtype=torch.bfloat16)))
        self.assertEqual(changed.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(changed[:, 0], captured['after']))
        with self.assertRaises(ValueError): unit(torch.zeros(4))


if __name__ == '__main__': unittest.main()
