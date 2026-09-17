"""Protect objective direction, decode-state semantics and ratio conditioning."""
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from length_budget_distill.asc_ces import (
    ResidualAddition, response_energy, contrastive_energy, forward_kl, trust_region_penalty,
)
from length_budget_distill.compression_baselines import (
    tokenskip_prompt, ratio_for_problem, compress_trace, dap_messages,
)


class TestASC(unittest.TestCase):
    def test_energy_masks_prompt_and_normalizes_length(self):
        logits = torch.zeros(1, 5, 3)
        ids = torch.tensor([[2, 0, 1, 0, 1]])
        mask = torch.tensor([[False, True, True, True, False]])
        first = response_energy(logits, ids, mask)
        logits[:, 0] = torch.tensor([100.0, -100.0, 0.0])
        self.assertAlmostEqual(float(first), float(response_energy(logits, ids, mask)))
        self.assertAlmostEqual(float(first), float(torch.log(torch.tensor(3.0))), places=6)

    def test_objective_gradient_prefers_short_and_hinge_respects_budget(self):
        short = torch.tensor(2.0, requires_grad=True)
        long = torch.tensor(1.0, requires_grad=True)
        contrastive_energy(short, long).backward()
        self.assertGreater(short.grad.item(), 0)
        self.assertLess(long.grad.item(), 0)
        self.assertEqual(trust_region_penalty(torch.tensor(0.01)).item(), 0)
        self.assertAlmostEqual(trust_region_penalty(torch.tensor(0.03)).item(), .2, places=6)

    def test_kl_direction_full_vocab_and_gradient(self):
        base = torch.tensor([[[1.0, 2.0, -1.0]]], requires_grad=True)
        steer = torch.tensor([[[.5, -.1, 1.0]]], requires_grad=True)
        mask = torch.ones(1, 1, dtype=torch.bool)
        actual = forward_kl(base, steer, mask, chunk_size=1)
        p, q = base.softmax(-1), steer.softmax(-1)
        expected = (p * (p.log() - q.log())).sum()
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNone(base.grad)
        self.assertGreater(steer.grad.abs().sum().item(), 0)
        self.assertAlmostEqual(forward_kl(base, base, mask).item(), 0, places=7)

    def test_teacher_forcing_matches_cached_decoding_and_base_stays_frozen(self):
        torch.manual_seed(17)
        cfg = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)
        model = Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        vector = torch.nn.Parameter(torch.randn(16) * .1)
        controller = ResidualAddition(model.model.layers[1], vector)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.tensor([[False, False, True, True, True]])
        with controller.applied(mask):
            full = model(ids, use_cache=False).logits
        with controller.applied():
            initial = model(ids[:, :3], use_cache=True)
            next1 = model(ids[:, 3:4], past_key_values=initial.past_key_values, use_cache=True)
            next2 = model(ids[:, 4:5], past_key_values=next1.past_key_values, use_cache=True)
        cached = torch.cat([initial.logits[:, -1:], next1.logits, next2.logits], dim=1)
        self.assertTrue(torch.allclose(full[:, 2:], cached, atol=1e-6))
        opt = torch.optim.Adam([vector], lr=.01)
        full.square().mean().backward()
        self.assertGreater(vector.grad.abs().sum().item(), 0)
        opt.step()
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with controller.applied():
                raise ValueError('sentinel')
        self.assertIsNone(controller.handle)


class TestTextBaselines(unittest.TestCase):
    def test_ratio_assignment_and_upstream_qwen_prompt(self):
        ratios = [.5, .6, .7, .8, .9, 1.]
        self.assertEqual(ratio_for_problem('q1', ratios, 17), ratio_for_problem('q1', ratios, 17))
        self.assertIn('question<|eot_id|>0.5<|eot_id|>', tokenskip_prompt('question', .5))
        self.assertNotIn('eot_id', tokenskip_prompt('question', 1.))
        with self.assertRaises(ValueError): ratio_for_problem('q1', [.5, .5], 17)

    def test_llmlingua_family_options_and_identity(self):
        class Recorder:
            def compress_prompt(self, text, **kwargs): return dict(text=text, **kwargs)
        self.assertEqual(compress_trace(Recorder(), 'trace', 1.)['compressed_prompt'], 'trace')
        self.assertNotIn('force_tokens', compress_trace(Recorder(), 'trace', .5))
        self.assertTrue(compress_trace(Recorder(), 'trace', .5, family='llama3')['force_reserve_digit'])

    def test_dap_preserves_full_trace_and_problem(self):
        trace = 'long source\n' * 200
        messages = dap_messages('q', trace, {'framework':'three frameworks',
                                            'rewrite_wrapper':'{question}\n{source_trace}'})
        self.assertIn(trace, messages[1]['content'])
        self.assertEqual(messages[0]['content'], 'three frameworks')


if __name__ == '__main__': unittest.main()
