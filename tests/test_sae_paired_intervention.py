import unittest
import tempfile
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
from length_budget_distill.sae_paired_intervention import clone_cache, cache_length, MeasuredSAEController
from length_budget_distill.sae_intervention import InterventionSpec


class CacheInterventionTests(unittest.TestCase):
    def test_measured_suppression_and_norm_cap(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.safetensors'
            save_file({'activation_scale': torch.tensor(1.), 'activation_mean': torch.zeros(2),
                       'encoder_weight': torch.eye(2), 'encoder_bias': torch.zeros(2),
                       'decoder_weight': torch.eye(2), 'decoder_bias': torch.zeros(2)}, str(path))
            controller = MeasuredSAEController(torch_module=torch, checkpoint_path=path,
                layer_module=torch.nn.Identity(), short_feature_ids=[0], long_feature_ids=[1],
                random_feature_ids=[0], measured_feature_ids=[1], k=1,
                maximum_delta_fraction=.15, device='cpu', dtype=torch.float32)
            original = torch.tensor([[[3., 4.]]])
            with controller.activate(InterventionSpec('suppress', 'long_suppress', 1.)):
                changed = controller.layer_module(original)
                diagnostics = controller.diagnostics()
            self.assertTrue(torch.equal(original, torch.tensor([[[3., 4.]]])))
            self.assertLess(diagnostics['mean_target_activation_after'], diagnostics['mean_target_activation_before'])
            self.assertAlmostEqual(float((changed-original).norm()/original.norm()), .15, places=6)
            with controller.activate(InterventionSpec('suppress', 'long_suppress', 1.)):
                with self.assertRaisesRegex(RuntimeError, 'prefill'):
                    controller.layer_module(torch.ones(1,2,2))

    def test_dynamic_and_tuple_caches_are_independent(self):
        tensor = torch.ones(1, 2, 3, 4)
        legacy = ((tensor, tensor.clone()),)
        for cache in (legacy, DynamicCache.from_legacy_cache(legacy)):
            other = clone_cache(cache)
            other[0][0].zero_()
            self.assertEqual(float(cache[0][0].sum()), 24.)
            self.assertEqual(cache_length(other), 3)

    def test_prefix_replay_and_first_continuation_hook(self):
        torch.manual_seed(17)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, bos_token_id=1, eos_token_id=None, pad_token_id=0)).eval()
        prompt = torch.tensor([[1, 4, 5]])
        with torch.inference_mode():
            prefix = model.generate(prompt, max_new_tokens=4, do_sample=False,
                                    return_dict_in_generate=True)
            self.assertEqual(cache_length(prefix.past_key_values), 6)
            calls = []
            handle = model.model.layers[1].register_forward_hook(
                lambda module, inputs, output: calls.append(output[0].shape[1]))
            shared = clone_cache(prefix.past_key_values)
            generated = model.generate(prefix.sequences, past_key_values=shared,
                                       max_new_tokens=3, do_sample=False)
            handle.remove()
            fresh = model.generate(prefix.sequences, max_new_tokens=3, do_sample=False)
            self.assertTrue(torch.equal(generated, fresh))
            self.assertEqual(calls, [1, 1, 1])
            self.assertEqual(cache_length(prefix.past_key_values), 6)


if __name__ == '__main__':
    unittest.main()
