from copy import deepcopy
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from length_budget_distill.sae_mechanism_readback import response_hidden_states
from length_budget_distill.math_steering_directions import match_calibration_pairs


class MathDirectionTest(unittest.TestCase):
    def test_response_states_match_backbone_output_and_exclude_prompt(self):
        torch.manual_seed(17)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)).eval()
        ids = [1,4,5,7,8,9]
        with torch.inference_mode():
            expected = model.model(torch.tensor([ids]), output_hidden_states=True).hidden_states[2][0,3:]
        actual = response_hidden_states(model, model.model.layers[1], ids, 3)
        self.assertTrue(torch.equal(actual, expected)); self.assertFalse(actual.requires_grad)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)
        with self.assertRaises(ValueError): response_hidden_states(model, model.model.layers[1], ids, len(ids))

    def test_pair_binding_rejects_role_teacher_and_trace_changes(self):
        cfg = {'pairs':1, 'teacher':{'revision':'r', 'model_name':'m'}}
        pair = {'problem_id':'p', 'question_role':'calibration', 'verbose_teacher_revision':'r',
            'verbose_teacher':'m', 'verbose':'trace', 'answer':'2', 'verbose_tokens':2}
        attempt = {'problem_id':'p', 'solution':'trace', 'eligible_pair':True, 'is_correct':True,
            'hit_max_new_tokens':False, 'gold_answer':'2', 'generated_tokens':2,
            'sampled_token_ids':[4,5,31], 'candidate_index':0, 'seed':17}
        actual = match_calibration_pairs([pair], [attempt], cfg)[0]
        self.assertEqual(actual['verbose_sampled_token_ids'], [4,5,31])
        for field,value in [('question_role','student_pool'),('verbose_teacher_revision','other'),
                            ('verbose','different'),('reviewed_gold_override',True),('answer','3')]:
            bad = deepcopy(pair); bad[field] = value
            with self.assertRaises(ValueError): match_calibration_pairs([bad], [attempt], cfg)
        with self.assertRaises(ValueError): match_calibration_pairs([pair], [attempt,attempt], cfg)


if __name__ == '__main__': unittest.main()
