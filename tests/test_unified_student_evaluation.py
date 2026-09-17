"""Guard shared decoding controls, exact stop accounting and dataset grading."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

from length_budget_distill.student_evaluation import generate_native_greedy_batch
from length_budget_distill.unified_student_evaluation import grade_student_prediction


class SyntheticTokenizer:
    eos_token_id=0
    pad_token_id=1
    def apply_chat_template(self,messages,**kwargs):return messages[-1]['content']
    def __call__(self,texts,**kwargs):
        assert kwargs['add_special_tokens'] is False
        width=max(map(len,texts))
        return {'input_ids':torch.tensor([[1]*(width-len(t))+list(map(ord,t)) for t in texts]),
                'attention_mask':torch.tensor([[0]*(width-len(t))+[1]*len(t) for t in texts])}
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids if i>1)


class SyntheticModel:
    device=torch.device('cpu')
    config=SimpleNamespace(max_position_embeddings=4096)
    generation_config=SimpleNamespace(forced_eos_token_id=None,repetition_penalty=1.1)
    def generate(self,**kwargs):
        self.kwargs=kwargs
        suffix=torch.tensor([list(map(ord,r'\boxed{2}'))+[0],list(map(ord,r'D\boxed{3}'))])
        return torch.cat([kwargs['input_ids'],suffix],dim=1)


class StudentEvaluationTest(unittest.TestCase):
    def test_greedy_controls_and_eos_are_not_inferred_from_decoded_text(self):
        model=SyntheticModel();tok=SyntheticTokenizer();bundle={'model':model,'tokenizer':tok,'torch':torch}
        messages=[[{'role':'user','content':'short'}],[{'role':'user','content':'a longer prompt'}]]
        rows,_=generate_native_greedy_batch(bundle,messages,max_new_tokens=10,repetition_penalty=1.0)
        self.assertFalse(model.kwargs['do_sample']);self.assertEqual(model.kwargs['repetition_penalty'],1.0)
        self.assertEqual(rows[0]['sampled_token_ids'][-1],0);self.assertEqual(rows[0]['output_tokens'],9)
        self.assertFalse(rows[0]['hit_max_new_tokens']);self.assertTrue(rows[1]['hit_max_new_tokens'])
        self.assertEqual(rows[1]['output_tokens'],10);self.assertEqual(rows[0]['prompt_tokens'],5)
        self.assertEqual(rows[0]['native_prompt_token_ids'],list(map(ord,'short')))
        generate_native_greedy_batch(bundle,messages,max_new_tokens=10)
        self.assertNotIn('repetition_penalty',model.kwargs)

    def test_forced_eos_and_context_overflow_are_rejected(self):
        model=SyntheticModel();bundle={'model':model,'tokenizer':SyntheticTokenizer(),'torch':torch}
        model.generation_config=SimpleNamespace(forced_eos_token_id=0)
        with self.assertRaisesRegex(ValueError,'Forced EOS'):
            generate_native_greedy_batch(bundle,[[{'role':'user','content':'a'}]],max_new_tokens=10)
        model.generation_config=SimpleNamespace(forced_eos_token_id=None)
        with self.assertRaisesRegex(ValueError,'context'):
            generate_native_greedy_batch(bundle,[[{'role':'user','content':'a'*4090}]],max_new_tokens=10)

    def test_reviewed_multiple_answers_and_hard_integer_precision(self):
        grading=json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())
        cfg={'grading':grading,'grading_methods':{'math500':'reviewed_math_v1','gsm8k_hard':'typed_math_v2'}}
        source={'dataset':'math500','answer':'3/2',
                'reviewed_answer':{'mode':'all','kind':'symbolic','answers':['-3','3/2']}}
        self.assertTrue(grade_student_prediction(r'\boxed{-3;3/2}',source,cfg)['is_correct'])
        self.assertFalse(grade_student_prediction(r'\boxed{3/2}',source,cfg)['is_correct'])
        hard={'dataset':'gsm8k_hard','question':'Compute the exact integer.','answer':'100000000000000003'}
        self.assertTrue(grade_student_prediction(r'\boxed{100000000000000003}',hard,cfg)['is_correct'])
        self.assertFalse(grade_student_prediction(r'\boxed{100000000000000004}',hard,cfg)['is_correct'])
        missing=deepcopy(source);missing.pop('reviewed_answer')
        with self.assertRaises(ValueError):grade_student_prediction(r'\boxed{3/2}',missing,cfg)


if __name__=='__main__':unittest.main()
