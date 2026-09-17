import unittest
from unittest.mock import patch
import torch
from transformers import BatchEncoding,Qwen2Config,Qwen2ForCausalLM
from length_budget_distill.math_cap_calibration import cap_view,merge_extended_traces
from length_budget_distill.baseline_reproduction import generate_text


class Tokenizer:
    eos_token_id=31
    pad_token_id=0
    def apply_chat_template(self,*args,**kwargs):return 'prompt'
    def __call__(self,*args,**kwargs):return BatchEncoding({'input_ids':torch.tensor([[1,4,5]]),'attention_mask':torch.ones(1,3,dtype=torch.long)})
    def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
    def encode(self,text,**kwargs):return [int(x) for x in text.split()] if text else []


class CapTests(unittest.TestCase):
    def test_extension_requires_complete_capped_support_and_exact_prefixes(self):
        row={'problem_id':'p','question':'Q','answer':'2','seed':17,'prompt_tokens':3,'model_role':'r1',
             'sampled_token_ids':[1,2,3],'hit_max_new_tokens':True}
        ended=dict(row,problem_id='q',sampled_token_ids=[2,31],hit_max_new_tokens=False)
        replacement=dict(row,sampled_token_ids=[1,2,3,4,31],hit_max_new_tokens=False)
        merged=merge_extended_traces([row,ended],[replacement])
        self.assertEqual(merged,[replacement,ended])
        for bad in ([],[replacement,replacement],[ended],[dict(replacement,sampled_token_ids=[1,2,4,31])],[dict(replacement,seed=18)]):
            with self.assertRaises(ValueError):merge_extended_traces([row,ended],bad)

    def test_eos_at_and_after_cap_are_distinct(self):
        a=cap_view([2,3,31],2,31);b=cap_view([2,3,31],3,31)
        self.assertTrue(a['hit_max_new_tokens']);self.assertFalse(b['hit_max_new_tokens'])
        self.assertEqual(a['body_token_ids'],b['body_token_ids']);self.assertEqual(b['generated_tokens'],2)
        self.assertFalse(cap_view([31],10,31)['hit_max_new_tokens'])
        with self.assertRaises(ValueError):cap_view([2],0,31)

    def test_real_decoder_greedy_and_sampled_prefixes_match_direct_caps(self):
        torch.manual_seed(17)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,eos_token_id=31,pad_token_id=0)).eval()
        tok=Tokenizer()
        for sample in (False,True):
            spec={'do_sample':sample,'temperature':.7,'top_p':.95,'max_new_tokens':12,'retain_token_ids':True}
            with patch('torch.cuda.synchronize'):
                full=generate_text(model,tok,[{'role':'user','content':'x'}],spec,17)
                short=generate_text(model,tok,[{'role':'user','content':'x'}],{**spec,'max_new_tokens':4},17)
            self.assertEqual(short['sampled_token_ids'],full['sampled_token_ids'][:4])
            view=cap_view(full['sampled_token_ids'],4,31)
            self.assertEqual(view['hit_max_new_tokens'],short['hit_max_new_tokens'])
            self.assertEqual(view['generated_tokens'],short['generated_tokens'])


if __name__=='__main__':unittest.main()
