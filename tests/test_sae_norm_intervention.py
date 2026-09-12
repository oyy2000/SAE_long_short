"""Behavioral contracts for live-token intervention timing and paired decoding."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from length_budget_distill.sae_norm_intervention import NormMatchedController, generate_condition, sample_from_uniform


class TinyTokenizer:
    eos_token_id = 31
    pad_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return 'prompt'

    def __call__(self, rendered, **kwargs):
        return {'input_ids':torch.tensor([[1,4,5]]*len(rendered)),
                'attention_mask':torch.ones(len(rendered),3,dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return 'Answer: 0'


class NormInterventionTests(unittest.TestCase):
    def controller(self, folder, layer, dim):
        decoder=torch.eye(dim)[:3]
        path=Path(folder)/'sae.safetensors'
        save_file({'encoder_weight':decoder.clone(),'decoder_weight':decoder,
                   'encoder_bias':torch.zeros(3),'decoder_bias':torch.zeros(dim),
                   'activation_mean':torch.zeros(dim),'activation_scale':torch.ones(1)},str(path))
        return NormMatchedController(short_ids=[0],long_ids=[1],random_ids=[2],
            torch_module=torch,checkpoint_path=path,layer_module=layer,k=2,
            maximum_delta_fraction=.30,device='cpu',dtype=torch.float32)

    def test_norm_prefill_last_position_and_finished_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            layer=torch.nn.Identity()
            control=self.controller(folder,layer,3)
            control.begin(dict(mode='short',count=1,rho=.15,start=0),2)
            control.live[1]=False
            hidden=torch.tensor([[[1.,2.,3.],[3.,4.,2.]],[[2.,4.,3.],[3.,1.,4.]]])
            original=hidden.clone()
            changed=layer(hidden)
            diag=control.end()
            self.assertTrue(torch.equal(hidden,original))
            self.assertTrue(torch.equal(changed[:,:-1],original[:,:-1]))
            self.assertTrue(torch.equal(changed[1],original[1]))
            self.assertAlmostEqual(float((changed[0,-1]-hidden[0,-1]).norm()/hidden[0,-1].norm()),.15,places=6)
            self.assertEqual(diag[0]['modified_positions'],1)
            self.assertEqual(diag[1]['eligible_forward_positions'],0)

    def test_independent_cache_noop_and_late_prefix(self):
        torch.manual_seed(17)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
            bos_token_id=1,eos_token_id=31,pad_token_id=0)).eval()
        tokenizer=TinyTokenizer()
        rows=[dict(prompt='prompt',problem_id='q1',question_split='dev',gold_answer='#### 0')]
        settings=dict(seed=17,max_new_tokens=8,temperature=0.,top_p=.95)
        with tempfile.TemporaryDirectory() as folder:
            control=self.controller(folder,model.model.layers[1],16)
            base=dict(name='none',mode='short',count=1,rho=0.,start=0)
            a=generate_condition(model,tokenizer,control,base,rows,settings)
            b=generate_condition(model,tokenizer,control,base,rows,settings)
            inputs=tokenizer(['prompt'])
            with torch.inference_mode():
                expected=model.generate(**inputs,max_new_tokens=8,do_sample=False)[0,3:].tolist()
            self.assertEqual(a[0]['token_ids'],expected)
            self.assertEqual(a,b)
            late=generate_condition(model,tokenizer,control,dict(base,name='late',rho=.3,start=3),rows,settings)
            self.assertEqual(a[0]['token_ids'][:3],late[0]['token_ids'][:3])

    def test_nucleus_support_and_row_local_uniforms(self):
        logits=torch.tensor([[2.,1.,-100.],[1.,2.,-100.]])
        uniform=torch.tensor([.25,.75])
        both=sample_from_uniform(logits,uniform,1.,.99)
        separate=torch.cat([sample_from_uniform(logits[i:i+1],uniform[i:i+1],1.,.99) for i in range(2)])
        self.assertTrue(torch.equal(both,separate))
        top=sample_from_uniform(logits,uniform,1.,.5)
        self.assertEqual(top.tolist(),[0,1])


if __name__=='__main__': unittest.main()
