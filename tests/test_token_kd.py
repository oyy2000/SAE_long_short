import unittest
import torch
import torch.nn.functional as F
from length_budget_distill.token_kd import completion_kd_loss

class TokenKDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.s=torch.randn(1,5,7,requires_grad=True)
        self.t=torch.randn(1,5,9,requires_grad=True)
        self.labels=torch.tensor([[-100,-100,2,3,-100]])
    def call(self,s=None,t=None,alpha=.5,chunk_tokens=1):
        return completion_kd_loss(self.s if s is None else s,self.t if t is None else t,self.labels,
            vocabulary_size=6,alpha=alpha,temperature=2.,chunk_tokens=chunk_tokens)
    def test_ce_matches_native_causal_completion_loss(self):
        loss,_=self.call(alpha=0)
        expected=F.cross_entropy(self.s[:,:-1].reshape(-1,7),self.labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(loss,expected)
    def test_same_distribution_has_zero_kl_even_with_extra_output_rows(self):
        s=self.s.detach().clone();t=torch.cat([s,torch.zeros(1,5,2)],-1)
        loss,_=self.call(s,t,alpha=1)
        self.assertLess(abs(loss.item()),1e-6)
    def test_prompt_padding_and_unshifted_positions_do_not_affect_loss(self):
        loss,_=self.call();s=self.s.detach().clone();t=self.t.detach().clone()
        s[:,[0,3,4]]=50;t[:,[0,3,4]]=-50
        other,_=self.call(s,t)
        torch.testing.assert_close(loss,other)
    def test_teacher_detached_student_learns_and_chunks_agree(self):
        first,_=self.call();other,_=self.call(chunk_tokens=64)
        torch.testing.assert_close(first,other)
        first.backward();self.assertIsNone(self.t.grad)
        self.assertGreater(self.s.grad.abs().sum().item(),0)
        self.assertEqual(self.s.grad[:,[0,3,4]].abs().sum().item(),0)
    def test_manual_forward_kl_and_temperature_scaling(self):
        loss,_=self.call(alpha=1)
        p=(self.t[:,1:3,:6]/2).softmax(-1)
        q=(self.s[:,1:3,:6]/2).log_softmax(-1)
        expected=4*(p*(p.log()-q)).sum()/2
        torch.testing.assert_close(loss,expected)
    def test_unequal_microbatches_match_combined_loss_and_gradients(self):
        s=torch.randn(2,5,7,requires_grad=True);t=torch.randn(2,5,9)
        labels=torch.tensor([[-100,-100,2,3,-100],[-100,1,2,3,4]])
        for alpha in (0.,.5,1.):
            kw=dict(vocabulary_size=6,alpha=alpha,temperature=2.,chunk_tokens=2)
            combined,_=completion_kd_loss(s,t,labels,**kw)
            split=sum(completion_kd_loss(s[i:i+1],t[i:i+1],labels[i:i+1],num_items_in_batch=6,**kw)[0] for i in range(2))
            torch.testing.assert_close(split,combined)
            g1=torch.autograd.grad(combined,s,retain_graph=True)[0]
            g2=torch.autograd.grad(split,s,retain_graph=True)[0]
            torch.testing.assert_close(g1,g2)
    def test_zero_kd_matches_pinned_transformers_accumulation_loss(self):
        from transformers.loss.loss_utils import ForCausalLMLoss
        loss,_=completion_kd_loss(self.s,None,self.labels,vocabulary_size=6,alpha=0.,temperature=2.,num_items_in_batch=11)
        expected=ForCausalLMLoss(self.s,self.labels,vocab_size=7,num_items_in_batch=11)
        torch.testing.assert_close(loss,expected)
