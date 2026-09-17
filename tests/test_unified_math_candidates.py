from copy import deepcopy
import unittest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from length_budget_distill.asc_ces import ResidualAddition
from length_budget_distill.sae_norm_intervention import generate_condition_raw,sample_from_uniform,generation_stream_seed
from length_budget_distill.unified_math_candidates import FixedVectorController,audit_candidate_grid,select_candidates,settings_for_candidate


class Tokenizer:
    eos_token_id=31
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
    def __call__(self,texts,**kwargs):
        rows=[[int(x) for x in t.split()] for t in texts];width=max(map(len,rows))
        return {'input_ids':torch.tensor([[0]*(width-len(r))+r for r in rows]),
                'attention_mask':torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in rows])}
    def decode(self,tokens,**kwargs):return ' '.join(str(i) for i in tokens if i not in (0,31))


class UnifiedCandidatesTest(unittest.TestCase):
    def test_raw_decoder_accepts_symbolic_questions_and_preserves_greedy_batching(self):
        torch.manual_seed(17)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
            bos_token_id=1,eos_token_id=31,pad_token_id=0)).eval()
        tok=Tokenizer();rows=[{'problem_id':'a','prompt':'1 4 5','question_role':'development'},
                             {'problem_id':'b','prompt':'1 3','question_role':'development'}]
        cfg={'seed':17,'temperature':0.,'top_p':.95,'top_k':20,'max_new_tokens':8,'add_special_tokens':False}
        both=generate_condition_raw(model,tok,None,{'name':'B1'},rows,cfg)
        with torch.inference_mode():expected=model.generate(**tok([r['prompt'] for r in rows]),do_sample=False,max_new_tokens=8)
        for row,raw,suffix in zip(rows,both,expected[:,3:].tolist()):
            if 31 in suffix:suffix=suffix[:suffix.index(31)+1]
            self.assertEqual(raw['token_ids'],suffix)
            self.assertNotIn('is_correct',raw)
            solo=generate_condition_raw(model,tok,None,{'name':'B1'},[row],cfg)[0]
            self.assertEqual(raw['token_ids'],solo['token_ids'])
        sampled={**cfg,'temperature':.7,'top_k':5}
        first=generate_condition_raw(model,tok,None,{'name':'B1'},rows,sampled)
        reordered=generate_condition_raw(model,tok,None,{'name':'B1'},rows[::-1],sampled)[::-1]
        self.assertEqual([r['token_ids'] for r in first],[r['token_ids'] for r in reordered])

    def test_top_k_is_applied_before_nucleus_and_preserves_threshold_ties(self):
        logits=torch.tensor([[4.,3.,2.,1.]]).repeat(3,1);u=torch.tensor([.01,.8,.999])
        self.assertTrue(set(sample_from_uniform(logits,u,1.,1.,top_k=2).tolist()) <= {0,1})
        self.assertEqual(sample_from_uniform(logits,u,1.,.5,top_k=2).tolist(),[0,0,0])
        tied=torch.tensor([[4.,3.,3.,1.]]).repeat(3,1)
        self.assertIn(2,sample_from_uniform(tied,u,1.,1.,top_k=2).tolist())

    def test_absolute_vector_matches_asc_and_finished_rows_stay_unchanged(self):
        layer=torch.nn.Identity();h=torch.arange(1,25).reshape(2,3,4).to(torch.bfloat16)
        vector=torch.tensor([.4,-.7,1.8,.12]);mask=torch.tensor([[0,0,1],[0,0,0]])
        old=ResidualAddition(layer,vector*.25)
        with old.applied(mask):expected=layer(h)
        controller=FixedVectorController(layer,vector,'absolute_vector',.25);controller.begin({},2);controller.live[1]=False
        actual=layer(h);diag=controller.end()
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(diag[1]['modified_positions'],0)

    def test_relative_vector_matches_requested_norm_without_encoder(self):
        layer=torch.nn.Identity();h=torch.tensor([[[1.,2.,3.,4.],[2.,1.,4.,3.]]]);before=h.clone()
        control=FixedVectorController(layer,torch.tensor([1.,2.,3.,4.]),'relative_vector',.2)
        control.begin({},1);actual=layer(h);diag=control.end()
        self.assertTrue(torch.equal(actual[:,:-1],before[:,:-1]));self.assertTrue(torch.equal(h,before))
        self.assertAlmostEqual(diag[0]['max_delta_to_hidden_norm_fraction'],.2,places=6)

    def pool(self):
        cfg={'methods':{'B1':{},'B2':{}},'candidates_per_question':4,'candidate_seed_stride':100003,
             'generation':{'seed':17},'selection_seed':73}
        qs=[{'problem_id':p,'question':'Q'+p,'answer':'2'} for p in ('p','q')];rows=[]
        for q in qs:
            for method in cfg['methods']:
                for i in range(4):
                    rows.append({**q,'gold_answer':q['answer'],'condition':method,'candidate_index':i,
                        'seed':generation_stream_seed(settings_for_candidate(cfg,i)['seed'],q['problem_id']),
                        'response':'duplicate' if i<2 else str(i),'output_token_count':[8,4,7,2][i],
                        'is_correct':not (q['problem_id']=='q' and method=='B2'),'hit_max_new_tokens':i==3})
        return cfg,qs,rows

    def test_selection_keeps_full_coverage_and_deduplicates_before_ranking(self):
        cfg,qs,rows=self.pool();selected,coverage,common=select_candidates(rows,qs,cfg)
        self.assertEqual(common,['p']);self.assertEqual(coverage['B1']['eligible_questions'],2)
        self.assertEqual(coverage['B2']['eligible_questions'],1)
        self.assertEqual(selected['B1']['p']['candidate_index'],2)
        self.assertIn(selected['B0']['p']['candidate_index'],(0,2))
        again=select_candidates(rows[::-1],qs,cfg)[0]
        self.assertEqual(selected,again)

    def test_missing_duplicates_and_changed_streams_are_rejected(self):
        cfg,qs,rows=self.pool();audit_candidate_grid(rows,qs,cfg)
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):audit_candidate_grid(bad,qs,cfg)
        bad=deepcopy(rows);bad[0]['seed']+=1
        with self.assertRaises(ValueError):audit_candidate_grid(bad,qs,cfg)


if __name__=='__main__':unittest.main()
