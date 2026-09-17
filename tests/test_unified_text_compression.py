from collections import Counter
import json
from pathlib import Path
import hashlib
import tempfile
import unittest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from length_budget_distill.unified_text_compression import build_sources, grade_candidate, stage_tokenskip_assets
from length_budget_distill.compression_baselines import balanced_ratio_assignment,dap_messages,tokenskip_completion,tokenskip_prompt
from length_budget_distill.sae_norm_intervention import generate_condition_raw


class TextCompressionTest(unittest.TestCase):
    def test_tokenskip_offline_assets_are_staged_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';scratch=root/'scratch'
            url='https://example.invalid/cl100k_base.tiktoken';key=hashlib.sha1(url.encode()).hexdigest()
            assets=[(cache/'tiktoken'/key,b'tokenizer bytes'),
                    (cache/'nltk/tokenizers/punkt_tab/english/abbrev_types.txt',b'English abbreviations')]
            for path,data in assets:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            cfg={'runtime':{'auxiliary_cache_root':str(cache)},'tokenskip':{'offline_assets':{
                'tiktoken':{'url':url,'sha256':hashlib.sha256(assets[0][1]).hexdigest()},
                'nltk_punkt_english_sha256':{'abbrev_types.txt':hashlib.sha256(assets[1][1]).hexdigest()}}}}
            staged=stage_tokenskip_assets(cfg,scratch)
            self.assertEqual((scratch/'tiktoken'/key).read_bytes(),assets[0][1])
            self.assertEqual((scratch/'nltk/tokenizers/punkt_tab/english/abbrev_types.txt').read_bytes(),assets[1][1])
            self.assertEqual(staged['tiktoken_cache_key'],key)
            assets[0][0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'missing or changed'):stage_tokenskip_assets(cfg,root/'other_scratch')

    def test_reviewed_dap_and_tokenskip_keep_all_requested_roots(self):
        grading=json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())
        cfg={'grading_method':'reviewed_math_v1','grading':grading}
        source={'answer':'3/2','reviewed_answer':{'mode':'all','kind':'symbolic','answers':['-3','3/2']}}
        grade=grade_candidate(cfg,r'The solutions are \boxed{-3} and \boxed{3/2}.',source)
        self.assertTrue(grade['is_correct'])
        completion=tokenskip_completion('Short derivation.',grade['predicted_answer'])
        appended=grade_candidate(cfg,completion,source)
        self.assertTrue(appended['is_correct']);self.assertEqual(appended['component_count'],2)
        self.assertFalse(grade_candidate(cfg,r'Only \boxed{3/2}.',source)['is_correct'])
        with self.assertRaisesRegex(ValueError,'declared grading backend'):
            grade_candidate({'grading':grading},completion,source)

    def test_ratio_balance_is_frozen_over_full_question_pool(self):
        ids=[f'q{i}' for i in range(17)];ratios=[.5,.6,.7,.8,.9,1.]
        a=balanced_ratio_assignment(ids,ratios,42);b=balanced_ratio_assignment(ids[::-1],ratios,42)
        self.assertEqual(a,b);counts=Counter(a.values());self.assertEqual(max(counts.values())-min(counts.values()),1)
        with self.assertRaises(ValueError):balanced_ratio_assignment(['q','q'],ratios,42)

    def test_dap_uses_all_unique_correct_sources_skip_uses_shortest(self):
        questions=[{'problem_id':'a'},{'problem_id':'b'},{'problem_id':'c'}]
        rows=[]
        for pid in ('a','b'):
            for i in range(4):
                rows.append({'problem_id':pid,'condition':'B1','candidate_index':i,'response':['long','long','short','capped'][i],
                             'output_token_count':[10,9,3,2][i],'is_correct':pid=='a','hit_max_new_tokens':i==3})
        dap,skip,assignment,missing=build_sources(questions,rows,[.5,1.],17)
        self.assertEqual([(r['problem_id'],r['source_candidate_index']) for r in dap],[('a',0),('a',2)])
        self.assertEqual(skip[0]['source_candidate_index'],2);self.assertEqual(missing,['b','c'])
        self.assertEqual(assignment,balanced_ratio_assignment(['a','b','c'],[.5,1.],17))
        self.assertEqual(skip[0]['assigned_ratio'],assignment['a'])

    def test_math_answer_append_and_ratio_prompt_preserve_tuple(self):
        text=tokenskip_completion('A shortened derivation.','(3,-1)')
        self.assertIn(r'\boxed{(3,-1)}',text)
        self.assertTrue(tokenskip_prompt('Find center.',.5).endswith('<|eot_id|>0.5<|eot_id|>'))
        self.assertNotIn('<|eot_id|>',tokenskip_prompt('Find center.',1.))

    def test_raw_decoder_passes_complete_system_and_user_messages(self):
        class Tok:
            eos_token_id=31
            def apply_chat_template(self,messages,**kwargs):self.seen=messages;return 'rendered'
            def __call__(self,texts,**kwargs):return {'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
            def decode(self,tokens,**kwargs):return 'response'
        tok=Tok();model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,eos_token_id=31)).eval()
        messages=dap_messages('question','complete original trace',{'framework':'full system framework','rewrite_wrapper':'Q: {question}\nOriginal: {source_trace}'})
        generate_condition_raw(model,tok,None,{'name':'B5'},[{'problem_id':'a','messages':messages}],
                               {'seed':17,'max_new_tokens':2,'temperature':0.,'top_p':.95,'top_k':20,'add_special_tokens':False})
        self.assertEqual(tok.seen,messages);self.assertEqual(tok.seen[0]['role'],'system')
        self.assertIn('complete original trace',tok.seen[1]['content'])


if __name__=='__main__':unittest.main()
