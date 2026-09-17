"""Native ASC audits must preserve sampled tokens and reviewed multi-answer targets."""
import json
from pathlib import Path
import unittest

from length_budget_distill.baseline_method_analysis import audit_typed_native_prediction
from length_budget_distill.factorial import canonical_sha256


class NativeMathAuditTest(unittest.TestCase):
    def setUp(self):
        grading=json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())
        self.cfg={'seed':17,'grading':{'method':'reviewed_math_v1','config':grading},
                  'validation':{'max_new_tokens':512}}
        self.source={'problem_id':'p','answer':'3/2','question_role':'development',
                     'reviewed_answer':{'mode':'all','kind':'symbolic','answers':['-3','3/2']}}
        class Tokenizer:
            eos_token_id=0
            def decode(self,tokens,**kwargs):return ''.join(chr(t) for t in tokens if t)
        self.tokenizer=Tokenizer()

    def row(self,text):
        return {'gold_answer':'3/2','seed':int(canonical_sha256([17,'p'])[:8],16),
                'sampled_token_ids':[ord(c) for c in text]+[0],'solution':text,
                'generated_tokens':len(text),'hit_max_new_tokens':False}

    def test_complete_roots_pass_and_last_root_alone_fails(self):
        good=self.row(r'The solutions are \boxed{-3} and \boxed{3/2}.')
        self.assertTrue(audit_typed_native_prediction(good,self.source,self.cfg,self.tokenizer)['is_correct'])
        incomplete=self.row(r'\boxed{3/2}')
        self.assertFalse(audit_typed_native_prediction(incomplete,self.source,self.cfg,self.tokenizer)['is_correct'])
        old={**self.cfg,'grading':{**self.cfg['grading'],'method':'typed_math_v2'}}
        with self.assertRaisesRegex(ValueError,'declared grading backend'):
            audit_typed_native_prediction(good,self.source,old,self.tokenizer)

    def test_changed_native_seed_text_and_eos_fail_before_grading(self):
        row=self.row(r'\boxed{-3;3/2}')
        for change in ({'seed':0},{'solution':r'\boxed{0}'},{'sampled_token_ids':row['sampled_token_ids'][:-1]}):
            with self.assertRaises(ValueError):
                audit_typed_native_prediction({**row,**change},self.source,self.cfg,self.tokenizer)


if __name__=='__main__':unittest.main()
