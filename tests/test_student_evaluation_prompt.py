from copy import deepcopy
import unittest
from length_budget_distill.student_prompts import build_unified_evaluation_prompt
from length_budget_distill.student_evaluation_preflight import compare_prompt_records


class TestStudentEvaluationPrompt(unittest.TestCase):
    def test_reviewed_choice_type_overrides_old_scalar_without_answer_leakage(self):
        cfg={'choice_instruction':'Return the option letter in a box.','question_template':'{question}\nReason.',
             'tokenskip_ratios':[.5,1.]}
        q={'question':'Which equation? (A) first (B) second','answer_spec':{'kind':'symbolic'},
           'reviewed_answer':{'kind':'choice','answers':['B']}}
        for ratio in (None,.5,1.):
            prompt=build_unified_evaluation_prompt(q,cfg,ratio=ratio)
            self.assertIn(cfg['choice_instruction'],prompt)
            changed=deepcopy(q);changed['reviewed_answer']['answers']=['A']
            self.assertEqual(prompt,build_unified_evaluation_prompt(changed,cfg,ratio=ratio))

    def test_revision_cannot_hide_changed_cohort_or_duplicate_ratio_keys(self):
        rows=[{'problem_id':'q','ratio':r,'prompt_text_sha256':'original'} for r in (None,.5,1.)]
        revised=deepcopy(rows);revised[1]['prompt_text_sha256']='revised'
        changes=compare_prompt_records(rows,revised)
        self.assertEqual([(r['problem_id'],r['ratio']) for r in changes],[('q',.5)])
        with self.assertRaises(ValueError):compare_prompt_records(rows,revised[1:])
        with self.assertRaises(ValueError):compare_prompt_records(rows,revised+[revised[0]])

    def test_choice_requirement_shared_and_no_gold_leakage(self):
        cfg={'choice_instruction':'Return the option letter in a box.','question_template':'{question}\nSolve step by step.',
             'tokenskip_ratios':[.5,1.]}
        q={'question':'Choose one:\nA)1\nB)2','answer':'A','reference_solution':'private gold rationale',
           'answer_spec':{'kind':'choice','gold_components':['A']}}
        for ratio in (None,.5,1.):
            actual=build_unified_evaluation_prompt(q,cfg,ratio=ratio)
            self.assertIn(cfg['choice_instruction'],actual);self.assertNotIn(q['reference_solution'],actual)
            changed=deepcopy(q);changed['answer']='B';changed['answer_spec']['gold_components']=['B'];changed['reference_solution']='other reference'
            self.assertEqual(actual,build_unified_evaluation_prompt(changed,cfg,ratio=ratio))

    def test_ratio_control_is_retained_and_invalid_ratio_fails(self):
        cfg={'choice_instruction':'Letter only.','question_template':'{question}\nReason.', 'tokenskip_ratios':[.5,1.]}
        q={'question':'Compute 1 + 1.','answer_spec':{'kind':'integer'}}
        self.assertEqual(build_unified_evaluation_prompt(q,cfg),'Compute 1 + 1.\nReason.')
        self.assertTrue(build_unified_evaluation_prompt(q,cfg,ratio=.5).endswith('<|eot_id|>0.5<|eot_id|>'))
        with self.assertRaises(ValueError):build_unified_evaluation_prompt(q,cfg,ratio=.4)


if __name__=='__main__':unittest.main()
