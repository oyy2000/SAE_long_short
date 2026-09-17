"""Pilot isolation, fourteen-arm support, shortening constraints and seed gates."""
import copy
import re
import unittest
from unittest.mock import patch

from length_budget_distill.gsm8k_student_pilot import condition_ids,common_training_rows,isolate_sources,ANSWER_REGION_PATTERN
from length_budget_distill.gsm8k_pilot_runtime import select_points,check_cell
from length_budget_distill.compression_baselines import tokenskip_prompt
from length_budget_distill.baseline_reproduction import grade_prediction
from length_budget_distill.unified_math_candidates import grade_candidate


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'strengths':{'B3':[.1,.2,.3],'B4':[.1,.25,.5],'B7':[.1,.2,.3]},
            'target_train_questions':16,'minimum_train_questions':8,'selection_seed':17,
            'tokenskip_ratios':[.5,.6,.7,.8,.9,1.],'question_template':'{question}\nReason.',
            'gate_minimum_accuracy_gain':.01,'seed':31,
            'decontamination':{'shingle_size':5,'jaccard_threshold':.8}}
        self.questions=[{'problem_id':f'q{i}','question':f'Compute question number {i}.','answer':str(i),
            'question_role':'student_pool','dataset':'gsm8k','near_component_id':str(i)} for i in range(20)]
        self.selected={}
        for method in condition_ids(self.cfg):
            self.selected[method]={q['problem_id']:{'problem_id':q['problem_id'],'question':q['question'],
                'gold_answer':q['answer'],'is_correct':True,'hit_max_new_tokens':False,'response':method+' solution.',
                'assigned_for_sft':True,'ratio':.6,'final_answer_grade':{'is_correct':True},
                'prompt':tokenskip_prompt(q['question'],.6),'completion':'Compressed answer.'} for q in self.questions}

    def test_fourteen_conditions_same_questions_and_one_ratio_per_question(self):
        self.assertEqual(len(condition_ids(self.cfg)),14)
        rows,audit=common_training_rows(self.questions,self.selected,self.cfg)
        self.assertEqual(audit['training_questions'],16)
        for values in rows.values():self.assertEqual([r['problem_id'] for r in values],audit['common_ids'])
        self.assertEqual(len(rows['B6']),16)
        self.assertEqual(rows['B2'][0]['prompt'],rows['B1'][0]['prompt'])
        self.assertNotEqual(rows['B0'][0]['completion'],rows['B1'][0]['completion'])

    def test_weak_strength_support_cannot_be_ignored(self):
        for i in range(5):del self.selected['B7__0.1'][f'q{i}']
        rows,audit=common_training_rows(self.questions,self.selected,self.cfg)
        self.assertIsNone(rows);self.assertEqual(audit['all_condition_common_questions'],15)
        rows,audit=common_training_rows(self.questions,self.selected,self.cfg,allow_reduced=True)
        self.assertEqual(audit['training_questions'],8)
        for values in rows.values():self.assertFalse(any(r['problem_id']=='q0' for r in values))

    def test_empty_or_missing_matrix_cannot_train(self):
        wrong=copy.deepcopy(self.selected);del wrong['B4__0.1']
        with self.assertRaises(ValueError):common_training_rows(self.questions,wrong,self.cfg)
        wrong=copy.deepcopy(self.selected);wrong['B4__0.1']={}
        with self.assertRaises(ValueError):common_training_rows(self.questions,wrong,self.cfg,allow_reduced=True)

    def test_heldout_duplicate_foreign_changed_capped_rejected(self):
        qs=copy.deepcopy(self.questions);qs[0]['question_role']='development'
        with self.assertRaises(ValueError):common_training_rows(qs,self.selected,self.cfg)
        with self.assertRaises(ValueError):common_training_rows(self.questions+[self.questions[0]],self.selected,self.cfg)
        for field,value in [('question','Different'),('is_correct',False),('hit_max_new_tokens',True),('gold_answer','other')]:
            wrong=copy.deepcopy(self.selected);wrong['B7__0.2']['q0'][field]=value
            with self.assertRaises(ValueError):common_training_rows(self.questions,wrong,self.cfg)
        wrong=copy.deepcopy(self.selected);wrong['B1']['foreign']=wrong['B1']['q0']
        with self.assertRaises(ValueError):common_training_rows(self.questions,wrong,self.cfg)

    def test_tokenskip_prompt_and_assignment_are_required(self):
        for field,value in [('prompt','plain question'),('assigned_for_sft',False),('ratio',.2)]:
            wrong=copy.deepcopy(self.selected);wrong['B6']['q0'][field]=value
            with self.assertRaises(ValueError):common_training_rows(self.questions,wrong,self.cfg)

    def test_sae_accuracy_selection_not_shortest_selection(self):
        metrics={m:{'accuracy':.7} for m in condition_ids(self.cfg)};metrics['base']={'accuracy':.6}
        lengths={m:200 for m in condition_ids(self.cfg)};lengths['B0']=300
        metrics['B7__0.1']['accuracy']=.8;metrics['B7__0.3']['accuracy']=.72
        lengths['B7__0.1']=280;lengths['B7__0.3']=100
        result=select_points(self.cfg,metrics,lengths)
        self.assertEqual(result['selected_conditions']['B7'],'B7__0.1');self.assertTrue(result['repeat_gate_passed'])
        self.assertEqual(len(result['repeat_conditions']),4)
        self.assertNotIn('base',result['repeat_conditions'])

    def test_nonshortening_high_score_cannot_be_selected_as_sae(self):
        metrics={m:{'accuracy':.7} for m in condition_ids(self.cfg)};metrics['base']={'accuracy':.6}
        lengths={m:200 for m in condition_ids(self.cfg)}
        metrics['B7__0.3']['accuracy']=.99
        result=select_points(self.cfg,metrics,lengths)
        self.assertIsNone(result['selected_conditions']['B7']);self.assertFalse(result['repeat_gate_passed'])

    def test_baseline_strongest_point_kept_even_if_longer(self):
        metrics={m:{'accuracy':.7} for m in condition_ids(self.cfg)};metrics['base']={'accuracy':.6}
        lengths={m:200 for m in condition_ids(self.cfg)};lengths['B0']=300
        metrics['B3__0.1']['accuracy']=.9;lengths['B3__0.1']=500
        result=select_points(self.cfg,metrics,lengths)
        self.assertEqual(result['champion_condition'],'B3__0.1');self.assertFalse(result['repeat_gate_passed'])

    def test_isolation_blocks_history_and_preserves_one_near_group_member(self):
        history=[{'problem_id':'h','question':'Jane owns 15 red marbles and buys 20 more. How many marbles does she own?'}]
        rows=[{'problem_id':'a','question':history[0]['question']},
              {'problem_id':'b','question':history[0]['question']+' Explain.'},
              {'problem_id':'c','question':'A train travels 90 miles in two hours. What is its average speed?'},
              {'problem_id':'d','question':'A train travels 90 miles in two hours. What is its average speed? Explain.'}]
        kept,excluded,_=isolate_sources(rows,history,self.cfg)
        self.assertEqual(len(kept),1);self.assertIn(kept[0]['problem_id'],('c','d'))
        self.assertEqual(len(excluded),3)

    def test_gsm_v3_dispatch_keeps_timeout_and_never_uses_legacy_fallback(self):
        with patch('length_budget_distill.gsm8k_grading_v3.grade_gsm8k_response',return_value={'is_correct':False}) as grade:
            result=grade_candidate({'grading_method':'gsm8k_explicit_answer_quantity_v3','grading':{'timeout_seconds':9}},'Answer: 3',{'answer':'4'})
            self.assertFalse(result['is_correct']);grade.assert_called_once_with('Answer: 3','4',timeout_seconds=9)

    def test_unknown_or_extra_seed_requires_frozen_repeat_gate(self):
        with self.assertRaises(ValueError):check_cell(self.cfg,'B8',17)

    def test_body_regions_recognize_both_boxed_and_explicit_answers(self):
        pattern=re.compile(ANSWER_REGION_PATTERN)
        self.assertEqual(pattern.search(r'Reasoning. \\boxed{3}').group(),r'\boxed{')
        self.assertEqual(pattern.search('Reasoning. Final Answer: 3').group(),'Final Answer:')
        self.assertIsNone(pattern.search('A body-only mathematical explanation.'))


if __name__=='__main__':unittest.main()
