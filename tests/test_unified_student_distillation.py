"""Reject partial method support, held-out leakage and TokenSkip format loss."""
from copy import deepcopy
import unittest

from length_budget_distill.compression_baselines import tokenskip_prompt, tokenskip_completion
from length_budget_distill.unified_student_distillation import METHODS, common_student_rows, unique_by_id


class TestUnifiedStudentData(unittest.TestCase):
    def setUp(self):
        self.questions = [{'problem_id':f'q{i}','question':f'Question {i}?','answer':str(i),
            'dataset':'MATH','question_role':'student_pool','near_component_id':f'group{i}'} for i in range(3)]
        self.selected = {}
        for method in METHODS:
            self.selected[method] = {}
            for q in self.questions:
                row = {'problem_id':q['problem_id'],'question':q['question'],'gold_answer':q['answer'],
                    'is_correct':True,'hit_max_new_tokens':False,'response':method+' full reasoning.', 'candidate_index':1}
                if method == 'B6':
                    row.update(ratio=.6,assigned_for_sft=True,final_answer_grade={'is_correct':True},
                        prompt=tokenskip_prompt(q['question'],.6),completion=tokenskip_completion('Compressed text.',q['answer']))
                self.selected[method][q['problem_id']] = row
        self.template = '{question}\nReason step by step and use a boxed final answer.'

    def build(self, questions=None, selected=None):
        return common_student_rows(self.questions if questions is None else questions,
                                   self.selected if selected is None else selected,self.template,[.5,.6,.7,.8,.9,1.])

    def test_all_eight_intersection_preserves_full_coverage_and_order(self):
        del self.selected['B7']['q2']; del self.selected['B4']['q1']
        datasets,audit=self.build()
        self.assertEqual(audit['common_ids'],['q0']); self.assertEqual(audit['source_questions'],3)
        self.assertEqual(audit['method_support']['B1'],3)
        self.assertEqual(audit['exclusions'],[{'problem_id':'q1','missing_methods':['B4']},
                                             {'problem_id':'q2','missing_methods':['B7']}])
        for rows in datasets.values(): self.assertEqual([r['problem_id'] for r in rows],['q0'])

    def test_baseline_identity_and_conditional_prompt_are_preserved(self):
        data,_=self.build()
        self.assertEqual(data['B0'][0]['completion'],'B0 full reasoning.')
        self.assertEqual(data['B1'][0]['completion'],'B1 full reasoning.')
        self.assertEqual(data['B2'][0]['prompt'],data['B1'][0]['prompt'])
        self.assertEqual(data['B6'][0]['prompt'],tokenskip_prompt('Question 0?',.6))
        self.assertEqual(data['B6'][0]['compression_ratio'],.6)
        self.assertEqual(len(data['B6']),3)

    def test_missing_method_empty_common_or_duplicate_question_fails(self):
        with self.assertRaises(ValueError):self.build(selected={k:v for k,v in self.selected.items() if k!='B7'})
        wrong=deepcopy(self.selected);wrong['B7']={}
        with self.assertRaises(ValueError):self.build(selected=wrong)
        with self.assertRaises(ValueError):unique_by_id(self.questions+[self.questions[0]])

    def test_held_out_or_foreign_or_changed_question_fails(self):
        wrong=deepcopy(self.questions);wrong[0]['question_role']='development'
        with self.assertRaises(ValueError):self.build(questions=wrong)
        wrong=deepcopy(self.selected);wrong['B1']['q0']['question']='Different problem'
        with self.assertRaises(ValueError):self.build(selected=wrong)
        wrong=deepcopy(self.selected);wrong['B1']['foreign']=wrong['B1']['q0']
        with self.assertRaises(ValueError):self.build(selected=wrong)

    def test_wrong_capped_or_unconditioned_selected_trace_fails(self):
        for key,value in [('is_correct',False),('hit_max_new_tokens',True),('gold_answer','wrong')]:
            wrong=deepcopy(self.selected);wrong['B5']['q0'][key]=value
            with self.assertRaises(ValueError):self.build(selected=wrong)
        wrong=deepcopy(self.selected);wrong['B6']['q0']['prompt']='Question 0?'
        with self.assertRaises(ValueError):self.build(selected=wrong)
        wrong=deepcopy(self.selected);wrong['B6']['q0']['assigned_for_sft']=False
        with self.assertRaises(ValueError):self.build(selected=wrong)


if __name__=='__main__':unittest.main()
