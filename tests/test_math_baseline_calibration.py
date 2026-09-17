import unittest
from length_budget_distill.math_baseline_calibration import validate_roles, make_reference_pool
from length_budget_distill.baseline_reproduction import grade_prediction


GRADING = {'timeout_seconds':5, 'max_expression_characters':4000,
           'gsmhard_absolute_tolerance':1e-6, 'gsmhard_relative_tolerance':1e-6,
           'olympiad_absolute_tolerance':0.}


class Tokenizer:
    def encode(self, text, **kwargs): return list(text)
    def apply_chat_template(self, messages, **kwargs): return messages[0]['content']


class MathCalibrationTest(unittest.TestCase):
    def test_role_and_near_component_separation(self):
        roles = [[{'problem_id':str(i),'near_component_id':str(i),'question_role':role}]
                 for i, role in enumerate(('calibration','development','student_pool','dap_development_reserved'))]
        validate_roles(*roles)
        roles[2][0]['near_component_id'] = '0'
        with self.assertRaises(ValueError): validate_roles(*roles)

    def test_context_is_required_and_tuple_order_is_preserved(self):
        cfg = {'grading':{'method':'typed_math_v2','config':GRADING}}
        row = {'dataset':'math_train','question':'Find the center.','answer':'(3,-1)'}
        with self.assertRaises(ValueError): grade_prediction(cfg, r'\boxed{(3,-1)}', row['answer'])
        self.assertTrue(grade_prediction(cfg,r'\boxed{(3,-1)}',row['answer'],source=row)['is_correct'])
        self.assertFalse(grade_prediction(cfg,r'\boxed{(-1,3)}',row['answer'],source=row)['is_correct'])

    def test_radix_is_not_silently_graded_as_decimal(self):
        cfg = {'grading':{'method':'typed_math_v2','config':GRADING}}
        row = {'dataset':'math_train','question':'Write in base 2.','answer':'101_2'}
        self.assertTrue(grade_prediction(cfg,r'\boxed{101_2}',row['answer'],source=row)['is_correct'])
        self.assertFalse(grade_prediction(cfg,r'\boxed{101_{10}}',row['answer'],source=row)['is_correct'])

    def test_reference_filter_retains_exclusion_reasons(self):
        row = {'problem_id':'a','dataset':'math_train','question':'Find the center.',
               'answer':'(3,-1)','reference_solution':r'The center is \boxed{(3,-1)}.'}
        cfg = {'seed':17,'grading':{'method':'typed_math_v2','config':GRADING},'asc':{'pairs':1,'max_sequence_length':1000},
               'prompt_policy':{'question_template':'{question}'}}
        rows = [row,dict(row,problem_id='b',reviewed_gold_override=True),
                dict(row,problem_id='c',reference_solution=r'\boxed{(-1,3)}')]
        pool, excluded = make_reference_pool(rows,Tokenizer(),cfg)
        self.assertEqual([r['problem_id'] for r in pool],['a'])
        self.assertEqual({r['reason'] for r in excluded},
                         {'gold_corrected_original_reasoning_not_repaired','reference_not_verified'})


if __name__ == '__main__': unittest.main()
