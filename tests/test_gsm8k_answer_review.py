import unittest
from unittest.mock import patch
from length_budget_distill.gsm8k_answer_review import inspect_answer,validate_decisions,case_key,score_answer,truncated_without_final_answer,apply_reviewed_grade,VERSION


class AnswerReviewTests(unittest.TestCase):
    def test_no_guess_on_multiple_prose_quantities(self):
        for text in ('Answer: Each of the 5 employees pays $11.',
                     'Answer: The team has 300 points after 5 games.',
                     'Answer: 5 tickets cost $11.', 'Answer: $11 for each of 5 people.',
                     'Answer: They need 760 cards to reach 1000.'):
            value=inspect_answer(text)
            self.assertEqual(value['status'],'needs_review');self.assertIsNone(value['answer'])

    def test_unambiguous_formats_remain_automatic(self):
        for text,answer in ((r'\boxed{\frac12}',r'\frac12'),('Inline result. Answer: 400.','400'),
                            ('Answer: There are 42 objects.','42'),
                            ('Answer: There are 42 objects, exactly 42.','42')):
            value=inspect_answer(text);self.assertEqual(value['status'],'automatic');self.assertEqual(value['answer'],answer)

    def test_unsupported_final_forms_require_review(self):
        for text in ('Answer: 41 or 42','Answer: 6 * 7','Therefore each pays eleven.',
                     'Answer: There are not 42 objects.'):
            self.assertEqual(inspect_answer(text)['status'],'needs_review')
        self.assertEqual(inspect_answer('')['status'],'invalid')

    def test_decision_coverage_and_staleness(self):
        cases=[{'case_id':'x'}];valid={'case_id':'x','decision':'answer','answer':'11','reason':'The asserted payment is eleven.'}
        self.assertEqual(validate_decisions(cases,[valid])['x']['answer'],'11')
        for rows in ([],[valid,valid],[{**valid,'case_id':'y'}],[{**valid,'answer':None}],[{**valid,'reason':''}]):
            with self.assertRaises(ValueError):validate_decisions(cases,rows)
        self.assertNotEqual(case_key('How many?', 'Answer: 5'),case_key('How much?', 'Answer: 5'))

    def test_invalid_generated_box_does_not_abort_the_full_audit(self):
        with patch('length_budget_distill.gsm8k_answer_review.grade_gsm8k_response',return_value={'status':'unparsed_prediction','is_correct':False}):
            self.assertFalse(score_answer('bad expression','42',reviewed=False,timeout_seconds=5)['is_correct'])
            with self.assertRaises(ValueError):score_answer('bad expression','42',reviewed=True,timeout_seconds=5)
        with patch('length_budget_distill.gsm8k_answer_review.grade_gsm8k_response',return_value={'status':'unparsed_gold','is_correct':False}):
            with self.assertRaises(ValueError):score_answer('42','bad gold',reviewed=False,timeout_seconds=5)

    def test_truncation_policy_cannot_discard_a_declared_answer(self):
        def record(text,cap=True):return {'prediction_text':text,'hit_max_new_tokens':cap,'inspection':inspect_answer(text)}
        self.assertTrue(truncated_without_final_answer(record('First calculate 6 times 7 and then')))
        for text in ('Answer: 42',r'\boxed{42}','The final answer is 42.'):
            self.assertFalse(truncated_without_final_answer(record(text)))
        self.assertFalse(truncated_without_final_answer(record('First calculate 6 times 7 and then',False)))

    def test_reviewed_grade_requires_identical_prediction_and_accounting(self):
        old={'problem_id':'p1','question':'How much does each pay?',
            'prediction_text':'Answer: Each of the 5 people pays $11.',
            'answer':'11','output_tokens':16,'hit_max_new_tokens':False,
            'is_correct':False,'predicted_answer':'5'}
        reviewed={**old,'group':'example','gold_answer':'11','grader_version':VERSION,
            'case_id':case_key(old['question'],old['prediction_text']),
            'is_correct':True,'predicted_answer':'11','status':'graded','review_decision':{'answer':'11'}}
        def apply(value):return apply_reviewed_grade(old,value,text_field='prediction_text',gold_field='answer',tokens_field='output_tokens')
        result=apply(reviewed)
        self.assertTrue(result['is_correct']);self.assertFalse(result['legacy_is_correct'])
        self.assertFalse(old['is_correct'])
        for field,value in [('prediction_text','Answer: 11.'),('gold_answer','12'),
                            ('output_tokens',17),('hit_max_new_tokens',True),
                            ('question','A different question?'),('case_id','stale'),('problem_id','p2')]:
            with self.subTest(field=field),self.assertRaises(ValueError):apply({**reviewed,field:value})


if __name__=='__main__':unittest.main()
