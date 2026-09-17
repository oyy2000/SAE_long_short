import unittest
from length_budget_distill.math_reference_integrity import diagnostic, reference_boxes


class IntegrityTests(unittest.TestCase):
    def test_two_answers_remain_distinct_and_gold_is_not_replaced(self):
        row = {'problem_id': 'q', 'dataset': 'math_train', 'question_role': 'student_pool',
               'question': 'Find all t, separated by commas.', 'answer': '-5/2',
               'reference_solution': r'The roots are \boxed{\frac{1}{3}} and \boxed{-\frac{5}{2}}.'}
        result = diagnostic(row)
        self.assertEqual([r['expression'] for r in result['reference_boxes']], [r'\frac{1}{3}', r'-\frac{5}{2}'])
        self.assertIn('multiple_reference_boxes', result['review_flags'])
        self.assertEqual(result['registered_gold'], '-5/2')
        self.assertFalse(result['gold_automatically_corrected'])

    def test_escaped_set_braces_and_nested_fraction(self):
        self.assertEqual(reference_boxes(r'\boxed{\{\frac{1}{2}, 3\}}')[0]['expression'], r'\{\frac{1}{2}, 3\}')
        self.assertIsNone(reference_boxes(r'\boxed{\boxed{3}}')[0]['expression'])

    def test_bare_base_answer_needs_context_metadata(self):
        row = {'problem_id': 'q', 'dataset': 'math_train', 'question_role': 'calibration',
               'question': 'Find the base five product.', 'answer': '1331',
               'reference_solution': r'The answer is \boxed{1331}_5.', 'answer_spec': {'kind': 'symbolic'}}
        self.assertIn('radix_context_without_typed_metadata', diagnostic(row)['review_flags'])
        row['answer_spec']['kind'] = 'radix'
        self.assertNotIn('radix_context_without_typed_metadata', diagnostic(row)['review_flags'])
