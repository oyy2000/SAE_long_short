import json
from pathlib import Path
import unittest

from length_budget_distill.openthoughts_reference_audit import check_reference, propagated_holdouts
from length_budget_distill.math_cohort_freeze import components

CFG = json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())


class ReferenceTests(unittest.TestCase):
    def test_parsed_wrong_answer_fails_independent_anchor(self):
        row = {'problem_id': 'q', 'question': 'What is 2+2?', 'reference_box_candidate': '5',
               'math_anchor': {'problem_id': 'math', 'question': 'What is 2+2?', 'answer': '4', 'dataset': 'math_train'}}
        result = check_reference(row, CFG)
        self.assertTrue(result['reference_self_grade']['is_correct'])
        self.assertFalse(result['math_anchor_grade']['is_correct'])
        self.assertTrue(result['requires_review'])

    def test_equivalent_anchor_agreement_is_not_training_certificate(self):
        row = {'problem_id': 'q', 'question': 'Compute a half.', 'reference_box_candidate': r'\frac{1}{2}',
               'math_anchor': {'problem_id': 'math', 'question': 'Compute a half.', 'answer': '0.5', 'dataset': 'math_train'}}
        result = check_reference(row, CFG)
        self.assertTrue(result['math_anchor_grade']['is_correct'])
        self.assertFalse(result['formal_training_ready'])
        row['math_anchor']['question'] = 'Compute a third.'
        with self.assertRaises(ValueError): check_reference(row, CFG)

    def test_generated_answer_cannot_fill_reference(self):
        with self.assertRaises(ValueError):
            check_reference({'problem_id': 'q', 'question': '2+2?', 'deepseek_solution': r'\boxed{4}'}, CFG)

    def test_holdout_propagates_through_non_candidate_bridge(self):
        groups = components(['holdout_match', 'proof_bridge', 'candidate', 'separate'],
                            [('holdout_match', 'proof_bridge'), ('proof_bridge', 'candidate')])
        self.assertEqual(propagated_holdouts(groups, ['holdout_match']),
                         {'holdout_match', 'proof_bridge', 'candidate'})
        with self.assertRaises(ValueError): propagated_holdouts(groups, ['missing'])
