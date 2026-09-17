"""Source identity and unreviewed-flag admission tests for cohort migration."""
import unittest

from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.reviewed_math_cohort import overlay_row, ADDED_FIELDS, unique_index
from length_budget_distill.typed_math_grading import compile_answer_spec


class CohortBindingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {'timeout_seconds': 5, 'max_expression_characters': 4000}
        self.row = {'problem_id': 'fixture', 'dataset': 'math_train', 'question': 'Find the center.',
                    'answer': '(3,-1)', 'question_role': 'development', 'near_component_id': 'keep'}
        self.row['answer_spec'] = compile_answer_spec(self.row, self.cfg)
        self.diag = {'problem_id': 'fixture', 'input_row_sha256': canonical_sha256(self.row),
                     'question_role': 'development', 'review_flags': []}

    def test_preserves_every_source_field_and_labels_automatic_provenance(self):
        result = overlay_row(self.row, self.diag, None, self.cfg)
        self.assertEqual({k: v for k, v in result.items() if k not in ADDED_FIELDS}, self.row)
        self.assertEqual(result['reviewed_answer'], {'mode': 'single', 'kind': 'tuple', 'answers': ['(3,-1)']})
        self.assertTrue(result['answer_definition_provenance']['origin'].startswith('automatic_'))

    def test_flagged_row_cannot_silently_inherit_old_gold(self):
        with self.assertRaisesRegex(ValueError, 'lacks a validated'):
            overlay_row(self.row, {**self.diag, 'review_flags': ['multiple_reference_boxes']}, None, self.cfg)

    def test_review_cannot_bind_different_role_or_source(self):
        reviewed = {**self.diag, 'reviewed_answer': {'mode': 'all', 'kind': 'symbolic', 'answers': ['-3', '5']}}
        for changed in ({'question_role': 'student_pool'}, {'input_row_sha256': 'stale'}):
            with self.assertRaisesRegex(ValueError, 'does not bind'):
                overlay_row(self.row, self.diag, {**reviewed, **changed}, self.cfg)

    def test_duplicate_definition_ids_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            unique_index([self.row, self.row], 'fixture')
