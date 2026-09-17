import unittest

from length_budget_distill.reviewed_math_calibration import replay_saved_prefix
from length_budget_distill.baseline_reproduction import grade_prediction


class CalibrationReplayTest(unittest.TestCase):
    def test_new_first_correct_candidate_changes_stop_without_consuming_later_attempts(self):
        pool = [{'problem_id': p} for p in ['first', 'newly_correct', 'old_last']]
        attempts = [{'problem_id': 'old_last', 'candidate_index': 0, 'eligible_pair': True},
                    {'problem_id': 'first', 'candidate_index': 0, 'eligible_pair': False},
                    {'problem_id': 'first', 'candidate_index': 1, 'eligible_pair': True},
                    {'problem_id': 'newly_correct', 'candidate_index': 0, 'eligible_pair': True},
                    {'problem_id': 'newly_correct', 'candidate_index': 1, 'eligible_pair': True}]
        chosen, consumed, visited = replay_saved_prefix(pool, attempts, pairs=2, max_candidates=2)
        self.assertEqual([r['problem_id'] for r, _ in chosen], ['first', 'newly_correct'])
        self.assertEqual(visited, ['first', 'newly_correct'])
        self.assertEqual(len(consumed), 3)

    def test_missing_earlier_candidate_does_not_allow_selecting_a_later_correct_one(self):
        pool = [{'problem_id': 'p'}]
        rows = [{'problem_id': 'p', 'candidate_index': 1, 'eligible_pair': True}]
        with self.assertRaisesRegex(ValueError, 'Missing necessary'):
            replay_saved_prefix(pool, rows, pairs=1, max_candidates=2)

    def test_duplicate_candidate_cannot_supply_two_pairs(self):
        row = {'problem_id': 'p', 'candidate_index': 0, 'eligible_pair': True}
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            replay_saved_prefix([{'problem_id': 'p'}], [row, row], pairs=1, max_candidates=2)

    def test_dispatch_requires_complete_reviewed_target_and_original_gold_binding(self):
        cfg = {'grading': {'method': 'reviewed_math_v1', 'config': {'timeout_seconds': 5, 'max_expression_characters': 4000}}}
        source = {'answer': '3/2', 'reviewed_answer': {'mode': 'all', 'kind': 'symbolic', 'answers': ['-3', '3/2']}}
        self.assertTrue(grade_prediction(cfg, r'\boxed{-3,3/2}', '3/2', source=source)['is_correct'])
        self.assertFalse(grade_prediction(cfg, r'\boxed{3/2}', '3/2', source=source)['is_correct'])
        with self.assertRaises(ValueError):
            grade_prediction(cfg, r'\boxed{-3,3/2}', '-3', source=source)
