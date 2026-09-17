"""Protect cap selection from test leakage, missing ratios and wrong adapters."""
from copy import deepcopy
import unittest

from length_budget_distill.student_cap_validation import audit_cap_grid, summarize_cap_grid
from length_budget_distill.student_evaluation_protocol import (evaluation_cell, evaluation_ratios,
                                                               validate_training_identity)


class StudentCapValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'students': {'qwen': {}}, 'methods': ['B'+str(i) for i in range(8)],
                    'student_seeds': [17, 42, 73], 'tokenskip_ratios': [.5, 1.],
                    'caps': [10, 20], 'proposal_maximum_cap_hit_rate': 0.,
                    'proposal_maximum_accuracy_drop': 0.}
        self.questions = [{'problem_id': str(i), 'question_role': 'development'} for i in range(2)]
        self.rows = [{'problem_id': q['problem_id'], 'question_role': 'development', 'ratio': r,
                      'max_new_tokens': c, 'grade': {'is_correct': True},
                      'output_tokens': 8, 'hit_max_new_tokens': False}
                     for q in self.questions for r in [.5, 1.] for c in [10, 20]]

    def test_every_ratio_must_pass_and_no_point_is_invented(self):
        summary = summarize_cap_grid(self.rows, self.questions, [.5, 1.], self.cfg)
        self.assertEqual(summary['proposed_cap'], 10)
        rows = deepcopy(self.rows)
        next(r for r in rows if r['ratio'] == .5 and r['max_new_tokens'] == 10)['grade']['is_correct'] = False
        self.assertEqual(summarize_cap_grid(rows, self.questions, [.5, 1.], self.cfg)['proposed_cap'], 20)
        next(r for r in rows if r['ratio'] == .5 and r['max_new_tokens'] == 20)['hit_max_new_tokens'] = True
        self.assertIsNone(summarize_cap_grid(rows, self.questions, [.5, 1.], self.cfg)['proposed_cap'])

    def test_missing_duplicate_and_test_cohorts_cannot_select_caps(self):
        for rows in (self.rows[:-1], self.rows[:-1]+[self.rows[0]], self.rows+self.rows[:1]):
            with self.assertRaises(ValueError): audit_cap_grid(rows, self.questions, [.5, 1.], [10, 20])
        for role in ('student_pool', 'evaluation', 'synthetic_interface_only'):
            questions = deepcopy(self.questions); questions[0]['question_role'] = role
            with self.assertRaises(ValueError): audit_cap_grid(self.rows, questions, [.5, 1.], [10, 20])
        rows = deepcopy(self.rows); rows[0]['question_role'] = 'evaluation'
        with self.assertRaises(ValueError): audit_cap_grid(rows, self.questions, [.5, 1.], [10, 20])

    def test_base_is_not_repeated_as_training_seeds_and_tokenskip_keeps_ratios(self):
        self.assertEqual(evaluation_ratios('base', self.cfg), [None, .5, 1.])
        self.assertEqual(evaluation_ratios('B6', self.cfg), [.5, 1.])
        self.assertEqual(evaluation_ratios('B4', self.cfg), [None])
        self.assertEqual(evaluation_cell('qwen', 'B4', 17, self.cfg), 'qwen/B4/seed_17')
        for args in [('qwen', 'base', 17), ('qwen', 'B4', None), ('qwen', 'B4', 99), ('other', 'base', None)]:
            with self.assertRaises(ValueError): evaluation_cell(*args, self.cfg)

    def test_synthetic_wrong_seed_wrong_base_and_incomplete_adapters_are_rejected(self):
        marker = {'training_complete': True}
        metrics = {'training_complete': True, 'student': 'qwen', 'method': 'B4', 'seed': 17,
                   'optimizer_steps': 100, 'nonzero_lora_b_tensors': 12}
        run = {'student': {'model_name': '/model'}, 'training': {'seed': 17, 'data_seed': 17}}
        args = ['qwen', 'B4', 17, {'snapshot_path': '/model'}]
        validate_training_identity(marker, metrics, run, *args)
        for bad in ({'training_complete': False}, {'training_complete': True, 'synthetic_only': True}):
            with self.assertRaises(ValueError): validate_training_identity(bad, metrics, run, *args)
        for field, value in [('seed', 42), ('method', 'B7'), ('optimizer_steps', 0), ('nonzero_lora_b_tensors', 0)]:
            changed = dict(metrics, **{field: value})
            with self.assertRaises(ValueError): validate_training_identity(marker, changed, run, *args)
        changed = deepcopy(run); changed['student']['model_name'] = '/wrong'
        with self.assertRaises(ValueError): validate_training_identity(marker, metrics, changed, *args)


if __name__ == '__main__': unittest.main()
