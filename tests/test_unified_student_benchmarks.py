"""Check locked cohorts, complete model grids and nonduplicated timing."""
from copy import deepcopy
import unittest

from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.unified_student_benchmarks import (
    COUNTS, required_cells, benchmark_ratios, common_development_cap,
    audit_benchmark_cohort, audit_benchmark_grid, audit_batch_timings, cap_summary_for_evaluation)


class StudentBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'students': {'qwen': {}}, 'methods': ['B'+str(i) for i in range(8)],
                    'student_seeds': [17, 42, 73], 'tokenskip_ratios': [.5, .6, .7, .8, .9, 1.],
                    'evaluation_stage': {'students': ['qwen'], 'methods': ['B1', 'B4', 'B5', 'B7'],
                                         'seeds': [17, 42, 73]}}

    def test_complete_stage_has_one_base_and_all_training_seeds(self):
        cells = required_cells(self.cfg)
        self.assertEqual(len(cells), 13)
        self.assertEqual(sum(c['method'] == 'base' for c in cells), 1)
        self.assertEqual({c['cell'] for c in cells if c['method'] == 'B4'},
                         {f'qwen/B4/seed_{s}' for s in [17, 42, 73]})
        self.assertEqual(benchmark_ratios('base', self.cfg), [None])
        changed = deepcopy(self.cfg); changed['evaluation_stage']['methods'].append('B6')
        self.assertEqual(benchmark_ratios('base', changed), [None])
        self.assertEqual(benchmark_ratios('B6', changed), [.5, .6, .7, .8, .9, 1.])
        for field in ('students', 'methods', 'seeds'):
            changed = deepcopy(self.cfg); changed['evaluation_stage'][field] *= 2
            with self.assertRaises(ValueError): required_cells(changed)

    def test_common_cap_requires_every_adapter_and_no_favorable_subset(self):
        summaries = {'base': {'admissible_caps': [1024, 2048, 4096]},
                     'B4/17': {'admissible_caps': [4096]}}
        self.assertEqual(common_development_cap(summaries, ['base', 'B4/17']), 4096)
        with self.assertRaises(ValueError): common_development_cap(summaries, ['base', 'B4/17', 'B7/17'])
        summaries['B4/17']['admissible_caps'] = []
        with self.assertRaises(ValueError): common_development_cap(summaries, ['base', 'B4/17'])

    def test_budget_uses_actual_prompts_and_keeps_every_tokenskip_ratio(self):
        q = [{'problem_id': 'd0', 'question_role': 'development'}]
        dcfg = {'caps': [10, 20], 'proposal_maximum_cap_hit_rate': 0., 'proposal_maximum_accuracy_drop': 0.}
        rows = [{'problem_id': 'd0', 'question_role': 'development', 'ratio': ratio, 'max_new_tokens': cap,
                 'grade': {'is_correct': True}, 'output_tokens': 8,
                 'hit_max_new_tokens': ratio == .5 and cap == 10}
                for ratio in [None, *self.cfg['tokenskip_ratios']] for cap in [10, 20]]
        base = cap_summary_for_evaluation(rows, q, 'base', dcfg, self.cfg)
        skip = cap_summary_for_evaluation(rows, q, 'B6', dcfg, self.cfg)
        self.assertEqual(base['evaluated_ratios'], [None]); self.assertEqual(base['proposed_cap'], 10)
        self.assertEqual(skip['evaluated_ratios'], self.cfg['tokenskip_ratios'])
        self.assertEqual(skip['proposed_cap'], 20)
        with self.assertRaises(ValueError):
            cap_summary_for_evaluation([r for r in rows if r['ratio'] != 1.], q, 'B6', dcfg, self.cfg)

    def test_all_five_cohorts_split_and_hard_parent_mapping_are_locked(self):
        questions = []
        for dataset, count in COUNTS.items():
            for index in range(count):
                source = index+50 if dataset == 'gsm8k' else index
                row = {'dataset': dataset, 'problem_id': f'{dataset}-{source:05d}',
                       'source_index': source, 'question_role': 'locked_evaluation'}
                if dataset == 'gsm8k_hard': row['parent_problem_ids'] = [f'gsm8k-{index+50:05d}']
                questions.append(row)
        audit_benchmark_cohort(questions)
        for altered in [questions[:-1], questions+questions[:1]]:
            with self.assertRaises(ValueError): audit_benchmark_cohort(altered)
        for role in ['development', 'student_pool', 'smoke']:
            changed = deepcopy(questions); changed[0]['question_role'] = role
            with self.assertRaises(ValueError): audit_benchmark_cohort(changed)
        changed = deepcopy(questions); changed[0]['source_index'] = 0
        with self.assertRaises(ValueError): audit_benchmark_cohort(changed)
        changed = deepcopy(questions)
        next(q for q in changed if q['dataset'] == 'gsm8k_hard')['parent_problem_ids'] = ['gsm8k-00000']
        with self.assertRaises(ValueError): audit_benchmark_cohort(changed)

    def test_missing_rows_wrong_model_cap_and_parent_mapping_are_rejected(self):
        q = {'problem_id': 'hard-0', 'dataset': 'gsm8k_hard', 'question_role': 'locked_evaluation',
             'parent_problem_ids': ['gsm8k-00050']}
        rows = [{'problem_id': q['problem_id'], 'ratio': ratio, 'cell': 'qwen/B6/seed_17',
                 'max_new_tokens': 2048, 'dataset': q['dataset'], 'question_role': q['question_role'],
                 'source_record_sha256': canonical_sha256(q), 'parent_problem_ids': q['parent_problem_ids']}
                for ratio in [.5, 1.]]
        args = [[q], [.5, 1.], 2048, 'qwen/B6/seed_17']
        audit_benchmark_grid(rows, *args)
        for bad in [rows[:-1], rows+rows[:1], [rows[0], rows[0]]]:
            with self.assertRaises(ValueError): audit_benchmark_grid(bad, *args)
        for key, value in [('cell', 'qwen/base'), ('max_new_tokens', 1024),
                           ('parent_problem_ids', []), ('source_record_sha256', 'other')]:
            bad = deepcopy(rows); bad[0][key] = value
            with self.assertRaises(ValueError): audit_benchmark_grid(bad, *args)

    def test_batch_cost_is_measured_once_and_must_be_finite(self):
        rows = [{'batch_id': 'a', 'problem_id': str(i), 'ratio': None,
                 'amortized_generation_wall_seconds': 1.5} for i in range(2)]
        batches = [{'batch_id': 'a', 'problem_ids': ['0', '1'], 'ratio': None, 'generation_wall_seconds': 3.}]
        audit_batch_timings(rows, batches)
        for bad in [[], batches+batches]:
            with self.assertRaises(ValueError): audit_batch_timings(rows, bad)
        bad = deepcopy(rows); bad[0]['amortized_generation_wall_seconds'] = 3.
        with self.assertRaises(ValueError): audit_batch_timings(bad, batches)
        for seconds in [0., -1., float('nan'), float('inf')]:
            bad = deepcopy(batches); bad[0]['generation_wall_seconds'] = seconds
            with self.assertRaises(ValueError): audit_batch_timings(rows, bad)


if __name__ == '__main__': unittest.main()
