import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from length_budget_distill.gsm8k_recipe_alignment import aligned_config, queue
from length_budget_distill.gsm8k_pilot_runtime import check_cell


class AlignmentTests(unittest.TestCase):
    def test_parent_kept_and_conditional_seed_gate_replaced_by_fixed_matrix(self):
        parent={'training':{'learning_rate':5e-5},'lora':{'r':8},'execution_marker':'old',
                'child_roots':{'sft':'old'},'question_template':'KEEP','grading':{'version':3}}
        reference={'training':{'learning_rate':2e-5,'max_length':2048,'num_train_epochs':1},
                   'student':{'lora':{'r':4,'dropout':.05}},'student_seeds':[17,42,73],
                   'evaluation':{'max_new_tokens':512,'batch_size':32}}
        spec={'checkpoint_root':'checkpoints/test','methods':['B1','B7__0.1','B7__0.3'],
              'evaluation_repetition_penalty':1.1}
        original=copy.deepcopy(parent)
        cfg=aligned_config(parent,reference,spec,Path('/example/result'))
        self.assertEqual(parent,original)
        self.assertEqual(cfg['execution_marker'],'/example/result/protocol/FROZEN.json')
        self.assertNotIn('child_roots',cfg)
        self.assertEqual(cfg['question_template'],'KEEP');self.assertEqual(cfg['grading'],parent['grading'])
        self.assertEqual(cfg['training']['learning_rate'],2e-5);self.assertEqual(cfg['lora']['r'],4)
        self.assertEqual(cfg['caps'],[512]);self.assertEqual(len(cfg['registered_student_cells']),10)
        self.assertEqual(cfg['repetition_penalty'],1.1)
        for method,seed in cfg['registered_student_cells']:check_cell(cfg,method,seed)
        for method,seed in [('B7__0.2',17),('B1',101),('base',42),('B0',17)]:
            with self.assertRaises(ValueError):check_cell(cfg,method,seed)

    def test_legacy_seed_gate_is_preserved(self):
        cfg={'strengths':{'B3':[.1],'B4':[.1],'B7':[.1]}}
        check_cell(cfg,'B1',17)
        with self.assertRaises(ValueError):check_cell(cfg,'foreign',17)

    def test_large_allocator_cache_keeps_h200_and_reuses_completed_first_cell(self):
        cfg={'result_root':'/run','checkpoint_root':'/adapters','execution_marker':'/marker',
             'alignment':{'gpu_memory_safety_margin_mib':8192,'l40s_runtime_multiplier':5},
             'runtime':{'minimum_free_mib':40000,'maximum_gpu_lanes':4},
             'registered_student_cells':[['base',17],['B1',17]],'evaluation_shards':4}
        training={'optimizer_steps':256,'actual_epoch':1,'peak_gpu_reserved_mib':37920,
                  'peak_gpu_allocated_mib':13026,'elapsed_seconds_including_batch_audit':62}
        evaluation={'peak_gpu_allocated_mib':11000,'elapsed_seconds':500}
        with patch('length_budget_distill.gsm8k_recipe_alignment.verify'), \
             patch('length_budget_distill.gsm8k_recipe_alignment.read_json',side_effect=[training,evaluation]), \
             patch('length_budget_distill.gsm8k_recipe_alignment.save') as save, \
             patch('length_budget_distill.gsm8k_recipe_alignment.runtime.submit',side_effect=range(100,108)) as submit:
            queue(cfg)
        calls=submit.call_args_list
        self.assertEqual(len(calls),8)
        self.assertTrue(all(c.kwargs['route']=='h200' for c in calls[:-1]))
        self.assertFalse(any(c.args[2]=='train' for c in calls))
        self.assertFalse(save.call_args.args[1]['l40s_eligible'])
