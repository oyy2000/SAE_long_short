import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import torch
from safetensors.torch import save_file
from length_budget_distill.sae_generation_controls import (WindowDirectionController, direction_feature_sets,
    conditions, validate_confirmation_transfer, select_generation_cohort)


class MarkerTokenizer:
    def decode(self, tokens, **kwargs):
        pieces = {1:'Answer', 2:':', 3:'word ', 4:' 2'}
        return ''.join(pieces[i] for i in tokens)


class GenerationControlTests(unittest.TestCase):
    def control(self, temp, layer):
        path = Path(temp)/'sae.safetensors'
        save_file({'activation_scale':torch.tensor(1.), 'activation_mean':torch.zeros(3),
            'encoder_weight':torch.eye(3), 'encoder_bias':torch.zeros(3),
            'decoder_weight':torch.eye(3), 'decoder_bias':torch.zeros(3)}, str(path))
        return WindowDirectionController(directions={'d':torch.tensor([1., 0., 0.])}, tokenizer=MarkerTokenizer(),
            marker_pattern=r'(?m)^(?P<marker>Answer):', short_ids=[0], long_ids=[1], random_ids=[2],
            torch_module=torch, checkpoint_path=path, layer_module=layer, k=2,
            maximum_delta_fraction=.3, device='cpu', dtype=torch.float32)

    def spec(self, **kwargs):
        return dict(name='window', mode='short', count=1, rho=.3, direction='d', start=0, end=None, window='full', **kwargs)

    def test_fixed_window_changes_only_last_live_position(self):
        with tempfile.TemporaryDirectory() as temp:
            layer = torch.nn.Identity(); controller = self.control(temp, layer)
            spec = self.spec(); spec.update(window='fixed', start=1, end=2)
            controller.begin(spec, 2); controller.live[1] = False
            h = torch.ones(2, 3, 3)
            for step in (0, 1, 2):
                controller.step = step; changed = layer(h)
                self.assertTrue(torch.equal(changed[:, :-1], h[:, :-1]))
                self.assertTrue(torch.equal(changed[1], h[1]))
                self.assertEqual(torch.equal(changed[0], h[0]), step != 1)
            result = controller.end()
            self.assertEqual(result[0]['modified_positions'], 1)
            self.assertEqual(result[1]['modified_positions'], 0)

    def test_marker_completed_then_next_logit_activated_per_row(self):
        with tempfile.TemporaryDirectory() as temp:
            layer = torch.nn.Identity(); controller = self.control(temp, layer)
            spec = self.spec(); spec['window'] = 'after_marker'
            controller.begin(spec, 2); h = torch.ones(2, 1, 3)
            self.assertTrue(torch.equal(layer(h), h))
            controller.observe_tokens(torch.tensor([1, 3]), 0)
            self.assertTrue(torch.equal(layer(h), h))
            controller.observe_tokens(torch.tensor([2, 4]), 1)
            controller.step = 2; changed = layer(h)
            self.assertFalse(torch.equal(changed[0], h[0])); self.assertTrue(torch.equal(changed[1], h[1]))
            self.assertTrue(controller.live.all())
            result = controller.end()
            self.assertEqual(result[0]['first_marker_completed_step'], 1)
            self.assertIsNone(result[1]['first_marker_completed_step'])

    def test_before_marker_stops_without_ending_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            layer = torch.nn.Identity(); controller = self.control(temp, layer)
            spec = self.spec(); spec['window'] = 'before_marker'
            controller.begin(spec, 1); h = torch.ones(1, 1, 3)
            self.assertFalse(torch.equal(layer(h), h))
            controller.observe_tokens(torch.tensor([1]), 0); controller.observe_tokens(torch.tensor([2]), 1)
            controller.step = 2
            self.assertTrue(torch.equal(layer(h), h)); self.assertTrue(controller.live.all())
            self.assertEqual(controller.end()[0]['modified_positions'], 1)

    def test_feature_subsets_have_fixed_membership_and_no_duplicate_single_zero(self):
        sae = {'short_features':list(range(8)), 'random_feature_sets':[list(range(8, 16)), list(range(16, 24)), list(range(24, 32))]}
        sets = direction_feature_sets(sae)
        self.assertEqual(sets['sae_top_1'], [0]); self.assertNotIn('sae_single_0', sets)
        for i in range(8): self.assertEqual(set(sets[f'sae_leave_out_{i}']), set(range(8))-{i})
        cfg = {'sae':sae, 'dose_directions':['sae_short_8','random_sae_1','random_sae_2','random_sae_3','dense_reference_minus_generated','answer_format'],
               'doses':[.05,.1,.2,.3], 'ablation_rho':.3, 'window_tokens':64}
        self.assertEqual(len(conditions(cfg, 'dose')), 25)
        self.assertEqual(len(conditions(cfg, 'ablation')), 24)
        top = next(r for r in conditions(cfg, 'ablation') if r['name'] == 'sae_top_1')
        self.assertEqual(top['injected_feature_ids'], [0]); self.assertEqual(top['measured_short_feature_ids'], list(range(8)))

    def test_confirmation_role_isolation_and_duplicate_ids(self):
        rows=[{'problem_id':'a','question_split':'dev'}, {'problem_id':'b','question_split':'confirmation'}]
        self.assertEqual(select_generation_cohort(rows,'confirmation',1),[rows[1]])
        for values,split,count in [(rows,'discovery',1),(rows,'confirmation',2),
                                    (rows+[dict(rows[0],question_split='confirmation')],'confirmation',2)]:
            with self.assertRaises(ValueError):select_generation_cohort(values,split,count)

    def test_confirmation_cannot_reselect_dose_or_change_controls(self):
        cfg={key:{} for key in ('generation','teacher','sae')}
        cfg.update(evaluation_split='confirmation',dose_directions=['sae_short_8','answer_format','dense_reference_minus_generated',
            'random_sae_1','random_sae_2','random_sae_3'],ablation_rho=.3,window_tokens=64,ablation_questions=64,
            ablation_subset_seed=3,smoke_conditions=['unmodified'],shards=4,norm_rounding_tolerance=.005,
            answer_heading_pattern='Answer:',readback_root='readback',input_root='inputs',doses=[.3])
        cfg['confirmation_plan']={'reference_rho':.3,'primary_metrics':['is_correct','generated_tokens','reasoning_body'],
            'references':['unmodified']+[f'{name}__rho0.3' for name in cfg['dose_directions'][1:]]}
        development=deepcopy(cfg);development['doses']=[.05,.1,.2,.3]
        validate_confirmation_transfer(cfg,development,.3)
        for key,value in [('doses',[.2]),('generation',{'seed':99}),('window_tokens',32),
                          ('dose_directions',['sae_short_8']),('evaluation_split','dev')]:
            changed=deepcopy(cfg);changed[key]=value
            with self.assertRaises(ValueError):validate_confirmation_transfer(changed,development,.3)


if __name__ == '__main__': unittest.main()
