from copy import deepcopy
import unittest
from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.sae_seed_stability import seeded_sample_view, scratch_requirement


class SeedSampleViewTests(unittest.TestCase):
    def fixture(self):
        config={'sae':{'seed':17,'learning_rate':.0003},'token_sampling':{'seed':20260904}}
        manifest={'config_hash':canonical_sha256(config),'layers':[{'samples':[{'path':'immutable','sha256':'a'}],
            'normalizer_path':'norm','normalizer_sha256':'b'}]}
        return config,manifest

    def test_only_training_seed_changes_and_inputs_remain_identical(self):
        config,manifest=self.fixture();original=deepcopy((config,manifest))
        child,view=seeded_sample_view(config,manifest,42,'config.json','manifest.json')
        self.assertEqual((config,manifest),original)
        self.assertEqual(view['layers'],manifest['layers'])
        self.assertEqual(child['token_sampling']['seed'],20260904)
        self.assertEqual(view['config_hash'],canonical_sha256(child))
        child['sae']['seed']=17;self.assertEqual(child,config)
        self.assertEqual(view['training_seed_view']['parent_manifest_hash'],canonical_sha256(manifest))

    def test_wrong_parent_or_invalid_seed_is_rejected(self):
        config,manifest=self.fixture()
        with self.assertRaises(ValueError):seeded_sample_view(config,{**manifest,'config_hash':'changed'},42,'c','m')
        for seed in (-1,True,42.5):
            with self.assertRaises(ValueError):seeded_sample_view(config,manifest,seed,'c','m')

    def test_storage_covers_periodic_final_and_spare_checkpoints(self):
        spec={'max_steps':1500,'checkpoint_interval_steps':500}
        value=scratch_requirement(822000000,spec)
        self.assertEqual(value['checkpoint_sized_files'],5)
        self.assertGreater(value['required_bytes'],4*822000000+2**30)
        spec['max_steps']=1499
        self.assertEqual(scratch_requirement(822000000,spec)['checkpoint_sized_files'],4)
        with self.assertRaises(ValueError):scratch_requirement(1,{**spec,'checkpoint_interval_steps':0})


if __name__=='__main__':unittest.main()
