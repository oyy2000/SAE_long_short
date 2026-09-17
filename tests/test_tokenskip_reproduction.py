"""Guard exact author-data text and distinct ratio/cap evaluation conditions."""
import unittest
from length_budget_distill.tokenskip_reproduction import unpack_author_input,evaluation_cells,expected_training_steps
from length_budget_distill.compression_baselines import TOKEN_SKIP_INSTRUCTION

class TestTokenSkipReproduction(unittest.TestCase):
    def test_source_text_and_ratio_roundtrip(self):
        row={'instruction':TOKEN_SKIP_INSTRUCTION,'input':'Question?<|eot_id|>0.6<|eot_id|>'}
        q,r,p=unpack_author_input(row)
        self.assertEqual(q,'Question?');self.assertEqual(r,.6)
        self.assertEqual(p,row['instruction']+'\n'+row['input'])
        with self.assertRaises(ValueError):unpack_author_input({**row,'input':'Question?<|eot_id|>0.3<|eot_id|>'})

    def test_all_ratios_and_two_caps_without_double_counting_identity(self):
        cfg={'ratios':[.5,.6,.7,.8,.9,1.], 'evaluation':{'cap_policies':['fixed','scaled'],'max_new_tokens':512}}
        cells=evaluation_cells(cfg,'replica')
        self.assertEqual(len(cells),11);self.assertEqual(len({c['name'] for c in cells}),11)
        self.assertEqual(next(c['max_new_tokens'] for c in cells if c['name']=='ratio_0.6__scaled'),307)
        self.assertEqual(len(evaluation_cells(cfg,'base')),1)

    def test_pinned_trainer_step_count_matches_author_state(self):
        cfg={'training':{'num_train_epochs':3,'per_device_train_batch_size':1,'gradient_accumulation_steps':8}}
        self.assertEqual(expected_training_steps(5974,cfg),2238)

if __name__=='__main__':unittest.main()
