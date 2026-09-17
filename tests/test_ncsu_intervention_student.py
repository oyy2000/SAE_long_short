"""Missing-shard and common-support safeguards for conditional distillation."""
import copy
import unittest
from length_budget_distill.ncsu_intervention_student import select_common

class TestCommonSupport(unittest.TestCase):
    def fixture(self):
        qs={p:{'answer':'1'} for p in ('a','b')}
        rows=[dict(problem_id=p,condition=c,candidate_index=0,seed=17,gold_answer='1',response='Answer: 1',
                   is_correct=True,hit_max_new_tokens=False,output_token_count=4)
              for p in qs for c in ('no_steering','selected_target','matched_random')]
        return qs,rows,{p:{} for p in qs}

    def test_cap_excludes_question_from_every_arm(self):
        qs,rows,natural=self.fixture()
        rows[1]['hit_max_new_tokens']=True
        support,selected,_=select_common(qs,rows,natural,1)
        self.assertEqual(support,['b'])
        self.assertTrue(all([r['problem_id'] for r in arm]==['b'] for arm in selected.values()))

    def test_duplicate_cannot_substitute_for_missing_candidate(self):
        qs,rows,natural=self.fixture()
        rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Missing, duplicate'):
            select_common(qs,rows,natural,1)

    def test_independently_seeded_arm_rejected(self):
        qs,rows,natural=self.fixture()
        rows[1]['seed']=42
        with self.assertRaisesRegex(ValueError,'not paired'):
            select_common(qs,rows,natural,1)

    def test_incorrect_verifier_metadata_rejected(self):
        qs,rows,natural=self.fixture()
        rows[0]['response']='Answer: 2'
        with self.assertRaisesRegex(ValueError,'Verifier or gold'):
            select_common(qs,rows,natural,1)

if __name__=='__main__':unittest.main()
