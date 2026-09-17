"""Data integrity tests for paired supplementation and answer multiplicity."""
import copy
import hashlib
import unittest
from length_budget_distill.ncsu_multi_answer import (
    unique_correct, pool_shortfalls, make_datasets, grouped_pool,
    check_candidate_records, contrast_definitions,
)


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class MultiAnswerTests(unittest.TestCase):
    def fixture(self):
        conditions=['no_steering','selected_target']
        specs=[{'name':c,'rho':0. if c=='no_steering' else .3} for c in conditions]
        cfg={'conditions':conditions,'main_specs':specs,'intervention':{'generation_seed':99},
             'supplement':{'initial_candidates':4,'max_candidates_per_question_condition':32,'round_size':4,'seed_offset':10000}}
        questions={'q1':{'answer':'1','student_prompt':'Question'}}
        rows=[]
        for c in conditions:
            for i in range(4):
                seed=int.from_bytes(hashlib.sha256(f'{99+10000+i}:q1'.encode()).digest()[:4],'little')
                rows.append(dict(problem_id='q1',condition=c,candidate_index=i,seed=seed,
                    response=f'{"reason "*(i+1)}Answer: 1',is_correct=True,hit_max_new_tokens=False,
                    output_token_count=5+i,token_ids=list(range(5+i)),gold_answer='1',
                    spec=next(s for s in specs if s['name']==c),diagnostics={
                        'max_delta_to_hidden_norm_fraction':0. if c=='no_steering' else .3,
                        'modified_positions':0 if c=='no_steering' else 5+i}))
        return cfg,questions,rows

    def test_pair_integrity_and_streams(self):
        cfg,qs,rows=self.fixture()
        check_candidate_records(rows,qs,cfg,initial_only=True)
        rows[4]['seed']+=1
        with self.assertRaisesRegex(ValueError,'random stream'):
            check_candidate_records(rows,qs,cfg)

    def test_duplicate_cannot_replace_missing_key(self):
        cfg,qs,rows=self.fixture();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            check_candidate_records(rows,qs,cfg)

    def test_wrong_gold_and_false_correct_flag_fail(self):
        cfg,qs,rows=self.fixture();rows[0]['response']='Answer: 2'
        with self.assertRaisesRegex(ValueError,'gold/verifier'):
            check_candidate_records(rows,qs,cfg)

    def test_whitespace_duplicates_and_caps_do_not_fill_shortfall(self):
        cfg,qs,rows=self.fixture()
        rows[1]['response']=' '+rows[0]['response']+'\n'
        rows[2]['hit_max_new_tokens']=True
        pool=grouped_pool(rows,qs,cfg['conditions'])
        self.assertEqual(len(unique_correct(pool['q1']['no_steering'])),2)
        self.assertEqual(pool_shortfalls(pool,4)['q1']['no_steering'],2)

    def test_nested_unique_and_exact_repetition(self):
        cfg,qs,rows=self.fixture()
        variants=[{'name':f'k{k}_{mode}','k':k,'mode':mode} for k,mode in ((1,'unique'),(2,'unique'),(4,'unique'),(2,'repeat'),(4,'repeat'))]
        pool=grouped_pool(rows,qs,cfg['conditions'])
        datasets=make_datasets(pool,qs,variants,cfg['conditions'],WordTokenizer())
        for c in cfg['conditions']:
            one=datasets[f'k1_unique__{c}'][0]['completion']
            two=[r['completion'] for r in datasets[f'k2_unique__{c}']]
            four=[r['completion'] for r in datasets[f'k4_unique__{c}']]
            self.assertEqual([one],two[:1]);self.assertEqual(two,four[:2]);self.assertEqual(len(set(four)),4)
            repeated=datasets[f'k4_repeat__{c}']
            self.assertEqual([r['completion'] for r in repeated],[one]*4)
            self.assertEqual([r['occurrence_index'] for r in repeated],[0,1,2,3])
            self.assertEqual(len({r['trace_id'] for r in repeated}),4)

    def test_new_shortest_changes_all_nested_levels(self):
        cfg,qs,rows=self.fixture()
        extra=copy.deepcopy(rows[3]);extra.update(candidate_index=4,response='Answer: 1',output_token_count=2)
        ranked=unique_correct(rows[:4]+[extra])
        self.assertEqual(ranked[0]['candidate_index'],4)

    def test_insufficient_unique_never_falls_back_to_repeat(self):
        cfg,qs,rows=self.fixture();pool=grouped_pool(rows,qs,cfg['conditions']);pool['q1']['no_steering'].pop()
        with self.assertRaisesRegex(ValueError,'Four unique'):
            make_datasets(pool,qs,[{'name':'k4_unique','k':4,'mode':'unique'}],cfg['conditions'],WordTokenizer())

    def test_nine_registered_contrasts_and_interaction_signs(self):
        contrasts=contrast_definitions();self.assertEqual(len(contrasts),9)
        self.assertEqual(contrasts['method_gap_k4_minus_k1'],{
            'k4_unique__selected_target':1,'k4_unique__no_steering':-1,
            'k1_unique__selected_target':-1,'k1_unique__no_steering':1})

if __name__=='__main__':unittest.main()
