from copy import deepcopy
import unittest
from length_budget_distill.sae_generation_analysis import audit_cell,audit_batches,confirmation_tests
from length_budget_distill.utility_analysis import paired_sign_randomization


class GenerationAuditTest(unittest.TestCase):
    def fixtures(self):
        spec={'name':'unmodified','rho':0.};question={'problem_id':'p','question':'Q','answer':'2'}
        row={**question,'gold_answer':'2','condition':'unmodified','spec':spec,'token_ids':[3,4,9],
             'generated_tokens':2,'hit_max_new_tokens':False,'retokenized_tokens':2,
             'region_token_counts':{'reasoning_body':1,'answer_marker':0,'answer_suffix':1},
             'diagnostics':{'max_delta_to_hidden_norm_fraction':0.,'modified_positions':0}}
        return spec,question,row

    def test_duplicate_and_wrong_condition_are_rejected(self):
        spec,q,row=self.fixtures();audit_cell([row],[q],spec,4,9,.005)
        with self.assertRaises(ValueError):audit_cell([row,row],[q],spec,4,9,.005)
        row['spec']={**spec,'rho':.3}
        with self.assertRaises(ValueError):audit_cell([row],[q],spec,4,9,.005)

    def test_eos_and_unmodified_perturbation_are_checked(self):
        spec,q,row=self.fixtures()
        for bad in (dict(row,token_ids=[3,9,4]),dict(row,generated_tokens=3),dict(row,hit_max_new_tokens=True)):
            with self.assertRaises(ValueError):audit_cell([bad],[q],spec,4,9,.005)
        row['diagnostics']['modified_positions']=1
        with self.assertRaises(ValueError):audit_cell([row],[q],spec,4,9,.005)

    def test_batch_cost_is_counted_once_and_missing_batches_fail(self):
        rows=[{'batch_id':'b','condition':'c','amortized_generation_wall_seconds':2.} for _ in range(3)]
        batch={'batch_id':'b','condition':'c','questions':3,'generation_wall_seconds':6.}
        self.assertEqual(audit_batches(rows,[batch]),{'c':6.})
        with self.assertRaises(ValueError):audit_batches(rows,[batch,batch])
        with self.assertRaises(ValueError):audit_batches(rows,[])
        rows[0]['amortized_generation_wall_seconds']=6.
        with self.assertRaises(ValueError):audit_batches(rows,[batch])

    def test_randomization_exact_enumeration_and_support(self):
        left={str(i):1. for i in range(4)};right={p:0. for p in left}
        result=paired_sign_randomization(left,right,samples=100,seed=17)
        self.assertEqual(result['draws'],16);self.assertEqual(result['p_value'],.125)
        self.assertEqual(paired_sign_randomization(left,left,samples=100,seed=17)['p_value'],1.)
        with self.assertRaises(ValueError):paired_sign_randomization(left,{'x':0.},samples=100,seed=17)
        with self.assertRaises(ValueError):paired_sign_randomization(left,{**right,'0':float('nan')},samples=100,seed=17)

    def test_randomization_monte_carlo_is_seeded_and_nonzero(self):
        left={str(i):float(i+1) for i in range(20)};right={p:0. for p in left}
        result=paired_sign_randomization(left,right,samples=999,seed=17)
        self.assertEqual(result,paired_sign_randomization(left,right,samples=999,seed=17))
        self.assertEqual(result['p_value'],.001)
        self.assertEqual(result['method'],'monte_carlo_paired_sign_randomization')

    def test_confirmation_retains_all_metrics_in_one_holm_family(self):
        plan={'reference_rho':.3,'references':['unmodified','answer_format__rho0.3'],
            'primary_metrics':['is_correct','generated_tokens','reasoning_body'],'questions':4,
            'randomization_samples':100,'randomization_seed':17,'alpha':.05}
        rows=[]
        for condition in ['sae_short_8__rho0.3',*plan['references']]:
            for i in range(4):
                sae=condition.startswith('sae_short')
                rows.append({'problem_id':str(i),'condition':condition,'family':'dose','is_correct':sae,
                    'generated_tokens':10 if sae else 20,'region_token_counts':{'reasoning_body':5 if sae else 15}})
        result=confirmation_tests(rows,plan)
        self.assertEqual(len(result),6)
        self.assertTrue(all(r['holm_family_size']==6 and r['holm_adjusted_p_value']==.75 for r in result))
        with self.assertRaises(ValueError):confirmation_tests(rows[:-1],plan)
        with self.assertRaises(ValueError):confirmation_tests(rows+[rows[0]],plan)


if __name__=='__main__':unittest.main()
