import copy
import unittest
from length_budget_distill.sae_full_sequence_screen import full_sequence_decision, stable_features


class FullSequenceScreenTests(unittest.TestCase):
    selection={'require_first64_direction':False,'primary_metric':'token_mean_activation',
               'confirmation_holm_alpha':.05,'minimum_test_absolute_paired_d':.15,
               'minimum_decoder_cosine':.5,'minimum_activation_correlation':.5}

    def candidate(self,fid=1):
        return {'feature_id':fid,'direction':'long','confirmed':False,
                'first_64_direction_replicated':False,'confirmation_holm_p_value':1e-5,
                'metrics':{'token_mean_activation':{'test':{'paired_d':-1.}},
                           'first_64_token_mean_activation':{'test':{'paired_d':.01}}}}

    def test_early_sign_flip_no_longer_disqualifies(self):
        result=full_sequence_decision(self.candidate(),self.selection)
        self.assertTrue(result['full_sequence_confirmed'])
        self.assertFalse(result['early_direction_diagnostic_only'])

    def test_wrong_full_sign_still_fails(self):
        row=self.candidate();row['metrics']['token_mean_activation']['test']['paired_d']=1.
        self.assertFalse(full_sequence_decision(row,self.selection)['full_sequence_confirmed'])

    def test_significance_and_effect_thresholds_unchanged(self):
        for p,d in [(.051,-1.),(.001,-.149),(float('nan'),-1.),(.01,float('inf'))]:
            row=self.candidate();row['confirmation_holm_p_value']=p
            row['metrics']['token_mean_activation']['test']['paired_d']=d
            self.assertFalse(full_sequence_decision(row,self.selection)['full_sequence_confirmed'])

    def test_boundaries(self):
        row=self.candidate();row['confirmation_holm_p_value']=.05
        row['metrics']['token_mean_activation']['test']['paired_d']=-.15
        self.assertTrue(full_sequence_decision(row,self.selection)['full_sequence_confirmed'])

    def test_three_seed_triangle_retained_but_cosine_not_relaxed(self):
        config={'selection':self.selection,'conditions':['full'],'seeds':[17,42,73]}
        candidates={('full',s):[self.candidate(fid)] for s,fid in [(17,1),(42,2),(73,3)]}
        pairs={}
        for a,b,fa,fb in [(17,42,1,2),(17,73,1,3),(42,73,2,3)]:
            pairs[f'full__{a}_{b}']=[{'left_feature_id':fa,'right_feature_id':fb,
                'decoder_cosine':.55,'test_trace_activation_correlation':.8,
                'mutual_nearest_neighbor':True,'same_direction':True}]
        original=copy.deepcopy(candidates)
        self.assertEqual(len(stable_features({'pair_matches':pairs},candidates,config)[0]),1)
        self.assertEqual(candidates,original)
        pairs['full__17_73'][0]['decoder_cosine']=.495
        self.assertEqual(stable_features({'pair_matches':pairs},candidates,config)[0],[])


if __name__=='__main__':unittest.main()
