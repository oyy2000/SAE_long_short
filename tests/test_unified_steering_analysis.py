import unittest
from length_budget_distill.unified_steering_analysis import operating_points


class OperatingPointTest(unittest.TestCase):
    def test_filters_pair_objective_and_tie_break(self):
        ref={'candidate_accuracy':.8,'correct_support_fraction':.9}
        rules={'maximum_candidate_accuracy_drop':.05,'maximum_correct_support_drop':.05,'maximum_cap_hit_rate':.01}
        def row(family,strength,delta,**extra):
            return dict(baseline=family,strength=strength,paired_selected_token_difference=delta,
                paired_selected_questions=50,candidate_accuracy=.8,correct_support_fraction=.9,cap_hit_rate=0.,**extra)
        data={'B3_low':row('B3',.1,-20),'B3_high':row('B3',.3,-100),'B3_tied':row('B3',.2,-20),
              'B4_bad':row('B4',1.,-200),'B7_bad':row('B7',.3,-200)}
        data['B3_high']['candidate_accuracy']=.74
        data['B4_bad']['cap_hit_rate']=.02;data['B7_bad']['correct_support_fraction']=.84
        chosen,assessed=operating_points(data,ref,rules)
        self.assertEqual(chosen['B3']['condition'],'B3_low')
        self.assertIsNone(chosen['B4']);self.assertIsNone(chosen['B7'])
        self.assertEqual(sum(r['admissible'] for r in assessed),2)

    def test_empty_common_support_is_not_an_identity_fallback(self):
        ref={'candidate_accuracy':1.,'correct_support_fraction':1.}
        rules={'maximum_candidate_accuracy_drop':0.,'maximum_correct_support_drop':0.,'maximum_cap_hit_rate':0.}
        row={'baseline':'B3','strength':.1,'candidate_accuracy':1.,'correct_support_fraction':1.,'cap_hit_rate':0.,
             'paired_selected_questions':0,'paired_selected_token_difference':None}
        chosen,assessed=operating_points({'B3_empty':row},ref,rules)
        self.assertTrue(all(v is None for v in chosen.values()));self.assertIn('no_paired_correct_support',assessed[0]['failed_filters'])


if __name__=='__main__':unittest.main()
