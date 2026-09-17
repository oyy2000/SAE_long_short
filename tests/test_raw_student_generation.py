"""The raw student stage must preserve actual development settings and scope."""
from copy import deepcopy
import unittest
from length_budget_distill.unified_math_candidates import validate_raw_student_transfer, selected_student_methods


class TestRawStudentTransfer(unittest.TestCase):
    def setUp(self):
        self.cfg = {'student_generation_stage':'raw_B0_B1_B2_sources',
            'methods':{'B1':{'kind':'unmodified'},'B2':{'kind':'unmodified'}},
            'teacher':{'revision':'fixed'},'grading':{'version':'fixed'},'generation':{'seed':17,'max_new_tokens':4096},
            'candidates_per_question':4,'candidate_seed_stride':100003,'selection_seed':73,
            'decode_policy':{'sampler':'fixed'},'selection_policy':'shortest correct'}

    def test_identical_source_recipe_is_allowed(self):
        validate_raw_student_transfer(self.cfg,deepcopy(self.cfg))

    def test_changed_cap_seed_selection_or_methods_is_rejected(self):
        for key in ('generation','selection_seed','teacher','decode_policy'):
            changed=deepcopy(self.cfg);changed[key]='changed'
            with self.assertRaises(ValueError):validate_raw_student_transfer(changed,self.cfg)
        changed=deepcopy(self.cfg);changed['methods']['B7']={'kind':'relative_vector'}
        with self.assertRaises(ValueError):validate_raw_student_transfer(changed,self.cfg)
        changed=deepcopy(self.cfg);del changed['student_generation_stage']
        with self.assertRaises(ValueError):validate_raw_student_transfer(changed,self.cfg)

    def test_selected_steering_preserves_real_directions_and_rejects_identity(self):
        conditions={f+'__dose':{'kind':'absolute_vector' if f=='B4' else 'relative_vector',
                    'strength':.25,'layer_index':16 if f=='B4' else 17,'vector_key':f} for f in ('B3','B4','B7')}
        points={'all_families_have_admissible_point':True,'selected_conditions':{f:f+'__dose' for f in ('B3','B4','B7')},
                'selected_method_specs':{f:deepcopy(conditions[f+'__dose']) for f in ('B3','B4','B7')}}
        selected=selected_student_methods(points,{'methods':conditions})
        self.assertEqual(selected['B4']['layer_index'],16)
        self.assertEqual(selected['B7']['kind'],'relative_vector')
        wrong=deepcopy(points);wrong['selected_method_specs']['B4']['strength']=0.
        with self.assertRaises(ValueError):selected_student_methods(wrong,{'methods':conditions})
        wrong=deepcopy(points);wrong['all_families_have_admissible_point']=False
        with self.assertRaises(ValueError):selected_student_methods(wrong,{'methods':conditions})
        wrong=deepcopy(points);wrong['selected_conditions']['B7']='B3__dose'
        with self.assertRaises(ValueError):selected_student_methods(wrong,{'methods':conditions})


if __name__=='__main__':unittest.main()
