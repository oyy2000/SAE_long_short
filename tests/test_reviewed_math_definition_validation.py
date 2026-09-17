from copy import deepcopy
import unittest

from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.reviewed_math_definition_validation import validate_binding


class ReviewBindingTests(unittest.TestCase):
    def test_changed_source_or_role_is_rejected(self):
        source = {'problem_id':'example','question_role':'student_pool','question':'Find both roots.','answer':'2'}
        case = {**source, 'integrity_diagnostic':{'flags':['multiple_reference_boxes']}}
        decision = {'problem_id':'example','question_role':'student_pool','input_row_sha256':canonical_sha256(source),
                    'review_case_sha256':canonical_sha256(case), 'applied':False,
                    'reviewed_scope':'Full source reference', 'reason':'Both roots are required.',
                    'reviewed_answer_proposal':{'mode':'all','kind':'symbolic','answers':['1','2']}}
        self.assertEqual(validate_binding(decision, case), source)
        altered = deepcopy(case)
        altered['question'] = 'Find the larger root.'
        with self.assertRaises(ValueError): validate_binding(decision, altered)
        altered = deepcopy(decision)
        altered['question_role'] = 'locked_evaluation'
        with self.assertRaises(ValueError): validate_binding(altered, case)

    def test_applied_or_unsupported_proposal_is_rejected(self):
        source = {'problem_id':'example','question_role':'student_pool'}
        case = {**source, 'integrity_diagnostic':{}}
        decision = {'problem_id':'example','question_role':'student_pool','input_row_sha256':canonical_sha256(source),
                    'review_case_sha256':canonical_sha256(case), 'applied':True,
                    'reviewed_scope':'Source review', 'reason':'Complete target',
                    'reviewed_answer_proposal':{'mode':'guess','kind':'symbolic','answers':['1']}}
        with self.assertRaises(ValueError): validate_binding(decision, case)
        decision['applied'] = False
        with self.assertRaises(ValueError): validate_binding(decision, case)
