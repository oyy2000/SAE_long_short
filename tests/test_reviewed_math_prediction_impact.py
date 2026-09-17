import unittest

from length_budget_distill.reviewed_math_prediction_impact import validate_prediction_identity, nested_value


class PredictionIdentityTest(unittest.TestCase):
    def test_rejects_changed_gold_question_and_role(self):
        source = {'problem_id': 'fixture', 'question': 'Question', 'answer': '0', 'question_role': 'development'}
        group = {'question_role': 'development'}
        prediction = {'problem_id': 'fixture', 'question': 'Question', 'gold_answer': '0'}
        validate_prediction_identity(prediction, source, group)
        for change in ({'gold_answer': '1'}, {'question': 'Changed'}, {'question_role': 'student_pool'}):
            with self.assertRaises(ValueError):
                validate_prediction_identity({**prediction, **change}, source, group)

    def test_nested_tokenskip_score_is_required_without_fallback(self):
        self.assertFalse(nested_value({'final_answer_grade': {'is_correct': False}}, 'final_answer_grade.is_correct'))
        with self.assertRaises(KeyError):
            nested_value({'is_correct': True}, 'final_answer_grade.is_correct')
