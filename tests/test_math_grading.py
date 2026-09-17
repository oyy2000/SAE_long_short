"""Regression cases guarding against numeric-prefix grading on symbolic answers."""
import unittest
from length_budget_distill.math_grading import extract_boxed, grade_boxed_math
from length_budget_distill.gsm8k_grading import grade_gsm8k_response


class TestMathGrading(unittest.TestCase):
    def test_nested_and_final_boxes(self):
        self.assertEqual(extract_boxed(r'\boxed{0}, finally \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(extract_boxed(r'\boxed{\{1,2\}}'), r'\{1,2\}')
        self.assertIsNone(extract_boxed(r'\boxed{1} then \boxed{\frac{1}{2}'))

    def test_equivalent_fraction(self):
        self.assertTrue(grade_boxed_math(r'\boxed{0.5}', r'\frac{1}{2}')['is_correct'])
        self.assertFalse(grade_boxed_math(r'\boxed{1}', r'\frac{1}{2}')['is_correct'])

    def test_algebra_and_sign(self):
        self.assertTrue(grade_boxed_math(r'\boxed{(x+1)^2}', r'x^2+2x+1')['is_correct'])
        self.assertFalse(grade_boxed_math(r'\boxed{-2}', '2')['is_correct'])

    def test_intermediate_value_cannot_override_final(self):
        self.assertFalse(grade_boxed_math(r'2 then \boxed{3}', '2')['is_correct'])
        self.assertEqual(grade_boxed_math('The calculation used 2.', '2')['status'], 'missing_final_box')

    def test_tuple_order_and_interval_boundary(self):
        self.assertFalse(grade_boxed_math(r'\boxed{(2,1)}', '(1,2)')['is_correct'])
        self.assertFalse(grade_boxed_math(r'\boxed{[0,1]}', '(0,1)')['is_correct'])

    def test_markdown_units_and_tex_thousands(self):
        self.assertTrue(grade_gsm8k_response('**Answer:**\nJessica has **\\$15** more.', '15')['is_correct'])
        self.assertTrue(grade_gsm8k_response(r'\boxed{8\ \text{minutes}}', '8')['is_correct'])
        self.assertTrue(grade_gsm8k_response(r'\boxed{\$37,\!500}', '37500')['is_correct'])
        self.assertFalse(grade_gsm8k_response('**Answer:**\nIt could be 15 or 16.', '15')['is_correct'])
        self.assertFalse(grade_gsm8k_response('Answer: 1/2', '1')['is_correct'])


if __name__=='__main__':unittest.main()
