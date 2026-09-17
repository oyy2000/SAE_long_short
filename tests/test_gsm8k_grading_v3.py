import unittest
from unittest.mock import patch
from length_budget_distill.gsm8k_grading_v3 import grade_gsm8k_response as grade


class FinalQuantityTests(unittest.TestCase):
    def test_inline_and_quoted_markers(self):
        for text in ('The price is $400. Answer: 400', '“Answer: 400”', '**Answer:** 400', '### Answer:\n400', 'Answer: 400.'):
            self.assertTrue(grade(text, '400')['is_correct'], text)

    def test_explanatory_quantities_are_not_alternatives(self):
        for text, gold in (('Answer: The team has 300 points after 5 games.', '300'),
                           ('Answer: They need 760 more cards to reach 1000.', '760'),
                           ('Answer: Mark will spend $90 over 6 weeks.', '90'),
                           ('Answer: 60\n\nThere are 60 pieces left.', '60')):
            self.assertTrue(grade(text, gold)['is_correct'])

    def test_reject_ambiguity_and_do_not_search_for_gold(self):
        for text in ('Answer: 41 or 42', 'Answer: 41-42', 'Answer: 40 + 2',
                     'Intermediate result 42; no conclusion.', 'Answer: '):
            self.assertFalse(grade(text, '42')['is_correct'], text)
        self.assertFalse(grade('Answer: They need 41 cards to reach 42.', '42')['is_correct'])
        self.assertFalse(grade('Answer: 42\nAnswer: 43', '42')['is_correct'])

    def test_number_formats_and_box_priority(self):
        for text, gold in ((r'Answer: \$32,\!348', '32348'), ('#### -0.5', '-0.5'),
                           ('Answer: .5', '0.5'), ('Answer: 1.2e3', '1200'),
                           (r'Answer: \boxed{\frac{1}{2}}', '0.5'),
                           (r'Answer: 40; final $\boxed{42}$', '42')):
            self.assertTrue(grade(text, gold)['is_correct'], text)

    def test_timeout_fails_closed(self):
        from math_verify.errors import TimeoutException
        with patch('length_budget_distill.gsm8k_grading_v3.grade_boxed_math', side_effect=TimeoutException('fixture')):
            result = grade('Answer: 42', '42')
        self.assertFalse(result['is_correct'])
        self.assertEqual(result['error_type'], 'TimeoutException')


if __name__ == '__main__':
    unittest.main()
