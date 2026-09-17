import json
from pathlib import Path
import unittest

from length_budget_distill.reviewed_math_grading import grade_reviewed_response, final_value

CFG = json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())


def grade(value, answers, mode='single', kind='symbolic', **kwargs):
    return grade_reviewed_response(r'\boxed{'+value+'}', {'reviewed_answer': dict(mode=mode, kind=kind, answers=answers, **kwargs)}, CFG)['is_correct']


class ReviewedGradingTests(unittest.TestCase):
    def test_tex_single_token_arguments_preserve_visible_matrix_signs(self):
        self.assertTrue(grade(r'2007+\frac\pi 2', [r'2007+\frac{\pi}{2}']))
        target = r'\begin{pmatrix}1&-1\\1&1\end{pmatrix}'
        self.assertTrue(grade(r'\begin{pmatrix}1&-1\\1&\phantom -1\end{pmatrix}', [target]))
        self.assertFalse(grade(r'\begin{pmatrix}1&-1\\1&-1\end{pmatrix}', [target]))

    def test_spaced_digit_groups_do_not_merge_required_lists(self):
        self.assertTrue(grade(r'11,\! 111,\! 111,\! 100', ['11111111100']))
        self.assertTrue(grade('1, 234', ['1', '234'], 'all'))
        self.assertFalse(grade('1234', ['1', '234'], 'all'))

    def test_requested_rounding_preserves_units(self):
        options = {'unit_aliases': ['mph', 'miles per hour'], 'rounded_answer_requested': True}
        self.assertTrue(grade(r'\approx 8.24\text{ mph}', ['8.24'], **options))
        self.assertFalse(grade(r'\approx 8.24\text{ km/h}', ['8.24'], **options))
        self.assertFalse(grade('8.2447', ['8.24'], **options))

    def test_reviewed_case_alias_does_not_accept_swapped_labels(self):
        options = {'literal_aliases': {'ab=9, cd=4': 'D'}, 'choice_options': {'D': 'AB=9, CD=4'}}
        self.assertTrue(grade('AB=9, CD=4', ['D'], kind='choice', **options))
        self.assertTrue(grade('D', ['D'], kind='choice', **options))
        self.assertFalse(grade('AB=4, CD=9', ['D'], kind='choice', **options))

    def test_all_roots_require_full_cardinality(self):
        answers = [r'-\frac{5}{2}', r'\frac{1}{3}']
        self.assertTrue(grade(r'\frac{1}{3}, -\frac{5}{2}', answers, 'all'))
        self.assertFalse(grade(r'\frac{1}{3}', answers, 'all'))
        self.assertFalse(grade(r'\frac{1}{3}, -\frac{5}{2}, 0', answers, 'all'))

    def test_angle_multiplicity_and_point_order(self):
        self.assertTrue(grade(r'75^\circ,30^\circ,75^\circ', ['30','75','75'], 'all', 'degree'))
        self.assertFalse(grade('30,75', ['30','75','75'], 'all', 'degree'))
        points = ['(-2,18)', '(8,38)']
        self.assertTrue(grade('(-2,18);(8,38)', points, 'all', 'tuple', ordered=True))
        self.assertFalse(grade('(8,38);(-2,18)', points, 'all', 'tuple', ordered=True))

    def test_either_focus_accepts_one_but_not_both(self):
        answers = ['(-5,7)', '(-5,1)']
        for answer in answers: self.assertTrue(grade(answer, answers, 'any', 'tuple'))
        self.assertFalse(grade('(-5,7),(-5,1)', answers, 'any', 'tuple'))
        self.assertFalse(grade('(-5,4)', answers, 'any', 'tuple'))

    def test_vector_alternatives_preserve_coordinates(self):
        answers = ['(2/3,-2/3,-1/3)', '(-2/3,2/3,1/3)']
        self.assertTrue(grade(r'\begin{pmatrix}2/3\\-2/3\\-1/3\end{pmatrix}', answers, 'any', 'vector'))
        self.assertTrue(grade('(-2/3,2/3,1/3)', answers, 'any', 'vector'))
        self.assertFalse(grade('(2/3,2/3,-1/3)', answers, 'any', 'vector'))

    def test_percentage_units_do_not_create_global_factor_100_tolerance(self):
        for value in ['56', r'56\%', r'0.56=56\%']:
            self.assertTrue(grade(value, ['56'], kind='percentage'))
        self.assertFalse(grade('0.56', ['56'], kind='percentage'))
        self.assertFalse(grade(r'0.57=56\%', ['56'], kind='percentage'))
        self.assertFalse(grade('100', ['1']))

    def test_radix_checks_inside_and_outside_base(self):
        spec = {'mode':'single','kind':'radix','answers':['1331'],'base':5}
        for value in [r'\boxed{1331}', r'\boxed{1331}_5', r'\boxed{1331_5}']:
            self.assertTrue(grade_reviewed_response(value, {'reviewed_answer': spec}, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{1331}_8', {'reviewed_answer': spec}, CFG)['is_correct'])
        self.assertFalse(grade('216', ['1331'], kind='radix', base=5))

    def test_clock_meridiem_and_invalid_times(self):
        for value in ['20:34', '8:34 pm', r'8\!:\!34 \text{p.m.}']:
            self.assertTrue(grade(value, ['20:34'], kind='clock'))
        for value in ['8:34 am','20:64','24:34']:
            self.assertFalse(grade(value, ['20:34'], kind='clock'))
        self.assertTrue(grade_reviewed_response(r'\boxed{8:34}\text{p.m.}', {'reviewed_answer':{'mode':'single','kind':'clock','answers':['20:34']}}, CFG)['is_correct'])

    def test_periodic_phase_shift(self):
        for value in [r'\pi/3', r'-\pi/3', r'\pi']:
            self.assertTrue(grade(value, [r'\pi/3'], kind='periodic', period=r'2\pi/3'))
        self.assertFalse(grade('0', [r'\pi/3'], kind='periodic', period=r'2\pi/3'))

    def test_choice_label_must_agree_with_letter(self):
        options = {'B':'Circle','C':'Parabola'}
        self.assertTrue(grade(r'\text{(C) Parabola}', ['C'], kind='choice', choice_options=options))
        self.assertFalse(grade(r'\text{(C) Circle}', ['C'], kind='choice', choice_options=options))
        self.assertFalse(grade('C or B', ['C'], kind='choice', choice_options=options))

    def test_grouped_digits_and_number_words(self):
        self.assertTrue(grade(r'\dfrac{1}{29,\!322,\!216}', [r'\frac{1}{29322216}']))
        self.assertTrue(grade('four', ['4'], literal_aliases={'four':'4'}))
        self.assertFalse(grade('four or five', ['4'], literal_aliases={'four':'4'}))
        self.assertFalse(grade('(1,234)', ['(1234,0)'], kind='tuple'))

    def test_external_unary_sign_and_nested_box(self):
        row = {'reviewed_answer': {'mode':'single','kind':'symbolic','answers':['-15']}}
        self.assertTrue(grade_reviewed_response(r'\[-\boxed{15}.\]', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{15}', row, CFG)['is_correct'])
        self.assertEqual(final_value(r'2-\boxed{15}', 'symbolic'), '15')
        with self.assertRaises(ValueError): final_value(r'\boxed{1+\boxed{3}}', 'symbolic')
        with self.assertRaises(ValueError): final_value(r'\boxed{4} then \boxed{5', 'symbolic')

    def test_plus_minus_and_polynomial_assignments(self):
        self.assertTrue(grade(r'\pm4', ['-4','4'], 'all'))
        self.assertTrue(grade('g(x)=2x+1;g(x)=-2x-1', ['2x+1','-2x-1'], 'all', assignment_lhs=['g(x)']))
        self.assertFalse(grade('g(x)=2x+1', ['2x+1','-2x-1'], 'all', assignment_lhs=['g(x)']))

    def test_separate_final_boxes_and_intermediate_work(self):
        spec = {'mode':'all','kind':'symbolic','answers':['12','18'],'assignment_lhs':['k']}
        response = r'An intermediate value is $\boxed{6}$. The answers are $k=\boxed{12}$ and $k=\boxed{18}$.'
        result = grade_reviewed_response(response, {'reviewed_answer':spec}, CFG)
        self.assertTrue(result['is_correct'])
        self.assertEqual(result['final_box_count'], 2)
        self.assertFalse(grade_reviewed_response(response.replace('12', '13'), {'reviewed_answer':spec}, CFG)['is_correct'])
        scalar = {'reviewed_answer':{'mode':'single','kind':'symbolic','answers':['4']}}
        self.assertTrue(grade_reviewed_response(r'$\log_2 4=\boxed{2}$, so $\log_2(4^2)=\boxed{4}$', scalar, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{3} or \boxed{4}', scalar, CFG)['is_correct'])

    def test_separate_boxes_do_not_hide_extra_focus(self):
        spec = {'mode':'any','kind':'tuple','answers':['(-5,7)','(-5,1)']}
        result = grade_reviewed_response(r'$\boxed{(-5,7)}$ and $\boxed{(-5,1)}$', {'reviewed_answer':spec}, CFG)
        self.assertFalse(result['is_correct'])

    def test_equivalent_scalar_boxes_and_grouping(self):
        spec = {'mode':'single','kind':'clock','answers':['20:34']}
        for text in [r'$\boxed{20:34}$ or $\boxed{8:34\text{p.m.}}$', r'\boxed{20:34,8:34 pm}']:
            self.assertTrue(grade_reviewed_response(text, {'reviewed_answer':spec}, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{20:34} or \boxed{8:34 am}', {'reviewed_answer':spec}, CFG)['is_correct'])
        self.assertTrue(grade('1,008,016', ['1008016']))
        self.assertTrue(grade('1,234', ['1','234'], 'all'))
        self.assertFalse(grade('1,234', ['1234','0'], 'all'))

    def test_interval_endpoints_remain_typed(self):
        self.assertTrue(grade(r'(-\infty,2]\cup(3,4)', [r'(-\infty,2]\cup(3,4)'], kind='interval'))
        self.assertFalse(grade(r'(-\infty,2)\cup(3,4)', [r'(-\infty,2]\cup(3,4)'], kind='interval'))

    def test_unbraced_single_token_does_not_reuse_earlier_answer(self):
        row = {'reviewed_answer':{'mode':'single','kind':'symbolic','answers':['2']}}
        self.assertTrue(grade_reviewed_response(r'The largest root is $\boxed 2$.', row, CFG)['is_correct'])
        self.assertTrue(grade_reviewed_response(r'The largest root is \boxed 2.', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{2} then \boxed 9$', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{2} then \boxed 23$', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{2} then \boxed x+2$', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{2} then \boxed 2.5$', row, CFG)['is_correct'])
        self.assertFalse(grade_reviewed_response(r'\boxed{1+\boxed 2}', row, CFG)['is_correct'])

    def test_alphanumeric_radix_and_explicit_output_base(self):
        for value in [r'\text{A5}_{11}', 'A5', r'A5_{11}']:
            self.assertTrue(grade(value, ['A5'], kind='radix', base=11))
        for value in ['115',r'A5_{16}']:
            self.assertFalse(grade(value, ['A5'], kind='radix', base=11))
        self.assertTrue(grade('110111', ['110111'], kind='radix', base=2))
        self.assertFalse(grade('55', ['110111'], kind='radix', base=2))
        self.assertTrue(grade('10000', ['10000'], kind='radix', base=4))
        self.assertFalse(grade('256', ['10000'], kind='radix', base=4))

    def test_percentage_words_and_points_preserve_scale(self):
        for text in [r'17\text{ percentage points}', '17 percent', '17 pp', r'17\text{\%}']:
            self.assertTrue(grade(text, ['17'], kind='percentage'))
        self.assertFalse(grade('0.17 percent', ['17'], kind='percentage'))
        self.assertTrue(grade(r'0.56=56\text{ percent}', ['56'], kind='percentage'))
        self.assertFalse(grade(r'0.57=56\text{ percent}', ['56'], kind='percentage'))

    def test_choice_math_label_and_text_aliases(self):
        options = {'E':{'kind':'interval','answer':r'\{-1,1\}'}}
        self.assertTrue(grade(r'\text{(e) }\{1,-1\}', ['E'], kind='choice', choice_options=options))
        self.assertFalse(grade(r'\text{(e) }\{-1,0,1\}', ['E'], kind='choice', choice_options=options))
        options = {'D':'a non-horizontal line'}
        aliases = {'D':['non-horizontal line']}
        self.assertTrue(grade(r'\text{(D), non-horizontal line}', ['D'], kind='choice', choice_options=options, choice_label_aliases=aliases))
        self.assertFalse(grade(r'\text{(D), horizontal line}', ['D'], kind='choice', choice_options=options, choice_label_aliases=aliases))

    def test_registered_quantity_units_do_not_disappear(self):
        aliases = ['pound','pounds','lb','lbs']
        for value in ['13.5',r'13.5\text{ pounds}','13.5 lbs']:
            self.assertTrue(grade(value, ['13.5'], unit_aliases=aliases))
        for value in ['13.5 kg',r'13.5\text{ kg}','13.5kg',r'13.5\mathrm{kg}',r'13.5\kg']:
            self.assertFalse(grade(value, ['13.5'], unit_aliases=aliases))
        self.assertTrue(grade(r'480+480\sqrt{3}\text{ feet}', ['480+480\\sqrt{3}'], unit_aliases=['feet','ft']))
        self.assertFalse(grade(r'480+480\sqrt{3}\text{ meters}', ['480+480\\sqrt{3}'], unit_aliases=['feet','ft']))

    def test_quantity_grouping_currency_and_squared_units(self):
        aliases = ['USD','dollar','dollars']
        self.assertTrue(grade(r'1,850\text{ USD}', ['1850'], unit_aliases=aliases, currency_symbol='$'))
        self.assertTrue(grade(r'\$1.85', ['1.85'], unit_aliases=aliases, currency_symbol='$'))
        self.assertFalse(grade(r'\$2', ['2'], unit_aliases=['euros','euro']))
        aliases = ['square centimeters','square cm','cm^2','cm^{2}']
        self.assertTrue(grade(r'428\text{ cm}^2', ['428'], unit_aliases=aliases))
        self.assertFalse(grade(r'428\text{ cm}', ['428'], unit_aliases=aliases))

    def test_letter_lists_and_conic_initials(self):
        self.assertTrue(grade(r'\text{H,E}', ['E','H'], 'all', 'choice'))
        self.assertFalse(grade(r'\text{H,H}', ['E','H'], 'all', 'choice'))
        self.assertTrue(grade(r'\text{A,D,F,G,H}', ['A','D','F','G','H'], 'all', 'choice'))
        self.assertFalse(grade(r'\text{A,D,F,G}', ['A','D','F','G','H'], 'all', 'choice'))
        self.assertTrue(grade(r'\text{(P) parabola}', ['P'], kind='choice', choice_options={'P':'parabola'}))
        self.assertFalse(grade(r'\text{(P) hyperbola}', ['P'], kind='choice', choice_options={'P':'parabola'}))

    def test_ordered_text_and_explicit_spelling_alias(self):
        answers = ['Softball','Kickball','Picnic']
        self.assertTrue(grade(r'\text{Softball, Kickball, Picnic}', answers, 'all', 'text', ordered=True))
        self.assertFalse(grade(r'\text{Picnic, Kickball, Softball}', answers, 'all', 'text', ordered=True))
        self.assertTrue(grade(r'\text{Oct 30}', ['October 30'], kind='text', literal_aliases={'oct 30':'October 30'}))
        self.assertFalse(grade(r'\text{Oct 31}', ['October 30'], kind='text', literal_aliases={'oct 30':'October 30'}))

    def test_clock_seconds_and_registered_omitted_meridiem(self):
        for value in ['04:51:06', '16:51:06', '4:51:06 pm']:
            self.assertTrue(grade(value, ['16:51:06'], kind='clock', clock_meridiem='pm'))
        for value in ['4:51:06 am', '4:51:60', '4:51:05', '4:51']:
            self.assertFalse(grade(value, ['16:51:06'], kind='clock', clock_meridiem='pm'))
        self.assertFalse(grade('4:51:06', ['16:51:06'], kind='clock'))
        self.assertFalse(grade('20:34:01', ['20:34'], kind='clock'))
