import json
from pathlib import Path
import unittest
from unittest.mock import patch

from length_budget_distill.typed_math_grading import grade_typed_response,split_top_level

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/phase13_typed_math_grading_v2.json').read_text())


def grade(answer,gold,**metadata):
    row={'dataset':'math_train','question':'Compute the requested value.','answer':gold,**metadata}
    return grade_typed_response(r'\boxed{'+answer+'}',row,CFG)


def test_radix_keeps_the_base_and_does_not_accept_decimal_value():
    meta={'question':'Express the quotient in base 5.'}
    assert grade('43','43_5',**meta)['is_correct']
    assert grade('043_{5}','43_5',**meta)['is_correct']
    assert not grade('23','43_5',**meta)['is_correct']
    assert not grade('43_{10}','43_5',**meta)['is_correct']
    assert not grade('48_5','43_5',**meta)['is_correct']


def test_choice_is_a_letter_not_a_symbolic_variable_or_number():
    assert grade(r'\text{(E)}','E',dataset='aqua_rat')['is_correct']
    assert not grade('D or E','E',dataset='aqua_rat')['is_correct']
    assert not grade('2.718281828','E',dataset='aqua_rat')['is_correct']


def test_symbolic_letter_only_and_trigonometric_answers():
    assert grade('c+a+b','a+b+c')['is_correct']
    assert grade(r'1-\cos^2(t)',r'\sin^2 t')['is_correct']
    assert not grade('a+b-c','a+b+c')['is_correct']


def test_multi_answer_cardinality_and_tuple_order():
    meta={'dataset':'olympiadbench','answer_type':'Tuple','is_multiple_answer':True,'error_tolerance':None}
    assert grade('(2,7,13); (1,8,19)','(1,8,19),(2,7,13)',**meta)['is_correct']
    assert not grade('(19,8,1); (2,7,13)','(1,8,19),(2,7,13)',**meta)['is_correct']
    meta['answer_type']='Numerical'
    assert grade('90;90','90,90',**meta)['is_correct']
    assert not grade('90','90,90',**meta)['is_correct']
    assert not grade('90;90;90','90,90',**meta)['is_correct']


def test_absolute_tolerance_and_no_implicit_percentage_scaling():
    meta={'dataset':'olympiadbench','answer_type':'Numerical','is_multiple_answer':False,'error_tolerance':'1e-1'}
    assert grade('1.05','1',**meta)['is_correct']
    assert not grade('1.2','1',**meta)['is_correct']
    assert not grade('100','1',**meta)['is_correct']


def test_gsmhard_rounded_noninteger_and_exact_large_integer():
    assert grade(r'\frac{6}{4808993}','1.2477e-06',dataset='gsm8k_hard')['is_correct']
    assert not grade('100000001','100000000.0',dataset='gsm8k_hard')['is_correct']
    assert not grade('0.0000015','1.2477e-06',dataset='gsm8k_hard')['is_correct']


def test_interval_boundaries_and_union_order():
    meta={'dataset':'olympiadbench','answer_type':'Interval','is_multiple_answer':False,'error_tolerance':None}
    assert grade(r'(2,3]\cup[0,1)','[0,1)\\cup(2,3]',**meta)['is_correct']
    assert not grade('[0,1]','[0,1)',**meta)['is_correct']
    assert not grade('[0,1,2]','[0,1)',**meta)['is_correct']


def test_missing_invalid_and_timeout_fail_closed():
    from math_verify.errors import TimeoutException
    import length_budget_distill.typed_math_grading as module
    assert not grade_typed_response('Intermediate 42',{'dataset':'math_train','answer':'42'},CFG)['is_correct']
    assert not grade('', '42')['is_correct']
    with patch.object(module,'_symbolic',side_effect=TimeoutException('test timeout')):
        result=grade('41','42')
    assert not result['is_correct'] and result['error_type']=='TimeoutException'


def test_nested_components_and_unbalanced_input():
    assert split_top_level(r'(1,2);\frac{1}{2};[3,4)')==['(1,2)',r'\frac{1}{2}','[3,4)']
    with unittest.TestCase().assertRaises(ValueError):split_top_level('(1,2')


def test_currency_products_clock_and_signed_base():
    assert grade('32348',r'\$32,\!348')['is_correct']
    assert grade('23000',r'\$23{,}000')['is_correct']
    assert grade('t^2-49','(t-7)(t+7)')['is_correct']
    assert not grade('t^2+49','(t-7)(t+7)')['is_correct']
    assert grade('5:00',r'05\!:\!00',question='Give the time in AB:CD format.')['is_correct']
    assert not grade('5:60',r'05\!:\!00',question='Give the time in AB:CD format.')['is_correct']
    assert grade('-37','-37_8',question='Express in base 8.')['is_correct']


def test_context_preserves_ordered_points_and_quadruples():
    meta={'question':'List the points in order of increasing x-coordinate, separated by semicolons.'}
    assert grade('(-4,27);(2,15)','(-4,27);(2,15)',**meta)['is_correct']
    assert not grade('(2,15);(-4,27)','(-4,27);(2,15)',**meta)['is_correct']
    assert grade('(-1,1,0,1)','(-1,1,0,1)',question='Given the domain and range, enter the ordered quadruple.')['is_correct']


def test_interval_finite_set_and_quantity_labels():
    meta={'dataset':'olympiadbench','answer_type':'Interval','is_multiple_answer':False,'error_tolerance':None}
    assert grade(r'\{1\}\cup(-\infty,0)',r'(-\infty,0)\cup\{1\}',**meta)['is_correct']
    assert grade('(0,4]',r't(0,4]',**meta)['is_correct']
    assert not grade('[0,4]',r't(0,4]',**meta)['is_correct']
    meta['answer_type']='Expression'
    assert grade('n^2-n-1',r'm_{\max}=n^2-n-1',**meta)['is_correct']


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__=='__main__':unittest.main()
