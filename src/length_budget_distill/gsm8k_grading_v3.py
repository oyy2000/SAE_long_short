"""Final-answer extraction repair; frozen v2 results remain unchanged.

The unboxed numeric policy selects the first quantity on the first nonempty
line after the last explicit answer marker. This is a syntactic GSM8K score,
not a claim that a fluent or incoherent explanation is mathematically valid.
"""
import re

from .math_grading import extract_boxed, grade_boxed_math

VERSION = 'gsm8k_explicit_answer_quantity_v3'
DECIMAL = r'[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
NUMBER = DECIMAL + r'(?:[eE][-+]?\d+)?'
QUANTITY = re.compile(r'(?<![\w.])' + NUMBER + r'(?:\s*/\s*' + NUMBER + r')?(?!\w|\.\d)')
HEADING = re.compile(r'(?i)\b(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*|(?m:^\s*####\s*)')


def extract_explicit_quantity(prediction):
    boxed = extract_boxed(prediction)
    if boxed is not None:
        return boxed, 'final_box'
    headings = list(HEADING.finditer(prediction))
    if not headings:
        return None, 'missing_final_answer'
    tail = prediction[headings[-1].end():].lstrip()
    if not tail:
        return None, 'empty_answer_line'
    # Later explanatory lines are not additional final-answer candidates.
    line = tail.splitlines()[0].replace('−', '-').replace(r'\$', '$')
    line = re.sub(r'\\[,!;:]', '', line).replace('{,}', ',')
    if re.search(r'(?i)\bor\s+\$?\s*' + NUMBER, line):
        return None, 'explicit_numeric_alternatives'
    match = QUANTITY.search(line)
    if match is None:
        return None, 'missing_answer_quantity'
    # Plain ranges and arithmetic are not single numeric answers. Boxed
    # expressions continue through the established symbolic backend.
    rest = line[match.end():]
    if re.match(r'\s*(?:[-+*/=]|to\b)\s*\$?\s*' + NUMBER, rest, re.I):
        return None, 'non_scalar_answer_expression'
    return match.group(), 'first_explicit_answer_quantity'


def grade_gsm8k_response(prediction, gold, *, timeout_seconds=5):
    from math_verify.errors import TimeoutException
    answer, rule = extract_explicit_quantity(prediction)
    if answer is None:
        return {'is_correct': False, 'status': rule, 'predicted_answer': None,
                'grader_version': VERSION, 'extraction_rule': rule}
    answer = re.sub(r'\\[,!;:]', '', answer).replace(r'\$', '').replace('{,}', ',')
    answer = re.sub(r'(?<![\w.])(' + DECIMAL + r')[eE]([-+]?\d+)',
                    lambda m: m[1] + r'\cdot10^{' + m[2] + '}', answer)
    try:
        result = grade_boxed_math(r'\boxed{' + answer + '}', str(gold), timeout_seconds=timeout_seconds)
    except TimeoutException as error:
        result = {'is_correct': False, 'status': 'grading_error', 'predicted_answer': answer,
                  'error_type': type(error).__name__, 'error': str(error)}
    return {**result, 'grader_version': VERSION, 'extraction_rule': rule}
