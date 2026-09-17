"""Symbolic grading for the expanded mathematics branch; historical GSM8K is unchanged."""
from __future__ import annotations
import re


def extract_boxed(text):
    """Read the final balanced boxed/fbox expression, including nested fractions."""
    starts = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        # Escaped braces are literal set delimiters, not TeX grouping braces.
        backslashes, cursor = 0, index-1
        while cursor >= 0 and text[cursor] == '\\':
            backslashes += 1; cursor -= 1
        if backslashes % 2:
            continue
        if text[index] == '{': depth += 1
        elif text[index] == '}': depth -= 1
        if depth == 0:
            return text[start:index].strip()
    return None


def parse_math_answer(answer, *, timeout_seconds=5):
    from math_verify import parse, LatexExtractionConfig
    return parse('$'+answer.strip().strip('$')+'$',
                 extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                 fallback_mode='no_fallback', extraction_mode='first_match',
                 parsing_timeout=timeout_seconds, raise_on_error=True)


def grade_boxed_math(prediction, gold, *, timeout_seconds=5):
    """Fail closed on missing/invalid final expressions and parser timeouts.

    This protocol requests a final boxed expression. It never falls back to the
    first number in a fraction, question, or intermediate reasoning step.
    """
    from math_verify import verify
    answer = extract_boxed(prediction)
    if answer is None:
        return {'is_correct':False, 'status':'missing_final_box', 'predicted_answer':None}
    try:
        parsed_gold = parse_math_answer(gold, timeout_seconds=timeout_seconds)
        parsed_pred = parse_math_answer(answer, timeout_seconds=timeout_seconds)
        if not parsed_gold or not parsed_pred:
            return {'is_correct':False,'status':'unparsed_gold' if not parsed_gold else 'unparsed_prediction',
                    'predicted_answer':answer}
        correct = bool(verify(parsed_gold, parsed_pred, strict=True,
                              timeout_seconds=timeout_seconds, raise_on_error=True))
        return {'is_correct':correct,'status':'graded','predicted_answer':answer,
                'parsed_gold':[str(x) for x in parsed_gold],'parsed_prediction':[str(x) for x in parsed_pred]}
    except Exception as error:
        return {'is_correct':False,'status':'grading_error','predicted_answer':answer,
                'error_type':type(error).__name__,'error':str(error)}
