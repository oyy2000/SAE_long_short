"""Versioned robust numeric grading for new GSM8K baseline runs."""
import re
from .math_grading import extract_boxed, grade_boxed_math


def grade_gsm8k_response(prediction, gold, *, timeout_seconds=5):
    """Versioned GSM8K grader for new baselines, preserving historical scores.

    Prefer the final balanced box. Otherwise require an explicit final-answer
    heading and exactly one numerical value in its suffix. Handle Markdown,
    currency and TeX spacing without selecting an intermediate reasoning number.
    """
    boxed = extract_boxed(prediction)
    if boxed is not None:
        answer = boxed
    else:
        headings=list(re.finditer(
            r'(?im)^\s*(?:#+\s*)?(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:\s*(?:\*\*)?\s*', prediction))
        if not headings:
            return {'is_correct':False,'status':'missing_final_answer','predicted_answer':None}
        suffix=prediction[headings[-1].end():]
        values=re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?',suffix)
        if len(values)!=1:
            return {'is_correct':False,'status':'ambiguous_answer_suffix','predicted_answer':None}
        answer=values[0]
    answer=re.sub(r'\\[,!;:]', '', answer).replace(r'\$', '')
    return grade_boxed_math('\\boxed{'+answer+'}',str(gold),timeout_seconds=timeout_seconds)
