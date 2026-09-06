"""GSM8K final-answer extraction and numeric verification."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
GSM8K_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def clean_answer(answer: str) -> str:
    return answer.strip().strip("$").strip().rstrip(".,").strip()


def extract_final_answer(text: str) -> Optional[str]:
    for pattern in (GSM8K_RE, BOXED_RE, ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return clean_answer(matches[-1])
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    cleaned = clean_answer(answer)
    match = NUMBER_RE.search(cleaned)
    if match:
        try:
            number = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            pass
        else:
            if number == number.to_integral_value():
                return str(int(number))
            return format(number.normalize(), "f")
    return re.sub(r"\s+", "", cleaned.replace(",", "")).lower()


def verify_answer(predicted: Optional[str], gold: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)
