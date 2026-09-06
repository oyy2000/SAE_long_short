"""Candidate deduplication helpers."""

from __future__ import annotations

import re


def normalized_completion_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def build_length_agnostic_teacher_prompt(question: str) -> str:
    """Reproduce the sealed parent pool's unconstrained teacher prompt."""

    return (
        "You are a careful math teacher. Solve the problem correctly with visible "
        "step-by-step reasoning. Do not target a particular response length; use the "
        "amount of detail that follows naturally from your solution. End with a line "
        "in the form: Answer: <final answer>.\n\n"
        f"Problem:\n{question}"
    )
