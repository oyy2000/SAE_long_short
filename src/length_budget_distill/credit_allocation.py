"""Step segmentation and fixed-context supervision-mask construction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, List, Sequence


ANSWER_MARKER = re.compile(r"(?i)(?:\b(?:final\s+)?answer\s*[:=]|####\s*|\\boxed\{)")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class ReasoningStep:
    index: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    is_answer: bool

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


def segment_reasoning_steps(
    completion: str,
    tokenizer: Any,
    *,
    minimum_step_tokens: int = 8,
) -> List[ReasoningStep]:
    """Segment on lines first, split long sentences, then merge short spans."""

    if minimum_step_tokens <= 0:
        raise ValueError("minimum_step_tokens must be positive.")
    raw_spans = _split_answer_suffixes(completion, _line_spans(completion))
    expanded = []
    for start, end in raw_spans:
        text = completion[start:end]
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count >= 2 * minimum_step_tokens and not ANSWER_MARKER.search(text):
            expanded.extend(_sentence_spans(completion, start, end))
        else:
            expanded.append((start, end))
    merged: List[tuple[int, int]] = []
    for start, end in expanded:
        tokens = len(tokenizer.encode(completion[start:end], add_special_tokens=False))
        if (
            tokens < minimum_step_tokens
            and merged
            and not ANSWER_MARKER.search(completion[start:end])
        ):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) >= 2:
        first_tokens = len(
            tokenizer.encode(
                completion[merged[0][0] : merged[0][1]], add_special_tokens=False
            )
        )
        if first_tokens < minimum_step_tokens and not ANSWER_MARKER.search(
            completion[merged[0][0] : merged[0][1]]
        ):
            merged[1] = (merged[0][0], merged[1][1])
            merged = merged[1:]
    encoding = tokenizer(
        completion, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = [tuple(value) for value in encoding["offset_mapping"]]
    steps = []
    for index, (start, end) in enumerate(merged):
        covered = [
            token_index
            for token_index, (left, right) in enumerate(offsets)
            if right > start and left < end
        ]
        if not covered:
            continue
        text = completion[start:end]
        steps.append(
            ReasoningStep(
                index=len(steps),
                text=text,
                char_start=start,
                char_end=end,
                token_start=min(covered),
                token_end=max(covered) + 1,
                is_answer=bool(ANSWER_MARKER.search(text)),
            )
        )
    if not steps or not any(step.is_answer for step in steps):
        raise ValueError("Completion did not yield an explicit answer step.")
    return steps


def select_step_knapsack(
    steps: Sequence[ReasoningStep],
    scores: Sequence[float],
    *,
    target_reasoning_tokens: int,
    maximum_relative_gap: float = 0.05,
) -> List[int]:
    """Maximize step score among whole-step subsets that match the token budget."""

    if len(steps) != len(scores):
        raise ValueError("Step and score lengths differ.")
    if target_reasoning_tokens <= 0:
        raise ValueError("target_reasoning_tokens must be positive.")
    if not 0.0 <= maximum_relative_gap < 1.0:
        raise ValueError("maximum_relative_gap must lie in [0, 1).")
    reasoning = [
        (step, float(score)) for step, score in zip(steps, scores) if not step.is_answer
    ]
    maximum_tokens = int(
        target_reasoning_tokens * (1.0 + maximum_relative_gap) + 0.999999
    )
    # budget -> (score, tuple(step indices)); deterministic lexicographic tie break.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for step, score in reasoning:
        updated = dict(states)
        for used, (value, indices) in states.items():
            candidate_used = used + step.token_count
            if candidate_used > maximum_tokens:
                continue
            candidate = (value + score, indices + (step.index,))
            incumbent = updated.get(candidate_used)
            if (
                incumbent is None
                or candidate[0] > incumbent[0]
                or (candidate[0] == incumbent[0] and candidate[1] < incumbent[1])
            ):
                updated[candidate_used] = candidate
        states = updated
    feasible = {
        used: value
        for used, value in states.items()
        if relative_budget_gap(used, target_reasoning_tokens) <= maximum_relative_gap
    }
    if not feasible:
        raise ValueError(
            "No whole-step subset satisfies the registered token-budget tolerance."
        )
    _, (_, selected) = max(
        feasible.items(),
        key=lambda item: (item[1][0], item[0], tuple(-value for value in item[1][1])),
    )
    return sorted(selected)


def supervision_weights(
    target_token_count: int,
    steps: Sequence[ReasoningStep],
    selected_reasoning_step_indices: Sequence[int],
) -> List[float]:
    selected = set(int(value) for value in selected_reasoning_step_indices)
    weights = [0.0] * target_token_count
    for step in steps:
        if step.is_answer or step.index in selected:
            for position in range(
                step.token_start, min(step.token_end, target_token_count)
            ):
                weights[position] = 1.0
    if not any(step.is_answer for step in steps):
        raise ValueError("Answer supervision is mandatory.")
    return weights


def deterministic_random_step_scores(
    problem_id: str, steps: Sequence[ReasoningStep], seed: int
) -> List[float]:
    values = []
    for step in steps:
        digest = hashlib.sha256(
            f"{seed}|{problem_id}|{step.index}".encode("utf-8")
        ).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64))
    return values


def equal_step_total_weights(
    steps: Sequence[ReasoningStep], total_reasoning_weight: float
) -> List[float]:
    reasoning = [step for step in steps if not step.is_answer]
    if not reasoning or total_reasoning_weight <= 0:
        raise ValueError("Step-balanced weighting requires positive reasoning weight.")
    per_step = float(total_reasoning_weight) / len(reasoning)
    length = max(step.token_end for step in steps)
    weights = [0.0] * length
    for step in reasoning:
        per_token = per_step / step.token_count
        for position in range(step.token_start, step.token_end):
            weights[position] = per_token
    for step in steps:
        if step.is_answer:
            for position in range(step.token_start, step.token_end):
                weights[position] = 1.0
    return weights


def relative_budget_gap(actual: int, target: int) -> float:
    if target <= 0:
        raise ValueError("target must be positive.")
    return abs(int(actual) - int(target)) / int(target)


def _line_spans(text: str) -> List[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"[^\n]+(?:\n+|$)", text):
        if text[match.start() : match.end()].strip():
            spans.append((match.start(), match.end()))
    return spans


def _sentence_spans(text: str, start: int, end: int) -> List[tuple[int, int]]:
    segment = text[start:end]
    boundaries = [
        0,
        *(match.end() for match in SENTENCE_BREAK.finditer(segment)),
        len(segment),
    ]
    spans = []
    for left, right in zip(boundaries, boundaries[1:]):
        if segment[left:right].strip():
            spans.append((start + left, start + right))
    return spans or [(start, end)]


def _split_answer_suffixes(
    text: str, spans: Sequence[tuple[int, int]]
) -> List[tuple[int, int]]:
    output = []
    for start, end in spans:
        match = ANSWER_MARKER.search(text, start, end)
        if match is not None and text[start : match.start()].strip():
            output.append((start, match.start()))
            output.append((match.start(), end))
        else:
            output.append((start, end))
    return output
