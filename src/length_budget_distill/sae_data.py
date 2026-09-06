"""Corpus construction and deterministic splits for mixed-trace SAE pilots."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_LABELS = ("short", "medium", "long", "other_correct", "incorrect")


def stable_question_split(
    problem_ids: Iterable[str],
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    seed: int,
) -> dict[str, str]:
    """Assign whole questions to deterministic, disjoint splits."""

    unique = sorted(set(str(value) for value in problem_ids))
    if train_count + dev_count + test_count != len(unique):
        raise ValueError("Question split counts do not cover the unique problem IDs.")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    boundaries = (train_count, train_count + dev_count)
    result = {}
    for index, problem_id in enumerate(ordered):
        if index < boundaries[0]:
            split = "train"
        elif index < boundaries[1]:
            split = "dev"
        else:
            split = "test"
        result[problem_id] = split
    return result


def assign_within_question_length_labels(
    rows: Sequence[Mapping[str, Any]], *, minimum_correct: int = 4
) -> dict[str, str]:
    """Label tails and one median trace without using labels for SAE training."""

    labels = {str(row["trace_id"]): "incorrect" for row in rows}
    correct = [row for row in rows if bool(row.get("is_correct"))]
    if len(correct) < minimum_correct:
        raise ValueError(
            f"Question has {len(correct)} correct traces; expected at least {minimum_correct}."
        )
    ordered = sorted(
        correct,
        key=lambda row: (
            int(row["solution_token_count"]),
            int(row.get("candidate_index", 0)),
            str(row["trace_id"]),
        ),
    )
    tail_count = max(1, math.ceil(0.2 * len(ordered)))
    short_ids = {str(row["trace_id"]) for row in ordered[:tail_count]}
    long_ids = {str(row["trace_id"]) for row in ordered[-tail_count:]}
    if short_ids & long_ids:
        raise ValueError("Short and long tail labels overlap.")
    eligible_medians = [
        (index, row)
        for index, row in enumerate(ordered)
        if str(row["trace_id"]) not in short_ids | long_ids
    ]
    if not eligible_medians:
        raise ValueError("No correct trace remains for the median label.")
    median_rank = (len(ordered) - 1) / 2.0
    _, median_row = min(
        eligible_medians,
        key=lambda pair: (
            abs(pair[0] - median_rank),
            int(pair[1]["solution_token_count"]),
            int(pair[1].get("candidate_index", 0)),
            str(pair[1]["trace_id"]),
        ),
    )
    for row in correct:
        labels[str(row["trace_id"])] = "other_correct"
    for trace_id in short_ids:
        labels[trace_id] = "short"
    labels[str(median_row["trace_id"])] = "medium"
    for trace_id in long_ids:
        labels[trace_id] = "long"
    return labels


def build_mixed_trace_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    question_splits: Mapping[str, str],
    minimum_correct: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a label-agnostic training corpus with analysis-only labels."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    trace_ids = set()
    for row in rows:
        problem_id = str(row["problem_id"])
        trace_id = str(row["trace_id"])
        if trace_id in trace_ids:
            raise ValueError(f"Duplicate trace ID: {trace_id}")
        trace_ids.add(trace_id)
        grouped[problem_id].append(row)
    if set(grouped) != set(question_splits):
        raise ValueError(
            "Corpus question IDs do not match the registered question split."
        )
    corpus = []
    for problem_id in sorted(grouped):
        question_rows = grouped[problem_id]
        labels = assign_within_question_length_labels(
            question_rows, minimum_correct=minimum_correct
        )
        for row in sorted(
            question_rows,
            key=lambda value: (
                int(value.get("candidate_index", 0)),
                str(value["trace_id"]),
            ),
        ):
            copied = dict(row)
            copied["question_split"] = str(question_splits[problem_id])
            copied["analysis_length_label"] = labels[str(row["trace_id"])]
            copied["sae_training_included"] = True
            corpus.append(copied)
    for index, row in enumerate(corpus):
        row["corpus_index"] = index
    split_counts = Counter(str(row["question_split"]) for row in corpus)
    label_counts = Counter(str(row["analysis_length_label"]) for row in corpus)
    return corpus, {
        "trajectory_count": len(corpus),
        "question_count": len(grouped),
        "split_trajectory_counts": dict(sorted(split_counts.items())),
        "analysis_label_counts": dict(sorted(label_counts.items())),
        "all_trajectories_mixed_for_training": all(
            bool(row["sae_training_included"]) for row in corpus
        ),
    }


def trace_shard(trace_id: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(str(trace_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count
