"""Controlled Phase-0 trace-length observation helpers.

The module deliberately contains only deterministic, reusable logic.  File-system
or model side effects live in the phase-24 entrypoints under ``scripts/``.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.ranked_sampling import normalized_completion_key


LENGTH_RANKS = ("short", "medium", "long")
BUDGET_REGIMES = ("equal_examples", "equal_target_tokens", "equal_processed_tokens")
LOSS_NORMALIZATIONS = ("token_mean", "sequence_mean")
LOSS_MASKS = ("full_completion", "answer_only")
SELECTION_METHOD = "within_quantile_band_median_v1"
ANSWER_SPAN_PATTERNS = (
    ("answer_line", re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(?P<answer>[^\n]+?)\s*$")),
    ("gsm8k_marker", re.compile(r"(?im)^\s*####\s*(?P<answer>[^\n]+?)\s*$")),
    ("boxed", re.compile(r"(?P<prefix>\\boxed\{)(?P<answer>[^{}]+)(?P<suffix>\})")),
    ("inline_answer", re.compile(r"(?i)(?:final\s+answer|answer)\s*[:=]\s*(?P<answer>[^\n]+)")),
)


def validate_trace_observation_config(config: Mapping[str, Any]) -> None:
    """Reject protocol drift before materializing or training any condition."""

    if config.get("protocol_variant") != "phase0_controlled_trace_length_observation_v1":
        raise ValueError("Unexpected Phase-0 protocol_variant.")
    if config.get("scope") != "GSM8K only":
        raise ValueError("Phase 0 is restricted to GSM8K.")
    teacher = dict(config.get("teacher", {}))
    student = dict(config.get("student", {}))
    if teacher.get("model_name") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Phase-0 teacher must be Qwen2.5-7B-Instruct.")
    if student.get("model_name") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Phase-0 student must be Qwen2.5-1.5B-Instruct.")

    candidate_pool = dict(config.get("candidate_pool", {}))
    if int(candidate_pool.get("candidates_per_problem", 0)) != 16:
        raise ValueError("Phase 0 requires 16 teacher rollouts per problem.")
    if int(candidate_pool.get("minimum_unique_correct", 0)) < 4:
        raise ValueError("Phase 0 requires at least four unique correct traces per problem.")
    if float(candidate_pool.get("tail_fraction", 0.0)) != 0.2:
        raise ValueError("Phase-0 tail_fraction must equal 0.2.")
    if candidate_pool.get("selection_method") != SELECTION_METHOD:
        raise ValueError(f"selection_method must equal {SELECTION_METHOD!r}.")

    training = dict(config.get("training", {}))
    if tuple(training.get("seeds", [])) != (17, 42, 73):
        raise ValueError("Phase-0 training seeds must be 17, 42, and 73.")
    if tuple(training.get("budget_regimes", [])) != BUDGET_REGIMES:
        raise ValueError(f"budget_regimes must equal {BUDGET_REGIMES}.")
    if tuple(training.get("loss_normalizations", [])) != LOSS_NORMALIZATIONS:
        raise ValueError(f"loss_normalizations must equal {LOSS_NORMALIZATIONS}.")
    if tuple(training.get("loss_masks", [])) != LOSS_MASKS:
        raise ValueError(f"loss_masks must equal {LOSS_MASKS}.")
    if int(training.get("per_device_train_batch_size", 0)) <= 0:
        raise ValueError("per_device_train_batch_size must be positive.")
    if int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if bool(training.get("packing", False)):
        raise ValueError("Packing is disabled because it obscures sequence-normalized loss.")
    if int(training.get("num_train_epochs", 0)) != 1:
        raise ValueError("The registered Phase-0 schedules require exactly one training pass.")
    if training.get("completion_separator") != "":
        raise ValueError("Chat-formatted Phase-0 training must not add a completion separator.")
    if training.get("budget_count_basis") != "completion_tokens_excluding_terminal_eos":
        raise ValueError("Unexpected training budget_count_basis.")
    if training.get("optimizer_budget_count_basis") != (
        "nonpadding_model_input_tokens_including_prompt_completion_terminal_eos"
    ):
        raise ValueError("Unexpected training optimizer_budget_count_basis.")

    evaluation = dict(config.get("evaluation", {}))
    locked = {
        "dataset_split": "test",
        "start_index": 50,
        "limit": 1269,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 512,
    }
    for key, value in locked.items():
        if evaluation.get(key) != value:
            raise ValueError(f"Locked evaluation field changed: {key}.")

    gate = dict(config.get("gate0", {}))
    required_cells = {
        ("equal_target_tokens", "token_mean"),
        ("equal_target_tokens", "sequence_mean"),
        ("equal_processed_tokens", "token_mean"),
        ("equal_processed_tokens", "sequence_mean"),
    }
    configured_cells = {
        (str(item["budget_regime"]), str(item["loss_normalization"]))
        for item in gate.get("required_short_vs_long_cells", [])
    }
    if configured_cells != required_cells:
        raise ValueError("Gate 0 must cover equal-token/update budgets under both normalizations.")


def protocol_hash(config: Mapping[str, Any]) -> str:
    normalized = dict(config)
    normalized.pop("_config_path", None)
    return canonical_sha256(normalized)


def unique_correct_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return exact-text-deduplicated rows already verified as correct."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row.get("candidate_index", 0)), str(row.get("trace_id", ""))),
    )
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []
    for row in ordered:
        if not bool(row.get("is_correct")):
            continue
        key = normalized_completion_key(str(row.get("solution", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def select_quantile_band_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_unique_correct: int = 4,
    tail_fraction: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """Select one representative from short/median/long within-question ranks.

    Short and long representatives are the median-length members of the bottom
    and top quantile bands.  This prevents the main condition from defaulting to
    an absolute extreme whenever a band contains multiple traces.
    """

    if minimum_unique_correct < 4:
        raise ValueError("minimum_unique_correct must be at least four.")
    if not 0.0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be between zero and one half.")
    problem_ids = {str(row.get("problem_id")) for row in rows}
    if len(problem_ids) > 1:
        raise ValueError("Candidate rows must belong to one problem.")
    ranked = sorted(
        unique_correct_rows(rows),
        key=lambda row: (
            int(row["solution_token_count"]),
            int(row.get("candidate_index", 0)),
            str(row["trace_id"]),
        ),
    )
    if len(ranked) < minimum_unique_correct:
        return {}
    band_size = max(1, int(math.ceil(len(ranked) * tail_fraction)))
    short_band = ranked[:band_size]
    long_band = ranked[-band_size:]
    global_median = float(statistics.median(int(row["solution_token_count"]) for row in ranked))
    selected = {
        "short": _band_representative(short_band, prefer_higher_on_tie=True),
        "medium": _closest_to_value(ranked, global_median),
        "long": _band_representative(long_band, prefer_higher_on_tie=False),
    }
    if len({str(row["trace_id"]) for row in selected.values()}) != 3:
        raise AssertionError("Quantile selection did not produce three distinct traces.")
    rank_by_trace = {str(row["trace_id"]): index for index, row in enumerate(ranked)}
    band_ids = {
        "short": [str(row["trace_id"]) for row in short_band],
        "medium": [
            str(row["trace_id"])
            for row in ranked
            if int(row["solution_token_count"]) == min(
                (int(item["solution_token_count"]) for item in ranked),
                key=lambda value: abs(value - global_median),
            )
        ],
        "long": [str(row["trace_id"]) for row in long_band],
    }
    output: Dict[str, Dict[str, Any]] = {}
    for label, row in selected.items():
        copied = deepcopy(row)
        metadata = dict(copied.get("metadata", {}))
        metadata["phase0_length_selection"] = {
            "label": label,
            "method": SELECTION_METHOD,
            "tail_fraction": tail_fraction,
            "band_size": len(short_band) if label == "short" else len(long_band) if label == "long" else len(band_ids["medium"]),
            "band_trace_ids": band_ids[label],
            "selected_rank_zero_based": rank_by_trace[str(row["trace_id"])],
            "eligible_unique_correct_count": len(ranked),
            "candidate_pool_count": len(rows),
            "absolute_extreme": rank_by_trace[str(row["trace_id"])] in {0, len(ranked) - 1},
        }
        copied["metadata"] = metadata
        copied["length_rank"] = label
        copied["selected_for_sft"] = True
        output[label] = copied
    return output


def select_by_problem(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_unique_correct: int = 4,
    tail_fraction: float = 0.2,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem_id"])].append(dict(row))
    selected: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dropped: List[str] = []
    for problem_id in sorted(grouped):
        choice = select_quantile_band_candidates(
            grouped[problem_id],
            minimum_unique_correct=minimum_unique_correct,
            tail_fraction=tail_fraction,
        )
        if choice:
            selected[problem_id] = choice
        else:
            dropped.append(problem_id)
    return selected, dropped


def completion_token_count(tokenizer: Any, completion: str) -> int:
    token_ids = tokenizer.encode(str(completion), add_special_tokens=False)
    if not token_ids:
        raise ValueError("Completion tokenized to zero tokens.")
    return len(token_ids)


def training_prompt_token_ids(tokenizer: Any, prompt: str) -> List[int]:
    """Format the registered student prompt exactly as the trainer does."""

    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(prompt)}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(str(prompt), add_special_tokens=True))


def model_input_token_count(tokenizer: Any, prompt: str, completion: str) -> int:
    """Count non-padding training tokens, including the terminal EOS target."""

    prompt_ids = training_prompt_token_ids(tokenizer, prompt)
    completion_ids = list(tokenizer.encode(str(completion), add_special_tokens=False))
    if not completion_ids:
        raise ValueError("Completion tokenized to zero tokens.")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    append_eos = eos_id is not None and completion_ids[-1] != int(eos_id)
    return len(prompt_ids) + len(completion_ids) + int(append_eos)


def materialize_budget_schedules(
    selected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    tokenizer: Any,
    seed: int,
    max_token_gap: int,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, Any]]:
    """Create the three registered training-fairness schedules."""

    by_rank: Dict[str, List[Dict[str, Any]]] = {rank: [] for rank in LENGTH_RANKS}
    for problem_id in sorted(selected):
        for rank in LENGTH_RANKS:
            row = deepcopy(dict(selected[problem_id][rank]))
            row["budget_token_count"] = completion_token_count(tokenizer, str(row["solution"]))
            row["model_input_token_count"] = model_input_token_count(
                tokenizer,
                str(row["student_prompt"]),
                str(row["solution"]),
            )
            by_rank[rank].append(row)
    completion_totals = {
        rank: sum(int(row["budget_token_count"]) for row in rows)
        for rank, rows in by_rank.items()
    }
    model_input_totals = {
        rank: sum(int(row["model_input_token_count"]) for row in rows)
        for rank, rows in by_rank.items()
    }
    # Both token-controlled estimands retain the complete paired question
    # cohort. Shorter conditions receive deterministic repeated exposures until
    # they approach the largest one-pass token total; no question is removed.
    equal_target = max(completion_totals.values())
    equal_processed = max(model_input_totals.values())
    schedules: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rank in LENGTH_RANKS:
        schedules[("equal_examples", rank)] = _annotate_occurrences(by_rank[rank])
        repeated_target = deterministic_repeated_token_schedule(
            by_rank[rank],
            equal_target,
            seed=seed,
            salt=f"equal_target_tokens:{rank}",
            token_field="budget_token_count",
        )
        schedules[("equal_target_tokens", rank)] = _annotate_occurrences(repeated_target)
        repeated_processed = deterministic_repeated_token_schedule(
            by_rank[rank],
            equal_processed,
            seed=seed,
            salt=f"equal_processed_tokens:{rank}",
            token_field="model_input_token_count",
        )
        schedules[("equal_processed_tokens", rank)] = _annotate_occurrences(repeated_processed)
    schedule_summaries: Dict[str, Dict[str, Any]] = {}
    for (regime, rank), rows in sorted(schedules.items()):
        completion_total = sum(int(row["budget_token_count"]) for row in rows)
        model_input_total = sum(int(row["model_input_token_count"]) for row in rows)
        if regime == "equal_target_tokens":
            token_basis = "completion_tokens"
            target = equal_target
            actual = completion_total
        elif regime == "equal_processed_tokens":
            token_basis = "model_input_tokens"
            target = equal_processed
            actual = model_input_total
        else:
            token_basis = "complete_question_cohort"
            target = len(by_rank[rank])
            actual = len(rows)
        gap = target - actual
        if regime != "equal_examples" and not 0 <= gap <= max_token_gap:
            raise ValueError(
                f"Token schedule gap exceeded limit: regime={regime} rank={rank} gap={gap}"
            )
        schedule_summaries[f"{regime}__{rank}"] = {
            "budget_regime": regime,
            "length_rank": rank,
            "record_count": len(rows),
            "unique_problem_count": len({str(row["problem_id"]) for row in rows}),
            "budget_token_basis": token_basis,
            "target_budget_tokens": target,
            "actual_budget_tokens": actual,
            "actual_completion_tokens": completion_total,
            "actual_model_input_tokens": model_input_total,
            "token_gap": gap,
            "duplicate_exposures": len(rows) - len({str(row["problem_id"]) for row in rows}),
        }
    return schedules, {
        "equal_target_tokens": equal_target,
        "equal_processed_tokens": equal_processed,
        "full_dataset_completion_tokens": completion_totals,
        "full_dataset_model_input_tokens": model_input_totals,
        "schedules": schedule_summaries,
    }


def deterministic_token_subset(
    rows: Sequence[Mapping[str, Any]],
    target_tokens: int,
    *,
    seed: int,
    salt: str,
    token_field: str = "budget_token_count",
) -> List[Dict[str, Any]]:
    """Choose whole records without exceeding a completion-token target."""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    ordered = _stable_shuffle(rows, seed=seed, salt=salt)
    chosen: List[Dict[str, Any]] = []
    total = 0
    for row in ordered:
        tokens = int(row[token_field])
        if tokens <= 0:
            raise ValueError("budget_token_count must be positive.")
        if total + tokens <= target_tokens:
            chosen.append(deepcopy(dict(row)))
            total += tokens
    if not chosen:
        raise ValueError("Token subset is empty.")
    return chosen


def deterministic_repeated_token_schedule(
    rows: Sequence[Mapping[str, Any]],
    target_tokens: int,
    *,
    seed: int,
    salt: str,
    token_field: str = "budget_token_count",
) -> List[Dict[str, Any]]:
    """Cycle whole records to match a training token-update budget."""

    if not rows:
        raise ValueError("Repeated schedule requires at least one record.")
    base_total = sum(int(row[token_field]) for row in rows)
    if base_total > target_tokens:
        raise ValueError("Repeated schedule target cannot be smaller than one full pass.")
    scheduled = _stable_shuffle(rows, seed=seed, salt=f"{salt}:pass:0")
    total = base_total
    pass_index = 1
    while total < target_tokens:
        made_progress = False
        for row in _stable_shuffle(rows, seed=seed, salt=f"{salt}:pass:{pass_index}"):
            tokens = int(row[token_field])
            if total + tokens <= target_tokens:
                scheduled.append(deepcopy(dict(row)))
                total += tokens
                made_progress = True
        if not made_progress:
            break
        pass_index += 1
    return scheduled


def build_run_matrix(
    config: Mapping[str, Any],
    schedule_evidence: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand all registered Phase-0 training cells deterministically."""

    training = dict(config["training"])
    seeds = [int(value) for value in training["seeds"]]
    runs: List[Dict[str, Any]] = []
    for regime in BUDGET_REGIMES:
        for normalization in LOSS_NORMALIZATIONS:
            for loss_mask in LOSS_MASKS:
                for rank in LENGTH_RANKS:
                    evidence = dict(schedule_evidence[(regime, rank)])
                    for seed in seeds:
                        run_name = phase0_run_name(regime, normalization, loss_mask, rank, seed)
                        runs.append(
                            {
                                "run_name": run_name,
                                "budget_regime": regime,
                                "loss_normalization": normalization,
                                "loss_mask": loss_mask,
                                "length_rank": rank,
                                "seed": seed,
                                **evidence,
                            }
                        )
    expected = len(BUDGET_REGIMES) * len(LOSS_NORMALIZATIONS) * len(LOSS_MASKS) * len(LENGTH_RANKS) * len(seeds)
    if len(runs) != expected or len({run["run_name"] for run in runs}) != expected:
        raise AssertionError("Phase-0 run matrix is incomplete or contains duplicates.")
    return runs


def phase0_run_name(
    budget_regime: str,
    loss_normalization: str,
    loss_mask: str,
    length_rank: str,
    seed: int,
) -> str:
    for value, allowed, label in (
        (budget_regime, BUDGET_REGIMES, "budget regime"),
        (loss_normalization, LOSS_NORMALIZATIONS, "loss normalization"),
        (loss_mask, LOSS_MASKS, "loss mask"),
        (length_rank, LENGTH_RANKS, "length rank"),
    ):
        if value not in allowed:
            raise ValueError(f"Unknown {label}: {value}")
    return f"{budget_regime}__{loss_normalization}__{loss_mask}__{length_rank}__seed_{int(seed)}"


def answer_character_span(completion: str) -> Tuple[int, int]:
    """Locate the final answer content in any verifier-recognized format."""

    matched = _final_answer_span_match(completion)
    if matched is None:
        raise ValueError("Completion has no maskable final-answer span.")
    _, match = matched
    start, end = match.span("answer")
    while end > start and completion[end - 1].isspace():
        end -= 1
    if start >= end:
        raise ValueError("Final-answer span contains no answer content.")
    return start, end


def answer_span_format(completion: str) -> str:
    matched = _final_answer_span_match(completion)
    if matched is None:
        raise ValueError("Completion has no maskable final-answer span.")
    return matched[0]


def _final_answer_span_match(completion: str) -> Tuple[str, Any] | None:
    candidates = []
    for name, pattern in ANSWER_SPAN_PATTERNS:
        candidates.extend((match.start(), name, match) for match in pattern.finditer(completion))
    if not candidates:
        return None
    _, name, match = max(candidates, key=lambda value: value[0])
    return name, match


def answer_token_mask(tokenizer: Any, completion: str) -> List[int]:
    """Return a completion-token mask for the final answer content."""

    start, end = answer_character_span(completion)
    encoded = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer must provide offset_mapping for answer-only supervision.")
    mask = [int(offset_end > start and offset_start < end) for offset_start, offset_end in offsets]
    if not any(mask):
        raise ValueError("Answer span did not align to any completion token.")
    return mask


def summarize_selected_rows(
    selected: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Dict[str, Any]:
    by_rank: Dict[str, List[Mapping[str, Any]]] = {rank: [] for rank in LENGTH_RANKS}
    eligible_counts: Counter[int] = Counter()
    extremes: Counter[str] = Counter()
    for choices in selected.values():
        eligible_counts[int(choices["short"]["metadata"]["phase0_length_selection"]["eligible_unique_correct_count"])] += 1
        for rank in LENGTH_RANKS:
            row = choices[rank]
            by_rank[rank].append(row)
            if bool(row["metadata"]["phase0_length_selection"]["absolute_extreme"]):
                extremes[rank] += 1
    return {
        "problem_count": len(selected),
        "eligible_unique_correct_histogram": {str(key): value for key, value in sorted(eligible_counts.items())},
        "by_rank": {
            rank: {
                "record_count": len(rows),
                "mean_source_solution_tokens": statistics.fmean(int(row["solution_token_count"]) for row in rows),
                "median_source_solution_tokens": statistics.median(int(row["solution_token_count"]) for row in rows),
                "absolute_extreme_count": extremes[rank],
                "hit_source_ceiling_count": sum(
                    int(row["solution_token_count"]) >= int(row.get("max_solution_tokens", 0))
                    for row in rows
                ),
            }
            for rank, rows in by_rank.items()
        },
    }


def _band_representative(
    rows: Sequence[Mapping[str, Any]], *, prefer_higher_on_tie: bool
) -> Dict[str, Any]:
    target = float(statistics.median(int(row["solution_token_count"]) for row in rows))
    winner = min(
        rows,
        key=lambda row: (
            abs(int(row["solution_token_count"]) - target),
            -int(row["solution_token_count"])
            if prefer_higher_on_tie
            else int(row["solution_token_count"]),
            int(row.get("candidate_index", 0)),
            str(row["trace_id"]),
        ),
    )
    return deepcopy(dict(winner))


def _closest_to_value(rows: Sequence[Mapping[str, Any]], target: float) -> Dict[str, Any]:
    winner = min(
        rows,
        key=lambda row: (
            abs(int(row["solution_token_count"]) - target),
            int(row["solution_token_count"]),
            int(row.get("candidate_index", 0)),
            str(row["trace_id"]),
        ),
    )
    return deepcopy(dict(winner))


def _stable_shuffle(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    salt: str,
) -> List[Dict[str, Any]]:
    copied = [deepcopy(dict(row)) for row in rows]
    stable = sorted(
        copied,
        key=lambda row: hashlib.sha256(
            f"{seed}:{salt}:{row.get('problem_id')}:{row.get('trace_id')}".encode("utf-8")
        ).hexdigest(),
    )
    # A local shuffle after hash sorting guards against structure in trace IDs
    # while remaining independent of process-level hash randomization.
    random.Random(int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:16], 16)).shuffle(stable)
    return stable


def _annotate_occurrences(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: Counter[str] = Counter()
    result: List[Dict[str, Any]] = []
    for row in rows:
        copied = deepcopy(dict(row))
        problem_id = str(copied["problem_id"])
        copied["occurrence_index"] = seen[problem_id]
        seen[problem_id] += 1
        result.append(copied)
    return result
