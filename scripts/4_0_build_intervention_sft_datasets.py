#!/usr/bin/env python3
"""Build paired equal-example and approximately equal-target-token SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker
from length_budget_distill.student_prompts import build_student_math_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase3_sae_intervention_distillation_pilot_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    result_root = _resolve(config["outputs"]["result_root"])
    generation_dir = result_root / "main_generation"
    marker = read_key_value_marker(generation_dir / "MAIN_GENERATION_COMPLETE")
    manifest_path = generation_dir / "generation_merge_manifest.json"
    if marker.get("status") != "complete" or marker.get(
        "manifest_sha256"
    ) != file_sha256(manifest_path):
        raise RuntimeError("Main intervention generation has not completed.")
    generation_manifest = read_json(manifest_path)
    selected_path = Path(
        generation_manifest["artifacts"]["common_support_selected_traces.jsonl"][
            "path"
        ]
    )
    if file_sha256(selected_path) != generation_manifest["artifacts"][
        "common_support_selected_traces.jsonl"
    ]["sha256"]:
        raise ValueError("Selected common-support trace hash mismatch.")
    selected = [json.loads(line) for line in selected_path.open("r", encoding="utf-8")]
    conditions = list(config["main_generation"]["conditions"])
    grouped = {
        condition: sorted(
            (row for row in selected if row["condition"] == condition),
            key=lambda row: row["problem_id"],
        )
        for condition in conditions
    }
    problem_sets = [{row["problem_id"] for row in rows} for rows in grouped.values()]
    if any(values != problem_sets[0] for values in problem_sets[1:]):
        raise ValueError("SFT conditions do not share identical problem IDs.")

    from transformers import AutoTokenizer

    student = config["student"]
    tokenizer = AutoTokenizer.from_pretrained(
        student["model_name"],
        revision=student["revision"],
        cache_dir=student["cache_dir"],
        local_files_only=True,
    )
    encoded = {
        condition: [
            {
                "problem_id": row["problem_id"],
                "trace_id": f"{row['pair_id']}:{condition}",
                "occurrence_index": 0,
                # The teacher generation instruction is not a student training
                # condition. Use the same standard student prompt at SFT and eval.
                "prompt": build_student_math_prompt(str(row["question"])),
                "completion": row["response"],
                "source_candidate_index": int(row["candidate_index"]),
                "completion_token_count": len(
                    tokenizer.encode(row["response"], add_special_tokens=False)
                ),
            }
            for row in rows
        ]
        for condition, rows in grouped.items()
    }
    equal_example_totals = {
        condition: sum(row["completion_token_count"] for row in rows)
        for condition, rows in encoded.items()
    }
    target = max(equal_example_totals.values())
    equal_target = {
        condition: _repeat_to_target(
            rows,
            target,
            seed=int(config["teacher"]["base_seed"]),
            condition=condition,
        )
        for condition, rows in encoded.items()
    }
    final_totals = {
        condition: sum(row["completion_token_count"] for row in rows)
        for condition, rows in equal_target.items()
    }
    if max(final_totals.values()) - min(final_totals.values()) > int(
        config["student_sft"]["maximum_equal_token_gap"]
    ):
        raise ValueError("Equal-target-token datasets exceed the registered gap.")
    output_dir = result_root / "sft_data"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_rows = []
    for budget, collection in (
        ("equal_examples", encoded),
        ("equal_target_tokens", equal_target),
    ):
        budget_dir = output_dir / budget
        budget_dir.mkdir()
        for condition in conditions:
            path = budget_dir / f"{condition}.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for row in collection[condition]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows = collection[condition]
            artifact_rows.append(
                {
                    "budget_regime": budget,
                    "condition": condition,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "record_count": len(rows),
                    "unique_problem_count": len({row["problem_id"] for row in rows}),
                    "completion_token_count": sum(
                        row["completion_token_count"] for row in rows
                    ),
                }
            )
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_generation_manifest_path": str(manifest_path),
        "source_generation_manifest_sha256": file_sha256(manifest_path),
        "common_support_question_count": len(problem_sets[0]),
        "problem_ids_sha256": _list_hash(sorted(problem_sets[0])),
        "equal_example_completion_tokens": equal_example_totals,
        "equal_target_reference_tokens": target,
        "equal_target_completion_tokens": final_totals,
        "equal_target_maximum_gap": max(final_totals.values())
        - min(final_totals.values()),
        "datasets": artifact_rows,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path_out = output_dir / "sft_data_manifest.json"
    write_json_exclusive(manifest_path_out, manifest)
    (output_dir / "SFT_DATA_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"manifest_sha256={file_sha256(manifest_path_out)}\n"
        f"common_support_question_count={len(problem_sets[0])}\n"
        f"equal_target_maximum_gap={manifest['equal_target_maximum_gap']}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _repeat_to_target(
    base_rows: Sequence[Mapping[str, Any]], target: int, *, seed: int, condition: str
) -> list[dict[str, Any]]:
    output = [dict(row) for row in base_rows]
    total = sum(int(row["completion_token_count"]) for row in output)
    occurrences = Counter(row["problem_id"] for row in output)
    cycle = 0
    while total < target:
        remaining = target - total
        ranked = sorted(
            base_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{condition}:{cycle}:{row['problem_id']}".encode("utf-8")
            ).hexdigest(),
        )
        fitting = [
            row for row in ranked if int(row["completion_token_count"]) <= remaining
        ]
        if fitting:
            chosen = fitting[0]
        else:
            chosen = min(
                ranked,
                key=lambda row: (
                    abs(int(row["completion_token_count"]) - remaining),
                    hashlib.sha256(str(row["problem_id"]).encode("utf-8")).hexdigest(),
                ),
            )
        copy = dict(chosen)
        problem_id = str(copy["problem_id"])
        copy["occurrence_index"] = int(occurrences[problem_id])
        copy["trace_id"] = f"{copy['trace_id']}:repeat_{copy['occurrence_index']:03d}"
        occurrences[problem_id] += 1
        output.append(copy)
        total += int(copy["completion_token_count"])
        cycle += 1
    return output


def _list_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
