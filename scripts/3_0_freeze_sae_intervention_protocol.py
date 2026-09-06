#!/usr/bin/env python3
"""Freeze the length-feature intervention and downstream SFT pilot protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase3_sae_intervention_distillation_pilot_v1.json",
    )
    args = parser.parse_args()
    overlay_path = _resolve(args.config)
    overlay = read_json(overlay_path)
    parent = overlay["parent_sae"]
    parent_marker_path = _resolve(parent["completion_marker_path"])
    parent_marker = read_key_value_marker(parent_marker_path)
    if parent_marker.get("status") != "passed" or parent_marker.get(
        "validated_condition_count"
    ) != "3":
        raise RuntimeError("Parent SAE sampling ablation has not passed audit.")
    feature_path = _resolve(parent["feature_path"])
    feature_marker_path = _resolve(parent["feature_scoring_marker_path"])
    feature_marker = read_key_value_marker(feature_marker_path)
    if feature_marker.get("status") != "complete" or feature_marker.get(
        "summary_sha256"
    ) != file_sha256(feature_path.parent / "scoring_summary.json"):
        raise RuntimeError("Parent feature scoring evidence is incomplete.")
    scoring_summary = read_json(feature_path.parent / "scoring_summary.json")
    if scoring_summary["artifacts"][feature_path.name]["sha256"] != file_sha256(
        feature_path
    ):
        raise ValueError("Parent feature artifact hash mismatch.")
    candidates = read_json(feature_path)["candidates"]
    selected = {
        direction: sorted(
            int(row["feature_id"])
            for row in candidates
            if row["direction"] == direction and bool(row["confirmed"])
        )
        for direction in ("short", "long")
    }
    if not selected["short"] or not selected["long"]:
        raise ValueError("Both confirmed short and long feature sets are required.")
    excluded = {int(row["feature_id"]) for row in candidates}
    population = [
        feature_id
        for feature_id in range(int(parent["feature_count"]))
        if feature_id not in excluded
    ]
    random_features = sorted(
        random.Random(int(overlay["intervention"]["random_seed"])).sample(
            population, len(selected["short"])
        )
    )
    checkpoint_path = _resolve(parent["checkpoint_path"])
    training_marker_path = _resolve(parent["training_marker_path"])
    training_marker = read_key_value_marker(training_marker_path)
    if training_marker.get("status") != "complete" or training_marker.get(
        "model_sha256"
    ) != file_sha256(checkpoint_path):
        raise ValueError("Parent SAE checkpoint is not hash-bound to training evidence.")
    corpus_path = _resolve(parent["corpus_path"])
    split_rows = _unique_problem_rows(corpus_path)
    calibration_ids = _select_ids(
        split_rows[overlay["calibration"]["question_split"]],
        int(overlay["calibration"]["question_count"]),
        int(overlay["teacher"]["base_seed"]),
        "calibration",
    )
    main_ids = _select_ids(
        split_rows[overlay["main_generation"]["question_split"]],
        int(overlay["main_generation"]["question_count"]),
        int(overlay["teacher"]["base_seed"]),
        "main",
    )
    frozen = copy.deepcopy(overlay)
    frozen["selected_features"] = {
        "short_feature_ids": selected["short"],
        "long_feature_ids": selected["long"],
        "random_feature_ids": random_features,
        "selection_source_path": str(feature_path),
        "selection_source_sha256": file_sha256(feature_path),
        "positive_means": "short-associated",
        "negative_means": "long-associated",
        "student_utility_interpretation": False,
    }
    frozen["question_cohorts"] = {
        "calibration_problem_ids": calibration_ids,
        "calibration_problem_ids_sha256": _list_hash(calibration_ids),
        "main_problem_ids": main_ids,
        "main_problem_ids_sha256": _list_hash(main_ids),
    }
    frozen["parent_evidence"] = {
        "parent_completion_marker_path": str(parent_marker_path),
        "parent_completion_marker_sha256": file_sha256(parent_marker_path),
        "parent_protocol_path": str(_resolve(parent["protocol_path"])),
        "parent_protocol_sha256": file_sha256(_resolve(parent["protocol_path"])),
        "feature_scoring_marker_path": str(feature_marker_path),
        "feature_scoring_marker_sha256": file_sha256(feature_marker_path),
        "feature_scoring_summary_path": str(feature_path.parent / "scoring_summary.json"),
        "feature_scoring_summary_sha256": file_sha256(feature_path.parent / "scoring_summary.json"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_marker_path": str(training_marker_path),
        "training_marker_sha256": file_sha256(training_marker_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
    }
    result_root = _resolve(overlay["outputs"]["result_root"])
    if result_root.exists():
        raise FileExistsError(f"Refusing to overwrite {result_root}")
    protocol_dir = result_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = protocol_dir / "frozen_protocol.json"
    write_json_exclusive(frozen_path, frozen)
    manifest = {
        "status": "frozen",
        "experiment_name": frozen["experiment_name"],
        "config_hash": canonical_sha256(frozen),
        "config_path": str(frozen_path),
        "config_sha256": file_sha256(frozen_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": file_sha256(overlay_path),
        "short_feature_count": len(selected["short"]),
        "long_feature_count": len(selected["long"]),
        "random_feature_count": len(random_features),
        "calibration_question_count": len(calibration_ids),
        "main_question_count": len(main_ids),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = protocol_dir / "protocol_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (protocol_dir / "PROTOCOL_FROZEN").write_text(
        f"status=frozen\nconfig_hash={manifest['config_hash']}\n"
        f"config_sha256={manifest['config_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _unique_problem_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["question_split"])
            problem_id = str(row["problem_id"])
            cell = result.setdefault(split, {})
            candidate = {
                "problem_id": problem_id,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "prompt": str(row["prompt"]),
                "source_index": int(row["metadata"]["problem_metadata"].get("source_index", row["metadata"].get("source_index", -1))) if isinstance(row["metadata"].get("problem_metadata"), dict) else int(row["metadata"].get("source_index", -1)),
            }
            # The corpus stores source_index beside problem_metadata in current evidence.
            candidate["source_index"] = int(row["metadata"].get("source_index", candidate["source_index"]))
            if problem_id in cell and any(
                cell[problem_id][key] != candidate[key]
                for key in ("question", "answer", "prompt", "source_index")
            ):
                raise ValueError(f"Inconsistent problem metadata: {problem_id}")
            cell[problem_id] = candidate
    return result


def _select_ids(
    rows: Mapping[str, Mapping[str, Any]], count: int, seed: int, namespace: str
) -> list[str]:
    if count > len(rows):
        raise ValueError(f"Requested {count} questions from only {len(rows)} available.")
    ranked = sorted(
        rows,
        key=lambda problem_id: hashlib.sha256(
            f"{namespace}:{seed}:{problem_id}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:count]


def _list_hash(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
