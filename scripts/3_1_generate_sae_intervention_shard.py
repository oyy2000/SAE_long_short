#!/usr/bin/env python3
"""Generate one registered shard of common-prefix SAE intervention traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
    runtime_metadata,
)
from length_budget_distill.sae_intervention import (
    InterventionSpec,
    SAEInterventionController,
    generate_common_prefix_branches,
)
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase3_sae_intervention_distillation_pilot_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    parser.add_argument("--stage", choices=("calibration", "main"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    _validate_protocol(config_path, config)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count).")
    registered_shards = int(
        config[args.stage if args.stage == "calibration" else "main_generation"][
            "generation_shards"
        ]
    )
    if args.shard_count != registered_shards:
        raise ValueError("Runtime shard count differs from the frozen protocol.")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    problem_ids = list(
        config["question_cohorts"][
            "calibration_problem_ids"
            if args.stage == "calibration"
            else "main_problem_ids"
        ]
    )
    shard_ids = problem_ids[args.shard_index :: args.shard_count]
    problem_map = _load_problem_map(Path(config["parent_evidence"]["corpus_path"]))
    problems = [problem_map[problem_id] for problem_id in shard_ids]
    specs, strength_evidence = _specifications(config, args.stage)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher intervention generation.")
    teacher = config["teacher"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        teacher["torch_dtype"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        teacher["snapshot_path"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        teacher["snapshot_path"],
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    features = config["selected_features"]
    intervention = config["intervention"]
    layer_index = int(config["parent_sae"]["layer_index"])
    controller = SAEInterventionController(
        torch_module=torch,
        checkpoint_path=config["parent_evidence"]["checkpoint_path"],
        layer_module=model.model.layers[layer_index],
        short_feature_ids=features["short_feature_ids"],
        long_feature_ids=features["long_feature_ids"],
        random_feature_ids=features["random_feature_ids"],
        k=int(config["parent_sae"]["k"]),
        maximum_delta_fraction=float(
            intervention["maximum_delta_to_hidden_norm_fraction"]
        ),
        device=model.device,
        dtype=dtype,
    )
    candidates = int(
        config[args.stage if args.stage == "calibration" else "main_generation"][
            "candidates_per_question"
        ]
    )
    batch_size = int(config["main_generation"]["batch_size"])
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "generations.jsonl"
    condition_counts = {
        spec.name: {"records": 0, "correct": 0, "tokens": 0} for spec in specs
    }
    pair_count = 0
    with records_path.open("x", encoding="utf-8") as handle:
        for candidate_index in range(candidates):
            for batch_start in range(0, len(problems), batch_size):
                batch = problems[batch_start : batch_start + batch_size]
                prefix_seed = _stable_seed(
                    int(teacher["base_seed"]),
                    args.stage,
                    args.shard_index,
                    candidate_index,
                    batch_start,
                    "prefix",
                )
                continuation_seed = _stable_seed(
                    int(teacher["base_seed"]),
                    args.stage,
                    args.shard_index,
                    candidate_index,
                    batch_start,
                    "continuation",
                )
                generated = generate_common_prefix_branches(
                    torch_module=torch,
                    model=model,
                    tokenizer=tokenizer,
                    controller=controller,
                    prompts=[row["prompt"] for row in batch],
                    specs=specs,
                    prefix_seed=prefix_seed,
                    continuation_seed=continuation_seed,
                    common_prefix_tokens=int(teacher["common_prefix_tokens"]),
                    max_new_tokens=int(teacher["max_new_tokens"]),
                    temperature=float(teacher["temperature"]),
                    top_p=float(teacher["top_p"]),
                )
                for row_index, problem in enumerate(batch):
                    prefix_hashes = set()
                    for spec in specs:
                        cell = generated[spec.name][row_index]
                        prefix_count = int(cell["common_prefix_token_count"])
                        prefix_ids = cell["response_token_ids"][:prefix_count]
                        prefix_sha256 = hashlib.sha256(
                            json.dumps(prefix_ids, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                        prefix_hashes.add(prefix_sha256)
                        predicted = extract_final_answer(cell["response"])
                        correct = verify_answer(predicted, problem["answer"])
                        record = {
                            "stage": args.stage,
                            "condition": spec.name,
                            "intervention_mode": spec.mode,
                            "intervention_strength": float(spec.strength),
                            "problem_id": problem["problem_id"],
                            "source_index": int(problem["source_index"]),
                            "question": problem["question"],
                            "answer": problem["answer"],
                            "prompt": problem["prompt"],
                            "candidate_index": candidate_index,
                            "pair_id": f"{problem['problem_id']}:candidate_{candidate_index:02d}",
                            "prefix_seed": prefix_seed,
                            "continuation_seed": continuation_seed,
                            "common_prefix_sha256": prefix_sha256,
                            "predicted_answer": predicted,
                            "is_correct": bool(correct),
                            **cell,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts = condition_counts[spec.name]
                        counts["records"] += 1
                        counts["correct"] += int(correct)
                        counts["tokens"] += int(cell["output_token_count"])
                    if len(prefix_hashes) != 1:
                        raise AssertionError("Intervention branches do not share a token-identical prefix.")
                    pair_count += 1
                print(
                    json.dumps(
                        {
                            "event": "generation_progress",
                            "stage": args.stage,
                            "shard_index": args.shard_index,
                            "candidate_index": candidate_index,
                            "processed_pairs": pair_count,
                            "total_pairs": len(problems) * candidates,
                        }
                    ),
                    flush=True,
                )
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "stage": args.stage,
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "problem_ids": shard_ids,
        "problem_ids_sha256": _list_hash(shard_ids),
        "problem_count": len(problems),
        "candidate_count_per_problem": candidates,
        "pair_count": pair_count,
        "conditions": [spec.__dict__ for spec in specs],
        "condition_counts": condition_counts,
        "common_prefix_audit": "passed",
        "strength_evidence": strength_evidence,
        "records_path": str(records_path),
        "records_sha256": file_sha256(records_path),
        "elapsed_seconds": time.time() - started,
        "runtime": runtime_metadata(
            ("python", "torch", "transformers", "safetensors")
        ),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_intervention.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "generation_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "GENERATION_SHARD_COMPLETE").write_text(
        f"status=complete\nstage={args.stage}\n"
        f"config_hash={manifest['config_hash']}\n"
        f"records_sha256={manifest['records_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        f"shard_index={args.shard_index}\nshard_count={args.shard_count}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _specifications(
    config: Mapping[str, Any], stage: str
) -> tuple[list[InterventionSpec], dict[str, Any] | None]:
    if stage == "calibration":
        specs = [InterventionSpec("no_steering", "none", 0.0)]
        specs.extend(
            InterventionSpec(
                f"short_feature_enhance__a_{_strength_name(value)}",
                "short_enhance",
                float(value),
            )
            for value in config["intervention"]["short_strength_grid"]
        )
        specs.extend(
            InterventionSpec(
                f"long_feature_suppress__a_{_strength_name(value)}",
                "long_suppress",
                float(value),
            )
            for value in config["intervention"]["long_suppression_grid"]
        )
        specs.extend(
            InterventionSpec(
                f"random_feature_enhance__a_{_strength_name(value)}",
                "random_enhance",
                float(value),
            )
            for value in config["intervention"].get("random_strength_grid", [])
        )
        return specs, None
    selection_path = _resolve(config["outputs"]["result_root"]) / "calibration" / "selected_strengths.json"
    marker_path = selection_path.parent / "CALIBRATION_COMPLETE"
    marker = read_key_value_marker(marker_path)
    if marker.get("status") != "complete" or marker.get(
        "selection_sha256"
    ) != file_sha256(selection_path):
        raise RuntimeError("Intervention calibration has not completed.")
    selection = read_json(selection_path)
    if selection["config_hash"] != canonical_sha256(config):
        raise ValueError("Calibration was produced under another protocol.")
    short_strength = float(selection["selected"]["short_feature_enhance"]["strength"])
    long_strength = float(selection["selected"]["long_feature_suppress"]["strength"])
    return [
        InterventionSpec("no_steering", "none", 0.0),
        InterventionSpec("short_feature_enhance", "short_enhance", short_strength),
        InterventionSpec("long_feature_suppress", "long_suppress", long_strength),
        InterventionSpec("random_feature_enhance", "random_enhance", short_strength),
    ], {
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "marker_path": str(marker_path),
        "marker_sha256": file_sha256(marker_path),
    }


def _load_problem_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            problem_id = str(row["problem_id"])
            candidate = {
                "problem_id": problem_id,
                "source_index": int(row["metadata"]["source_index"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "prompt": str(row["prompt"]),
            }
            if problem_id in result and result[problem_id] != candidate:
                raise ValueError(f"Inconsistent problem metadata: {problem_id}")
            result[problem_id] = candidate
    return result


def _validate_protocol(config_path: Path, config: Mapping[str, Any]) -> None:
    marker = read_key_value_marker(config_path.parent / "PROTOCOL_FROZEN")
    if marker.get("status") != "frozen" or marker.get("config_hash") != canonical_sha256(config):
        raise ValueError("Intervention protocol is not frozen or its hash changed.")
    for key in (
        "parent_completion_marker",
        "parent_protocol",
        "feature_scoring_marker",
        "feature_scoring_summary",
        "checkpoint",
        "training_marker",
        "corpus",
    ):
        path = Path(config["parent_evidence"][f"{key}_path"])
        if file_sha256(path) != config["parent_evidence"][f"{key}_sha256"]:
            raise ValueError(f"Parent evidence changed: {path}")


def _stable_seed(*values: Any) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _strength_name(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _list_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
