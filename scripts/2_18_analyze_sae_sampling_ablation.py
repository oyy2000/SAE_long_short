#!/usr/bin/env python3
"""Match SAE dictionaries and render the registered sampling-ablation analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker
from length_budget_distill.sae_ablation import mutual_decoder_matches


CONDITIONS = (
    "full_token_uniform",
    "full_trace_balanced",
    "prefix64_trace_balanced",
)
DISPLAY_NAMES = {
    "full_token_uniform": "Full sequence\ntoken-uniform",
    "full_trace_balanced": "Full sequence\ntrace-balanced",
    "prefix64_trace_balanced": "First 64 tokens\ntrace-balanced",
}
SHORT_COLOR = "#2166AC"
LONG_COLOR = "#B2182B"
EARLY_COLOR = "#1B9E77"
FULL_COLOR = "#7570B3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_sae_sampling_ablation_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    ablation = config["sampling_ablation"]
    result_root = _resolve(ablation["outputs"]["result_root"])
    analysis_dir = result_root / "analysis"
    figure_dir = _resolve(ablation["outputs"]["figure_root"])
    if analysis_dir.exists() or figure_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {analysis_dir} or {figure_dir}")
    runs = {name: _load_condition(result_root, name, config) for name in CONDITIONS}
    _validate_common_trace_order(runs)

    matching = ablation["feature_matching"]
    pair_names = (
        (CONDITIONS[0], CONDITIONS[1]),
        (CONDITIONS[0], CONDITIONS[2]),
        (CONDITIONS[1], CONDITIONS[2]),
    )
    pair_matches = {}
    all_matches = []
    for left, right in pair_names:
        rows = mutual_decoder_matches(
            left_name=left,
            right_name=right,
            left_candidates=runs[left]["candidates"],
            right_candidates=runs[right]["candidates"],
            left_decoders=runs[left]["decoder"],
            right_decoders=runs[right]["decoder"],
            left_test_activations=runs[left]["test_activations"],
            right_test_activations=runs[right]["test_activations"],
            minimum_decoder_cosine=float(matching["minimum_decoder_cosine"]),
            minimum_activation_correlation=float(
                matching["minimum_test_trace_activation_correlation"]
            ),
            require_same_direction=bool(matching["require_same_short_long_direction"]),
        )
        pair_matches[(left, right)] = rows
        all_matches.extend(rows)
    stable_triples = _stable_triples(pair_matches)

    analysis_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    match_path = analysis_dir / "candidate_dictionary_matches.csv"
    _write_csv(match_path, all_matches)
    triple_path = analysis_dir / "stable_three_dictionary_features.csv"
    _write_csv(triple_path, stable_triples, allow_empty=True)
    decoded_path = analysis_dir / "decoded_token_associations.csv"
    decoded_rows = _decode_token_associations(config, runs)
    _write_csv(decoded_path, decoded_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.dpi": 180,
        }
    )
    figure_paths = []
    figure_paths.extend(_plot_sampling_exposure(plt, figure_dir, runs))
    figure_paths.extend(_plot_reconstruction_and_yield(plt, figure_dir, runs))
    figure_paths.extend(_plot_position_control(plt, figure_dir, runs))
    figure_paths.extend(
        _plot_dictionary_matching(plt, figure_dir, runs, pair_matches)
    )
    figure_paths.extend(_plot_relative_profiles(plt, figure_dir, runs))
    figure_paths.extend(
        _plot_token_signatures(plt, figure_dir, runs, decoded_rows)
    )

    findings = _findings(runs, pair_matches, stable_triples)
    findings_path = analysis_dir / "ablation_findings.json"
    write_json_exclusive(findings_path, findings)
    report_path = analysis_dir / "analysis_report.md"
    _write_report(report_path, config, findings, figure_paths)
    summary = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "condition_count": len(runs),
        "condition_summaries": {
            name: {
                "heldout_confirmed_count": int(run["summary"]["heldout_confirmed_count"]),
                "short_confirmed_count": int(run["summary"]["short_confirmed_count"]),
                "long_confirmed_count": int(run["summary"]["long_confirmed_count"]),
                "test_full_explained_variance": float(
                    run["summary"]["common_set_reconstruction"]["test"]["full"]["explained_variance"]
                ),
                "test_first64_explained_variance": float(
                    run["summary"]["common_set_reconstruction"]["test"]["first_64"]["explained_variance"]
                ),
                "length_sample_spearman_r": float(
                    run["summary"]["training_sample_contribution"]["length_sample_count_spearman_r"]
                ),
            }
            for name, run in runs.items()
        },
        "pairwise_stable_match_counts": findings["pairwise_stable_match_counts"],
        "stable_three_dictionary_feature_count": len(stable_triples),
        "findings": findings,
        "artifacts": {},
        "input_scoring_evidence": [
            {
                "condition": name,
                "marker_path": str(run["marker_path"]),
                "marker_sha256": file_sha256(run["marker_path"]),
                "summary_path": str(run["summary_path"]),
                "summary_sha256": file_sha256(run["summary_path"]),
            }
            for name, run in runs.items()
        ],
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_ablation.py"
        ),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    for path in (
        match_path,
        triple_path,
        decoded_path,
        findings_path,
        report_path,
        *figure_paths,
    ):
        summary["artifacts"][path.name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    summary_path = analysis_dir / "analysis_summary.json"
    write_json_exclusive(summary_path, summary)
    (analysis_dir / "ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={summary['config_hash']}\n"
        f"analysis_summary_sha256={file_sha256(summary_path)}\n"
        f"figure_count={len(figure_paths)}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _load_condition(result_root: Path, name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.torch import load_file

    run_dir = result_root / "condition_scores" / name
    marker_path = run_dir / "CONDITION_SCORING_COMPLETE"
    summary_path = run_dir / "scoring_summary.json"
    marker = read_key_value_marker(marker_path)
    summary = read_json(summary_path)
    if marker.get("status") != "complete" or marker.get("condition") != name:
        raise RuntimeError(f"Incomplete condition score: {name}")
    if marker.get("summary_sha256") != file_sha256(summary_path):
        raise ValueError(f"Condition summary hash mismatch: {name}")
    if summary["config_hash"] != canonical_sha256(config):
        raise ValueError(f"Condition config hash mismatch: {name}")
    for evidence in summary["artifacts"].values():
        if file_sha256(evidence["path"]) != evidence["sha256"]:
            raise ValueError(f"Condition artifact hash mismatch: {evidence['path']}")
    candidates = read_json(run_dir / "discovered_features.json")["candidates"]
    feature_ids = [int(row["feature_id"]) for row in candidates]
    checkpoint_path = Path(summary["input_evidence"]["checkpoint_path"])
    checkpoint = load_file(str(checkpoint_path), device="cpu")
    decoder = checkpoint["decoder_weight"][feature_ids].float().numpy()
    trace_rows = _read_csv(run_dir / "selected_trace_metrics.csv")
    test_rows = sorted(
        (row for row in trace_rows if row["question_split"] == "test"),
        key=lambda row: int(row["corpus_index"]),
    )
    test_activations = np.column_stack(
        [
            np.asarray(
                [float(row[f"feature_{feature_id}_token_mean_activation"]) for row in test_rows]
            )
            for feature_id in feature_ids
        ]
    )
    return {
        "name": name,
        "run_dir": run_dir,
        "marker_path": marker_path,
        "summary_path": summary_path,
        "summary": summary,
        "candidates": candidates,
        "feature_ids": feature_ids,
        "decoder": decoder,
        "test_rows": test_rows,
        "test_activations": test_activations,
        "contributions": _read_csv(run_dir / "training_sample_contributions.csv"),
        "relative": _read_csv(run_dir / "test_relative_position_profiles.csv"),
        "tokens": _read_csv(run_dir / "test_token_associations.csv"),
    }


def _validate_common_trace_order(runs: Mapping[str, Mapping[str, Any]]) -> None:
    reference = None
    for name in CONDITIONS:
        identity = [int(row["corpus_index"]) for row in runs[name]["test_rows"]]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"Test trace order differs for condition {name}.")


def _stable_triples(pair_matches: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    a, b, c = CONDITIONS
    ab = {
        int(row["left_feature_id"]): row
        for row in pair_matches[(a, b)]
        if row["stable_match"]
    }
    ac = {
        int(row["left_feature_id"]): row
        for row in pair_matches[(a, c)]
        if row["stable_match"]
    }
    bc = {
        (int(row["left_feature_id"]), int(row["right_feature_id"])): row
        for row in pair_matches[(b, c)]
        if row["stable_match"]
    }
    rows = []
    for feature_a in sorted(set(ab) & set(ac)):
        feature_b = int(ab[feature_a]["right_feature_id"])
        feature_c = int(ac[feature_a]["right_feature_id"])
        if (feature_b, feature_c) not in bc:
            continue
        rows.append(
            {
                "full_token_uniform_feature_id": feature_a,
                "full_trace_balanced_feature_id": feature_b,
                "prefix64_trace_balanced_feature_id": feature_c,
                "direction": ab[feature_a]["left_direction"],
                "cosine_full_token_vs_full_trace": ab[feature_a]["decoder_cosine"],
                "correlation_full_token_vs_full_trace": ab[feature_a]["test_trace_activation_correlation"],
                "cosine_full_token_vs_prefix64": ac[feature_a]["decoder_cosine"],
                "correlation_full_token_vs_prefix64": ac[feature_a]["test_trace_activation_correlation"],
                "cosine_full_trace_vs_prefix64": bc[(feature_b, feature_c)]["decoder_cosine"],
                "correlation_full_trace_vs_prefix64": bc[(feature_b, feature_c)]["test_trace_activation_correlation"],
            }
        )
    return rows


def _decode_token_associations(
    config: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["teacher"]["snapshot_path"], local_files_only=True
    )
    rows = []
    for name, run in runs.items():
        directions = {
            int(row["feature_id"]): row["direction"] for row in run["candidates"]
        }
        confirmed = {
            int(row["feature_id"]): bool(row["confirmed"]) for row in run["candidates"]
        }
        for source in run["tokens"]:
            token_id = int(source["token_id"])
            rows.append(
                {
                    "condition": name,
                    "feature_id": int(source["feature_id"]),
                    "direction": directions[int(source["feature_id"])],
                    "confirmed": confirmed[int(source["feature_id"])],
                    "analysis_length_label": source["analysis_length_label"],
                    "rank": int(source["rank"]),
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r"),
                    "activation_mass": float(source["activation_mass"]),
                    "within_label_mass_share": float(source["within_label_mass_share"]),
                }
            )
    return rows


def _plot_sampling_exposure(plt: Any, figure_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for axis, name in zip(axes, CONDITIONS):
        rows = runs[name]["contributions"]
        x = np.asarray([int(row["solution_token_count"]) for row in rows])
        y = np.asarray([int(row["sampled_token_count"]) for row in rows])
        rho = float(runs[name]["summary"]["training_sample_contribution"]["length_sample_count_spearman_r"])
        axis.scatter(x, y, s=5, alpha=0.12, color="#333333", rasterized=True)
        axis.set_title(f"{DISPLAY_NAMES[name]}\nSpearman r = {rho:.3f}")
        axis.set_xlabel("Trace length (tokens)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Tokens sampled for SAE training")
    figure.suptitle("Training exposure: token-uniform sampling implicitly weights long traces", fontsize=13)
    figure.tight_layout()
    return _save(figure, figure_dir / "01_training_sampling_exposure", plt)


def _plot_reconstruction_and_yield(plt: Any, figure_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(CONDITIONS))
    width = 0.34
    full = [
        runs[name]["summary"]["common_set_reconstruction"]["test"]["full"]["explained_variance"]
        for name in CONDITIONS
    ]
    early = [
        runs[name]["summary"]["common_set_reconstruction"]["test"]["first_64"]["explained_variance"]
        for name in CONDITIONS
    ]
    axes[0].bar(x - width / 2, full, width, color=FULL_COLOR, label="All tokens")
    axes[0].bar(x + width / 2, early, width, color=EARLY_COLOR, label="First 64 tokens")
    axes[0].set_xticks(x, [DISPLAY_NAMES[name] for name in CONDITIONS])
    axes[0].set_ylabel("Explained variance on common test traces")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    short = [int(runs[name]["summary"]["short_confirmed_count"]) for name in CONDITIONS]
    long = [int(runs[name]["summary"]["long_confirmed_count"]) for name in CONDITIONS]
    axes[1].bar(x, short, color=SHORT_COLOR, label="Short-associated")
    axes[1].bar(x, long, bottom=short, color=LONG_COLOR, label="Long-associated")
    axes[1].set_xticks(x, [DISPLAY_NAMES[name] for name in CONDITIONS])
    axes[1].set_ylabel("Held-out confirmed candidates")
    axes[1].legend(frameon=False)
    figure.suptitle("Common-set reconstruction and replicated short/long feature yield", fontsize=13)
    figure.tight_layout()
    return _save(figure, figure_dir / "02_reconstruction_and_feature_yield", plt)


def _plot_position_control(plt: Any, figure_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    values = []
    for name in CONDITIONS:
        for candidate in runs[name]["candidates"]:
            values.extend(
                [
                    float(candidate["metrics"]["token_mean_activation"]["test"]["paired_d"]),
                    float(candidate["metrics"]["first_64_token_mean_activation"]["test"]["paired_d"]),
                ]
            )
    limit = max(1.0, np.percentile(np.abs(values), 98) * 1.1)
    for axis, name in zip(axes, CONDITIONS):
        for candidate in runs[name]["candidates"]:
            x = float(candidate["metrics"]["token_mean_activation"]["test"]["paired_d"])
            y = float(candidate["metrics"]["first_64_token_mean_activation"]["test"]["paired_d"])
            color = SHORT_COLOR if candidate["direction"] == "short" else LONG_COLOR
            marker = "o" if candidate["confirmed"] else "x"
            axis.scatter(x, y, color=color, marker=marker, s=35, alpha=0.8)
        axis.plot([-limit, limit], [-limit, limit], color="#AAAAAA", linestyle="--", linewidth=1)
        axis.axhline(0, color="#CCCCCC", linewidth=0.8)
        axis.axvline(0, color="#CCCCCC", linewidth=0.8)
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xlabel("All-token short-minus-long paired d")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
    axes[0].set_ylabel("First-64-token short-minus-long paired d")
    figure.suptitle("Position control: circles replicate on held-out test; crosses do not", fontsize=13)
    figure.tight_layout()
    return _save(figure, figure_dir / "03_full_vs_first64_feature_effects", plt)


def _plot_dictionary_matching(
    plt: Any,
    figure_dir: Path,
    runs: Mapping[str, Mapping[str, Any]],
    pair_matches: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    image = None
    for axis, pair in zip(axes, pair_matches):
        left, right = pair
        a = runs[left]["decoder"].astype(float)
        b = runs[right]["decoder"].astype(float)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        cosine = a @ b.T
        image = axis.imshow(cosine, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        stable = [row for row in pair_matches[pair] if row["stable_match"]]
        for row in stable:
            axis.scatter(row["right_slot"], row["left_slot"], facecolors="none", edgecolors="white", s=55, linewidths=1.2)
        axis.set_title(f"{left.replace('_', ' ')}\nvs {right.replace('_', ' ')}\n{len(stable)} stable matches")
        axis.set_xlabel("Right candidate slot")
        axis.set_ylabel("Left candidate slot")
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Decoder cosine")
    figure.suptitle("Candidate dictionary alignment; white rings pass cosine, activation, and direction gates", fontsize=13)
    figure.subplots_adjust(left=0.06, right=0.92, bottom=0.13, top=0.78, wspace=0.28)
    return _save(figure, figure_dir / "04_candidate_dictionary_matching", plt)


def _plot_relative_profiles(plt: Any, figure_dir: Path, runs: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for axis, name in zip(axes, CONDITIONS):
        run = runs[name]
        candidate_direction = {
            int(row["feature_id"]): row["direction"] for row in run["candidates"] if row["confirmed"]
        }
        grouped: dict[tuple[int, str], list[float]] = {}
        for row in run["relative"]:
            feature_id = int(row["feature_id"])
            if feature_id not in candidate_direction:
                continue
            key = (feature_id, row["analysis_length_label"])
            grouped.setdefault(key, [0.0] * 5)[int(row["relative_position_bin"])] = float(row["mean_activation"])
        for direction, color in (("short", SHORT_COLOR), ("long", LONG_COLOR)):
            curves = []
            for feature_id, candidate_direction_value in candidate_direction.items():
                if candidate_direction_value != direction:
                    continue
                label = direction
                curve = np.asarray(grouped.get((feature_id, label), [0.0] * 5), dtype=float)
                curve /= max(curve.sum(), 1e-12)
                curves.append(curve)
                axis.plot(np.arange(5), curve, color=color, alpha=0.18, linewidth=1)
            if curves:
                axis.plot(np.arange(5), np.mean(curves, axis=0), color=color, linewidth=2.5, label=direction.capitalize())
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xlabel("Relative trace-position bin")
        axis.set_xticks(range(5), ["0-20", "20-40", "40-60", "60-80", "80-100"])
        axis.legend(frameon=False)
    axes[0].set_ylabel("Within-feature activation-mass share")
    figure.suptitle("Where confirmed features activate along short and long traces", fontsize=13)
    figure.tight_layout()
    return _save(figure, figure_dir / "05_confirmed_feature_position_profiles", plt)


def _plot_token_signatures(
    plt: Any,
    figure_dir: Path,
    runs: Mapping[str, Mapping[str, Any]],
    decoded_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    figure, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row_index, name in enumerate(CONDITIONS):
        candidates = runs[name]["candidates"]
        for column, direction in enumerate(("short", "long")):
            axis = axes[row_index, column]
            selected = [row for row in candidates if row["direction"] == direction and row["confirmed"]]
            selected.sort(
                key=lambda row: -abs(float(row["metrics"]["token_mean_activation"]["test"]["paired_d"]))
            )
            selected = selected[:3]
            lines = []
            for candidate in selected:
                feature_id = int(candidate["feature_id"])
                tokens = [
                    row
                    for row in decoded_rows
                    if row["condition"] == name
                    and int(row["feature_id"]) == feature_id
                    and row["analysis_length_label"] == direction
                    and int(row["rank"]) <= 4
                ]
                token_text = ", ".join(
                    f"{row['token_text']!r} ({100 * float(row['within_label_mass_share']):.1f}%)"
                    for row in tokens
                )
                lines.append(f"F{feature_id}: {token_text}")
            if not lines:
                lines = ["No held-out confirmed feature"]
            axis.text(0.02, 0.94, "\n\n".join(lines), transform=axis.transAxes, va="top", ha="left", fontsize=9, wrap=True)
            axis.set_title(f"{DISPLAY_NAMES[name].replace(chr(10), ' ')}: {direction}-associated", color=SHORT_COLOR if direction == "short" else LONG_COLOR)
            axis.set_axis_off()
    figure.suptitle("Top token signatures of strongest held-out confirmed features\n(share of feature activation mass within the associated trace group)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, figure_dir / "06_confirmed_feature_token_signatures", plt)


def _findings(
    runs: Mapping[str, Mapping[str, Any]],
    pair_matches: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    triples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pair_counts = {
        f"{left}__{right}": sum(bool(row["stable_match"]) for row in rows)
        for (left, right), rows in pair_matches.items()
    }
    return {
        "length_exposure_spearman_r": {
            name: float(run["summary"]["training_sample_contribution"]["length_sample_count_spearman_r"])
            for name, run in runs.items()
        },
        "test_explained_variance": {
            name: {
                "full": float(run["summary"]["common_set_reconstruction"]["test"]["full"]["explained_variance"]),
                "first64": float(run["summary"]["common_set_reconstruction"]["test"]["first_64"]["explained_variance"]),
            }
            for name, run in runs.items()
        },
        "heldout_confirmed_features": {
            name: {
                "short": int(run["summary"]["short_confirmed_count"]),
                "long": int(run["summary"]["long_confirmed_count"]),
            }
            for name, run in runs.items()
        },
        "pairwise_stable_match_counts": pair_counts,
        "stable_three_dictionary_feature_count": len(triples),
        "decision": {
            "compare_full_and_prefix64_training": True,
            "add_explicit_length_penalty": False,
            "reason": "Trace-balanced sampling directly controls representation weight without forcing length into the SAE objective; explicit length supervision would confound discovery with the label being explained.",
        },
    }


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    findings: Mapping[str, Any],
    figure_paths: Sequence[Path],
) -> None:
    exposure = findings["length_exposure_spearman_r"]
    ev = findings["test_explained_variance"]
    confirmed = findings["heldout_confirmed_features"]
    lines = [
        "# SAE full-sequence versus prefix-64 sampling ablation",
        "",
        "## Registered comparison",
        "",
        "All three TopK SAEs use layer 17, 28,672 features, k=64, seed 17, 250,000 training tokens, and 1,500 optimizer steps. The only intended difference is how training tokens are sampled: full-sequence token-uniform, full-sequence trace-balanced, or first-64-token trace-balanced. No explicit length penalty or short/long label enters the SAE loss.",
        "",
        "All SAEs are evaluated on the same complete dev/test traces. Candidate features are discovered on dev, confirmed on test, and compared across dictionaries by mutual decoder nearest neighbors, decoder cosine, common-test activation correlation, and short/long direction agreement.",
        "",
        "## Results",
        "",
        f"The training exposure Spearman correlations between trace length and sampled-token count are {exposure['full_token_uniform']:.3f}, {exposure['full_trace_balanced']:.3f}, and {exposure['prefix64_trace_balanced']:.3f}, respectively.",
        "",
        "Common-test explained variance (all tokens / first 64 tokens):",
        "",
    ]
    for name in CONDITIONS:
        lines.append(f"- `{name}`: {ev[name]['full']:.4f} / {ev[name]['first64']:.4f}")
    lines.extend(["", "Held-out confirmed short-associated / long-associated candidates:", ""])
    for name in CONDITIONS:
        lines.append(f"- `{name}`: {confirmed[name]['short']} / {confirmed[name]['long']}")
    lines.extend(["", "Stable candidate matches across dictionary pairs:", ""])
    for pair, count in findings["pairwise_stable_match_counts"].items():
        lines.append(f"- `{pair}`: {count}")
    lines.extend(
        [
            f"- Strict features stable across all three dictionaries: {findings['stable_three_dictionary_feature_count']}",
            "",
            "## Interpretation boundary",
            "",
            "This is a single-SAE-seed exploratory falsification. It can identify sampling and position artifacts, but it does not establish that a feature is semantic, student-utility-related, or causally changes reasoning. Token signatures and relative-position profiles should be treated as diagnostics requiring the lexical falsification tests in Phase 2.5.",
            "",
            "An explicit length penalty is not recommended for the discovery SAE: it would make length separation partly supervised by construction. Trace-balanced sampling is the clean control for long-trace over-weighting; downstream utility should be modeled after feature extraction.",
            "",
            "## Figures",
            "",
        ]
    )
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, allow_empty: bool = False) -> None:
    if not rows and not allow_empty:
        raise ValueError(f"No rows available for {path}")
    fields = list(rows[0]) if rows else ["status"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _save(figure: Any, stem: Path, plt: Any) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return paths


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
