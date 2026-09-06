#!/usr/bin/env python3
"""Merge SAE feature scores and render held-out short-versus-long diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


PRIMARY_METRIC = "token_mean_activation"
EARLY_METRIC = "first_64_token_mean_activation"
FREQUENCY_METRIC = "token_activation_frequency"
MAXIMUM_METRIC = "maximum_activation"
SHORT_COLOR = "#2166AC"
LONG_COLOR = "#B2182B"
NEUTRAL_COLOR = "#666666"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_short_long_feature_analysis_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    result_root = _resolve(config["outputs"]["result_root"])
    score_root = result_root / "feature_scores"
    analysis_dir = result_root / "analysis"
    figure_dir = _resolve(config["outputs"]["figure_root"])
    if analysis_dir.exists() or figure_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {analysis_dir} or {figure_dir}"
        )
    parent_config = read_json(_resolve(config["parent_sae"]["config_path"]))
    matrix = [
        (int(layer), int(k))
        for layer in parent_config["activation_extraction"]["layer_indices_zero_based"]
        for k in parent_config["sae"]["k_values"]
    ]
    runs = []
    for layer, k in matrix:
        run_dir = score_root / f"layer_{layer:02d}_k_{k:03d}"
        marker_path = run_dir / "FEATURE_SCORING_COMPLETE"
        summary_path = run_dir / "scoring_summary.json"
        marker = read_key_value_marker(marker_path)
        if marker.get("status") != "complete" or marker.get(
            "summary_sha256"
        ) != file_sha256(summary_path):
            raise RuntimeError(f"Incomplete feature-scoring evidence: {run_dir}")
        summary = read_json(summary_path)
        candidates_path = run_dir / "discovered_features.json"
        candidates = read_json(candidates_path)["candidates"]
        runs.append(
            {
                "layer_index": layer,
                "k": k,
                "run_dir": run_dir,
                "summary": summary,
                "candidates": candidates,
                "marker_path": marker_path,
                "summary_path": summary_path,
            }
        )
    primary_definition = config["comparison"]["primary_sae"]
    primary = next(
        row
        for row in runs
        if row["layer_index"] == int(primary_definition["layer_index"])
        and row["k"] == int(primary_definition["k"])
    )
    selected_primary = _figure_features(primary, config)
    if not selected_primary:
        raise RuntimeError("No primary-SAE candidates are available for figures.")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    parent_metrics = _read_csv(
        _resolve(config["parent_sae"]["training_root"]).parent / "audit/sae_metrics.csv"
    )
    figure_paths = []
    figure_paths.extend(_plot_overview(plt, figure_dir, runs, parent_metrics))
    full_primary_stats = _read_csv(primary["run_dir"] / "feature_statistics.csv")
    figure_paths.extend(
        _plot_volcano(plt, figure_dir, full_primary_stats, primary["candidates"])
    )
    figure_paths.extend(
        _plot_validation(plt, figure_dir, selected_primary, primary)
    )
    corpus_rows = _load_jsonl(_resolve(config["parent_sae"]["corpus_path"]))
    tokenizer = _load_tokenizer(parent_config)
    trace_rows = _read_csv(primary["run_dir"] / "selected_trace_metrics.csv")
    event_rows = _load_jsonl(
        primary["run_dir"] / "selected_feature_token_events.jsonl"
    )
    example = _choose_representative_pair(
        trace_rows, selected_primary, split=config["comparison"]["confirmation_split"]
    )
    figure_paths.extend(
        _plot_token_heatmap(
            plt,
            figure_dir,
            selected_primary,
            example,
            event_rows,
            corpus_rows,
            tokenizer,
            first_n_tokens=int(config["token_analysis"]["first_n_tokens"]),
        )
    )
    relative_rows = _read_csv(
        primary["run_dir"] / "selected_feature_relative_position.csv"
    )
    token_frequency = _read_csv(primary["run_dir"] / "test_token_frequency.csv")
    lexical = _lexical_associations(
        event_rows,
        token_frequency,
        selected_primary,
        tokenizer,
        minimum_count=int(
            config["token_analysis"]["minimum_token_occurrences_for_lexical_rate"]
        ),
    )
    figure_paths.extend(
        _plot_token_position_diagnostics(
            plt,
            figure_dir,
            relative_rows,
            selected_primary,
            lexical,
        )
    )
    figure_paths.extend(
        _plot_story(plt, figure_dir, runs, primary, selected_primary, example)
    )
    context_csv, context_md = _write_context_report(
        analysis_dir,
        event_rows,
        corpus_rows,
        tokenizer,
        selected_primary,
        lexical,
        window=int(config["token_analysis"]["context_window_tokens"]),
        per_cell=int(config["token_analysis"]["top_contexts_per_feature_and_label"]),
    )
    run_summary_path = analysis_dir / "sae_feature_difference_summary.csv"
    _write_run_summary(run_summary_path, runs)
    feature_summary_path = analysis_dir / "primary_heldout_features.csv"
    _write_primary_features(feature_summary_path, selected_primary)
    findings = _summarize_findings(runs, primary, selected_primary, example, lexical)
    report_path = analysis_dir / "analysis_report.md"
    _write_report(report_path, config, findings, figure_paths, context_md)
    summary = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "sae_run_count": len(runs),
        "primary_sae": primary_definition,
        "primary_candidate_count": len(primary["candidates"]),
        "primary_heldout_confirmed_count": sum(
            bool(row["confirmed"]) for row in primary["candidates"]
        ),
        "displayed_feature_count": len(selected_primary),
        "representative_example": example,
        "findings": findings,
        "artifacts": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                run_summary_path,
                feature_summary_path,
                context_csv,
                context_md,
                report_path,
                *figure_paths,
            )
        },
        "input_scoring_evidence": [
            {
                "layer_index": row["layer_index"],
                "k": row["k"],
                "marker_path": str(row["marker_path"]),
                "marker_sha256": file_sha256(row["marker_path"]),
                "summary_path": str(row["summary_path"]),
                "summary_sha256": file_sha256(row["summary_path"]),
            }
            for row in runs
        ],
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
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


def _figure_features(primary, config):
    maximum = int(config["token_analysis"]["maximum_features_per_direction_in_figures"])
    selected = []
    for direction in ("short", "long"):
        rows = [row for row in primary["candidates"] if row["direction"] == direction]
        rows.sort(
            key=lambda row: (
                not bool(row["confirmed"]),
                -abs(float(row["metrics"][PRIMARY_METRIC]["test"]["paired_d"])),
            )
        )
        selected.extend(rows[:maximum])
    return selected


def _plot_overview(plt, figure_dir, runs, parent_metrics):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, color, marker in ((32, "#4D9221", "o"), (64, "#7B3294", "s")):
        cells = sorted(
            (int(row["layer_index"]), float(row["test_explained_variance"]))
            for row in parent_metrics
            if int(row["k"]) == k
        )
        axes[0].plot(
            [row[0] for row in cells],
            [row[1] for row in cells],
            marker=marker,
            color=color,
            linewidth=2,
            label=f"TopK k={k}",
        )
    axes[0].set(title="SAE reconstruction on held-out tokens", xlabel="Layer", ylabel="Explained variance")
    axes[0].legend(frameon=False)
    labels = [f"L{row['layer_index']}\nk={row['k']}" for row in runs]
    x = np.arange(len(runs))
    dev_short = [
        sum(c["direction"] == "short" and c["passes_discovery_gate"] for c in row["candidates"])
        for row in runs
    ]
    dev_long = [
        sum(c["direction"] == "long" and c["passes_discovery_gate"] for c in row["candidates"])
        for row in runs
    ]
    test_short = [int(row["summary"]["short_confirmed_count"]) for row in runs]
    test_long = [int(row["summary"]["long_confirmed_count"]) for row in runs]
    width = 0.18
    axes[1].bar(x - 1.5 * width, dev_short, width, color=SHORT_COLOR, alpha=0.45, label="Short dev")
    axes[1].bar(x - 0.5 * width, test_short, width, color=SHORT_COLOR, label="Short test")
    axes[1].bar(x + 0.5 * width, dev_long, width, color=LONG_COLOR, alpha=0.45, label="Long dev")
    axes[1].bar(x + 1.5 * width, test_long, width, color=LONG_COLOR, label="Long test")
    axes[1].set(title="Discovered and held-out confirmed features", ylabel="Feature count", xticks=x, xticklabels=labels)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for run in runs:
        for candidate in run["candidates"]:
            dev = float(candidate["metrics"][PRIMARY_METRIC]["dev"]["paired_d"])
            test = float(candidate["metrics"][PRIMARY_METRIC]["test"]["paired_d"])
            color = SHORT_COLOR if candidate["direction"] == "short" else LONG_COLOR
            axes[2].scatter(dev, test, color=color, alpha=0.45, s=24, edgecolors="none")
    limits = axes[2].axis()
    low = min(limits[0], limits[2])
    high = max(limits[1], limits[3])
    axes[2].plot([low, high], [low, high], linestyle="--", color="#999999", linewidth=1)
    axes[2].axhline(0, color="#cccccc", linewidth=0.8)
    axes[2].axvline(0, color="#cccccc", linewidth=0.8)
    axes[2].set(xlim=(low, high), ylim=(low, high), title="Direction replication across splits", xlabel="Dev paired d", ylabel="Test paired d")
    figure.suptitle("Short-versus-long SAE feature analysis: reconstruction, discovery, replication", fontsize=14, y=1.02)
    figure.tight_layout()
    return _save_figure(figure, figure_dir / "01_sae_feature_difference_overview", plt)


def _plot_volcano(plt, figure_dir, statistics, candidates):
    x = np.asarray([float(row[f"dev_{PRIMARY_METRIC}_paired_d"]) for row in statistics])
    q = np.asarray([float(row[f"dev_{PRIMARY_METRIC}_bh_q_value"]) for row in statistics])
    y = np.minimum(-np.log10(np.clip(q, 1e-300, 1.0)), 30.0)
    candidate_map = {int(row["feature_id"]): row for row in candidates}
    colors = np.full(len(x), "#BDBDBD", dtype=object)
    sizes = np.full(len(x), 8.0)
    for feature_id, candidate in candidate_map.items():
        colors[feature_id] = SHORT_COLOR if candidate["direction"] == "short" else LONG_COLOR
        sizes[feature_id] = 34 if candidate["passes_discovery_gate"] else 22
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(x, y, c=colors, s=sizes, alpha=0.6, edgecolors="none", rasterized=True)
    axis.axvline(0, color="#888888", linewidth=0.8)
    axis.axhline(-math.log10(0.05), color="#555555", linestyle="--", linewidth=1, label="BH q = 0.05")
    for feature_id, candidate in candidate_map.items():
        if candidate["discovery_rank"] <= 4:
            axis.annotate(
                f"F{feature_id}",
                (x[feature_id], y[feature_id]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
    axis.text(0.02, 0.96, "Long-associated", color=LONG_COLOR, transform=axis.transAxes, va="top")
    axis.text(0.98, 0.96, "Short-associated", color=SHORT_COLOR, transform=axis.transAxes, va="top", ha="right")
    axis.set(title="Primary SAE dev discovery (layer 17, k=64)", xlabel="Question-paired standardized effect, short minus long", ylabel="-log10(BH q-value)")
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save_figure(figure, figure_dir / "02_primary_sae_discovery_volcano", plt)


def _plot_validation(plt, figure_dir, features, primary):
    columns = [
        ("Dev\nmean", "dev", PRIMARY_METRIC),
        ("Test\nmean", "test", PRIMARY_METRIC),
        ("Dev\nfirst 64", "dev", EARLY_METRIC),
        ("Test\nfirst 64", "test", EARLY_METRIC),
        ("Test\nfrequency", "test", FREQUENCY_METRIC),
        ("Test\nmaximum", "test", MAXIMUM_METRIC),
    ]
    matrix = np.asarray(
        [
            [float(row["metrics"][metric][split]["paired_d"]) for _, split, metric in columns]
            for row in features
        ]
    )
    limit = max(0.25, float(np.nanpercentile(np.abs(matrix), 95)))
    figure, axis = plt.subplots(figsize=(8.5, max(5.0, 0.42 * len(features))))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    labels = [
        f"{'S' if row['direction'] == 'short' else 'L'} F{row['feature_id']}"
        f"{'  confirmed' if row['confirmed'] else '  not confirmed'}"
        for row in features
    ]
    axis.set_xticks(np.arange(len(columns)), [row[0] for row in columns])
    axis.set_yticks(np.arange(len(features)), labels)
    axis.set_xlabel("Positive = more active in short; negative = more active in long")
    axis.set_title("Dev-discovered features evaluated on held-out questions")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.55 * limit else "black")
    figure.colorbar(image, ax=axis, label="Question-paired d", shrink=0.8)
    figure.tight_layout()
    return _save_figure(figure, figure_dir / "03_heldout_feature_validation", plt)


def _choose_representative_pair(trace_rows, features, *, split):
    test_rows = [row for row in trace_rows if row["question_split"] == split]
    feature_ids = [int(row["feature_id"]) for row in features]
    signs = np.asarray([1.0 if row["direction"] == "short" else -1.0 for row in features])
    matrix = np.asarray(
        [
            [float(row[f"feature_{feature_id}_{PRIMARY_METRIC}"]) for feature_id in feature_ids]
            for row in test_rows
        ]
    )
    scale = matrix.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    signature = ((matrix - matrix.mean(axis=0)) / scale * signs).mean(axis=1)
    by_problem = defaultdict(lambda: {"short": [], "long": []})
    for index, row in enumerate(test_rows):
        by_problem[row["problem_id"]][row["analysis_length_label"]].append(index)
    pairs = []
    for problem_id, cells in by_problem.items():
        short_index = max(cells["short"], key=lambda index: signature[index])
        long_index = min(cells["long"], key=lambda index: signature[index])
        pairs.append(
            {
                "problem_id": problem_id,
                "short_trace_id": test_rows[short_index]["trace_id"],
                "long_trace_id": test_rows[long_index]["trace_id"],
                "short_corpus_index": int(test_rows[short_index]["corpus_index"]),
                "long_corpus_index": int(test_rows[long_index]["corpus_index"]),
                "short_token_count": int(test_rows[short_index]["solution_token_count"]),
                "long_token_count": int(test_rows[long_index]["solution_token_count"]),
                "signature_gap": float(signature[short_index] - signature[long_index]),
            }
        )
    positive = sorted((row for row in pairs if row["signature_gap"] > 0), key=lambda row: row["signature_gap"])
    pool = positive or sorted(pairs, key=lambda row: row["signature_gap"])
    selected = dict(pool[len(pool) // 2])
    selected["selection_rule"] = "median positive held-out signature contrast"
    return selected


def _plot_token_heatmap(plt, figure_dir, features, example, events, corpus_rows, tokenizer, *, first_n_tokens):
    display = []
    for direction in ("short", "long"):
        candidates = [row for row in features if row["direction"] == direction]
        display.extend(candidates[:3])
    feature_ids = [int(row["feature_id"]) for row in display]
    event_map = defaultdict(float)
    for row in events:
        if int(row["feature_id"]) in feature_ids:
            event_map[(row["trace_id"], int(row["feature_id"]), int(row["position"]))] = float(row["activation"])
    corpus = {int(row["corpus_index"]): row for row in corpus_rows}
    panels = [
        ("Short", example["short_trace_id"], example["short_corpus_index"]),
        ("Long", example["long_trace_id"], example["long_corpus_index"]),
    ]
    matrices = []
    token_labels = []
    for _, trace_id, corpus_index in panels:
        token_ids = tokenizer.encode(corpus[corpus_index]["solution"], add_special_tokens=False)
        length = min(first_n_tokens, len(token_ids))
        matrices.append(
            np.asarray(
                [
                    [event_map.get((trace_id, feature_id, position), 0.0) for position in range(length)]
                    for feature_id in feature_ids
                ]
            )
        )
        token_labels.append([_clean_token(tokenizer.convert_ids_to_tokens(token_id)) for token_id in token_ids[:length]])
    maximum = max(float(np.percentile(matrix[matrix > 0], 98)) if np.any(matrix > 0) else 1.0 for matrix in matrices)
    figure, axes = plt.subplots(2, 1, figsize=(18, 7.5), sharey=True)
    for axis, panel, matrix, labels in zip(axes, panels, matrices, token_labels):
        image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=maximum)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yticks(
            np.arange(len(display)),
            [f"{'S' if row['direction'] == 'short' else 'L'} F{row['feature_id']}" for row in display],
        )
        axis.set_ylabel("SAE feature")
        axis.set_title(f"{panel[0]} trace: first {len(labels)} tokens (total {example[panel[0].lower() + '_token_count']} tokens)", loc="left")
    axes[-1].set_xlabel("Decoded completion token")
    figure.colorbar(image, ax=axes, label="TopK feature activation", fraction=0.015, pad=0.01)
    figure.suptitle(
        f"Same-question held-out token view: {example['problem_id']}\n"
        "Rows prefixed S/L were discovered as short-/long-associated on dev questions",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.22, hspace=0.45)
    return _save_figure(figure, figure_dir / "04_same_question_token_heatmap", plt)


def _lexical_associations(events, frequencies, features, tokenizer, *, minimum_count):
    denominators = Counter()
    for row in frequencies:
        denominators[(row["analysis_length_label"], int(row["token_id"]))] += int(row["count"])
    activation = Counter()
    for row in events:
        activation[(int(row["feature_id"]), row["analysis_length_label"], int(row["token_id"]))] += float(row["activation"])
    result = defaultdict(list)
    for feature in features:
        feature_id = int(feature["feature_id"])
        for label in ("short", "long"):
            entries = []
            for (candidate, candidate_label, token_id), total in activation.items():
                if candidate != feature_id or candidate_label != label:
                    continue
                count = denominators[(label, token_id)]
                if count < minimum_count:
                    continue
                entries.append(
                    {
                        "feature_id": feature_id,
                        "feature_direction": feature["direction"],
                        "trace_label": label,
                        "token_id": token_id,
                        "token": _clean_token(tokenizer.convert_ids_to_tokens(token_id)),
                        "token_count": count,
                        "activation_sum": total,
                        "conditional_mean_activation": total / count,
                    }
                )
            entries.sort(key=lambda row: row["conditional_mean_activation"], reverse=True)
            result[(feature_id, label)] = entries
    return result


def _plot_token_position_diagnostics(plt, figure_dir, relative_rows, features, lexical):
    feature_ids = {int(row["feature_id"]) for row in features}
    position = defaultdict(list)
    for row in relative_rows:
        feature_id = int(row["feature_id"])
        if feature_id not in feature_ids:
            continue
        position[(row["direction"], row["analysis_length_label"], int(row["relative_position_bin"]))].append(float(row["mean_activation"]))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    bins = sorted({key[2] for key in position})
    x = np.arange(len(bins))
    for direction, color in (("short", SHORT_COLOR), ("long", LONG_COLOR)):
        for trace_label, linestyle in (("short", "-"), ("long", "--")):
            means = [np.mean(position[(direction, trace_label, bin_index)]) for bin_index in bins]
            axes[0].plot(x, means, color=color, linestyle=linestyle, marker="o", label=f"{direction}-assoc in {trace_label} traces")
    axes[0].set_xticks(x, [f"{20*i}-{20*(i+1)}%" for i in bins])
    axes[0].set(title="Relative-position activation profiles", xlabel="Relative trace position", ylabel="Mean activation per token")
    axes[0].legend(frameon=False, fontsize=8)
    labels = []
    values = []
    colors = []
    for feature in features:
        feature_id = int(feature["feature_id"])
        cell = lexical[(feature_id, feature["direction"])]
        if cell:
            top = cell[0]
            labels.append(f"F{feature_id}: {top['token']}")
            values.append(float(top["conditional_mean_activation"]))
            colors.append(SHORT_COLOR if feature["direction"] == "short" else LONG_COLOR)
    order = np.argsort(values)
    axes[1].barh(np.arange(len(order)), np.asarray(values)[order], color=np.asarray(colors)[order])
    axes[1].set_yticks(np.arange(len(order)), np.asarray(labels)[order], fontsize=8)
    axes[1].set(title="Top token association per displayed feature", xlabel="Activation sum / token occurrences")
    figure.suptitle("Position and lexical diagnostics (descriptive, not semantic proof)", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, figure_dir / "05_token_and_position_diagnostics", plt)


def _plot_story(plt, figure_dir, runs, primary, features, example):
    figure = plt.figure(figsize=(16, 8.5))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle("What differs between naturally short and long correct traces?", fontsize=20, y=0.96, fontweight="bold")
    boxes = [
        (0.04, 0.63, 0.25, 0.19, "Controlled comparison", "Same teacher\nSame GSM8K question\nBoth verifier-correct\nQuestion-paired short minus long"),
        (0.375, 0.63, 0.25, 0.19, "SAE representation", "One mixed dictionary per layer/k\nNo separate short/long SAEs\nDev discovers features\nHeld-out test confirms them"),
        (0.71, 0.63, 0.25, 0.19, "Token interpretation", "Token-normalized activation\nFirst-64 position control\nRelative-position profiles\nTop tokens and text contexts"),
    ]
    for x, y, width, height, title, body in boxes:
        axis.add_patch(plt.Rectangle((x, y), width, height, facecolor="#F7F7F7", edgecolor="#555555", linewidth=1.2))
        axis.text(x + 0.015, y + height - 0.035, title, fontsize=13, fontweight="bold", va="top")
        axis.text(x + 0.015, y + height - 0.075, body, fontsize=10.5, va="top", linespacing=1.45)
    axis.annotate("", xy=(0.37, 0.725), xytext=(0.30, 0.725), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    axis.annotate("", xy=(0.705, 0.725), xytext=(0.635, 0.725), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    short_confirmed = int(primary["summary"]["short_confirmed_count"])
    long_confirmed = int(primary["summary"]["long_confirmed_count"])
    total_confirmed = sum(int(row["summary"]["heldout_confirmed_count"]) for row in runs)
    axis.text(0.04, 0.51, "Primary SAE held-out result", fontsize=15, fontweight="bold")
    count_scale = max(1, short_confirmed, long_confirmed)
    for x, count, color in ((0.095, short_confirmed, SHORT_COLOR), (0.215, long_confirmed, LONG_COLOR)):
        axis.add_patch(
            plt.Rectangle(
                (x, 0.405),
                0.07,
                0.07 * count / count_scale,
                facecolor=color,
                edgecolor="none",
            )
        )
    axis.text(0.13, 0.46, str(short_confirmed), ha="center", color=SHORT_COLOR, fontsize=16, fontweight="bold")
    axis.text(0.25, 0.46, str(long_confirmed), ha="center", color=LONG_COLOR, fontsize=16, fontweight="bold")
    axis.text(0.13, 0.38, "short-associated", ha="center", fontsize=10)
    axis.text(0.25, 0.38, "long-associated", ha="center", fontsize=10)
    axis.text(0.39, 0.49, f"Across six SAEs: {total_confirmed} held-out confirmations", fontsize=13)
    axis.text(0.39, 0.445, f"Displayed token example: {example['problem_id']}", fontsize=11)
    axis.text(0.39, 0.405, f"short {example['short_token_count']} tokens vs long {example['long_token_count']} tokens", fontsize=11)
    axis.text(0.04, 0.24, "Interpretation boundary", fontsize=15, fontweight="bold", color="#7F3B08")
    axis.text(
        0.04,
        0.18,
        "These results establish association and held-out replication only. A feature may encode wording, formatting,\n"
        "position, or reasoning state. Student utility and causal steering remain separate experiments.",
        fontsize=12,
        color="#7F3B08",
        linespacing=1.4,
    )
    axis.text(0.04, 0.08, "Blue: more active in short traces    Red: more active in long traces", fontsize=11, color=NEUTRAL_COLOR)
    return _save_figure(figure, figure_dir / "06_short_long_feature_story", plt)


def _write_context_report(analysis_dir, events, corpus_rows, tokenizer, features, lexical, *, window, per_cell):
    corpus = {int(row["corpus_index"]): row for row in corpus_rows}
    feature_map = {int(row["feature_id"]): row for row in features}
    chosen_events = defaultdict(list)
    for row in events:
        feature_id = int(row["feature_id"])
        if feature_id in feature_map:
            chosen_events[(feature_id, row["analysis_length_label"])].append(row)
    output = []
    for key, rows in chosen_events.items():
        rows.sort(key=lambda row: float(row["activation"]), reverse=True)
        for rank, event in enumerate(rows[:per_cell], start=1):
            source = corpus[int(event["corpus_index"])]
            token_ids = tokenizer.encode(source["solution"], add_special_tokens=False)
            position = int(event["position"])
            left = max(0, position - window)
            right = min(len(token_ids), position + window + 1)
            context = tokenizer.decode(token_ids[left:right]).replace("\n", " ").strip()
            output.append(
                {
                    "feature_id": key[0],
                    "feature_direction": feature_map[key[0]]["direction"],
                    "trace_label": key[1],
                    "rank": rank,
                    "activation": float(event["activation"]),
                    "problem_id": event["problem_id"],
                    "trace_id": event["trace_id"],
                    "position": position,
                    "token": _clean_token(tokenizer.convert_ids_to_tokens(int(event["token_id"]))),
                    "context": context,
                }
            )
    csv_path = analysis_dir / "top_activation_contexts.csv"
    _write_csv(csv_path, output)
    md_path = analysis_dir / "token_contexts.md"
    with md_path.open("x", encoding="utf-8") as handle:
        handle.write("# Token-level contexts for short- and long-associated features\n\n")
        handle.write("The features were selected on dev questions and the contexts below come from held-out test questions. High-activation text is descriptive evidence, not a semantic or causal label.\n\n")
        for feature in features:
            feature_id = int(feature["feature_id"])
            handle.write(f"## F{feature_id} ({feature['direction']}-associated)\n\n")
            for label in ("short", "long"):
                top_tokens = lexical[(feature_id, label)][:5]
                token_text = ", ".join(f"`{row['token']}` ({row['conditional_mean_activation']:.3g})" for row in top_tokens) or "none above frequency threshold"
                handle.write(f"{label.capitalize()} trace top token associations: {token_text}.\n\n")
                matching = [row for row in output if row["feature_id"] == feature_id and row["trace_label"] == label][:3]
                for row in matching:
                    handle.write(f"- activation={row['activation']:.3f}, token=`{row['token']}`, context: {row['context']}\n")
                handle.write("\n")
    return csv_path, md_path


def _summarize_findings(runs, primary, features, example, lexical):
    all_confirmed = sum(int(row["summary"]["heldout_confirmed_count"]) for row in runs)
    primary_short = int(primary["summary"]["short_confirmed_count"])
    primary_long = int(primary["summary"]["long_confirmed_count"])
    displayed_confirmed = sum(bool(row["confirmed"]) for row in features)
    lexical_tokens = []
    for row in features:
        entries = lexical[(int(row["feature_id"]), row["direction"])]
        if entries:
            lexical_tokens.append(entries[0]["token"])
    return {
        "all_sae_heldout_confirmation_count": all_confirmed,
        "primary_short_confirmed_count": primary_short,
        "primary_long_confirmed_count": primary_long,
        "displayed_confirmed_count": displayed_confirmed,
        "representative_problem_id": example["problem_id"],
        "representative_short_tokens": example["short_token_count"],
        "representative_long_tokens": example["long_token_count"],
        "example_top_lexical_associations": lexical_tokens[:12],
        "interpretation": "Replicated short/long association under token normalization and a first-64-token control; not yet utility or causality.",
    }


def _write_report(path, config, findings, figures, context_md):
    with path.open("x", encoding="utf-8") as handle:
        handle.write("# Short-versus-long SAE feature analysis\n\n")
        handle.write("## Result\n\n")
        handle.write(
            f"Across the six layer-by-k SAEs, {findings['all_sae_heldout_confirmation_count']} dev-discovered features passed the registered held-out confirmation rule. "
            f"For the primary layer-17, k=64 SAE, {findings['primary_short_confirmed_count']} were short-associated and {findings['primary_long_confirmed_count']} were long-associated.\n\n"
        )
        handle.write("The primary contrast is token-normalized and question-paired. Directional agreement in the first 64 completion tokens is required for held-out confirmation, so the result is not based on whole-trace activation area alone.\n\n")
        handle.write("## Figures\n\n")
        for figure in figures:
            if figure.suffix == ".png":
                handle.write(f"- [{figure.name}]({figure})\n")
        handle.write("\n## Token contexts\n\n")
        handle.write(f"See [{context_md.name}]({context_md}).\n\n")
        handle.write("## Claim boundary\n\n")
        handle.write(config["claim_boundary"] + "\n")


def _write_run_summary(path, runs):
    rows = []
    for run in runs:
        rows.append(
            {
                "layer_index": run["layer_index"],
                "k": run["k"],
                "candidate_count": run["summary"]["candidate_count"],
                "discovery_gate_pass_count": run["summary"]["discovery_gate_pass_count"],
                "heldout_confirmed_count": run["summary"]["heldout_confirmed_count"],
                "short_confirmed_count": run["summary"]["short_confirmed_count"],
                "long_confirmed_count": run["summary"]["long_confirmed_count"],
            }
        )
    _write_csv(path, rows)


def _write_primary_features(path, features):
    rows = []
    for feature in features:
        rows.append(
            {
                "feature_id": feature["feature_id"],
                "direction": feature["direction"],
                "discovery_rank": feature["discovery_rank"],
                "passes_discovery_gate": feature["passes_discovery_gate"],
                "confirmed": feature["confirmed"],
                "confirmation_holm_p_value": feature["confirmation_holm_p_value"],
                "dev_primary_paired_d": feature["metrics"][PRIMARY_METRIC]["dev"]["paired_d"],
                "test_primary_paired_d": feature["metrics"][PRIMARY_METRIC]["test"]["paired_d"],
                "test_first_64_paired_d": feature["metrics"][EARLY_METRIC]["test"]["paired_d"],
            }
        )
    _write_csv(path, rows)


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_tokenizer(parent_config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        parent_config["teacher"]["snapshot_path"], local_files_only=True
    )


def _clean_token(token):
    value = str(token).replace("Ġ", "▁").replace("Ċ", "\\n")
    value = value.replace("\n", "\\n").replace("\t", "\\t")
    return value[:18]


def _save_figure(figure, stem, plt):
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
