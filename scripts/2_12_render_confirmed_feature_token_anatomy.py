#!/usr/bin/env python3
"""Render a focused token anatomy of held-out-confirmed primary-SAE features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import file_sha256, read_key_value_marker


SHORT_COLOR = "#2166AC"
LONG_COLOR = "#B2182B"


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
    primary = config["comparison"]["primary_sae"]
    run_dir = (
        _resolve(config["outputs"]["result_root"])
        / "feature_scores"
        / f"layer_{int(primary['layer_index']):02d}_k_{int(primary['k']):03d}"
    )
    marker_path = run_dir / "FEATURE_SCORING_COMPLETE"
    summary_path = run_dir / "scoring_summary.json"
    marker = read_key_value_marker(marker_path)
    if marker.get("status") != "complete" or marker.get(
        "summary_sha256"
    ) != file_sha256(summary_path):
        raise RuntimeError("Primary feature-scoring evidence is incomplete.")
    output_dir = _resolve(config["outputs"]["result_root"]) / "supplementary_token_analysis_v1"
    figure_root = _resolve(config["outputs"]["figure_root"])
    png_path = figure_root / "07_confirmed_feature_token_anatomy.png"
    pdf_path = figure_root / "07_confirmed_feature_token_anatomy.pdf"
    if output_dir.exists() or png_path.exists() or pdf_path.exists():
        raise FileExistsError("Refusing to overwrite supplementary token analysis.")
    candidates = [
        row
        for row in read_json(run_dir / "discovered_features.json")["candidates"]
        if bool(row["confirmed"])
    ]
    if not candidates:
        raise RuntimeError("No held-out-confirmed features are available.")
    feature_map = {int(row["feature_id"]): row for row in candidates}
    event_path = run_dir / "selected_feature_token_events.jsonl"
    activation_mass = Counter()
    token_mass = Counter()
    with event_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            feature_id = int(row["feature_id"])
            if feature_id not in feature_map:
                continue
            value = float(row["activation"])
            activation_mass[feature_id] += value
            token_mass[(feature_id, int(row["token_id"]))] += value
    position_values = defaultdict(list)
    relative_path = run_dir / "selected_feature_relative_position.csv"
    with relative_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            feature_id = int(row["feature_id"])
            if feature_id not in feature_map:
                continue
            position_values[
                (
                    feature_map[feature_id]["direction"],
                    row["analysis_length_label"],
                    int(row["relative_position_bin"]),
                )
            ].append(float(row["mean_activation"]))
    parent_config = read_json(_resolve(config["parent_sae"]["config_path"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        parent_config["teacher"]["snapshot_path"], local_files_only=True
    )
    feature_rows = []
    for candidate in candidates:
        feature_id = int(candidate["feature_id"])
        top_token_id, top_mass = max(
            (
                (token_id, mass)
                for (candidate_id, token_id), mass in token_mass.items()
                if candidate_id == feature_id
            ),
            key=lambda row: row[1],
        )
        feature_rows.append(
            {
                "feature_id": feature_id,
                "direction": candidate["direction"],
                "test_token_mean_paired_d": float(
                    candidate["metrics"]["token_mean_activation"]["test"]["paired_d"]
                ),
                "test_first_64_paired_d": float(
                    candidate["metrics"]["first_64_token_mean_activation"]["test"]["paired_d"]
                ),
                "top_token_id": top_token_id,
                "top_token": _clean_token(
                    tokenizer.convert_ids_to_tokens(top_token_id)
                ),
                "top_token_activation_mass_share": top_mass
                / activation_mass[feature_id],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_csv = output_dir / "confirmed_feature_token_summary.csv"
    _write_csv(feature_csv, feature_rows)
    profile_rows = []
    for direction in ("short", "long"):
        for trace_label in ("short", "long"):
            for bin_index in range(int(config["token_analysis"]["relative_position_bins"])):
                values = position_values[(direction, trace_label, bin_index)]
                profile_rows.append(
                    {
                        "feature_direction": direction,
                        "trace_label": trace_label,
                        "relative_position_bin": bin_index,
                        "mean_activation_per_token": float(np.mean(values)),
                        "feature_trace_cell_count": len(values),
                    }
                )
    profile_csv = output_dir / "confirmed_feature_position_profiles.csv"
    _write_csv(profile_csv, profile_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    bins = np.arange(int(config["token_analysis"]["relative_position_bins"]))
    for direction, color in (("short", SHORT_COLOR), ("long", LONG_COLOR)):
        for trace_label, linestyle in (("short", "-"), ("long", "--")):
            values = [
                next(
                    row["mean_activation_per_token"]
                    for row in profile_rows
                    if row["feature_direction"] == direction
                    and row["trace_label"] == trace_label
                    and row["relative_position_bin"] == bin_index
                )
                for bin_index in bins
            ]
            axes[0].plot(
                bins,
                values,
                marker="o",
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{direction}-assoc in {trace_label}",
            )
    axes[0].set_xticks(bins, [f"{20*i}-{20*(i+1)}%" for i in bins])
    axes[0].set(
        title="A. Where the features activate",
        xlabel="Relative completion position",
        ylabel="Mean activation per token",
    )
    axes[0].legend(frameon=False, fontsize=8)
    ordered = sorted(
        feature_rows, key=lambda row: row["top_token_activation_mass_share"]
    )
    y = np.arange(len(ordered))
    axes[1].barh(
        y,
        [row["top_token_activation_mass_share"] for row in ordered],
        color=[SHORT_COLOR if row["direction"] == "short" else LONG_COLOR for row in ordered],
    )
    axes[1].set_yticks(
        y,
        [
            f"{'S' if row['direction'] == 'short' else 'L'} F{row['feature_id']}: {row['top_token']}"
            for row in ordered
        ],
        fontsize=8,
    )
    axes[1].set_xlim(0, 1)
    axes[1].set(
        title="B. Lexical concentration",
        xlabel="Share of activation mass on top token",
    )
    for row in feature_rows:
        color = SHORT_COLOR if row["direction"] == "short" else LONG_COLOR
        axes[2].scatter(
            row["test_token_mean_paired_d"],
            row["test_first_64_paired_d"],
            color=color,
            s=48,
        )
        axes[2].annotate(
            f"F{row['feature_id']}",
            (
                row["test_token_mean_paired_d"],
                row["test_first_64_paired_d"],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2].axhline(0, color="#999999", linewidth=0.8)
    axes[2].axvline(0, color="#999999", linewidth=0.8)
    axes[2].set(
        title="C. Whole trace versus first 64 tokens",
        xlabel="Held-out whole-trace paired d",
        ylabel="Held-out first-64 paired d",
    )
    figure.suptitle(
        "Token anatomy of held-out-confirmed features (primary SAE: layer 17, k=64)",
        fontsize=15,
        y=1.01,
    )
    figure.text(
        0.5,
        0.01,
        "Blue short-associated features concentrate at the Answer token near the end; red long-associated features are more distributed and emerge earlier.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    short_rows = [row for row in feature_rows if row["direction"] == "short"]
    long_rows = [row for row in feature_rows if row["direction"] == "long"]
    report_path = output_dir / "token_interpretation_zh.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("# Short/long SAE feature 的 token 解释\n\n")
        handle.write("## 核心结论\n\n")
        handle.write(
            "Primary SAE（layer 17, k=64）有 13 个 dev 发现、test 确认的 feature：5 个 short-associated，8 个 long-associated。"
            "但是 token 分析表明两侧不是对称的潜在推理机制。\n\n"
        )
        handle.write(
            f"- 5 个 short-associated feature 的最高 activation-mass token 全部是 `Answer`；该 token 单独解释每个 feature 总 activation mass 的 "
            f"{100*min(row['top_token_activation_mass_share'] for row in short_rows):.1f}%–{100*max(row['top_token_activation_mass_share'] for row in short_rows):.1f}%。\n"
        )
        handle.write(
            "- short-associated feature 在前 80% 轨迹中几乎不激活，主要在最后 20% 突增。因为 `Answer` 基本每条轨迹只出现一次，同样一次收尾激活除以更短的序列长度，会机械地产生更高的 token-normalized trace mean。\n"
        )
        handle.write(
            "- 8 个 long-associated feature 的最高 token mass 更分散，单一 token 占比为 "
            f"{100*min(row['top_token_activation_mass_share'] for row in long_rows):.1f}%–{100*max(row['top_token_activation_mass_share'] for row in long_rows):.1f}%；"
            "其 token/context 多为公式换行、章节/步骤编号、Substituting/Find/Conclusion 等过程展开，以及单位和变量标签。\n"
        )
        handle.write(
            "- long-associated feature 在首 64 token 已呈同方向（paired d 为负），说明长轨迹的展开风格在采样分岔后较早出现；但这仍然可能是格式和措辞，不足以证明 extra computation。\n\n"
        )
        handle.write("## 研究含义\n\n")
        handle.write(
            "当前结果支持的是可复现的轨迹状态/风格差异，而不是 pre-generation short intent，也不是 student utility。"
            "进入 steering 前应优先对 long-associated feature 做 lexical scrubbing、token injection 和 paraphrase invariance；"
            "short-associated `Answer` features 不应作为增强 short trace 的候选。\n"
        )
    summary = {
        "status": "complete",
        "primary_sae": primary,
        "confirmed_feature_count": len(feature_rows),
        "short_confirmed_count": len(short_rows),
        "long_confirmed_count": len(long_rows),
        "short_top_token_all_answer": all(row["top_token"] == "Answer" for row in short_rows),
        "short_top_token_mass_share_range": [
            min(row["top_token_activation_mass_share"] for row in short_rows),
            max(row["top_token_activation_mass_share"] for row in short_rows),
        ],
        "long_top_token_mass_share_range": [
            min(row["top_token_activation_mass_share"] for row in long_rows),
            max(row["top_token_activation_mass_share"] for row in long_rows),
        ],
        "input_artifacts": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in (marker_path, summary_path, event_path, relative_path)
        },
        "artifacts": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in (feature_csv, profile_csv, report_path, png_path, pdf_path)
        },
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    supplementary_summary = output_dir / "supplementary_summary.json"
    write_json_exclusive(supplementary_summary, summary)
    (output_dir / "SUPPLEMENTARY_TOKEN_ANALYSIS_COMPLETE").write_text(
        f"status=complete\nsummary_sha256={file_sha256(supplementary_summary)}\n"
        f"figure_sha256={file_sha256(png_path)}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clean_token(token: str) -> str:
    return str(token).replace("Ġ", "▁").replace("Ċ", "\\n").replace("\n", "\\n")[:18]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
