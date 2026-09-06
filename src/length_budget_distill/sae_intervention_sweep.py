"""Audited calibration loading and paired analysis for SAE intervention sweeps."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, read_key_value_marker


def analyze_strength_sweep(config_path: Path, project_root: Path) -> None:
    """Require every registered cell, then summarize question-paired effects."""
    import numpy as np

    config = read_json(config_path)
    marker = read_key_value_marker(config_path.parent / "PROTOCOL_FROZEN")
    if marker.get("status") != "frozen" or marker.get("config_hash") != canonical_sha256(config) or marker.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Frozen protocol hash mismatch")
    for key, path in config["parent_evidence"].items():
        if key.endswith("_path") and file_sha256(Path(path)) != config["parent_evidence"][key[:-5] + "_sha256"]:
            raise ValueError(f"Parent evidence hash mismatch: {path}")
    root = project_root / config["outputs"]["result_root"]
    records, evidence = load_calibration_shards(
        root / "calibration_shards", int(config["calibration"]["generation_shards"]), config
    )
    expected = {"no_steering": ("none", 0.0)}
    for prefix, mode, key in (
        ("short_feature_enhance", "short_enhance", "short_strength_grid"),
        ("long_feature_suppress", "long_suppress", "long_suppression_grid"),
        ("random_feature_enhance", "random_enhance", "random_strength_grid"),
    ):
        for value in config["intervention"].get(key, []):
            expected[f"{prefix}__a_{str(float(value)).replace('.', 'p')}"] = (mode, float(value))
    ids = config["question_cohorts"]["calibration_problem_ids"]
    candidates = int(config["calibration"]["candidates_per_question"])
    keys = [(pid, candidate) for pid in ids for candidate in range(candidates)]
    cells = {(r["condition"], r["problem_id"], int(r["candidate_index"])): r for r in records}
    if set(cells) != {(name, *key) for name in expected for key in keys}:
        raise ValueError("Missing or unexpected registered sweep cells")
    for key in keys:
        base = cells[("no_steering", *key)]
        for name, (mode, strength) in expected.items():
            row = cells[(name, *key)]
            if (row["intervention_mode"], row["intervention_strength"]) != (mode, strength):
                raise ValueError("Record strength differs from registered condition")
            for field in ("common_prefix_sha256", "prefix_seed", "continuation_seed", "prompt", "answer"):
                if row[field] != base[field]:
                    raise ValueError(f"Paired generation mismatch: {field}")
            n = int(row["common_prefix_token_count"])
            if row["response_token_ids"][:n] != base["response_token_ids"][:n]:
                raise ValueError("Paired token prefixes differ")
    sweep = config["strength_sweep"]
    rng = np.random.default_rng(int(sweep["bootstrap_seed"]))
    indices = rng.integers(0, len(ids), size=(int(sweep["bootstrap_samples"]), len(ids)))

    def paired(rows, controls, field):
        delta = np.array([float(r[field]) - float(b[field]) for r, b in zip(rows, controls)])
        question_delta = delta.reshape(len(ids), candidates).mean(axis=1)
        low, high = np.quantile(question_delta[indices].mean(axis=1), [0.025, 0.975])
        return float(delta.mean()), float(low), float(high)

    controls = [cells[("no_steering", *key)] for key in keys]
    baseline_length = statistics.mean(r["output_token_count"] for r in controls)
    summaries = []
    for name, (mode, strength) in expected.items():
        rows = [cells[(name, *key)] for key in keys]
        summary = summarize_condition(name, rows)
        for field, label in (("output_token_count", "length_delta"), ("is_correct", "accuracy_delta")):
            mean, low, high = paired(rows, controls, field)
            summary.update({label: mean, label + "_95_low": low, label + "_95_high": high})
        summary["relative_length_delta"] = summary["length_delta"] / baseline_length
        summary["changed_response_fraction"] = statistics.mean(
            r["response_token_ids"] != b["response_token_ids"] for r, b in zip(rows, controls)
        )
        summary["shorter_fraction"] = statistics.mean(r["output_token_count"] < b["output_token_count"] for r, b in zip(rows, controls))
        summary["longer_fraction"] = statistics.mean(r["output_token_count"] > b["output_token_count"] for r, b in zip(rows, controls))
        summary["at_token_cap_fraction"] = statistics.mean(r["output_token_count"] >= config["teacher"]["max_new_tokens"] for r in rows)
        summary["passes_exploratory_effect_screen"] = bool(
            mode in ("short_enhance", "long_suppress")
            and summary["relative_length_delta"] <= -float(sweep["minimum_relative_length_reduction"])
            and summary["length_delta_95_high"] < 0
            and summary["accuracy_delta"] >= -float(sweep["maximum_accuracy_drop"])
        )
        # Matched random directions are available for enhancement only.
        random_name = f"random_feature_enhance__a_{str(strength).replace('.', 'p')}"
        for label in ("length_vs_random", "accuracy_vs_random"):
            for suffix in ("", "_95_low", "_95_high"):
                summary[label + suffix] = None
        if mode == "short_enhance" and random_name in expected:
            random_rows = [cells[(random_name, *key)] for key in keys]
            for field, label in (("output_token_count", "length_vs_random"), ("is_correct", "accuracy_vs_random")):
                mean, low, high = paired(rows, random_rows, field)
                summary.update({label: mean, label + "_95_low": low, label + "_95_high": high})
        summaries.append(summary)
    output = root / "strength_analysis"
    output.mkdir(exist_ok=False)
    figure_dir = project_root / config["outputs"]["figure_root"] / "exploratory"
    figure_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output / "strength_metrics.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    merged_path = output / "merged_generations.jsonl"
    with merged_path.open("x") as handle:
        for row in sorted(records, key=lambda r: (r["problem_id"], r["candidate_index"], r["condition"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    figures = plot_strength_response(summaries, figure_dir)
    passed = [r["condition"] for r in summaries if r["passes_exploratory_effect_screen"]]
    report_path = output / "strength_report.md"
    lines = [
        "# 更强 SAE 干预：教师端探索性强度扫描", "",
        f"已审计 {len(ids)} 道开发集题目、每题 {candidates} 个候选、{len(expected)} 个条件，共 {len(records)} 条输出。",
        "完整单元格、重复记录、共同前缀、配对种子、配置与输入分片哈希检查通过。", "",
        "沿用第 17 层和前 64 tokens 共同前缀，扰动范数上限为隐藏状态的 15%。",
        "开发集包含前轮校准题；这是探索性复用，不是独立确认。",
        "长特征系数 2 表示减去两倍激活分量，超过完全消融，可能反转该方向；不应解释为普通的部分抑制。", "",
        "| 条件 | 正确率 | 平均 tokens | 配对长度差 [95% CI] | 平均扰动 | 达到 token 上限 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(f"| {r['condition']} | {100*r['accuracy']:.2f}% | {r['mean_output_tokens']:.2f} | {r['length_delta']:+.2f} [{r['length_delta_95_low']:.2f}, {r['length_delta_95_high']:.2f}] | {100*r['mean_delta_to_hidden_norm_fraction']:.2f}% | {100*r['at_token_cap_fraction']:.2f}% |")
    lines += ["", "探索性筛选条件：平均长度至少减少 5%，配对长度差区间上界小于 0，正确率点估计下降不超过 5 个百分点。", f"通过筛选：{', '.join(passed) if passed else '无'}。", "", "区间为按题配对 bootstrap 的逐点 95% 区间，未做多重比较校正；正确率点估计过线不证明非劣效。", "扰动诊断由每个生成 batch 汇总后复制到记录，包含 batch 中已结束序列的后续前向位置；它不是逐条有效输出 token 的精确扰动统计。", "本轮仅验证教师端行为，不包含新的学生训练或正式机制结论。", ""]
    report_path.write_text("\n".join(lines))
    artifacts = [csv_path, merged_path, report_path, *figures]
    manifest = {
        "status": "passed", "experiment_name": config["experiment_name"],
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "config_hash": canonical_sha256(config), "record_count": len(records),
        "expected_record_count": len(expected) * len(keys), "duplicate_cell_count": 0,
        "missing_cell_count": 0, "paired_prefix_audit": "passed", "input_shards": evidence,
        "source_sha256": file_sha256(Path(__file__)),
        "entrypoint_sha256": file_sha256(project_root / "scripts/3_4_analyze_sae_strength_sweep.py"),
        "passing_exploratory_conditions": passed, "formal_claim_allowed": False,
        "artifacts": {str(p): file_sha256(p) for p in artifacts},
    }
    manifest_path = output / "strength_analysis_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output / "STRENGTH_SWEEP_COMPLETE").write_text(
        f"status=passed\nmanifest_sha256={file_sha256(manifest_path)}\nconfig_hash={canonical_sha256(config)}\nformal_claim_allowed=false\n"
    )
    print(json.dumps({"status": "passed", "records": len(records), "passing_conditions": passed, "report": str(report_path)}))


def plot_strength_response(rows, output: Path) -> list[Path]:
    """Extend phase3 colors to dose-response lines with paired uncertainty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"short_enhance": "#4472C4", "long_suppress": "#70AD47", "random_enhance": "#ED7D31"}
    labels = {"short_enhance": "Enhance short features", "long_suppress": "Suppress long features", "random_enhance": "Random direction control"}
    for row_index, modes in enumerate((("short_enhance", "random_enhance"), ("long_suppress",))):
        for mode in modes:
            selected = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["strength"])
            baseline = next(r for r in rows if r["mode"] == "none")
            selected = [baseline] + selected
            strengths = [r["strength"] for r in selected]
            for column, (key, factor, ylabel) in enumerate((("length_delta", 1, "Output length change (tokens)"), ("accuracy_delta", 100, "Correctness change (percentage points)"))):
                mean = np.array([factor * r[key] for r in selected])
                low = np.array([factor * r[key + "_95_low"] for r in selected])
                high = np.array([factor * r[key + "_95_high"] for r in selected])
                axis = axes[row_index, column]
                axis.errorbar(strengths, mean, yerr=[mean-low, high-mean], fmt="o-", capsize=3, color=colors[mode], label=labels[mode])
                axis.axhline(0, color="gray", linewidth=0.8)
                axis.set_ylabel(ylabel)
            axes[row_index, 2].plot(strengths, [100*r["mean_delta_to_hidden_norm_fraction"] for r in selected], "o-", color=colors[mode], label=labels[mode])
        for axis in axes[row_index]:
            axis.set_xlabel("Enhancement strength" if row_index == 0 else "Suppression strength")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[row_index, 2].set_ylabel("Mean recorded ||delta h|| / ||h|| (%)")
    figure.suptitle(f"Stronger SAE intervention: paired teacher dose response\n{rows[0]['n']} traces per condition; pointwise 95% question-bootstrap intervals; exploratory", fontsize=12)
    figure.tight_layout()
    paths = [output / "teacher_strength_dose_response.png", output / "teacher_strength_dose_response.pdf"]
    for path in paths:
        figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths

def load_calibration_shards(
    root: Path, shard_count: int, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    evidence = []
    seen_pairs = set()
    source_hashes = set()
    library_hashes = set()
    for shard_index in range(shard_count):
        directory = root / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        marker_path = directory / "GENERATION_SHARD_COMPLETE"
        manifest_path = directory / "generation_manifest.json"
        marker = read_key_value_marker(marker_path)
        manifest = read_json(manifest_path)
        if marker.get("status") != "complete" or marker.get("stage") != "calibration":
            raise ValueError(f"Incomplete calibration shard: {directory}")
        if marker.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"Calibration manifest hash mismatch: {directory}")
        if manifest["config_hash"] != canonical_sha256(config):
            raise ValueError(f"Calibration config hash mismatch: {directory}")
        records_path = Path(manifest["records_path"])
        if file_sha256(records_path) != manifest["records_sha256"]:
            raise ValueError(f"Calibration records hash mismatch: {directory}")
        source_hashes.add(manifest["source_code_sha256"])
        library_hashes.add(manifest["library_code_sha256"])
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = (
                    row["condition"],
                    row["problem_id"],
                    int(row["candidate_index"]),
                )
                if identity in seen_pairs:
                    raise ValueError(f"Duplicate calibration record: {identity}")
                seen_pairs.add(identity)
                records.append(row)
        evidence.append(
            {
                "shard_index": shard_index,
                "marker_path": str(marker_path),
                "marker_sha256": file_sha256(marker_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "records_path": str(records_path),
                "records_sha256": manifest["records_sha256"],
            }
        )
    if len(source_hashes) != 1 or len(library_hashes) != 1:
        raise ValueError("Calibration shards used inconsistent source hashes.")
    return records, evidence


def summarize_condition(name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row["condition"] == name]
    correct = [row for row in rows if bool(row["is_correct"])]
    total_tokens = sum(int(row["output_token_count"]) for row in rows)
    diagnostics = [row["intervention_diagnostics"] for row in rows]
    return {
        "condition": name,
        "mode": rows[0]["intervention_mode"],
        "strength": float(rows[0]["intervention_strength"]),
        "n": len(rows),
        "correct": len(correct),
        "accuracy": len(correct) / len(rows),
        "total_generated_tokens": total_tokens,
        "mean_output_tokens": statistics.fmean(
            int(row["output_token_count"]) for row in rows
        ),
        "mean_correct_output_tokens": statistics.fmean(
            int(row["output_token_count"]) for row in correct
        )
        if correct
        else float("inf"),
        "correct_usable_per_million_tokens": len(correct)
        / max(total_tokens, 1)
        * 1_000_000,
        "mean_modified_token_fraction": statistics.fmean(
            float(row["modified_tokens"])
            / max(int(row["continuation_forward_tokens"]), 1)
            for row in diagnostics
        ),
        "mean_delta_to_hidden_norm_fraction": statistics.fmean(
            float(row["mean_delta_to_hidden_norm_fraction"])
            for row in diagnostics
        ),
        "maximum_delta_to_hidden_norm_fraction": max(
            float(row["max_delta_to_hidden_norm_fraction"])
            for row in diagnostics
        ),
    }
