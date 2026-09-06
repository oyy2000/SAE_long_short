#!/usr/bin/env python3
"""Render Chinese explanatory figures for SAE training and feature identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker
from length_budget_distill.sae_explainer import (
    configure_matplotlib,
    read_csv_rows,
    render_observed_short_long_features,
    render_sae_feature_anatomy,
    render_training_and_identification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_sae_explainer_v1.json")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    result_root = _resolve(config["outputs"]["result_root"])
    figure_root = _resolve(config["outputs"]["figure_root"])
    if result_root.exists() or figure_root.exists():
        raise FileExistsError(f"Refusing to overwrite {result_root} or {figure_root}")
    sources = {name: _resolve(path) for name, path in config["sources"].items()}
    sae_marker = read_key_value_marker(sources["sae_completion_marker"])
    feature_marker = read_key_value_marker(sources["feature_analysis_marker"])
    if sae_marker.get("status") != "passed" or sae_marker.get("validated_sae_count") != "6":
        raise RuntimeError("Six-SAE parent audit is incomplete.")
    if feature_marker.get("status") != "passed" or feature_marker.get("validated_sae_count") != "6":
        raise RuntimeError("Short/long feature analysis audit is incomplete.")
    result_root.mkdir(parents=True, exist_ok=False)
    figure_root.mkdir(parents=True, exist_ok=False)
    visual = config["visualization"]
    plt, font_name = configure_matplotlib(Path(visual["chinese_font_path"]))
    protocol = read_json(sources["sae_protocol"])
    training_metrics = read_json(sources["primary_training_metrics"])
    sae_metrics = read_csv_rows(sources["sae_metrics"])
    feature_rows = read_csv_rows(sources["confirmed_features"])
    profile_rows = read_csv_rows(sources["position_profiles"])
    common = {
        "short_color": visual["short_color"],
        "long_color": visual["long_color"],
        "neutral_color": visual["neutral_color"],
        "dpi": int(visual["output_dpi"]),
    }
    figures = []
    figures.extend(
        render_sae_feature_anatomy(
            plt, figure_root / "01_sae_feature_anatomy_zh", **common
        )
    )
    figures.extend(
        render_training_and_identification(
            plt,
            figure_root / "02_sae_training_and_identification_zh",
            protocol=protocol,
            training_metrics=training_metrics,
            sae_metrics=sae_metrics,
            **common,
        )
    )
    figures.extend(
        render_observed_short_long_features(
            plt,
            figure_root / "03_observed_short_long_features_zh",
            feature_rows=feature_rows,
            profile_rows=profile_rows,
            **common,
        )
    )
    missing = [
        name
        for name in config["outputs"]["required_figures"]
        if not (figure_root / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing required figures: {missing}")
    report_path = result_root / "sae_explainer_readme_zh.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("# SAE 与 short/long feature 图解\n\n")
        handle.write("1. `01_sae_feature_anatomy_zh.png`：一个 token 如何被编码为稀疏 feature，以及 feature ID 的含义。\n")
        handle.write("2. `02_sae_training_and_identification_zh.png`：混合轨迹训练、实际训练曲线、六个 SAE 重构结果，以及 short/long 标签只在训练后使用。\n")
        handle.write("3. `03_observed_short_long_features_zh.png`：如何用同题配对效应区分 short/long-associated features，以及实际 token/position 结果。\n\n")
        handle.write("这些图片是对已完成探索性实验的解释，不增加 student utility 或 causal steering 证据。\n")
    summary = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "font_name": font_name,
        "source_artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sources.items()
        },
        "artifacts": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in (*figures, report_path)
        },
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_explainer.py"
        ),
        "formal_claim_allowed": False,
    }
    summary_path = result_root / "sae_explainer_summary.json"
    write_json_exclusive(summary_path, summary)
    marker_text = (
        f"status=complete\nconfig_hash={summary['config_hash']}\n"
        f"summary_sha256={file_sha256(summary_path)}\n"
        f"required_figure_count={len(config['outputs']['required_figures'])}\n"
        "formal_claim_allowed=false\n"
    )
    (result_root / "SAE_EXPLAINER_COMPLETE").write_text(marker_text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
