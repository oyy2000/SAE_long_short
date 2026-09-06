"""Pedagogical, publication-quality diagrams for the SAE feature pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_matplotlib(font_path: Path) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": [font_name, "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    return plt, font_name


def render_sae_feature_anatomy(
    plt: Any,
    output_stem: Path,
    *,
    short_color: str,
    long_color: str,
    neutral_color: str,
    dpi: int,
) -> list[Path]:
    """Explain a token-level SAE and the meaning of one dictionary feature."""

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure = plt.figure(figsize=(17, 9.5))
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    figure.suptitle("一张图理解 SAE：feature 到底是什么？", fontsize=23, fontweight="bold", y=0.965)

    y = 0.69
    boxes = [
        (0.035, y, 0.15, 0.16, "Teacher hidden state", "$h_t \\in \\mathbb{R}^{3584}$\n某一个 completion token\n在 residual stream 的状态", "#F2F2F2"),
        (0.235, y, 0.15, 0.16, "Encoder", "$a_t=W_{enc}(h_t-b)+c$\n把 hidden state 投影到\n28,672 个候选方向", "#E8EEF7"),
        (0.435, y, 0.15, 0.16, "TopK 稀疏化", "$z_t=TopK(ReLU(a_t))$\n只保留最强的 64 个\n其余 28,608 个置零", "#EAF4E3"),
        (0.635, y, 0.15, 0.16, "Decoder", "$\\hat h_t=W_{dec}z_t+b$\n用激活 feature 的方向\n重构原 hidden state", "#F8EAD8"),
        (0.835, y, 0.13, 0.16, "训练目标", "$\\|h_t-\\hat h_t\\|_2^2$\n+ dead-feature auxiliary loss\ndecoder direction 单位化", "#F6E7E7"),
    ]
    for x, box_y, width, height, title, body, face in boxes:
        _box(canvas, FancyBboxPatch, x, box_y, width, height, title, body, face, neutral_color)
    for left, right in ((0.185, 0.235), (0.385, 0.435), (0.585, 0.635), (0.785, 0.835)):
        canvas.add_patch(
            FancyArrowPatch(
                (left + 0.005, y + 0.08),
                (right - 0.005, y + 0.08),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.7,
                color=neutral_color,
            )
        )

    canvas.text(0.04, 0.595, "一个 feature 的定义", fontsize=16, fontweight="bold")
    canvas.text(
        0.04,
        0.545,
        "Feature j 不是一个人工标签，也不等于一个单词。它由同一个 SAE 中的一对对象定义：",
        fontsize=12.5,
    )
    canvas.text(0.08, 0.49, "$z_{t,j}$", color=short_color, fontsize=22, fontweight="bold")
    canvas.text(0.145, 0.49, "该 token 上 feature j 的激活强度", fontsize=12.5)
    canvas.text(0.48, 0.49, "$w_j=W_{dec}[j,:]$", color=long_color, fontsize=20, fontweight="bold")
    canvas.text(0.665, 0.49, "feature j 指向 residual space 的 decoder direction", fontsize=12.5)

    matrix_axis = figure.add_axes([0.055, 0.115, 0.56, 0.30])
    tokens = ["To", "solve", "we", "calculate", "24", "+", "12", "=", "36", ".", "Answer", ":", "36"]
    feature_labels = ["F11983: answer 收尾", "F23254: 求解动作", "F3934: 公式格式", "F11432: 变量/单位", "其他 feature"]
    values = np.zeros((5, len(tokens)))
    values[0, 10] = 1.0
    values[0, 11] = 0.30
    values[1, 3] = 0.75
    values[2, 7] = 0.65
    values[3, 4] = 0.35
    values[3, 6] = 0.30
    values[4, [0, 2, 5, 8, 12]] = [0.25, 0.18, 0.30, 0.22, 0.20]
    image = matrix_axis.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    matrix_axis.set_xticks(np.arange(len(tokens)), tokens, rotation=45, ha="right")
    matrix_axis.set_yticks(np.arange(len(feature_labels)), feature_labels)
    matrix_axis.set_title("同一条轨迹中，每个 token 都有一组稀疏 feature activations", loc="left", pad=12)
    matrix_axis.set_xlabel("completion token $t$")
    figure.colorbar(image, ax=matrix_axis, fraction=0.025, pad=0.02, label="$z_{t,j}$")

    explanation = figure.add_axes([0.67, 0.12, 0.29, 0.29])
    explanation.axis("off")
    _explanation_box(
        explanation,
        FancyBboxPatch,
        0.0,
        0.48,
        1.0,
        0.50,
        "如何给 feature 起名字？",
        "1. 找它最高激活的 token/context\n2. 看跨问题是否重复出现同一模式\n3. 做 token injection、scrubbing、paraphrase\n4. 通过后才给语义标签",
        "#F5F5F5",
        neutral_color,
    )
    _explanation_box(
        explanation,
        FancyBboxPatch,
        0.0,
        0.0,
        1.0,
        0.38,
        "本项目中的 feature ID",
        "F11983 只在 layer 17, k=64 这个 SAE 内有意义。\n换 layer、换 k、换 SAE seed 后，需要匹配 decoder\ndirection，不能直接比较相同数字 ID。",
        "#FFF4E6",
        neutral_color,
    )
    return _save(figure, output_stem, plt, dpi)


def render_training_and_identification(
    plt: Any,
    output_stem: Path,
    *,
    protocol: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    sae_metrics: Sequence[Mapping[str, str]],
    short_color: str,
    long_color: str,
    neutral_color: str,
    dpi: int,
) -> list[Path]:
    """Show what enters SAE training and how post-hoc feature labels are assigned."""

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure = plt.figure(figsize=(17, 10))
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    figure.suptitle("SAE 是怎样训练的？short/long 标签什么时候使用？", fontsize=22, fontweight="bold", y=0.97)
    pipeline = [
        (0.035, "混合轨迹池", "14,096 条 raw trajectories\nshort / medium / long / other / incorrect\n训练时全部混在一起"),
        (0.245, "提取 hidden state", "Qwen2.5-7B-Instruct\ncompletion tokens only\nresidual layers 10 / 17 / 23"),
        (0.455, "确定性 token sample", "每层 train 250k\ndev 50k / test 50k\n先 normalize activation"),
        (0.665, "训练 6 个 SAE", "3 layers × k∈{32,64}\nfeature count = 28,672\n1,500 optimizer steps"),
        (0.835, "训练后才比较", "同一 feature j\n同题 short − long\ndev 发现，test 确认"),
    ]
    widths = [0.16, 0.16, 0.16, 0.14, 0.13]
    for (x, title, body), width in zip(pipeline, widths):
        face = "#EDF4FB" if x < 0.8 else "#FFF1E8"
        _box(canvas, FancyBboxPatch, x, 0.72, width, 0.16, title, body, face, neutral_color)
    for left, right in ((0.195, 0.245), (0.405, 0.455), (0.615, 0.665), (0.805, 0.835)):
        canvas.add_patch(FancyArrowPatch((left, 0.80), (right, 0.80), arrowstyle="-|>", mutation_scale=15, linewidth=1.7, color=neutral_color))
    canvas.text(
        0.035,
        0.665,
        "关键：SAE 训练不看 short/long 标签；否则 dictionary 会把标签差异直接编码进去。",
        fontsize=13,
        fontweight="bold",
        color="#7F3B08",
    )

    curve_axis = figure.add_axes([0.075, 0.15, 0.39, 0.42])
    logs = training_metrics["training_log"]
    steps = np.asarray([int(row["step"]) for row in logs])
    train_loss = np.asarray([float(row["train_reconstruction_loss"]) for row in logs])
    dev_ev = np.asarray([float(row["dev_explained_variance"]) for row in logs])
    curve_axis.plot(steps, train_loss, color=long_color, marker="o", linewidth=2, label="train reconstruction loss")
    curve_axis.set(xlabel="optimizer step", ylabel="reconstruction loss", title="实际训练曲线：primary SAE (layer 17, k=64)")
    curve_axis.tick_params(axis="y", colors=long_color)
    twin = curve_axis.twinx()
    twin.plot(steps, dev_ev, color=short_color, marker="s", linewidth=2, label="dev explained variance")
    twin.set_ylabel("dev explained variance", color=short_color)
    twin.tick_params(axis="y", colors=short_color)
    curve_axis.grid(axis="y", alpha=0.2)
    lines = curve_axis.lines + twin.lines
    curve_axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right", fontsize=9)

    matrix_axis = figure.add_axes([0.555, 0.15, 0.37, 0.42])
    layers = sorted({int(row["layer_index"]) for row in sae_metrics})
    ks = sorted({int(row["k"]) for row in sae_metrics})
    values = np.asarray(
        [
            [
                float(next(row["test_explained_variance"] for row in sae_metrics if int(row["layer_index"]) == layer and int(row["k"]) == k))
                for k in ks
            ]
            for layer in layers
        ]
    )
    image = matrix_axis.imshow(values, cmap="YlGnBu", vmin=0.70, vmax=0.86, aspect="auto")
    matrix_axis.set_xticks(np.arange(len(ks)), [f"k={k}" for k in ks])
    matrix_axis.set_yticks(np.arange(len(layers)), [f"layer {layer}" for layer in layers])
    matrix_axis.set_title("6 个 SAE 的 held-out reconstruction")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            matrix_axis.text(j, i, f"EV={values[i,j]:.3f}", ha="center", va="center", color="white" if values[i,j] > 0.79 else "black", fontweight="bold")
    figure.colorbar(image, ax=matrix_axis, fraction=0.045, pad=0.03, label="test explained variance")
    canvas.text(
        0.535,
        0.08,
        "重构好只说明 SAE 能表示 hidden states；\n不等于 feature 已经有可解释语义。",
        fontsize=12,
        color="#7F3B08",
    )
    return _save(figure, output_stem, plt, dpi)


def render_observed_short_long_features(
    plt: Any,
    output_stem: Path,
    *,
    feature_rows: Sequence[Mapping[str, str]],
    profile_rows: Sequence[Mapping[str, str]],
    short_color: str,
    long_color: str,
    neutral_color: str,
    dpi: int,
) -> list[Path]:
    """Explain how short- and long-associated features are distinguished in data."""

    figure = plt.figure(figsize=(18, 10))
    grid = figure.add_gridspec(2, 3, height_ratios=[0.95, 1.05], width_ratios=[1.05, 1.05, 1.0], hspace=0.34, wspace=0.34)
    figure.suptitle("什么叫 short feature / long feature？", fontsize=23, fontweight="bold", y=0.975)

    definition = figure.add_subplot(grid[0, :2])
    definition.axis("off")
    definition.set_xlim(0, 1)
    definition.set_ylim(0, 1)
    definition.text(0.0, 0.96, "1. 它们是训练完成后的统计关联，不是训练标签", fontsize=16, fontweight="bold", va="top")
    definition.text(0.02, 0.75, "$A_j^{full}(\\tau)=\\frac{1}{T}\\sum_{t=1}^{T} z_{t,j}$", fontsize=19)
    definition.text(0.52, 0.75, "$A_j^{64}(\\tau)=\\frac{1}{\\min(T,64)}\\sum_{t<64} z_{t,j}$", fontsize=19)
    definition.text(0.02, 0.55, "全 sequence：该 feature 在整条回答中平均有多强", fontsize=12)
    definition.text(0.52, 0.55, "前 64 token：该 feature 是否在轨迹早期已经出现", fontsize=12)
    definition.text(0.02, 0.32, "$\\Delta_j=mean_q[A_j(short)]-mean_q[A_j(long)]$", fontsize=20, fontweight="bold")
    definition.text(0.02, 0.12, "$\\Delta_j>0$：short-associated", color=short_color, fontsize=14, fontweight="bold")
    definition.text(0.40, 0.12, "$\\Delta_j<0$：long-associated", color=long_color, fontsize=14, fontweight="bold")
    definition.text(0.75, 0.12, "同一问题内配对，再跨问题检验", color=neutral_color, fontsize=12)

    cards = figure.add_subplot(grid[0, 2])
    cards.axis("off")
    cards.set_xlim(0, 1)
    cards.set_ylim(0, 1)
    cards.text(0.0, 0.96, "2. 真实 feature 例子", fontsize=16, fontweight="bold", va="top")
    _feature_card(cards, 0.02, 0.57, 0.96, 0.26, short_color, "Short F11983", "最高激活 token: Answer\n96.3% activation mass 在 Answer\nfull d=+2.28，first-64 d=+0.06")
    _feature_card(cards, 0.02, 0.25, 0.96, 0.26, long_color, "Long F3934", "最高激活 token: [\\n\n公式块、换行和步骤格式\nfull d=-0.84，first-64 d=-0.15")
    cards.text(0.02, 0.08, "所以 feature ID 本身没有名字；\n名字来自 activation contexts 和 falsification。", fontsize=11.5, color="#7F3B08")

    position_axis = figure.add_subplot(grid[1, 0])
    bins = sorted({int(row["relative_position_bin"]) for row in profile_rows})
    x = np.arange(len(bins))
    for direction, color in (("short", short_color), ("long", long_color)):
        for label, linestyle in (("short", "-"), ("long", "--")):
            values = [
                float(next(row["mean_activation_per_token"] for row in profile_rows if row["feature_direction"] == direction and row["trace_label"] == label and int(row["relative_position_bin"]) == bin_index))
                for bin_index in bins
            ]
            position_axis.plot(x, values, color=color, linestyle=linestyle, marker="o", linewidth=2, label=f"{direction}-assoc / {label} trace")
    position_axis.set_xticks(x, [f"{20*i}-{20*(i+1)}%" for i in bins])
    position_axis.set(title="3. feature 在轨迹的什么位置出现？", xlabel="relative completion position", ylabel="mean activation per token")
    position_axis.legend(frameon=False, fontsize=8)

    scatter_axis = figure.add_subplot(grid[1, 1])
    for row in feature_rows:
        direction = row["direction"]
        color = short_color if direction == "short" else long_color
        x_value = float(row["test_token_mean_paired_d"])
        y_value = float(row["test_first_64_paired_d"])
        scatter_axis.scatter(x_value, y_value, color=color, s=55)
        scatter_axis.annotate(f"F{row['feature_id']}", (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
    scatter_axis.axhline(0, color="#999999", linewidth=0.8)
    scatter_axis.axvline(0, color="#999999", linewidth=0.8)
    scatter_axis.set(title="4. 全 sequence 与前 64 token", xlabel="full-sequence paired d", ylabel="first-64 paired d")
    scatter_axis.text(0.55, 0.62, "short: full 很强\n但 early 接近 0", transform=scatter_axis.transAxes, ha="left", va="top", color=short_color, fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
    scatter_axis.text(0.03, 0.05, "long: full 与 early\n方向一致", transform=scatter_axis.transAxes, ha="left", va="bottom", color=long_color, fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    conclusion = figure.add_subplot(grid[1, 2])
    conclusion.axis("off")
    conclusion.set_xlim(0, 1)
    conclusion.set_ylim(0, 1)
    conclusion.text(0.0, 0.98, "5. 当前可以怎样解释？", fontsize=16, fontweight="bold", va="top")
    _feature_card(conclusion, 0.02, 0.67, 0.96, 0.23, short_color, "观察到的 short features", "主要是 Answer / 冒号等收尾信号\n更像 final-answer density\n不应直接拿来 steering")
    _feature_card(conclusion, 0.02, 0.38, 0.96, 0.23, long_color, "观察到的 long features", "公式换行、步骤编号、Find/Substituting\n在 early prefix 已部分出现\n仍可能只是 style/lexical cue")
    _feature_card(conclusion, 0.02, 0.04, 0.96, 0.28, "#7F3B08", "尚不能声称", "不是 pre-generation 的 short/long intent\n不是 student teaching utility\n不是已证实的 extra computation\n需要 injection/scrubbing/paraphrase/steering")
    return _save(figure, output_stem, plt, dpi)


def _box(axis: Any, patch_class: Any, x: float, y: float, width: float, height: float, title: str, body: str, face: str, edge: str) -> None:
    axis.add_patch(
        patch_class(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    axis.text(x + 0.014, y + height - 0.028, title, fontsize=12.5, fontweight="bold", va="top")
    axis.text(x + 0.014, y + height - 0.070, body, fontsize=10.2, va="top", linespacing=1.35)


def _feature_card(axis: Any, x: float, y: float, width: float, height: float, color: str, title: str, body: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=2,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            0.025,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            facecolor=color,
            edgecolor=color,
        )
    )
    axis.text(x + 0.055, y + height - 0.035, title, fontsize=11.5, fontweight="bold", color=color, va="top")
    axis.text(x + 0.055, y + height - 0.092, body, fontsize=9.5, va="top", linespacing=1.22)


def _explanation_box(axis: Any, patch_class: Any, x: float, y: float, width: float, height: float, title: str, body: str, face: str, edge: str) -> None:
    axis.add_patch(
        patch_class(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    axis.text(x + 0.025, y + height - 0.055, title, fontsize=11.5, fontweight="bold", va="top")
    axis.text(x + 0.025, y + height - 0.155, body, fontsize=8.8, va="top", linespacing=1.25)


def _save(figure: Any, output_stem: Path, plt: Any, dpi: int) -> list[Path]:
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]
