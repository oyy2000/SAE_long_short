import numpy as np

from length_budget_distill.sae_feature_analysis import (
    benjamini_hochberg,
    holm_adjust,
    paired_feature_statistics,
    relative_position_bin,
    select_discovery_features,
)


def test_paired_feature_statistics_uses_question_means() -> None:
    rows = [
        {"problem_id": "q1", "analysis_length_label": "short"},
        {"problem_id": "q1", "analysis_length_label": "short"},
        {"problem_id": "q1", "analysis_length_label": "long"},
        {"problem_id": "q2", "analysis_length_label": "short"},
        {"problem_id": "q2", "analysis_length_label": "long"},
    ]
    values = np.asarray([[3.0], [5.0], [1.0], [4.0], [2.0]])
    result = paired_feature_statistics(values, rows)
    assert result["question_count"] == 2
    assert np.allclose(result["question_differences"].ravel(), [3.0, 2.0])
    assert np.allclose(result["effect"], [2.5])


def test_adjustments_are_monotonic_in_rank() -> None:
    p_values = np.asarray([0.03, 0.001, 0.02, 0.8])
    for adjusted in (benjamini_hochberg(p_values), holm_adjust(p_values)):
        order = np.argsort(p_values)
        assert np.all(np.diff(adjusted[order]) >= -1e-12)
        assert np.all((0 <= adjusted) & (adjusted <= 1))


def test_discovery_selection_preserves_direction_and_gate_status() -> None:
    primary = {
        "paired_d": np.asarray([1.0, 0.5, -0.9, -0.4]),
        "prevalence": np.asarray([0.5, 0.5, 0.5, 0.5]),
        "bh_q_value": np.asarray([0.01, 0.2, 0.01, 0.2]),
    }
    early = {"paired_d": np.asarray([0.5, 0.2, -0.6, -0.1])}
    selected = select_discovery_features(
        primary,
        early,
        per_direction=2,
        minimum_prevalence=0.05,
        maximum_bh_q=0.05,
        minimum_abs_paired_d=0.2,
    )
    assert [row["feature_id"] for row in selected] == [0, 1, 2, 3]
    assert [row["direction"] for row in selected] == ["short", "short", "long", "long"]
    assert [row["passes_discovery_gate"] for row in selected] == [True, False, True, False]


def test_relative_position_bins_cover_boundaries() -> None:
    assert relative_position_bin(0, 10, 5) == 0
    assert relative_position_bin(9, 10, 5) == 4
    assert relative_position_bin(4, 10, 5) == 2
