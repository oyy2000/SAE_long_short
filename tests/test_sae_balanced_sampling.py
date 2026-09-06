import numpy as np

from length_budget_distill.sae_sampling import (
    deterministic_positions_for_quotas,
    maximally_equal_trace_quotas,
)


def test_maximally_equal_trace_quotas_are_exact_and_capacity_bounded() -> None:
    capacities = {0: 2, 1: 10, 2: 10, 3: 10}
    quotas = maximally_equal_trace_quotas(capacities, target_tokens=20, seed=7)
    assert sum(quotas.values()) == 20
    assert quotas[0] == 2
    assert max(quotas.values()) - min(quotas[index] for index in (1, 2, 3)) <= 1
    assert all(quotas[index] <= capacities[index] for index in capacities)


def test_position_selection_is_deterministic_and_without_replacement() -> None:
    capacities = {0: 8, 1: 9}
    quotas = {0: 3, 1: 4}
    first = deterministic_positions_for_quotas(
        capacities, quotas, seed=11, layer_index=17, split_code=0
    )
    second = deterministic_positions_for_quotas(
        capacities, quotas, seed=11, layer_index=17, split_code=0
    )
    for trace_index in capacities:
        assert np.array_equal(first[trace_index], second[trace_index])
        assert len(np.unique(first[trace_index])) == quotas[trace_index]
        assert np.all(first[trace_index] < capacities[trace_index])
