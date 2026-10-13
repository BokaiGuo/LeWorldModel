import math

import torch

from experiments.empirical_spectral_audit.run_action_sequence_composition_synthetic import (
    COMPOSITION_HORIZONS,
    _sequence_composition_error,
    _sequence_product,
    _correlations,
)


def test_sequence_product_uses_time_order_for_column_operators():
    operators = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )
    sequence = (0, 1)
    expected = operators[1] @ operators[0]
    assert torch.equal(_sequence_product(operators, sequence), expected)


def test_exact_products_are_zero_for_a_perfect_model():
    operators = torch.stack(
        [torch.eye(4), 0.5 * torch.eye(4)]
    )
    for horizon in COMPOSITION_HORIZONS:
        result = _sequence_composition_error(operators, operators, horizon)
        assert result["n_sequences"] == float(2**horizon)
        assert result["mean"] == 0.0
        assert result["p90"] == 0.0
        assert result["max"] == 0.0


def test_sequence_error_is_finite_and_nonzero_for_a_perturbed_operator():
    true = torch.stack([torch.eye(4), 0.5 * torch.eye(4)])
    learned = true.clone()
    learned[1, 0, 0] += 0.1
    result = _sequence_composition_error(learned, true, 4)
    assert result["n_sequences"] == 16.0
    assert result["mean"] > 0.0
    assert all(math.isfinite(value) for value in result.values())


def test_correlation_report_has_pooled_and_per_seed_blocks():
    records = []
    for seed in (0, 1):
        for condition in (0, 1, 2):
            records.append(
                {
                    "mode": ("none", "global", "conditional")[condition],
                    "seed": seed,
                    "sequence_composition_error": {
                        str(horizon): {"mean": float(condition + horizon)}
                        for horizon in COMPOSITION_HORIZONS
                    },
                    "random_rollout_mse": {
                        split: {str(horizon): float(condition + 2 * horizon)
                                for horizon in COMPOSITION_HORIZONS}
                        for split in ("id", "ood")
                    },
                }
            )
    result = _correlations(records)
    assert "composition_h8_vs_rollout" in result
    assert set(result["per_seed"]) == {"0", "1"}
    assert set(result["per_mode"]) == {"none", "global", "conditional"}
    assert "log10_pooled" in result
