import copy

import torch

from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import (
    TrainingConfig,
    _fit_condition,
    _sequence_split,
)
from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
    NonlinearWorldModel,
    _seed_everything,
)


def test_sequence_split_is_deterministic_and_has_clean_targets():
    first = _sequence_split(5, 4, torch.Generator().manual_seed(31), 0.02)
    second = _sequence_split(5, 4, torch.Generator().manual_seed(31), 0.02)
    for key in first:
        assert torch.equal(first[key], second[key])
    assert first["targets"].shape == (5, 4, 4)
    assert first["clean_targets"].shape == (5, 4, 4)


def test_multihorizon_fit_returns_finite_checkpoint():
    _seed_everything(37)
    config = TrainingConfig(
        train_sequences=32,
        validation_sequences=16,
        test_sequences=16,
        sequence_horizon=4,
        epochs=3,
    )
    generator = torch.Generator().manual_seed(37001)
    train = _sequence_split(config.train_sequences, config.sequence_horizon, generator, config.noise_std)
    validation = _sequence_split(config.validation_sequences, config.sequence_horizon, generator, 0.0)
    initial = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    for condition in ("one_step", "multi_horizon"):
        model, epoch, score, metrics = _fit_condition(
            condition,
            copy.deepcopy(initial.state_dict()),
            train,
            validation,
            config,
        )
        assert epoch > 0
        assert score >= 0.0
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert all(value >= 0.0 for value in metrics.values())
