import torch

from experiments.empirical_spectral_audit.run_predictor_equivariance_synthetic import (
    ACTION_TRANSFORM,
    LATENT_TRANSFORM,
    EquivarianceBenchmarkConfig,
    _dynamics_action,
    _transform,
    _validate_protocol,
)


def test_synthetic_equivariance_protocol_has_consistent_orthogonal_transforms():
    config = EquivarianceBenchmarkConfig()
    _validate_protocol(config)

    assert torch.allclose(
        ACTION_TRANSFORM.T @ ACTION_TRANSFORM,
        torch.eye(config.state_dim),
        atol=1e-6,
    )


def test_broken_control_dynamics_does_not_commute_with_rotation():
    config = EquivarianceBenchmarkConfig(symmetry_broken=True)
    _validate_protocol(config)
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    transformed_then_mapped = _dynamics_action(_transform(action, ACTION_TRANSFORM), config)
    mapped_then_transformed = _transform(_dynamics_action(action, config), ACTION_TRANSFORM)
    assert not torch.allclose(transformed_then_mapped, mapped_then_transformed, atol=1e-6)
    assert torch.allclose(
        LATENT_TRANSFORM.T @ LATENT_TRANSFORM,
        torch.eye(config.latent_dim),
        atol=1e-6,
    )

    state = torch.randn(11, config.state_dim, generator=torch.Generator().manual_seed(8))
    action = torch.randn(11, config.state_dim, generator=torch.Generator().manual_seed(9))
    assert torch.allclose(
        _transform(state + action, ACTION_TRANSFORM),
        _transform(state, ACTION_TRANSFORM) + _transform(action, ACTION_TRANSFORM),
        atol=1e-6,
    )
