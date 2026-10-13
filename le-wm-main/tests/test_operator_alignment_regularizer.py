import math

import torch

from experiments.empirical_spectral_audit.regularizers import (
    equivariance_diagnostics,
    latent_equivariance_loss,
    operator_alignment_diagnostic,
    operator_alignment_loss,
    operator_alignment_route_loss,
    predictor_equivariance_loss,
)


def _rotated_operator(angle: float, eigenvalues: tuple[float, float]) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    spectrum = torch.diag(torch.tensor(eigenvalues, dtype=torch.float32))
    return rotation @ spectrum @ rotation.T


def test_global_alignment_is_small_for_commuting_operator_and_large_for_rotated_operator():
    generator = torch.Generator().manual_seed(0)
    source = torch.randn(1024, 2, generator=generator)
    source[:, 0] *= 4.0
    diagonal = torch.diag(torch.tensor([0.8, 0.2]))
    rotated = _rotated_operator(math.pi / 4.0, (0.8, 0.2))

    commuting = operator_alignment_loss(source, source @ diagonal.T, mode="global", min_samples=32)
    noncommuting = operator_alignment_loss(source, source @ rotated.T, mode="global", min_samples=32)

    assert float(commuting) < 1e-5
    assert float(noncommuting) > 1e-3


def test_conditional_alignment_averages_action_local_groups():
    generator = torch.Generator().manual_seed(1)
    source = torch.randn(512, 2, generator=generator)
    source[:, 0] *= 4.0
    actions = torch.cat([torch.zeros(256, 1), torch.ones(256, 1)], dim=0)
    diagonal = torch.diag(torch.tensor([0.8, 0.2]))
    predicted = source @ diagonal.T

    value = operator_alignment_loss(
        source,
        predicted,
        actions=actions,
        mode="conditional",
        num_bins=2,
        min_samples=32,
    )

    assert float(value) < 1e-3
    diagnostic = operator_alignment_diagnostic(
        source, predicted, actions=actions, mode="conditional", num_bins=2, min_samples=32
    )
    assert diagnostic["operator_alignment_samples"] == 512


def test_direction_conditioning_has_non_degenerate_direction_groups():
    diagonal = 2.0 ** -0.5
    actions = torch.tensor(
        [[-diagonal, -diagonal], [diagonal, -diagonal],
         [diagonal, diagonal], [-diagonal, diagonal]]
    ).repeat(32, 1)
    source = torch.randn(128, 3, generator=torch.Generator().manual_seed(21))
    predicted = source @ torch.diag(torch.tensor([0.8, 0.5, 0.2])).T
    diagnostic = operator_alignment_diagnostic(
        source,
        predicted,
        actions=actions,
        mode="conditional",
        num_bins=4,
        min_samples=16,
        conditioning="direction",
    )
    assert diagnostic["operator_alignment_group_count"] == 4
    assert diagnostic["operator_alignment_min_group"] == 32


def test_local_conditioning_reports_finite_covered_groups():
    source = torch.randn(128, 4, generator=torch.Generator().manual_seed(22))
    predicted = source @ torch.diag(torch.tensor([0.8, 0.6, 0.4, 0.2])).T
    actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]).repeat(32, 1)
    diagnostic = operator_alignment_diagnostic(
        source,
        predicted,
        actions=actions,
        mode="conditional",
        num_bins=2,
        min_samples=8,
        conditioning="local",
    )
    assert diagnostic["operator_alignment_group_count"] >= 4
    assert diagnostic["operator_alignment_min_group"] >= 8
    assert math.isfinite(float(diagnostic["operator_alignment"]))


def test_alignment_loss_backpropagates_through_predictions_not_covariance():
    generator = torch.Generator().manual_seed(2)
    source = torch.randn(128, 2, generator=generator)
    source[:, 0] *= 3.0
    predicted = (source @ _rotated_operator(0.4, (0.9, 0.5)).T).requires_grad_()

    loss = operator_alignment_loss(source, predicted, mode="global", min_samples=16)
    loss.backward()

    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_alignment_mode_none_has_zero_loss_without_touching_shapes():
    source = torch.randn(8, 4)
    predicted = torch.randn(8, 4)
    value = operator_alignment_loss(source, predicted, mode="none")
    assert value.shape == torch.Size([])
    assert float(value) == 0.0


def test_alignment_gradient_routes_isolate_encoder_and_predictor_paths():
    generator = torch.Generator().manual_seed(11)
    raw_source = torch.randn(128, 3, generator=generator)
    encoder = torch.nn.Linear(3, 3, bias=False)
    predictor = torch.nn.Linear(3, 3, bias=False)
    source = encoder(raw_source)
    prediction = predictor(source)
    detached_prediction = predictor(source.detach())

    predictor_loss = operator_alignment_route_loss(
        source.unsqueeze(0),
        prediction.unsqueeze(0),
        mode="global",
        min_samples=16,
        gradient_route="predictor",
        predictor_detached_prediction=detached_prediction.unsqueeze(0),
    )
    predictor_grads = torch.autograd.grad(
        predictor_loss, (encoder.weight, predictor.weight), allow_unused=True
    )
    assert predictor_grads[0] is None
    assert predictor_grads[1] is not None
    assert torch.isfinite(predictor_grads[1]).all()

    encoder_loss = operator_alignment_route_loss(
        source.unsqueeze(0),
        prediction.unsqueeze(0),
        mode="global",
        min_samples=16,
        gradient_route="encoder",
    )
    encoder_grads = torch.autograd.grad(
        encoder_loss, (encoder.weight, predictor.weight), allow_unused=True
    )
    assert encoder_grads[0] is not None
    assert torch.isfinite(encoder_grads[0]).all()
    assert encoder_grads[1] is None


def test_joint_route_preserves_legacy_alignment_value_and_none_route_is_exact_zero():
    generator = torch.Generator().manual_seed(12)
    source = torch.randn(128, 3, generator=generator)
    predicted = torch.randn(128, 3, generator=generator)
    legacy = operator_alignment_loss(source, predicted, mode="global", min_samples=16)
    joint = operator_alignment_route_loss(
        source, predicted, mode="global", min_samples=16, gradient_route="joint"
    )
    assert torch.allclose(legacy, joint)

    none = operator_alignment_route_loss(
        source, predicted, mode="global", min_samples=16, gradient_route="none"
    )
    assert none.shape == torch.Size([])
    assert float(none) == 0.0


def test_latent_equivariance_loss_is_zero_for_a_known_representation_action():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    embeddings = torch.randn(32, 2, generator=torch.Generator().manual_seed(3))
    transformed = embeddings @ rotation.T

    loss = latent_equivariance_loss(embeddings, transformed, rotation)

    assert float(loss) < 1e-10


def test_predictor_equivariance_helper_distinguishes_equivariant_predictor():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    generator = torch.Generator().manual_seed(4)
    source = torch.randn(64, 2, generator=generator)
    actions = torch.randn(64, 2, generator=generator)
    transformed_source = source @ rotation.T
    transformed_actions = actions @ rotation.T

    def equivariant_predictor(state, action):
        return state + action

    def anisotropic_predictor(state, action):
        return (state + action) @ torch.diag(torch.tensor([2.0, 0.25])).T

    good = predictor_equivariance_loss(
        equivariant_predictor,
        source,
        actions,
        transformed_source,
        transformed_actions,
        rotation,
    )
    bad = predictor_equivariance_loss(
        anisotropic_predictor,
        source,
        actions,
        transformed_source,
        transformed_actions,
        rotation,
    )
    diagnostics = equivariance_diagnostics(
        anisotropic_predictor,
        source,
        actions,
        transformed_source,
        transformed_actions,
        rotation,
    )

    assert float(good) < 1e-10
    assert float(bad) > 1e-3
    assert diagnostics["predictor_equivariance_error"] > 1e-2
