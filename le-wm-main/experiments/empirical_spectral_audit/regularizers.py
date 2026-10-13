"""Training-time spectral regularizer helpers for LeWM.

The empirical audit code is offline.  These helpers expose the same directional
ideas as opt-in losses for train_lewm_directional.py without making them depend
on audit JSON/report code.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch


def _apply_representation_transform(
    embeddings: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Apply a representation transform supplied as a callable or matrix."""

    if callable(transform):
        return transform(embeddings)
    matrix = transform.to(device=embeddings.device, dtype=embeddings.dtype)
    return embeddings @ matrix.T


def latent_equivariance_loss(
    embeddings: torch.Tensor,
    transformed_embeddings: torch.Tensor,
    representation_transform: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Match a transformed observation embedding to its prescribed action.

    ``representation_transform`` is intentionally explicit.  It may be a known
    group representation (a matrix) or a differentiable callable.  The helper
    does not normalize or stop gradients, so callers can decide whether the
    encoder or a separate representation head receives the intervention.
    """

    if embeddings.shape != transformed_embeddings.shape:
        raise ValueError(
            "embeddings and transformed_embeddings must have identical shapes"
        )
    expected = _apply_representation_transform(embeddings, representation_transform)
    return (transformed_embeddings - expected).square().mean()


def predictor_equivariance_loss(
    predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source_embeddings: torch.Tensor,
    actions: torch.Tensor,
    transformed_source_embeddings: torch.Tensor,
    transformed_actions: torch.Tensor,
    representation_transform: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Enforce equivariance of an action-conditioned predictor.

    The loss is

    ``||P(rho(z), rho_a(a)) - rho(P(z, a))||^2``

    when ``transformed_source_embeddings`` already equals ``rho(z)``.  Keeping
    the transformed embedding as an input makes the helper usable with a
    learned encoder and prevents it from silently replacing predictor
    equivariance with a latent-only penalty.
    """

    if source_embeddings.shape != transformed_source_embeddings.shape:
        raise ValueError(
            "source_embeddings and transformed_source_embeddings must have identical shapes"
        )
    if actions.shape != transformed_actions.shape:
        raise ValueError("actions and transformed_actions must have identical shapes")
    prediction = predictor(source_embeddings, actions)
    transformed_prediction = predictor(transformed_source_embeddings, transformed_actions)
    expected = _apply_representation_transform(prediction, representation_transform)
    return (transformed_prediction - expected).square().mean()


@torch.no_grad()
def equivariance_diagnostics(
    predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source_embeddings: torch.Tensor,
    actions: torch.Tensor,
    transformed_source_embeddings: torch.Tensor,
    transformed_actions: torch.Tensor,
    representation_transform: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
) -> dict[str, float]:
    """Return normalized latent and predictor equivariance errors."""

    latent_expected = _apply_representation_transform(source_embeddings, representation_transform)
    prediction = predictor(source_embeddings, actions)
    transformed_prediction = predictor(transformed_source_embeddings, transformed_actions)
    prediction_expected = _apply_representation_transform(prediction, representation_transform)
    latent_scale = latent_expected.square().mean().sqrt().clamp_min(1e-8)
    prediction_scale = prediction_expected.square().mean().sqrt().clamp_min(1e-8)
    return {
        "latent_equivariance_error": float(
            (transformed_source_embeddings - latent_expected).square().mean().sqrt()
            / latent_scale
        ),
        "predictor_equivariance_error": float(
            (transformed_prediction - prediction_expected).square().mean().sqrt()
            / prediction_scale
        ),
    }


def covariance_eigenspace(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x.detach().reshape(-1, x.size(-1))
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(x.size(0) - 1, 1)
    vals, vecs = torch.linalg.eigh(cov)
    order = torch.argsort(vals, descending=True)
    return vecs[:, order[: min(int(k), vecs.size(1))]].detach()


def band_fraction(tangents: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    tangents = tangents.reshape(-1, tangents.size(-1))
    top = (tangents @ basis).square().sum(dim=1).mean()
    total = tangents.square().sum(dim=1).mean().clamp_min(1e-12)
    return top / total


def spectral_prediction_loss(pred_emb: torch.Tensor, target_emb: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return band_fraction(pred_emb - target_emb, basis)


def residual_band_regularization(residual: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return band_fraction(residual, basis)


def _normalized_commutator(covariance: torch.Tensor, operator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a scale-normalized Frobenius commutator penalty."""

    commutator = covariance @ operator - operator @ covariance
    denominator = covariance.square().sum() * operator.square().sum() + eps
    return commutator.square().sum() / denominator


def _quantile_labels(values: torch.Tensor, bins: int) -> torch.Tensor:
    """Assign detached scalar values to approximately balanced quantile bins."""

    if int(bins) <= 1 or values.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.long, device=values.device)
    quantiles = torch.linspace(
        0.0,
        1.0,
        int(bins) + 1,
        device=values.device,
        dtype=values.dtype,
    )[1:-1]
    thresholds = torch.quantile(values.detach(), quantiles) if quantiles.numel() else values.new_empty(0)
    return torch.bucketize(values.detach(), thresholds)


def _direction_labels(actions: torch.Tensor, bins: int) -> torch.Tensor:
    """Return detached action-direction labels without assuming a group action.

    For the common two-dimensional control interface, directions are angular
    bins.  For higher-dimensional actions we use the signed dominant coordinate
    as a deterministic, architecture-independent direction code.  Zero
    actions form a separate stable label.  This is a diagnostic grouping, not
    an equivariance assumption.
    """

    actions = torch.nan_to_num(actions.detach().float(), 0.0)
    if actions.size(-1) == 1:
        values = actions[:, 0]
        labels = (values >= 0).long()
        labels[values.abs() < 1e-8] = 2
        return labels
    norms = actions.norm(dim=-1)
    if actions.size(-1) == 2:
        angles = torch.atan2(actions[:, 1], actions[:, 0])
        labels = torch.floor(
            (angles + math.pi) / (2.0 * math.pi) * max(int(bins), 1)
        ).long()
        labels = labels.clamp_min(0).clamp_max(max(int(bins), 1) - 1)
        labels[norms < 1e-8] = max(int(bins), 1)
        return labels
    dominant = actions.abs().argmax(dim=-1)
    sign = (actions.gather(1, dominant[:, None]).squeeze(1) >= 0).long()
    labels = 2 * dominant + sign
    labels[norms < 1e-8] = 2 * actions.size(-1)
    return labels


def _alignment_group_masks(
    source_flat: torch.Tensor,
    action_flat: torch.Tensor | None,
    conditioning: str,
    num_bins: int,
) -> list[torch.Tensor]:
    """Build detached masks for the training/diagnostic operator groups."""

    valid = {"magnitude", "direction", "local"}
    if conditioning not in valid:
        raise ValueError(
            f"unknown alignment conditioning: {conditioning}; expected one of {sorted(valid)}"
        )
    n = source_flat.size(0)
    if action_flat is None or action_flat.size(0) != n:
        return [torch.ones(n, dtype=torch.bool, device=source_flat.device)]
    action_flat = torch.nan_to_num(action_flat.detach().float(), 0.0)
    if conditioning == "magnitude":
        labels = _quantile_labels(action_flat.norm(dim=-1), num_bins)
    elif conditioning == "direction":
        labels = _direction_labels(action_flat, num_bins)
    else:
        # Local groups combine a state-location proxy and action direction.
        # Both labels are detached: the intervention cannot optimize the
        # partition itself.  Quantile-binning source norm is deliberately
        # modest, keeping coverage auditable in small batches.
        state_labels = _quantile_labels(source_flat.detach().norm(dim=-1), max(int(num_bins), 2))
        direction_labels = _direction_labels(action_flat, max(int(num_bins), 2))
        direction_cardinality = int(direction_labels.max().detach().cpu()) + 1 if direction_labels.numel() else 1
        labels = state_labels * max(direction_cardinality, 1) + direction_labels
    unique = torch.unique(labels, sorted=True)
    return [labels == label for label in unique]


def _ridge_transition_operator(
    source: torch.Tensor,
    predicted: torch.Tensor,
    ridge: float,
    detach_covariance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a row-wise linear transition and return ``(covariance, operator)``.

    ``source`` has rows of latent states and ``predicted`` has the corresponding
    predicted next states.  The returned operator follows the column-vector
    convention, so the row-wise fit is ``predicted ~= source @ operator.T``.
    By default the covariance and source features are detached: the regularizer
    then trains the predictor output to respect the current latent geometry
    without rewarding a collapse of the encoder covariance.
    """

    source = source.float()
    predicted = predicted.float()
    source_centered = source - source.mean(dim=0, keepdim=True)
    predicted_centered = predicted - predicted.mean(dim=0, keepdim=True)
    if detach_covariance:
        source_centered = source_centered.detach()
    denom = max(source_centered.size(0) - 1, 1)
    covariance = source_centered.T @ source_centered / denom
    cross_covariance = source_centered.T @ predicted_centered / denom
    ridge_scale = covariance.trace().detach() / max(covariance.size(0), 1)
    regularized = covariance + float(ridge) * ridge_scale.clamp_min(1e-6) * torch.eye(
        covariance.size(0), device=covariance.device, dtype=covariance.dtype
    )
    row_operator = torch.linalg.solve(regularized, cross_covariance)
    return covariance, row_operator.T


def operator_alignment_loss(
    source_emb: torch.Tensor,
    predicted_emb: torch.Tensor,
    actions: torch.Tensor | None = None,
    mode: str = "conditional",
    num_bins: int = 2,
    min_samples: int = 16,
    ridge: float = 1e-3,
    detach_covariance: bool = True,
    conditioning: str = "magnitude",
) -> torch.Tensor:
    """Penalize covariance--predictor non-commutativity during training.

    This is an opt-in training intervention, not a claim that commutativity is
    universally optimal.  ``global`` fits one operator over all transitions;
    ``conditional`` partitions transitions according to ``conditioning``:
    ``magnitude`` preserves the historical action-norm bins, ``direction``
    uses action-direction bins, and ``local`` combines detached source-state
    and action-direction bins.  The partition is used only to define local
    groups, while gradients flow through ``predicted_emb``.
    """

    if mode in (None, "none", "off"):
        return predicted_emb.new_zeros(())
    if mode not in {"global", "conditional"}:
        raise ValueError(f"unknown operator-alignment mode: {mode}")

    source = source_emb[..., : predicted_emb.size(-2), :]
    predicted = predicted_emb
    source_flat = source.reshape(-1, source.size(-1))
    predicted_flat = predicted.reshape(-1, predicted.size(-1))
    if mode == "global":
        groups = [torch.ones(source_flat.size(0), dtype=torch.bool, device=source_flat.device)]
    else:
        action_flat = None
        if actions is not None:
            action_flat = torch.nan_to_num(
                actions[..., : predicted_emb.size(-2), :].float(), 0.0
            ).reshape(source_flat.size(0), -1)
        groups = _alignment_group_masks(
            source_flat, action_flat, conditioning=conditioning, num_bins=num_bins
        )

    penalties = []
    for mask in groups:
        if int(mask.sum()) < int(min_samples):
            continue
        covariance, operator = _ridge_transition_operator(
            source_flat[mask], predicted_flat[mask], ridge=ridge, detach_covariance=detach_covariance
        )
        penalties.append(_normalized_commutator(covariance, operator))
    if not penalties:
        return predicted_emb.new_zeros(())
    return torch.stack(penalties).mean().to(dtype=predicted_emb.dtype)


def operator_alignment_route_loss(
    source_emb: torch.Tensor,
    predicted_emb: torch.Tensor,
    actions: torch.Tensor | None = None,
    mode: str = "conditional",
    num_bins: int = 2,
    min_samples: int = 16,
    ridge: float = 1e-3,
    gradient_route: str = "joint",
    predictor_detached_prediction: torch.Tensor | None = None,
    conditioning: str = "magnitude",
) -> torch.Tensor:
    """Apply operator alignment through an explicitly auditable gradient route.

    ``joint`` preserves the historical training behavior: covariance features
    are detached, while the prediction branch remains connected to both the
    predictor and the encoder through the predictor input.  ``predictor``
    requires ``predictor_detached_prediction`` to have been computed from a
    detached source embedding; it therefore cannot send alignment gradients
    into the encoder.  ``encoder`` detaches the predictor output and keeps the
    covariance differentiable, so alignment gradients enter only the encoder.
    ``none`` returns an exact scalar zero.

    The separate detached-source prediction is intentional.  Merely detaching
    ``predicted_emb`` is not sufficient for a predictor-only intervention,
    because the predictor output may still retain a path through the encoder.
    """

    if gradient_route not in {"none", "predictor", "encoder", "joint"}:
        raise ValueError(f"unknown alignment gradient route: {gradient_route}")
    if gradient_route == "none":
        return predicted_emb.new_zeros(())
    if gradient_route == "predictor":
        if predictor_detached_prediction is None:
            raise ValueError(
                "predictor route requires a prediction computed from detached source embeddings"
            )
        return operator_alignment_loss(
            source_emb=source_emb.detach(),
            predicted_emb=predictor_detached_prediction,
            actions=actions,
            mode=mode,
            num_bins=num_bins,
            min_samples=min_samples,
            ridge=ridge,
            detach_covariance=True,
            conditioning=conditioning,
        )
    if gradient_route == "encoder":
        return operator_alignment_loss(
            source_emb=source_emb,
            predicted_emb=predicted_emb.detach(),
            actions=actions,
            mode=mode,
            num_bins=num_bins,
            min_samples=min_samples,
            ridge=ridge,
            detach_covariance=False,
            conditioning=conditioning,
        )
    return operator_alignment_loss(
        source_emb=source_emb,
        predicted_emb=predicted_emb,
        actions=actions,
        mode=mode,
        num_bins=num_bins,
        min_samples=min_samples,
        ridge=ridge,
        detach_covariance=True,
        conditioning=conditioning,
    )


@torch.no_grad()
def operator_alignment_diagnostic(
    source_emb: torch.Tensor,
    predicted_emb: torch.Tensor,
    actions: torch.Tensor | None = None,
    mode: str = "conditional",
    num_bins: int = 2,
    min_samples: int = 16,
    ridge: float = 1e-3,
    conditioning: str = "magnitude",
) -> dict[str, float | int]:
    """Report detached alignment values using the same grouping as training."""

    value = operator_alignment_loss(
        source_emb=source_emb,
        predicted_emb=predicted_emb,
        actions=actions,
        mode=mode,
        num_bins=num_bins,
        min_samples=min_samples,
        ridge=ridge,
        detach_covariance=True,
        conditioning=conditioning,
    )
    source = source_emb[..., : predicted_emb.size(-2), :].reshape(-1, source_emb.size(-1))
    action_flat = None
    if actions is not None:
        action_flat = torch.nan_to_num(
            actions[..., : predicted_emb.size(-2), :].float(), 0.0
        ).reshape(source.size(0), -1)
    groups = _alignment_group_masks(source, action_flat, conditioning, num_bins)
    sizes = [int(mask.sum()) for mask in groups]
    return {
        "operator_alignment": float(value.detach().cpu()),
        "operator_alignment_samples": int(predicted_emb.numel() // predicted_emb.size(-1)),
        "operator_alignment_group_count": int(len(groups)),
        "operator_alignment_min_group": int(min(sizes) if sizes else 0),
    }


def action_controllability_alignment_loss(
    dynamics: torch.Tensor,
    actions: torch.Tensor | None,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Align the action-predictable component of latent dynamics to top bands."""

    if actions is None:
        return dynamics.new_zeros(())
    dyn = dynamics.reshape(-1, dynamics.size(-1))
    act = torch.nan_to_num(actions, 0.0).reshape(dyn.size(0), -1)
    if act.size(0) != dyn.size(0) or act.size(0) < 2:
        return dynamics.new_zeros(())
    act = act - act.mean(dim=0, keepdim=True)
    dyn_centered = dyn - dyn.mean(dim=0, keepdim=True)
    coef = torch.linalg.pinv(act) @ dyn_centered
    controllable = act @ coef
    return -band_fraction(controllable, basis)


def teacher_anchor_loss(emb: torch.Tensor, teacher_emb: torch.Tensor | None) -> torch.Tensor:
    if teacher_emb is None:
        return emb.new_zeros(())
    if teacher_emb.shape != emb.shape:
        return emb.new_zeros(())
    return (emb - teacher_emb.detach()).square().mean()


def contact_event_spectral_auxiliary(
    dynamics: torch.Tensor,
    residual: torch.Tensor,
    basis: torch.Tensor,
    contact: torch.Tensor | None,
) -> torch.Tensor:
    if contact is None:
        return dynamics.new_zeros(())
    mask = contact.reshape(-1).bool()
    flat_dyn = dynamics.reshape(-1, dynamics.size(-1))
    flat_res = residual.reshape(-1, residual.size(-1))
    if mask.numel() != flat_dyn.size(0) or not bool(mask.any()):
        return dynamics.new_zeros(())
    contact_dyn = flat_dyn[mask]
    contact_res = flat_res[mask]
    return -band_fraction(contact_dyn, basis) + residual_band_regularization(contact_res, basis)


def nuisance_suppression_loss(
    batch: dict,
    emb: torch.Tensor,
    basis: torch.Tensor,
    encode_pixels: Callable[[dict], torch.Tensor],
    shift: int = 2,
) -> torch.Tensor:
    if "pixels" not in batch:
        return emb.new_zeros(())
    nuisance_batch = dict(batch)
    nuisance_batch["pixels"] = torch.roll(batch["pixels"], shifts=int(shift), dims=-1)
    nuisance_emb = encode_pixels(nuisance_batch)
    return band_fraction(nuisance_emb - emb.detach(), basis)


def directional_regularizer_losses(
    emb: torch.Tensor,
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    batch: dict,
    top_k: int,
    encode_pixels: Callable[[dict], torch.Tensor] | None = None,
    nuisance_enabled: bool = False,
    alignment_mode: str = "none",
    alignment_bins: int = 2,
    alignment_min_samples: int = 16,
    alignment_ridge: float = 1e-3,
    alignment_conditioning: str = "magnitude",
) -> dict[str, torch.Tensor]:
    basis = covariance_eigenspace(emb, top_k)
    dyn_tangent = emb[:, 1:] - emb[:, :-1]
    reach_tangent = emb[:, -1:] - emb[:, :-1]
    pred_residual = pred_emb - target_emb
    actions = batch.get("action") if isinstance(batch, dict) else None
    teacher_emb = batch.get("teacher_emb") if isinstance(batch, dict) else None
    actions_for_alignment = batch.get("action") if isinstance(batch, dict) else None
    losses = {
        "action_tangent_align_loss": -band_fraction(dyn_tangent, basis),
        "action_controllability_align_loss": action_controllability_alignment_loss(dyn_tangent, actions[:, :-1] if torch.is_tensor(actions) else None, basis),
        "reach_align_loss": -band_fraction(reach_tangent, basis),
        "spectral_prediction_loss": spectral_prediction_loss(pred_emb, target_emb, basis),
        "residual_band_regularization_loss": residual_band_regularization(pred_residual, basis),
        "teacher_anchor_loss": teacher_anchor_loss(emb, teacher_emb),
        "operator_alignment_loss": operator_alignment_loss(
            source_emb=emb[:, :-1],
            predicted_emb=pred_emb,
            actions=actions_for_alignment,
            mode=alignment_mode,
            num_bins=alignment_bins,
            min_samples=alignment_min_samples,
            ridge=alignment_ridge,
            conditioning=alignment_conditioning,
        ),
        "contact_event_spectral_auxiliary_loss": contact_event_spectral_auxiliary(
            dyn_tangent,
            pred_residual,
            basis,
            batch.get("contact") if isinstance(batch, dict) else None,
        ),
    }
    if nuisance_enabled and encode_pixels is not None:
        losses["nuisance_deemphasis_loss"] = nuisance_suppression_loss(batch, emb, basis, encode_pixels)
    else:
        losses["nuisance_deemphasis_loss"] = emb.new_zeros(())
    return losses
