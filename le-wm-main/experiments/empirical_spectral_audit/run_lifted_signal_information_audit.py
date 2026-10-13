"""Causal signal/information diagnostics for action-to-task sufficient lifting.

The module implements finite-history delay and short-time Fourier features
without future-frame access, plus a Gaussian plug-in conditional mutual
information estimate and cross-fitted transfer sufficiency.  These are
diagnostics for the hypothesis

    Delta y independent of hidden state | (lifted history, action),

not proofs of causal identification.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_action_lifted_system_audit import _matrix, _ridge_solve
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_action_lifted_system_audit import _matrix, _ridge_solve


def causal_history_indices(
    episode_id: np.ndarray,
    frame: np.ndarray,
    *,
    history_length: int,
) -> list[list[int]]:
    """Return chronological within-episode histories ending at each valid row."""

    episode_id = np.asarray(episode_id).reshape(-1)
    frame = np.asarray(frame).reshape(-1)
    if len(episode_id) != len(frame) or history_length < 1:
        raise ValueError("episode_id, frame and history_length are incompatible")
    histories: list[list[int]] = []
    by_episode: dict[Any, list[int]] = {}
    for index, episode in enumerate(episode_id.tolist()):
        by_episode.setdefault(episode, []).append(index)
    for episode in sorted(by_episode, key=str):
        indices = sorted(by_episode[episode], key=lambda index: (int(frame[index]), index))
        for position in range(history_length - 1, len(indices)):
            window = indices[position - history_length + 1 : position + 1]
            if len({int(frame[index]) for index in window}) != len(window):
                continue
            histories.append(window)
    return histories


def causal_delay_time_frequency_features(
    signal: np.ndarray,
    action: np.ndarray,
    episode_id: np.ndarray,
    frame: np.ndarray,
    *,
    history_length: int = 4,
) -> dict[str, Any]:
    """Build base, delay, interaction, and causal short-time spectral features."""

    signal = _matrix(signal, "signal")
    action = _matrix(action, "action")
    episode_id = np.asarray(episode_id).reshape(-1)
    frame = np.asarray(frame).reshape(-1)
    if len(signal) != len(action) or len(signal) != len(episode_id) or len(signal) != len(frame):
        raise ValueError("signal, action, episode_id and frame must align")
    histories = causal_history_indices(episode_id, frame, history_length=history_length)
    if not histories:
        raise ValueError("no causal histories are available")
    base = []
    delay = []
    interaction = []
    time_frequency = []
    current_indices = []
    for window in histories:
        values = signal[window]
        actions = action[window]
        current = values[-1]
        current_action = actions[-1]
        base.append(np.concatenate((current, current_action)))
        delay.append(np.concatenate((values.reshape(-1), actions.reshape(-1))))
        # The action block is a short sequence of 2-D controls flattened into
        # ten coordinates.  A full state x flattened-block product creates
        # hundreds of nearly collinear interaction coordinates for only a few
        # independent episodes.  Retain the complete action interface, but
        # use its block-mean direction for the physical bilinear term.  This
        # is a declared low-dimensional action summary, not a test-set fit.
        action_summary = np.mean(actions, axis=0)
        interaction.append(np.concatenate((
            current,
            current_action,
            (current[:, None] * action_summary[None, :]).reshape(-1),
        )))
        # The spectral lift must retain the action interface used by the
        # baseline.  The previous implementation supplied only an FFT of the
        # latent history, while ``base`` supplied current state + current
        # action.  That was an invalid comparison for action-sensitive tasks:
        # the frequency branch could not represent which action caused the
        # effect.  Keep the current state/action and append the causal
        # frequency content of the history residual relative to the current
        # state.  This is still a finite, past-only transform, but now it is a
        # genuine action-conditioned lift of the same decision interface.
        residual_history = values - current[None, :]
        residual_actions = actions - current_action[None, :]
        spectrum = np.fft.rfft(residual_history, axis=0)
        action_spectrum = np.fft.rfft(residual_actions, axis=0)
        time_frequency.append(np.concatenate((
            current,
            current_action,
            spectrum.real.reshape(-1),
            spectrum.imag.reshape(-1),
            action_spectrum.real.reshape(-1),
            action_spectrum.imag.reshape(-1),
        )))
        current_indices.append(window[-1])
    return {
        "indices": np.asarray(current_indices, dtype=np.int64),
        "base": np.asarray(base, dtype=np.float64),
        "delay": np.asarray(delay, dtype=np.float64),
        "interaction": np.asarray(interaction, dtype=np.float64),
        "time_frequency": np.asarray(time_frequency, dtype=np.float64),
        # A declared multi-view lift: causal delay coordinates plus the
        # action-conditioned spectral residual.  It is kept separate from
        # the interaction-only diagnostic because the latter is not stable in
        # the small PushT validation set.
        "delay_spectral": np.concatenate((
            np.asarray(delay, dtype=np.float64),
            np.asarray(time_frequency, dtype=np.float64),
        ), axis=1),
        "history_length": int(history_length),
        "causal_window": True,
    }


def adaptive_time_frequency_features(
    signal: np.ndarray,
    action: np.ndarray,
    episode_id: np.ndarray,
    frame: np.ndarray,
    *,
    history_length: int = 4,
) -> dict[str, Any]:
    """Build causal local/event-aware lifts for a short history window.

    The fixed Fourier branch is intentionally retained as a historical
    baseline.  It represents a window by one residual harmonic, which can
    erase endpoint jumps, phase direction, and contact-like transients.  This
    function exposes those quantities explicitly while keeping the current
    state/action interface in every variant.  No thresholds, statistics, or
    projections are estimated from outcome rows.

    The returned variants are deliberately redundant; downstream evaluation
    must apply the same fit-only feature-rank budget to every variant.  The
    feature families are diagnostics for finite-history action-to-task
    transfer, not evidence of causal identification.
    """

    signal = _matrix(signal, "signal")
    action = _matrix(action, "action")
    episode_id = np.asarray(episode_id).reshape(-1)
    frame = np.asarray(frame).reshape(-1)
    if len(signal) != len(action) or len(signal) != len(episode_id) or len(signal) != len(frame):
        raise ValueError("signal, action, episode_id and frame must align")
    if history_length < 3:
        raise ValueError("adaptive local features require history_length >= 3")

    histories = causal_history_indices(episode_id, frame, history_length=history_length)
    if not histories:
        raise ValueError("no causal histories are available")

    base_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    legacy_rows: list[np.ndarray] = []
    difference_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    compact_event_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    current_indices: list[int] = []
    event_scores: list[float] = []

    for window in histories:
        values = signal[window]
        actions = action[window]
        current = values[-1]
        previous = values[-2]
        previous_previous = values[-3]
        current_action = actions[-1]
        previous_action = actions[-2]
        action_mean = np.mean(actions, axis=0)
        action_delta = current_action - previous_action

        first_difference = current - previous
        previous_difference = previous - previous_previous
        second_difference = first_difference - previous_difference
        # This is a causal event magnitude, not a learned threshold.  It is
        # retained as both a scalar and a smooth bounded gate so short
        # discontinuities are not averaged into a global harmonic.
        event_energy = float(np.sqrt(np.mean(first_difference ** 2)))
        event_gate = event_energy / (1.0 + event_energy)
        event_scores.append(event_energy)

        base = np.concatenate((current, current_action))
        base_rows.append(base)
        raw_rows.append(np.concatenate((values[::-1].reshape(-1), actions[::-1].reshape(-1))))

        residual = values - current[None, :]
        state_spectrum = np.fft.rfft(residual, axis=0)
        action_residual = actions - current_action[None, :]
        action_spectrum = np.fft.rfft(action_residual, axis=0)
        first_harmonic = state_spectrum[1] if state_spectrum.shape[0] > 1 else np.zeros_like(current)
        # Polar coordinates preserve the phase explicitly.  The bounded
        # complex normalization avoids an unstable atan2 branch when a
        # coefficient is close to zero while still retaining direction.
        amplitude = np.abs(first_harmonic)
        phase_cos = first_harmonic.real / (amplitude + 1e-8)
        phase_sin = first_harmonic.imag / (amplitude + 1e-8)
        fixed_rows.append(np.concatenate((
            base,
            state_spectrum.real.reshape(-1),
            state_spectrum.imag.reshape(-1),
            action_spectrum.real.reshape(-1),
            action_spectrum.imag.reshape(-1),
        ))
        )

        # Exact feature-level analogue of the historical
        # ``state_history_mode=fourier_residual`` plus summary-action input.
        legacy_rows.append(np.concatenate((
            base,
            first_harmonic.real,
            first_harmonic.imag,
            action_mean,
            action_delta,
        )))

        # Local multi-scale basis: endpoint, recent velocity-like change,
        # acceleration-like change, local average, and action transition.
        difference_rows.append(np.concatenate((
            base,
            np.mean(values, axis=0),
            first_difference,
            previous_difference,
            second_difference,
            action_mean,
            action_delta,
        )))

        # Event-aware lift: keep signed and magnitude information.  The
        # scalar gate is deterministic and causal; it does not peek at the
        # held-out target or choose a percentile from the complete export.
        event_rows.append(np.concatenate((
            base,
            first_difference,
            np.abs(first_difference),
            second_difference,
            event_gate * first_difference,
            np.asarray([event_energy, event_gate]),
            action_mean,
            action_delta,
            (first_difference[:, None] * action_mean[None, :]).reshape(-1),
        )))

        # Compact event branch: preserve the same causal ingredients without
        # per-latent absolute values or bilinear expansion.  This is the
        # declared low-sample control for the high-dimensional event branch.
        compact_event_rows.append(np.concatenate((
            base,
            np.mean(values, axis=0),
            first_difference,
            previous_difference,
            second_difference,
            action_mean,
            action_delta,
            np.asarray([event_energy, event_gate]),
        )))

        phase_rows.append(np.concatenate((
            base,
            first_difference,
            second_difference,
            amplitude,
            phase_cos,
            phase_sin,
            action_mean,
            action_delta,
            action_spectrum[1].real if action_spectrum.shape[0] > 1 else np.zeros_like(current_action),
            action_spectrum[1].imag if action_spectrum.shape[0] > 1 else np.zeros_like(current_action),
        )))
        current_indices.append(window[-1])

    return {
        "indices": np.asarray(current_indices, dtype=np.int64),
        "base": np.asarray(base_rows, dtype=np.float64),
        "raw_history": np.asarray(raw_rows, dtype=np.float64),
        "fixed_fourier": np.asarray(fixed_rows, dtype=np.float64),
        "legacy_first_harmonic": np.asarray(legacy_rows, dtype=np.float64),
        "local_difference": np.asarray(difference_rows, dtype=np.float64),
        "event_aware": np.asarray(event_rows, dtype=np.float64),
        "compact_event_aware": np.asarray(compact_event_rows, dtype=np.float64),
        "spectral_phase": np.asarray(phase_rows, dtype=np.float64),
        "event_score": np.asarray(event_scores, dtype=np.float64),
        "history_length": int(history_length),
        "causal_window": True,
        "feature_contract": {
            "base_interface": "current_state_plus_current_action",
            "local_difference": "endpoint_mean_first_difference_previous_difference_second_difference_action_mean_delta",
            "event_aware": "signed_and_absolute_endpoint_difference_bounded_causal_gate_event_energy_action_interaction",
            "compact_event_aware": "endpoint_mean_signed_first_second_difference_bounded_event_energy_action_summary",
            "spectral_phase": "local_difference_first_harmonic_amplitude_and_normalized_phase_action_spectrum",
            "fixed_fourier": "full_window_rfft_residual_with_current_state_action",
            "legacy_first_harmonic": "historical_first_non_dc_residual_harmonic_with_summary_action",
            "threshold_source": "none",
        },
    }


def _covariance(value: np.ndarray, ridge: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    centered = value - np.mean(value, axis=0, keepdims=True)
    return (centered.T @ centered) / max(value.shape[0] - 1, 1) + ridge * np.eye(value.shape[1])


def _conditional_residual(target: np.ndarray, condition: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate((condition, np.ones((len(condition), 1))), axis=1)
    coefficients = _ridge_solve(design, target, ridge)
    return target - design @ coefficients


def gaussian_conditional_mutual_information(
    target: np.ndarray,
    extra: np.ndarray,
    condition: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Estimate I(target; extra | condition) with a Gaussian plug-in formula.

    This is a dependence diagnostic, not a nonparametric mutual-information
    estimator.  It is useful here only when paired with held-out transfer.
    """

    target = _matrix(target, "target")
    extra = _matrix(extra, "extra")
    condition = _matrix(condition, "condition")
    if len(target) != len(extra) or len(target) != len(condition):
        raise ValueError("target, extra and condition must align")
    residual_condition = _conditional_residual(target, condition, ridge)
    joint = np.concatenate((condition, extra), axis=1)
    residual_joint = _conditional_residual(target, joint, ridge)
    sign_a, logdet_a = np.linalg.slogdet(_covariance(residual_condition, ridge))
    sign_b, logdet_b = np.linalg.slogdet(_covariance(residual_joint, ridge))
    if sign_a <= 0 or sign_b <= 0:
        raise ValueError("residual covariance is not positive definite")
    value = max(0.0, 0.5 * float(logdet_a - logdet_b))
    return {
        "estimate_nats": value,
        "estimate_bits": value / math.log(2.0),
        "estimator": "gaussian_linear_residual_plugin",
        "ridge": float(ridge),
        "n_samples": int(len(target)),
    }


def _cross_fitted_rmse(
    features: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    outcome_mask: np.ndarray,
    ridge: float,
) -> float:
    fit_features = np.asarray(features)[fit_mask]
    fit_target = np.asarray(target)[fit_mask]
    outcome_features = np.asarray(features)[outcome_mask]
    outcome_target = np.asarray(target)[outcome_mask]
    design = np.concatenate((fit_features, np.ones((len(fit_features), 1))), axis=1)
    coefficients = _ridge_solve(design, fit_target, ridge)
    prediction = np.concatenate((outcome_features, np.ones((len(outcome_features), 1))), axis=1) @ coefficients
    return float(np.sqrt(np.mean((prediction - outcome_target) ** 2)))


def fit_only_feature_projection(
    features: np.ndarray,
    fit_mask: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a feature-space PCA only on the fit rows.

    Delay, interaction, and Fourier lifts can have many more coordinates than
    independent fit episodes.  Comparing such a lift to a lower-dimensional
    base with the same tiny ridge penalty confounds information with an
    underidentified regression.  This projection gives every feature family a
    declared, fit-only budget and is applied unchanged to outcome rows.
    """

    features = _matrix(features, "features")
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if len(features) != len(fit_mask) or int(np.sum(fit_mask)) < 2:
        raise ValueError("feature rows and fit mask are incompatible")
    requested = int(rank)
    if requested < 1:
        raise ValueError("rank must be positive")
    fit_values = features[fit_mask]
    center = np.mean(fit_values, axis=0, keepdims=True)
    centered_fit = fit_values - center
    _, singular_values, components = np.linalg.svd(centered_fit, full_matrices=False)
    retained = min(requested, components.shape[0])
    projected = (features - center) @ components[:retained].T
    energy = singular_values**2
    return projected, {
        "method": "fit_only_feature_pca",
        "original_dimension": int(features.shape[1]),
        "requested_rank": requested,
        "retained_rank": int(retained),
        "fit_explained_variance_fraction": float(
            np.sum(energy[:retained]) / max(float(np.sum(energy)), 1e-12)
        ),
    }


def fit_only_standardized_feature_projection(
    features: np.ndarray,
    fit_mask: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit feature z-scoring and PCA on fit rows only.

    High-dimensional event features can otherwise be dominated by a few
    latent coordinates with large scale.  The standardization is fit-only and
    is therefore a protocol control, not a test-set normalization.
    """

    features = _matrix(features, "features")
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if len(features) != len(fit_mask) or int(np.sum(fit_mask)) < 2:
        raise ValueError("feature rows and fit mask are incompatible")
    fit_values = features[fit_mask]
    center = np.mean(fit_values, axis=0, keepdims=True)
    scale = np.std(fit_values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (features - center) / scale
    _, singular_values, components = np.linalg.svd(normalized[fit_mask], full_matrices=False)
    retained = min(int(rank), components.shape[0])
    projected = normalized @ components[:retained].T
    energy = singular_values**2
    return projected, {
        "method": "fit_only_standardized_feature_pca",
        "original_dimension": int(features.shape[1]),
        "requested_rank": int(rank),
        "retained_rank": int(retained),
        "fit_explained_variance_fraction": float(np.sum(energy[:retained]) / max(float(np.sum(energy)), 1e-12)),
        "zero_variance_coordinates": int(np.sum(scale <= 1e-12)),
    }


def transfer_sufficiency_report(
    target_effect: np.ndarray,
    base: np.ndarray,
    lifted: np.ndarray,
    fit_mask: np.ndarray,
    outcome_mask: np.ndarray,
    *,
    ridge: float = 1e-5,
    feature_rank: int | None = None,
    feature_normalization: str = "raw",
) -> dict[str, Any]:
    """Compare held-out action-to-task effect prediction before/after lifting."""

    target_effect = _matrix(target_effect, "target_effect")
    base = _matrix(base, "base")
    lifted = _matrix(lifted, "lifted")
    fit_mask = np.asarray(fit_mask, dtype=bool)
    outcome_mask = np.asarray(outcome_mask, dtype=bool)
    if len(target_effect) != len(base) or len(base) != len(lifted):
        raise ValueError("target and feature arrays must align")
    base_rmse = _cross_fitted_rmse(base, target_effect, fit_mask, outcome_mask, ridge)
    lifted_rmse = _cross_fitted_rmse(lifted, target_effect, fit_mask, outcome_mask, ridge)
    report = {
        "base_outcome_rmse": base_rmse,
        "lifted_outcome_rmse": lifted_rmse,
        "relative_rmse_reduction": float((base_rmse - lifted_rmse) / max(base_rmse, 1e-12)),
        "lifted_beats_base": bool(lifted_rmse < base_rmse),
        "fit_count": int(np.sum(fit_mask)),
        "outcome_count": int(np.sum(outcome_mask)),
        "ridge": float(ridge),
    }
    if feature_rank is not None:
        if feature_normalization == "raw":
            projection_fn = fit_only_feature_projection
        elif feature_normalization == "zscore":
            projection_fn = fit_only_standardized_feature_projection
        else:
            raise ValueError("feature_normalization must be raw or zscore")
        base_budgeted, base_projection = projection_fn(base, fit_mask, rank=int(feature_rank))
        lifted_budgeted, lifted_projection = projection_fn(lifted, fit_mask, rank=int(feature_rank))
        budget_base_rmse = _cross_fitted_rmse(
            base_budgeted, target_effect, fit_mask, outcome_mask, ridge
        )
        budget_lifted_rmse = _cross_fitted_rmse(
            lifted_budgeted, target_effect, fit_mask, outcome_mask, ridge
        )
        report["fit_only_rank_matched"] = {
            "feature_rank": int(feature_rank),
            "base_outcome_rmse": budget_base_rmse,
            "lifted_outcome_rmse": budget_lifted_rmse,
            "relative_rmse_reduction": float(
                (budget_base_rmse - budget_lifted_rmse) / max(budget_base_rmse, 1e-12)
            ),
            "lifted_beats_base": bool(budget_lifted_rmse < budget_base_rmse),
            "base_projection": base_projection,
            "lifted_projection": lifted_projection,
            "fit_count": int(np.sum(fit_mask)),
            "outcome_count": int(np.sum(outcome_mask)),
            "ridge": float(ridge),
            "feature_normalization": feature_normalization,
        }
    return report


def fit_only_select_transfer_variant(
    target_effect: np.ndarray,
    candidates: dict[str, np.ndarray],
    episode_id: np.ndarray,
    fit_mask: np.ndarray,
    outcome_mask: np.ndarray,
    *,
    feature_rank: int,
    ridge: float = 1e-5,
    split_seed: int = 0,
    minimum_fit_episodes: int = 4,
) -> dict[str, Any]:
    """Select a memory family using only an inner split of outer-fit episodes.

    This is deliberately a selector, not an outcome-tuned ensemble.  The
    outer outcome rows are never consulted while choosing a candidate.  Each
    candidate receives the same fit-only PCA rank and ridge regression budget,
    so the result tests whether the historical-Fourier advantage is a fixed
    representation effect or can be recovered by a frozen, predeclared
    selection protocol.  With too few episodes the function fails closed.
    """

    target_effect = _matrix(target_effect, "target_effect")
    episode_id = np.asarray(episode_id).reshape(-1)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    outcome_mask = np.asarray(outcome_mask, dtype=bool)
    if len(target_effect) != len(episode_id) or len(fit_mask) != len(target_effect):
        raise ValueError("target, episode_id, and masks must align")
    if not candidates:
        raise ValueError("at least one candidate is required")
    candidate_names = tuple(sorted(candidates))
    for name in candidate_names:
        candidate = _matrix(candidates[name], f"candidate[{name}]")
        if candidate.shape[0] != len(target_effect):
            raise ValueError("candidate rows must align with target")
    fit_episodes = np.unique(episode_id[fit_mask])
    if len(fit_episodes) < int(minimum_fit_episodes):
        return {
            "status": "insufficient_support",
            "selected_variant": None,
            "fit_episode_count": int(len(fit_episodes)),
            "minimum_fit_episodes": int(minimum_fit_episodes),
            "inner_validation_rmse": {},
        }
    ordered = np.random.default_rng(int(split_seed)).permutation(fit_episodes)
    cut = max(1, min(len(ordered) - 1, len(ordered) // 2))
    inner_fit_episodes = ordered[:cut]
    inner_validation_episodes = ordered[cut:]
    inner_fit = fit_mask & np.isin(episode_id, inner_fit_episodes)
    inner_validation = fit_mask & np.isin(episode_id, inner_validation_episodes)
    if int(np.sum(inner_fit)) < 2 or int(np.sum(inner_validation)) < 1:
        return {
            "status": "insufficient_support",
            "selected_variant": None,
            "fit_episode_count": int(len(fit_episodes)),
            "minimum_fit_episodes": int(minimum_fit_episodes),
            "inner_validation_rmse": {},
        }

    validation_rmse: dict[str, float] = {}
    for name in candidate_names:
        projected, _ = fit_only_feature_projection(
            candidates[name], inner_fit, rank=int(feature_rank)
        )
        validation_rmse[name] = _cross_fitted_rmse(
            projected, target_effect, inner_fit, inner_validation, ridge
        )
    selected = min(candidate_names, key=lambda name: (validation_rmse[name], name))
    return {
        "status": "pass",
        "selected_variant": selected,
        "fit_episode_count": int(len(fit_episodes)),
        "minimum_fit_episodes": int(minimum_fit_episodes),
        "inner_fit_episode_ids": [int(item) for item in inner_fit_episodes],
        "inner_validation_episode_ids": [int(item) for item in inner_validation_episodes],
        "inner_fit_rows": int(np.sum(inner_fit)),
        "inner_validation_rows": int(np.sum(inner_validation)),
        "inner_validation_rmse": validation_rmse,
        "selection_seed": int(split_seed),
        "selection_outcome_leakage": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
