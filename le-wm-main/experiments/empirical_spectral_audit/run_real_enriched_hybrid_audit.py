"""Run the projected-hybrid audit on a canonical enriched learned-model export.

The runner is intentionally fail-closed.  It consumes only
``enriched_learned_model.v1`` payloads, splits by episode before fitting, and
does not infer a contact mode from RGB or from an endpoint label.  Its output
is a projected-system diagnostic, not a causal or planning result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import (  # noqa: E402
    SCHEMA,
    load_payload,
    validate_payload,
)
from experiments.empirical_spectral_audit.run_action_lifted_system_audit import (  # noqa: E402
    _ridge_solve,
    action_excitation,
    finite_horizon_diagnostics,
    fit_action_lifted_system,
    persistent_excitation,
    residualized_action_excitation,
)
from experiments.empirical_spectral_audit.run_projected_hybrid_physics_audit import (  # noqa: E402
    _json_safe,
    finite_horizon_hankel_geometry,
    frequency_response,
)
from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (  # noqa: E402
    causal_delay_time_frequency_features,
    gaussian_conditional_mutual_information,
    transfer_sufficiency_report,
)


def load_enriched_arrays(
    source: Path,
    *,
    horizon: int = 1,
    action_block: int = 0,
) -> dict[str, Any]:
    """Load one canonical export without fabricating missing fields."""

    metadata_migration = False
    try:
        payload = load_payload(source)
    except ValueError as error:
        # Some earlier files were emitted with the schema name but before the
        # three top-level provenance fields were added.  Permit only a
        # metadata-only, in-memory migration; vector-only or row-incomplete
        # artifacts still fail in validate_payload.
        raw = json.loads(source.read_text())
        if raw.get("record_schema") != SCHEMA or not isinstance(raw.get("enriched_records"), list):
            raise
        missing = {"study_scope", "claim_boundary", "runtime"}.difference(raw)
        if not missing:
            raise error
        raw.setdefault("study_scope", "legacy canonical enriched export; metadata-only migration")
        raw.setdefault(
            "claim_boundary",
            "Frozen-checkpoint projected diagnostic only; not causal identification, online planning, or universal transfer.",
        )
        raw.setdefault("runtime", "legacy_metadata_migration.v1")
        payload = validate_payload(raw)
        metadata_migration = True
    if str(horizon) not in {str(item) for item in payload["horizons"]}:
        raise ValueError(f"horizon {horizon} is absent from {source}")
    rows = payload["enriched_records"]
    if not rows:
        raise ValueError("canonical export has no rows")
    first_actions = np.asarray(rows[0]["action_blocks_normalized"], dtype=np.float64)
    if first_actions.ndim != 2 or not 0 <= action_block < first_actions.shape[0]:
        raise ValueError("action_block is outside the declared action block array")
    latent_dim = len(rows[0]["source_latent"])
    state_dim = len(rows[0]["source_state_raw"])
    action_dim = first_actions.shape[1]
    source_latent = []
    next_latent = []
    source_state = []
    target_state = []
    action = []
    episode = []
    frame = []
    sample_ids = []
    for row in rows:
        current = np.asarray(row["source_latent"], dtype=np.float64)
        state = np.asarray(row["source_state_raw"], dtype=np.float64)
        actions = np.asarray(row["action_blocks_normalized"], dtype=np.float64)
        successor = row["latent_by_horizon"][str(horizon)]
        target_latent = np.asarray(successor["target"], dtype=np.float64)
        successor_state = np.asarray(successor["target_state_raw"], dtype=np.float64)
        if current.shape != (latent_dim,) or target_latent.shape != (latent_dim,):
            raise ValueError("latent dimensions are not constant across the export")
        if state.shape != (state_dim,) or successor_state.shape != (state_dim,):
            raise ValueError("task-state dimensions are not constant across the export")
        if actions.ndim != 2 or actions.shape[1] != action_dim or action_block >= actions.shape[0]:
            raise ValueError("action block dimensions are not constant across the export")
        source_latent.append(current)
        next_latent.append(target_latent)
        source_state.append(state)
        target_state.append(successor_state)
        action.append(actions[action_block])
        episode.append(int(row["episode_id"]))
        frame.append(int(row["source_frame"]))
        sample_ids.append(str(row["sample_id"]))
    arrays = {
        "source_latent": np.asarray(source_latent),
        "next_latent": np.asarray(next_latent),
        "source_state": np.asarray(source_state),
        "target_state": np.asarray(target_state),
        "action": np.asarray(action),
        "episode_id": np.asarray(episode, dtype=np.int64),
        "source_frame": np.asarray(frame, dtype=np.int64),
        "sample_id": sample_ids,
    }
    arrays["metadata"] = {
        "record_schema": payload["record_schema"],
        "architecture": payload["architecture"],
        "environment": payload["environment"],
        "seed": payload["seed"],
        "training_stage": payload["training_stage"],
        "latent_dimension": payload["latent_dimension"],
        "action_distribution": payload["action_distribution"],
        "horizon": int(horizon),
        "action_block": int(action_block),
        "source": str(source),
        "metadata_only_migration": bool(metadata_migration),
    }
    return arrays


def episode_disjoint_split(
    episode_id: np.ndarray,
    *,
    fit_fraction: float = 0.6,
    split_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return deterministic row indices with no episode crossing.

    ``split_seed`` enables a reproducible randomized episode split.  The
    historical default (``None``) preserves sorted episode order for audit
    compatibility, while confirmatory runs should declare a seed to avoid
    conflating episode order with the fit/outcome assignment.
    """

    episode_id = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    episodes = np.unique(episode_id)
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for a disjoint split")
    cut = max(1, min(len(episodes) - 1, int(round(len(episodes) * fit_fraction))))
    if split_seed is None:
        ordered_episodes = episodes
    else:
        ordered_episodes = np.random.default_rng(int(split_seed)).permutation(episodes)
    fit_episodes = ordered_episodes[:cut]
    outcome_episodes = ordered_episodes[cut:]
    fit_index = np.flatnonzero(np.isin(episode_id, fit_episodes))
    outcome_index = np.flatnonzero(np.isin(episode_id, outcome_episodes))
    if set(episode_id[fit_index]).intersection(set(episode_id[outcome_index])):
        raise AssertionError("episode-disjoint split invariant failed")
    return fit_index, outcome_index, {
        "fit_episode_ids": [int(item) for item in fit_episodes],
        "outcome_episode_ids": [int(item) for item in outcome_episodes],
        "fit_row_count": int(len(fit_index)),
        "outcome_row_count": int(len(outcome_index)),
        "split_seed": None if split_seed is None else int(split_seed),
        "split_order": "sorted" if split_seed is None else "seeded_episode_permutation",
    }


def _predict_dynamics(fit: dict[str, Any], eta: np.ndarray, action: np.ndarray) -> np.ndarray:
    prediction = eta @ np.asarray(fit["A"]).T + action @ np.asarray(fit["B"]).T
    for index, matrix in enumerate(fit["bilinear_B"]):
        prediction += (eta * action[:, index, None]) @ np.asarray(matrix).T
    return prediction + np.asarray(fit["intercept"])


def _rmse(value: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(value) - np.asarray(target)) ** 2)))


def fit_only_pod_projection(
    eta: np.ndarray,
    fit_index: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Project latents using a fit-only POD/SVD basis.

    This is a declared computational reduction, not an unreported latent
    rewrite.  The basis is estimated from fit episodes only, then applied to
    every row, including outcome rows.  The report records the retained rank
    and variance fraction so downstream claims remain rank-qualified.
    """

    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or not len(fit_index):
        raise ValueError("eta must be a non-empty matrix with fit rows")
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    centered = eta[np.asarray(fit_index)] - np.mean(eta[np.asarray(fit_index)], axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    retained = min(rank, components.shape[0])
    basis = components[:retained]
    projected = (eta - np.mean(eta[np.asarray(fit_index)], axis=0, keepdims=True)) @ basis.T
    energy = singular_values**2
    fraction = float(np.sum(energy[:retained]) / max(float(np.sum(energy)), 1e-12))
    return projected, {
        "method": "fit_only_pod_svd",
        "original_dimension": int(eta.shape[1]),
        "retained_rank": int(retained),
        "requested_rank": int(rank),
        "truncated": bool(retained < eta.shape[1]),
        "fit_explained_variance_fraction": fraction,
    }, basis, np.mean(eta[np.asarray(fit_index)], axis=0, keepdims=True)


def apply_pod_projection(eta: np.ndarray, basis: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Apply an already fit-only POD basis without refitting on outcome rows."""

    return (np.asarray(eta, dtype=np.float64) - center) @ np.asarray(basis, dtype=np.float64).T


def _resolve_memory_action_mode(
    *,
    include_action_history: bool,
    action_history_mode: str | None,
) -> str:
    """Resolve the backward-compatible boolean into an explicit mode."""

    if action_history_mode is None:
        return "full" if include_action_history else "current"
    if action_history_mode not in {"current", "full", "summary"}:
        raise ValueError("action_history_mode must be current, full, or summary")
    return action_history_mode


def _memory_action_parts(
    action: np.ndarray,
    window: list[int],
    *,
    mode: str,
) -> list[np.ndarray]:
    """Return a fixed, causal action interface for one history window.

    ``summary`` is deliberately low-dimensional: current action, within-window
    action mean, and current-minus-previous action.  It preserves a declared
    action-history signal while avoiding one parameter block per latent history
    step in the small learned-export splits.
    """

    values = [np.asarray(action[index], dtype=np.float64) for index in window]
    if mode == "current":
        return [values[-1]]
    if mode == "full":
        return list(reversed(values))
    if mode == "summary":
        previous = values[-2] if len(values) > 1 else values[-1]
        return [values[-1], np.mean(values, axis=0), values[-1] - previous]
    raise ValueError(f"unsupported memory action mode: {mode}")


def _memory_state_parts(
    eta: np.ndarray,
    window: list[int],
    *,
    mode: str,
) -> list[np.ndarray]:
    """Return raw or first-harmonic causal state-memory coordinates."""

    values = np.asarray([eta[index] for index in window], dtype=np.float64)
    if mode == "raw":
        return [values[index] for index in range(len(values) - 1, -1, -1)]
    if mode == "fourier_residual":
        residual = values - values[-1][None, :]
        spectrum = np.fft.rfft(residual, axis=0)
        if spectrum.shape[0] < 2:
            return [values[-1]]
        return [values[-1], spectrum[1].real, spectrum[1].imag]
    raise ValueError("state_history_mode must be raw or fourier_residual")


def _fit_memory_windows(
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    episode_id: np.ndarray,
    frame: np.ndarray,
    indices: np.ndarray,
    *,
    memory_length: int,
    ridge: float,
    current_indices: np.ndarray | None = None,
    include_action_history: bool = False,
    action_history_mode: str | None = None,
    state_history_mode: str = "raw",
) -> dict[str, Any]:
    """Fit memory coefficients only from within-episode chronological windows.

    ``include_action_history`` is deliberately explicit.  The legacy model
    used the latent history but only the current action, which is not a
    complete finite-memory input for an input-driven system.  The corrected
    protocol includes the action attached to every state in the same causal
    window.  The default remains ``False`` so previously generated artifacts
    remain exactly reproducible.
    """

    mode = _resolve_memory_action_mode(
        include_action_history=include_action_history,
        action_history_mode=action_history_mode,
    )
    rows = []
    targets = []
    selected = set(int(item) for item in np.asarray(indices).tolist())
    eligible_current = selected if current_indices is None else set(
        int(item) for item in np.asarray(current_indices).tolist()
    )
    for episode in sorted(set(int(item) for item in episode_id[indices])):
        group = [int(item) for item in indices if int(episode_id[item]) == episode]
        group.sort(key=lambda item: (int(frame[item]), item))
        for position in range(memory_length - 1, len(group)):
            window = group[position - memory_length + 1 : position + 1]
            if any(item not in selected for item in window):
                continue
            current = group[position]
            if current not in eligible_current:
                continue
            row_parts = _memory_state_parts(eta, window, mode=state_history_mode)
            row_parts.extend(_memory_action_parts(action, window, mode=mode))
            rows.append(np.concatenate((*row_parts, [1.0])))
            targets.append(next_eta[current])
    if not rows:
        return {"status": "insufficient_contiguous_support", "memory_length": int(memory_length)}
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    coefficients = _ridge_solve(design, target, ridge)
    state_dim = eta.shape[1]
    action_dim = action.shape[1]
    predicted = design @ coefficients
    return {
        "status": "fit",
        "memory_length": int(memory_length),
        "n_windows": int(len(rows)),
        "coefficients": coefficients,
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "include_action_history": bool(mode == "full"),
        "action_history_mode": mode,
        "state_history_mode": state_history_mode,
        "feature_dimension": int(design.shape[1]),
        "train_rmse": _rmse(predicted, target),
    }


def _evaluate_memory_windows(
    fit: dict[str, Any],
    eta: np.ndarray,
    next_eta: np.ndarray,
    action: np.ndarray,
    episode_id: np.ndarray,
    frame: np.ndarray,
    indices: np.ndarray,
    *,
    current_indices: np.ndarray | None = None,
    include_action_history: bool = False,
    action_history_mode: str | None = None,
    state_history_mode: str = "raw",
) -> float | None:
    if fit.get("status") != "fit":
        return None
    mode = _resolve_memory_action_mode(
        include_action_history=include_action_history,
        action_history_mode=action_history_mode,
    )
    rows = []
    targets = []
    length = int(fit["memory_length"])
    selected = set(int(item) for item in np.asarray(indices).tolist())
    eligible_current = selected if current_indices is None else set(
        int(item) for item in np.asarray(current_indices).tolist()
    )
    for episode in sorted(set(int(item) for item in episode_id[indices])):
        group = [int(item) for item in indices if int(episode_id[item]) == episode]
        group.sort(key=lambda item: (int(frame[item]), item))
        for position in range(length - 1, len(group)):
            window = group[position - length + 1 : position + 1]
            if any(item not in selected for item in window):
                continue
            current = group[position]
            if current not in eligible_current:
                continue
            row_parts = _memory_state_parts(eta, window, mode=state_history_mode)
            row_parts.extend(_memory_action_parts(action, window, mode=mode))
            rows.append(np.concatenate((*row_parts, [1.0])))
            targets.append(next_eta[current])
    if not rows:
        return None
    return _rmse(np.asarray(rows) @ np.asarray(fit["coefficients"]), np.asarray(targets))


def _contiguous_current_indices(
    episode_id: np.ndarray,
    frame: np.ndarray,
    indices: np.ndarray,
    *,
    memory_length: int,
) -> np.ndarray:
    """Return current rows with a complete within-episode history.

    This is used to compare memory models on the *same current transitions*.
    The historical rows remain available to the estimator, while the returned
    rows define the evaluation target.  Without this distinction, a longer
    memory model is evaluated on fewer transitions and its RMSE is not paired
    with the Markov baseline.
    """

    if memory_length < 1:
        raise ValueError("memory_length must be positive")
    selected = set(int(item) for item in np.asarray(indices).tolist())
    eligible: list[int] = []
    for episode in sorted(set(int(item) for item in episode_id[indices])):
        group = [int(item) for item in indices if int(episode_id[item]) == episode]
        group.sort(key=lambda item: (int(frame[item]), item))
        for position in range(memory_length - 1, len(group)):
            window = group[position - memory_length + 1 : position + 1]
            if all(item in selected for item in window):
                eligible.append(group[position])
    return np.asarray(eligible, dtype=np.int64)


def build_real_report(
    arrays: dict[str, Any],
    *,
    fit_fraction: float = 0.6,
    ridge: float = 1e-5,
    horizons: tuple[int, ...] = (1, 2, 4),
    memory_lengths: tuple[int, ...] = (1, 2, 4),
    frequency_points: int = 9,
    state_rank: int = 32,
    split_seed: int | None = None,
    memory_include_action_history: bool = True,
    memory_action_mode: str | None = None,
    state_history_mode: str = "raw",
) -> dict[str, Any]:
    """Build a fit/outcome report for one canonical export."""

    eta = arrays["source_latent"]
    next_eta = arrays["next_latent"]
    action = arrays["action"]
    source_task = arrays["source_state"]
    target_task = arrays["target_state"]
    episode_id = arrays["episode_id"]
    frame = arrays["source_frame"]
    fit_index, outcome_index, split = episode_disjoint_split(
        episode_id, fit_fraction=fit_fraction, split_seed=split_seed
    )
    eta_model, projection, basis, center = fit_only_pod_projection(eta, fit_index, rank=state_rank)
    next_eta_model = apply_pod_projection(next_eta, basis, center)
    model = fit_action_lifted_system(
        eta_model[fit_index], next_eta_model[fit_index], action[fit_index], source_task[fit_index],
        ridge=ridge, include_interaction=True,
    )
    predicted_next = _predict_dynamics(model, eta_model[outcome_index], action[outcome_index])
    predicted_task = predicted_next @ np.asarray(model["D"]).T + np.asarray(model["task_intercept"])
    reconstructed_next = predicted_next @ basis + center
    diagnostics = {}
    for horizon in horizons:
        diagnostics[str(horizon)] = finite_horizon_diagnostics(
            model["A"], model["B"], model["D"], horizon=int(horizon)
        )
    hankel = {
        str(horizon): finite_horizon_hankel_geometry(
            model["A"], model["B"], model["D"], horizon=int(horizon)
        )
        for horizon in horizons
    }
    if frequency_points < 2:
        raise ValueError("frequency_points must be at least two")
    frequencies = np.linspace(0.0, np.pi, int(frequency_points))
    frequency = frequency_response(model["A"], model["B"], model["D"], frequencies)
    memory = {}
    max_memory_length = max(int(length) for length in memory_lengths)
    paired_fit_current = _contiguous_current_indices(
        episode_id, frame, fit_index, memory_length=max_memory_length
    )
    paired_outcome_current = _contiguous_current_indices(
        episode_id, frame, outcome_index, memory_length=max_memory_length
    )
    resolved_memory_action_mode = _resolve_memory_action_mode(
        include_action_history=memory_include_action_history,
        action_history_mode=memory_action_mode,
    )
    for length in memory_lengths:
        fitted = _fit_memory_windows(
            eta_model, next_eta_model, action, episode_id, frame, fit_index,
            memory_length=int(length), ridge=ridge,
            include_action_history=memory_include_action_history,
            action_history_mode=resolved_memory_action_mode,
            state_history_mode=state_history_mode,
        )
        fitted["outcome_rmse"] = _evaluate_memory_windows(
            fitted, eta_model, next_eta_model, action, episode_id, frame, outcome_index,
            include_action_history=memory_include_action_history,
            action_history_mode=resolved_memory_action_mode,
            state_history_mode=state_history_mode,
        )
        # Compare all memory lengths on the same current transitions.  The
        # historical rows remain available to construct each causal window;
        # only the current target rows are paired across models.
        paired_fit = _fit_memory_windows(
            eta_model, next_eta_model, action, episode_id, frame, fit_index,
            memory_length=int(length), ridge=ridge,
            current_indices=paired_fit_current,
            include_action_history=memory_include_action_history,
            action_history_mode=resolved_memory_action_mode,
            state_history_mode=state_history_mode,
        )
        fitted["paired_train_current_count"] = int(paired_fit.get("n_windows", 0))
        fitted["paired_outcome_rmse"] = _evaluate_memory_windows(
            paired_fit, eta_model, next_eta_model, action, episode_id, frame,
            outcome_index, current_indices=paired_outcome_current,
            include_action_history=memory_include_action_history,
            action_history_mode=resolved_memory_action_mode,
            state_history_mode=state_history_mode,
        )
        fitted["paired_outcome_current_count"] = int(len(paired_outcome_current))
        fitted.pop("coefficients", None)
        memory[str(length)] = fitted
    markov_outcome = memory.get("1", {}).get("outcome_rmse")
    supported_memory_outcomes = [
        item.get("outcome_rmse") for key, item in memory.items()
        if key != "1" and item.get("status") == "fit" and item.get("outcome_rmse") is not None
    ]
    memory_gate = {
        "status": "pass" if supported_memory_outcomes and markov_outcome is not None and min(supported_memory_outcomes) < markov_outcome else "fail_or_unsupported",
        "markov_outcome_rmse": markov_outcome,
        "best_supported_memory_outcome_rmse": min(supported_memory_outcomes) if supported_memory_outcomes else None,
        "criterion": "finite-memory outcome RMSE must strictly improve over Markov closure",
    }
    paired_markov_outcome = memory.get("1", {}).get("paired_outcome_rmse")
    paired_memory_outcomes = [
        item.get("paired_outcome_rmse") for key, item in memory.items()
        if key != "1" and item.get("status") == "fit" and item.get("paired_outcome_rmse") is not None
    ]
    memory_gate["paired_support"] = {
        "status": (
            "pass" if paired_memory_outcomes and paired_markov_outcome is not None
            and min(paired_memory_outcomes) < paired_markov_outcome
            else "fail_or_unsupported"
        ),
        "max_declared_memory_length": int(max_memory_length),
        "fit_current_count": int(len(paired_fit_current)),
        "outcome_current_count": int(len(paired_outcome_current)),
        "markov_outcome_rmse": paired_markov_outcome,
        "best_supported_memory_outcome_rmse": min(paired_memory_outcomes) if paired_memory_outcomes else None,
        "criterion": "compare every memory length on the same current transitions supported by the maximum history",
    }
    try:
        signal_features = causal_delay_time_frequency_features(
            eta_model, action, episode_id, frame, history_length=4
        )
        signal_indices = signal_features["indices"]
        signal_fit_mask = np.isin(signal_indices, fit_index)
        signal_outcome_mask = np.isin(signal_indices, outcome_index)
        signal_target = target_task[signal_indices] - source_task[signal_indices]
        base_features = signal_features["base"]
        # Equalize the effective feature budget using fit rows only.  This is
        # essential for PushT, where interaction/Fourier lifts otherwise have
        # hundreds of coordinates but only a few independent fit episodes.
        matched_feature_rank = min(
            int(base_features.shape[1]),
            max(1, int(np.sum(signal_fit_mask)) - 1),
        )
        signal_reports = {
            "delay": transfer_sufficiency_report(
                signal_target, base_features, signal_features["delay"], signal_fit_mask, signal_outcome_mask,
                feature_rank=matched_feature_rank,
            ),
            "interaction": transfer_sufficiency_report(
                signal_target, base_features, signal_features["interaction"], signal_fit_mask, signal_outcome_mask,
                feature_rank=matched_feature_rank,
            ),
            "time_frequency": transfer_sufficiency_report(
                signal_target, base_features, signal_features["time_frequency"], signal_fit_mask, signal_outcome_mask,
                feature_rank=matched_feature_rank,
            ),
            "delay_spectral": transfer_sufficiency_report(
                signal_target, base_features, signal_features["delay_spectral"], signal_fit_mask, signal_outcome_mask,
                feature_rank=matched_feature_rank,
            ),
        }
        fit_signal = signal_fit_mask
        signal_reports["gaussian_cmi_delay_given_base_fit"] = gaussian_conditional_mutual_information(
            signal_target[fit_signal],
            signal_features["delay"][fit_signal],
            base_features[fit_signal],
        )
        signal_information = {
            "status": "available",
            "history_length": signal_features["history_length"],
            "causal_window": signal_features["causal_window"],
            "n_valid_rows": int(len(signal_indices)),
            "matched_feature_rank": int(matched_feature_rank),
            "reports": signal_reports,
        }
    except ValueError as error:
        signal_information = {
            "status": "unavailable",
            "reason": str(error),
            "history_length": 4,
            "causal_window": True,
        }
    return _json_safe({
        "protocol": {
            "study_scope": "covariance_predictor_unified",
            "split": "episode_disjoint_fit_outcome",
            "fit_only_estimation": True,
            "claim_boundary": "projected input-output diagnostic; not causal identification, online planning, or universal world model",
            "task_readout_source": "source_state_raw; target_state_raw is held out for successor evaluation",
            "mode_labels": "unavailable; no contact mode inferred",
            "state_projection": projection,
            "ridge": float(ridge),
            "memory_comparison": {
                "legacy_unpaired_values_retained": True,
                "paired_max_memory_length": int(max_memory_length),
                "paired_fit_current_count": int(len(paired_fit_current)),
                "paired_outcome_current_count": int(len(paired_outcome_current)),
                "paired_comparison_required_for_gate": True,
                "feature_mode": (
                    resolved_memory_action_mode,
                ),
                "state_history_mode": state_history_mode,
            },
        },
        "metadata": arrays["metadata"],
        "split": split,
        "fit": {
            "model": model,
            "persistent_excitation": persistent_excitation(eta[fit_index], action[fit_index]),
            "action_excitation": action_excitation(action[fit_index]),
            "residualized_action_excitation": residualized_action_excitation(eta_model[fit_index], action[fit_index]),
            "n_rows": int(len(fit_index)),
        },
        "outcome": {
            "n_rows": int(len(outcome_index)),
            "latent_next_rmse_projected": _rmse(predicted_next, next_eta_model[outcome_index]),
            "latent_next_rmse_reconstructed": _rmse(reconstructed_next, next_eta[outcome_index]),
            "latent_projection_target_rmse": _rmse(next_eta_model[outcome_index] @ basis + center, next_eta[outcome_index]),
            "task_next_rmse": _rmse(predicted_task, target_task[outcome_index]),
        },
        "finite_horizon": diagnostics,
        "hankel_geometry": hankel,
        "frequency_response": frequency,
        "memory_closure": memory,
        "gates": {
            "action_excitation": {
                "status": "pass" if bool(report_action_pass := action_excitation(action[fit_index])["passes"]) else "fail",
                "residualized_status": "pass" if bool(residualized_action_excitation(eta_model[fit_index], action[fit_index])["passes"]) else "fail",
                "criterion": "input-only and state-residualized action Gramian minimum eigenvalue exceeds declared threshold; joint latent-action rank is reported but not used as the sole gate",
            },
            "memory_closure": memory_gate,
            "mode_observability": {
                "status": "unavailable",
                "criterion": "successor contact-mode labels or a separately validated mode probe are required",
            },
        },
        "signal_information": signal_information,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--action-block", type=int, default=0)
    parser.add_argument("--fit-fraction", type=float, default=0.6)
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Seeded episode permutation for fit/outcome assignment; omitted preserves legacy sorted order.",
    )
    parser.add_argument("--frequency-points", type=int, default=9)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument(
        "--legacy-memory-input",
        action="store_true",
        help="ablation: use state history plus current action only; default uses state and action history",
    )
    parser.add_argument(
        "--memory-action-mode",
        choices=("current", "full", "summary"),
        default=None,
        help="explicit causal action interface; summary uses current, mean, and current-minus-previous action",
    )
    parser.add_argument(
        "--memory-state-mode",
        choices=("raw", "fourier_residual"),
        default="raw",
        help="state-memory interface; fourier_residual uses current state plus the first non-DC residual harmonic",
    )
    args = parser.parse_args()
    print("Running root-disjoint real enriched export audit: observability, memory, transfer, Hankel, and frequency response.")
    arrays = load_enriched_arrays(args.source, horizon=args.horizon, action_block=args.action_block)
    report = build_real_report(
        arrays,
        fit_fraction=args.fit_fraction,
        frequency_points=args.frequency_points,
        state_rank=args.state_rank,
        split_seed=args.split_seed,
        memory_include_action_history=not args.legacy_memory_input,
        memory_action_mode=args.memory_action_mode,
        state_history_mode=args.memory_state_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source": str(args.source),
        "fit_rows": report["split"]["fit_row_count"],
        "outcome_rows": report["split"]["outcome_row_count"],
        "latent_next_rmse_projected": report["outcome"]["latent_next_rmse_projected"],
        "latent_next_rmse_reconstructed": report["outcome"]["latent_next_rmse_reconstructed"],
        "task_next_rmse": report["outcome"]["task_next_rmse"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
