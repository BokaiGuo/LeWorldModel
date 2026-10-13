"""Audit recurrent and event-gated latent memory against Fourier controls.

The experiment is intentionally conservative.  PCA, normalization, model
parameters, and optimization are fit on outer-fit episodes only.  The outcome
episodes are never used for model selection.  If the export has too little
episode support for a recurrent comparison, the result is ``unavailable``.
No contact-mode label is inferred from a task endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
    adaptive_time_frequency_features,
    causal_history_indices,
    fit_only_feature_projection,
    fit_only_select_transfer_variant,
    transfer_sufficiency_report,
)
from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import (
    episode_disjoint_split,
    fit_only_pod_projection,
    load_enriched_arrays,
)


class _MemoryRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


class _EventGatedMemoryRegressor(nn.Module):
    """Causal recurrent state update with an explicit event gate.

    The previous ``event_gated_gru`` name referred only to concatenating two
    event scalars to the GRU input.  This module makes the claimed mechanism
    literal: a candidate recurrent update is blended with the previous state
    by a learned, input-dependent gate at every causal step.  The gate is
    fit only on the outer-fit episodes through the same training call as the
    predictor; no outcome statistic is used to set a threshold.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            sequence.shape[0], self.cell.hidden_size,
            dtype=sequence.dtype, device=sequence.device,
        )
        for step in range(sequence.shape[1]):
            value = sequence[:, step]
            candidate = self.cell(value, hidden)
            update = self.gate(value)
            hidden = (1.0 - update) * hidden + update * candidate
        return self.head(hidden)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _windows(
    arrays: dict[str, Any],
    history_length: int,
    *,
    state_signal: np.ndarray | None = None,
) -> dict[str, Any]:
    histories = causal_history_indices(
        arrays["episode_id"], arrays["source_frame"], history_length=history_length
    )
    if not histories:
        raise ValueError("no causal histories are available")
    signal = np.asarray(
        arrays["source_latent"] if state_signal is None else state_signal,
        dtype=np.float64,
    )
    if signal.ndim != 2 or len(signal) != len(arrays["episode_id"]):
        raise ValueError("state_signal must be a row-aligned two-dimensional array")
    action = np.asarray(arrays["action"], dtype=np.float64)
    source_state = np.asarray(arrays["source_state"], dtype=np.float64)
    target_state = np.asarray(arrays["target_state"], dtype=np.float64)
    sequences = []
    events = []
    targets = []
    episodes = []
    for history in histories:
        z = signal[history]
        u = action[history]
        dz = np.diff(z, axis=0, prepend=z[:1])
        du = np.diff(u, axis=0, prepend=u[:1])
        event = np.sqrt(np.mean(dz * dz, axis=1, keepdims=True))
        sequences.append(np.concatenate((z, u), axis=1))
        events.append(np.concatenate((z, u, event, np.sqrt(np.mean(du * du, axis=1, keepdims=True))), axis=1))
        last = history[-1]
        targets.append(target_state[last] - source_state[last])
        episodes.append(int(arrays["episode_id"][last]))
    return {
        "sequence": np.asarray(sequences, dtype=np.float64),
        "event_sequence": np.asarray(events, dtype=np.float64),
        "target": np.asarray(targets, dtype=np.float64),
        "episode_id": np.asarray(episodes, dtype=np.int64),
    }


def _fit_normalization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True)
    return center, np.where(scale > 1.0e-8, scale, 1.0)


def _train_predict(
    train_sequence: np.ndarray,
    train_target: np.ndarray,
    test_sequence: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    event_gated: bool = False,
) -> tuple[np.ndarray, int]:
    _seed(seed)
    x_center, x_scale = _fit_normalization(train_sequence)
    y_center = train_target.mean(axis=0, keepdims=True)
    y_scale = np.where(train_target.std(axis=0, keepdims=True) > 1.0e-8, train_target.std(axis=0, keepdims=True), 1.0)
    x_train = torch.tensor((train_sequence - x_center) / x_scale, dtype=torch.float32)
    y_train = torch.tensor((train_target - y_center) / y_scale, dtype=torch.float32)
    x_test = torch.tensor((test_sequence - x_center) / x_scale, dtype=torch.float32)
    model_cls = _EventGatedMemoryRegressor if event_gated else _MemoryRegressor
    model = model_cls(x_train.shape[-1], hidden_dim, y_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x_test).numpy() * y_scale + y_center
    return prediction, int(sum(item.numel() for item in model.parameters()))


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def frozen_horizon_memory_variant(horizon: int) -> str:
    """Return the development-frozen causal spectral memory policy.

    The policy keeps the one-step interface conservative and uses the
    multi-view phase branch only once the target is genuinely multi-step.
    This is deliberately a declared policy, not an outcome-tuned selector.
    """

    if int(horizon) <= 1:
        return "fixed_fourier"
    if int(horizon) >= 4:
        return "fourier_plus_phase"
    return "spectral_phase"


def build_report(
    arrays: dict[str, Any],
    *,
    split_seed: int,
    history_length: int = 4,
    state_rank: int = 32,
    hidden_dim: int = 16,
    epochs: int = 120,
    learning_rate: float = 1.0e-3,
    horizon: int = 1,
    minimum_fit_episodes: int = 8,
    minimum_fit_sequences: int = 16,
) -> dict[str, Any]:
    fit_index, outcome_index, split = episode_disjoint_split(
        arrays["episode_id"], fit_fraction=0.6, split_seed=split_seed
    )
    fit_episodes = set(int(item) for item in arrays["episode_id"][fit_index])
    outcome_episodes = set(int(item) for item in arrays["episode_id"][outcome_index])
    # The recurrent and linear controls must see the same fit-only state
    # space.  The previous implementation projected only the linear controls
    # while feeding the raw latent to the GRU, creating a large parameterized
    # interface (384 dimensions in the JEPA export) for a few dozen windows.
    # That is an invalid comparison and can make the GRU appear to fail by
    # overfitting.  Fit the POD basis on fit episodes, then apply it to both
    # fit and outcome windows before any model is trained.
    all_latent = np.asarray(arrays["source_latent"], dtype=np.float64)
    projected, projection, _basis, _center = fit_only_pod_projection(
        all_latent, fit_index, rank=state_rank
    )
    windows = _windows(arrays, history_length, state_signal=projected)
    fit_mask = np.isin(windows["episode_id"], list(fit_episodes))
    outcome_mask = np.isin(windows["episode_id"], list(outcome_episodes))
    report: dict[str, Any] = {
        "artifact": "event_gated_recurrent_memory_audit",
        "protocol": {
            "version": "event_gated_recurrent_memory.v1",
            "history_length": int(history_length),
            "horizon": int(horizon),
            "state_rank": int(state_rank),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "fit_only_normalization": True,
            "episode_disjoint": True,
            "mode_labels": "unavailable; no privileged contact labels inferred",
            "claim_boundary": "held-out action-to-task memory diagnostic only; no causal or online-planning claim",
        },
        "metadata": arrays["metadata"],
        "split": split,
        "support": {
            "fit_episodes": len(fit_episodes),
            "outcome_episodes": len(outcome_episodes),
            "fit_sequences": int(fit_mask.sum()),
            "outcome_sequences": int(outcome_mask.sum()),
            "minimum_fit_episodes": int(minimum_fit_episodes),
            "minimum_fit_sequences": int(minimum_fit_sequences),
        },
    }
    if len(fit_episodes) < minimum_fit_episodes or int(fit_mask.sum()) < minimum_fit_sequences:
        report["status"] = "unavailable_insufficient_recurrent_support"
        report["claim_boundary"] = "recurrent result withheld because support is insufficient"
        return report

    # The same projected state is used by the causal linear controls above and
    # by the recurrent inputs below; no outcome rows enter the projection.
    feature_arrays = adaptive_time_frequency_features(
        projected,
        np.asarray(arrays["action"], dtype=np.float64),
        np.asarray(arrays["episode_id"]),
        np.asarray(arrays["source_frame"]),
        history_length=history_length,
    )
    # A declared multi-view lift: low-frequency history plus local phase/event
    # information.  It is projected back to the same fit-only rank below, so
    # the hybrid does not receive a larger downstream regression budget.
    feature_arrays["fourier_plus_phase"] = np.concatenate(
        (feature_arrays["fixed_fourier"], feature_arrays["spectral_phase"]), axis=1
    )
    signal_indices = np.asarray(feature_arrays["indices"], dtype=np.int64)
    feature_fit = np.isin(signal_indices, fit_index)
    feature_outcome = np.isin(signal_indices, outcome_index)
    target_effect = np.asarray(arrays["target_state"], dtype=np.float64)[signal_indices] - np.asarray(arrays["source_state"], dtype=np.float64)[signal_indices]
    rank = min(8, max(1, int(feature_fit.sum()) - 1))
    controls = {}
    for name in (
        "raw_history",
        "fixed_fourier",
        "legacy_first_harmonic",
        "local_difference",
        "event_aware",
        "compact_event_aware",
        "spectral_phase",
        "fourier_plus_phase",
    ):
        controls[name] = transfer_sufficiency_report(
            target_effect, feature_arrays["base"], feature_arrays[name], feature_fit, feature_outcome,
            feature_rank=rank, ridge=1.0e-5,
        )["fit_only_rank_matched"]

    selector_candidates = {
        name: feature_arrays[name]
        for name in (
            "fixed_fourier",
            "legacy_first_harmonic",
            "local_difference",
            "event_aware",
            "compact_event_aware",
            "spectral_phase",
            "fourier_plus_phase",
        )
    }
    selector = fit_only_select_transfer_variant(
        target_effect,
        selector_candidates,
        np.asarray(arrays["episode_id"])[signal_indices],
        feature_fit,
        feature_outcome,
        feature_rank=rank,
        ridge=1.0e-5,
        split_seed=split_seed + 7001,
    )
    if selector["status"] == "pass":
        selected_variant = str(selector["selected_variant"])
        selector["outer_selected_report"] = controls[selected_variant]
    else:
        selector["outer_selected_report"] = None
    frozen_variant = frozen_horizon_memory_variant(horizon)

    fit_sequence = windows["sequence"][fit_mask]
    outcome_sequence = windows["sequence"][outcome_mask]
    event_fit = windows["event_sequence"][fit_mask]
    event_outcome = windows["event_sequence"][outcome_mask]
    target_fit = windows["target"][fit_mask]
    target_outcome = windows["target"][outcome_mask]
    gru_prediction, gru_parameters = _train_predict(
        fit_sequence, target_fit, outcome_sequence, seed=split_seed + 101,
        hidden_dim=hidden_dim, epochs=epochs, learning_rate=learning_rate,
    )
    event_prediction, event_parameters = _train_predict(
        event_fit, target_fit, event_outcome, seed=split_seed + 202,
        hidden_dim=hidden_dim, epochs=epochs, learning_rate=learning_rate,
        event_gated=True,
    )
    report.update({
        "status": "available",
        "projection": projection,
        "feature_rank": rank,
        "controls": controls,
        "fit_only_spectral_selector": selector,
        "frozen_horizon_policy": {
            "variant": frozen_variant,
            "outcome_report": controls[frozen_variant],
            "selection_uses_outcome": False,
            "policy_status": "development_frozen; requires independent environment/architecture confirmation",
        },
        "recurrent": {
            "gru": {"outcome_rmse": _rmse(gru_prediction, target_outcome), "parameter_count": gru_parameters},
            "event_gated_gru": {"outcome_rmse": _rmse(event_prediction, target_outcome), "parameter_count": event_parameters},
        },
        "gates": {
            "gru_beats_legacy_fourier": bool(
                _rmse(gru_prediction, target_outcome) < controls["legacy_first_harmonic"]["lifted_outcome_rmse"]
            ),
            "fit_only_selector_beats_legacy_fourier": bool(
                selector.get("outer_selected_report") is not None
                and selector["outer_selected_report"]["lifted_outcome_rmse"]
                < controls["legacy_first_harmonic"]["lifted_outcome_rmse"]
            ),
            "fit_only_selector_beats_fixed_fourier": bool(
                selector.get("outer_selected_report") is not None
                and selector["outer_selected_report"]["lifted_outcome_rmse"]
                < controls["fixed_fourier"]["lifted_outcome_rmse"]
            ),
            "event_gated_gru_beats_gru": bool(
                _rmse(event_prediction, target_outcome) < _rmse(gru_prediction, target_outcome)
            ),
            "mode_conditioned_branch": "unavailable_without_independent_contact_mode_labels",
        },
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    arrays = load_enriched_arrays(args.source, horizon=args.horizon)
    report = build_report(
        arrays, split_seed=args.split_seed, history_length=args.history_length,
        state_rank=args.state_rank, hidden_dim=args.hidden_dim, epochs=args.epochs,
        horizon=args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "support": report["support"], "gates": report.get("gates", {})}, indent=2))


if __name__ == "__main__":
    main()
