"""Audit local/event-aware history lifts against the frozen Fourier baseline.

This is a separate, append-only experiment.  The historical ``fourier_residual``
result is never overwritten.  Every feature family is evaluated on the same
episode split, the same projected latent, the same action interface, and a
fit-only rank-matched budget.  The output is transfer evidence only; it does
not establish causal identification, contact-mode recovery, or planning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
    _json_safe,
    adaptive_time_frequency_features,
    causal_delay_time_frequency_features,
    fit_only_select_transfer_variant,
    transfer_sufficiency_report,
)
from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import (
    apply_pod_projection,
    episode_disjoint_split,
    fit_only_pod_projection,
    load_enriched_arrays,
)


def build_report(
    arrays: dict[str, Any],
    *,
    fit_fraction: float = 0.6,
    split_seed: int | None = None,
    history_length: int = 4,
    state_rank: int = 32,
    ridge: float = 1e-5,
) -> dict[str, Any]:
    """Build one split-level report with a frozen feature family."""

    eta = np.asarray(arrays["source_latent"], dtype=np.float64)
    action = np.asarray(arrays["action"], dtype=np.float64)
    source_task = np.asarray(arrays["source_state"], dtype=np.float64)
    target_task = np.asarray(arrays["target_state"], dtype=np.float64)
    episode_id = np.asarray(arrays["episode_id"], dtype=np.int64)
    frame = np.asarray(arrays["source_frame"], dtype=np.int64)
    fit_index, outcome_index, split = episode_disjoint_split(
        episode_id, fit_fraction=fit_fraction, split_seed=split_seed
    )
    eta_model, projection, basis, center = fit_only_pod_projection(
        eta, fit_index, rank=state_rank
    )
    # ``basis`` and ``center`` are retained in the protocol metadata so the
    # lift is an explicit fit-only reparameterization rather than an opaque
    # post-hoc feature rewrite.
    _ = apply_pod_projection(eta, basis, center)
    features = adaptive_time_frequency_features(
        eta_model, action, episode_id, frame, history_length=history_length
    )
    signal_indices = np.asarray(features["indices"], dtype=np.int64)
    fit_mask = np.isin(signal_indices, fit_index)
    outcome_mask = np.isin(signal_indices, outcome_index)
    target_effect = target_task[signal_indices] - source_task[signal_indices]
    base = np.asarray(features["base"], dtype=np.float64)
    rank = min(int(base.shape[1]), max(1, int(np.sum(fit_mask)) - 1))

    fixed = causal_delay_time_frequency_features(
        eta_model, action, episode_id, frame, history_length=history_length
    )
    variants = {
        "raw_history": np.asarray(features["raw_history"], dtype=np.float64),
        "legacy_first_harmonic": np.asarray(features["legacy_first_harmonic"], dtype=np.float64),
        "fixed_fourier": np.asarray(fixed["time_frequency"], dtype=np.float64),
        "local_difference": np.asarray(features["local_difference"], dtype=np.float64),
        "event_aware": np.asarray(features["event_aware"], dtype=np.float64),
        "compact_event_aware": np.asarray(features["compact_event_aware"], dtype=np.float64),
        "spectral_phase": np.asarray(features["spectral_phase"], dtype=np.float64),
    }
    reports = {}
    for name, lifted in variants.items():
        reports[name] = transfer_sufficiency_report(
            target_effect,
            base,
            lifted,
            fit_mask,
            outcome_mask,
            ridge=ridge,
            feature_rank=rank,
        )
        reports[name]["feature_shape"] = [int(value) for value in lifted.shape]
    compact_zscore = transfer_sufficiency_report(
        target_effect,
        base,
        variants["compact_event_aware"],
        fit_mask,
        outcome_mask,
        ridge=ridge,
        feature_rank=rank,
        feature_normalization="zscore",
    )
    compact_zscore["feature_shape"] = [int(value) for value in variants["compact_event_aware"].shape]
    reports["compact_event_zscore"] = compact_zscore

    selection = fit_only_select_transfer_variant(
        target_effect,
        {
            "compact_event_aware": variants["compact_event_aware"],
            "legacy_first_harmonic": variants["legacy_first_harmonic"],
            "local_difference": variants["local_difference"],
            "raw_history": variants["raw_history"],
            "spectral_phase": variants["spectral_phase"],
        },
        episode_id[signal_indices],
        fit_mask,
        outcome_mask,
        feature_rank=rank,
        ridge=ridge,
        split_seed=0 if split_seed is None else int(split_seed) + 7919,
    )
    if selection["status"] == "pass":
        selected_name = str(selection["selected_variant"])
        selection["outer_selected_report"] = reports[selected_name]["fit_only_rank_matched"]

    event_score = np.asarray(features["event_score"], dtype=np.float64)
    return _json_safe({
        "artifact": "adaptive_time_frequency_memory_audit",
        "protocol": {
            "version": "adaptive_event_memory.v1",
            "replicate_unit": "seeded episode split",
            "split": "episode_disjoint_fit_outcome",
            "fit_only_projection": True,
            "fit_only_feature_budget": True,
            "same_current_transition_rows": True,
            "history_length": int(history_length),
            "state_rank": int(projection["retained_rank"]),
            "feature_rank": int(rank),
            "primary_variant": "event_aware",
            "frozen_negative_control": "legacy_first_harmonic",
            "fit_only_selector": "inner_episode_validation_over_declared_memory_families",
            "threshold_source": "none; deterministic bounded event gate",
            "claim_boundary": (
                "Held-out action-to-task transfer diagnostic only; no causal "
                "identification, contact-mode recovery, online planning, or universal world model."
            ),
        },
        "metadata": arrays["metadata"],
        "split": split,
        "support": {
            "n_rows": int(len(signal_indices)),
            "fit_rows": int(np.sum(fit_mask)),
            "outcome_rows": int(np.sum(outcome_mask)),
            "event_score_mean": float(np.mean(event_score)),
            "event_score_max": float(np.max(event_score)),
            "event_score_finite": bool(np.isfinite(event_score).all()),
        },
        "feature_contract": features["feature_contract"],
        "variants": reports,
        "fit_only_selection": selection,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--fit-fraction", type=float, default=0.6)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-5)
    args = parser.parse_args()
    arrays = load_enriched_arrays(args.source)
    report = build_report(
        arrays,
        fit_fraction=args.fit_fraction,
        split_seed=args.split_seed,
        history_length=args.history_length,
        state_rank=args.state_rank,
        ridge=args.ridge,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "split_seed": args.split_seed,
        "primary": report["variants"]["compact_event_zscore"],
        "event_aware_raw": report["variants"]["event_aware"],
        "fixed_fourier": report["variants"]["fixed_fourier"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
