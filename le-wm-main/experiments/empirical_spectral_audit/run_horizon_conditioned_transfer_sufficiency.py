"""Audit horizon-conditioned, support-aware latent memory.

This runner implements the methodological change motivated by the literature
and the existing negative results:

* ``fixed_fourier`` is the low-variance carrier;
* phase/event information is a conditional residual, never an unconditional
  replacement for the carrier;
* horizon policy is frozen before outcome evaluation;
* fit-only selector instability is reported as a diagnostic and never used to
  tune an online controller from held-out outcomes;
* insufficient support or unstable selection causes a fail-closed fallback.

The artifact is still a transfer-sufficiency diagnostic.  It does not imply
action ranking, contact-mode identification, pose closure, or online planning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_event_gated_recurrent_memory_audit import (
    frozen_horizon_memory_variant,
)
from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
    adaptive_time_frequency_features,
    fit_only_select_transfer_variant,
    transfer_sufficiency_report,
)
from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import (
    episode_disjoint_split,
    fit_only_pod_projection,
    load_enriched_arrays,
)


def _entropy(values: list[str]) -> float:
    if not values:
        return float("nan")
    counts = Counter(values)
    total = float(len(values))
    raw = -sum((count / total) * math.log(count / total) for count in counts.values())
    normalizer = math.log(len(counts)) if len(counts) > 1 else 1.0
    return float(raw / normalizer) if normalizer > 0 else 0.0


def _conditional_event_residual(
    fixed_fourier: np.ndarray,
    spectral_phase: np.ndarray,
    event_score: np.ndarray,
) -> np.ndarray:
    """Add a causal, bounded event-gated phase residual to the carrier.

    The arrays have different native dimensions, so the implementation keeps
    the carrier and appends only the gated residual branch.  A common
    fit-only rank budget is applied later by ``transfer_sufficiency_report``.
    """

    event_score = np.asarray(event_score, dtype=np.float64).reshape(-1, 1)
    gate = event_score / (1.0 + np.abs(event_score))
    residual = gate * np.asarray(spectral_phase, dtype=np.float64)
    return np.concatenate((np.asarray(fixed_fourier, dtype=np.float64), residual), axis=1)


def declared_horizon_policy(horizon: int, profile: str = "development_v1") -> str:
    """Return a policy frozen independently of the current outcome rows.

    ``development_v1`` preserves the earlier audit policy.  The second profile
    is a new confirmatory candidate: keep the carrier for h=1 and use the
    multi-view phase lift for all genuinely multi-step targets.  It must be
    evaluated on fresh split seeds and is not retroactively applied to the
    earlier reports.
    """

    if profile == "development_v1":
        return frozen_horizon_memory_variant(horizon)
    if profile == "carrier_plus_phase_for_multistep_v2":
        return "fixed_fourier" if int(horizon) <= 1 else "fourier_plus_phase"
    raise ValueError(f"unknown policy profile: {profile}")


def _selector_instability(
    target_effect: np.ndarray,
    candidates: dict[str, np.ndarray],
    episode_id: np.ndarray,
    fit_mask: np.ndarray,
    *,
    feature_rank: int,
    trials: int,
    split_seed: int,
    minimum_fit_episodes: int = 4,
) -> dict[str, Any]:
    """Measure fit-only memory-family selection instability.

    Each trial removes a random 25% of the outer-fit episodes, then performs
    the predeclared inner split.  The outer outcome episodes are never passed
    to the selector.  This is a stability diagnostic, not a way to pick a
    winning variant.
    """

    episode_id = np.asarray(episode_id).reshape(-1)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_episodes = np.unique(episode_id[fit_mask])
    if len(fit_episodes) < int(minimum_fit_episodes):
        return {
            "status": "insufficient_support",
            "selected_variants": [],
            "selection_frequency": {},
            "stability": None,
            "normalized_entropy": None,
            "uses_outcome": False,
        }

    keep_count = max(int(minimum_fit_episodes), int(round(0.75 * len(fit_episodes))))
    keep_count = min(keep_count, len(fit_episodes))
    selected: list[str] = []
    trial_support: list[int] = []
    for trial in range(int(trials)):
        rng = np.random.default_rng(int(split_seed) + 1009 * (trial + 1))
        subset = rng.choice(fit_episodes, size=keep_count, replace=False)
        trial_mask = fit_mask & np.isin(episode_id, subset)
        result = fit_only_select_transfer_variant(
            target_effect,
            candidates,
            episode_id,
            trial_mask,
            np.zeros_like(fit_mask, dtype=bool),
            feature_rank=feature_rank,
            ridge=1.0e-5,
            split_seed=int(split_seed) + trial,
            minimum_fit_episodes=minimum_fit_episodes,
        )
        variant = result.get("selected_variant")
        if variant is not None:
            selected.append(str(variant))
        trial_support.append(int(np.sum(trial_mask)))
    counts = Counter(selected)
    total = max(len(selected), 1)
    frequencies = {name: float(count / total) for name, count in sorted(counts.items())}
    stability = max(frequencies.values()) if frequencies else None
    return {
        "status": "available" if selected else "insufficient_inner_support",
        "trial_count": int(trials),
        "successful_trial_count": int(len(selected)),
        "keep_episode_count": int(keep_count),
        "selected_variants": selected,
        "selection_frequency": frequencies,
        "stability": stability,
        "normalized_entropy": _entropy(selected),
        "trial_fit_row_counts": trial_support,
        "uses_outcome": False,
    }


def build_report(
    arrays: dict[str, Any],
    *,
    split_seed: int,
    history_length: int = 4,
    horizon: int = 1,
    state_rank: int = 32,
    feature_rank: int = 8,
    policy_profile: str = "development_v1",
    instability_trials: int = 16,
    minimum_fit_episodes: int = 4,
    stability_threshold: float = 0.75,
) -> dict[str, Any]:
    fit_index, outcome_index, split = episode_disjoint_split(
        arrays["episode_id"], fit_fraction=0.6, split_seed=split_seed
    )
    fit_episodes = set(int(item) for item in arrays["episode_id"][fit_index])
    outcome_episodes = set(int(item) for item in arrays["episode_id"][outcome_index])
    projected, projection, _basis, _center = fit_only_pod_projection(
        np.asarray(arrays["source_latent"], dtype=np.float64),
        fit_index,
        rank=state_rank,
    )
    features = adaptive_time_frequency_features(
        projected,
        np.asarray(arrays["action"], dtype=np.float64),
        np.asarray(arrays["episode_id"]),
        np.asarray(arrays["source_frame"]),
        history_length=history_length,
    )
    features["fourier_plus_phase"] = np.concatenate(
        (features["fixed_fourier"], features["spectral_phase"]), axis=1
    )
    features["fourier_gated_event_residual"] = _conditional_event_residual(
        features["fixed_fourier"],
        features["spectral_phase"],
        features["event_score"],
    )
    indices = np.asarray(features["indices"], dtype=np.int64)
    episode_rows = np.asarray(arrays["episode_id"])[indices]
    fit_rows = np.isin(episode_rows, list(fit_episodes))
    outcome_rows = np.isin(episode_rows, list(outcome_episodes))
    target = (
        np.asarray(arrays["target_state"], dtype=np.float64)[indices]
        - np.asarray(arrays["source_state"], dtype=np.float64)[indices]
    )
    if int(np.sum(fit_rows)) < 2 or int(np.sum(outcome_rows)) < 1:
        return {
            "artifact": "horizon_conditioned_transfer_sufficiency",
            "status": "unavailable_insufficient_rows",
            "split": split,
            "support": {
                "fit_episodes": len(fit_episodes),
                "outcome_episodes": len(outcome_episodes),
                "fit_rows": int(np.sum(fit_rows)),
                "outcome_rows": int(np.sum(outcome_rows)),
            },
        }

    candidates = {
        "fixed_fourier": features["fixed_fourier"],
        "spectral_phase": features["spectral_phase"],
        "fourier_plus_phase": features["fourier_plus_phase"],
        "fourier_gated_event_residual": features["fourier_gated_event_residual"],
    }
    rank = min(int(feature_rank), max(1, int(np.sum(fit_rows)) - 1))
    reports = {
        name: transfer_sufficiency_report(
            target,
            features["base"],
            value,
            fit_rows,
            outcome_rows,
            feature_rank=rank,
            ridge=1.0e-5,
        )["fit_only_rank_matched"]
        for name, value in candidates.items()
    }
    selector = fit_only_select_transfer_variant(
        target,
        candidates,
        episode_rows,
        fit_rows,
        outcome_rows,
        feature_rank=rank,
        ridge=1.0e-5,
        split_seed=int(split_seed) + 7001,
        minimum_fit_episodes=minimum_fit_episodes,
    )
    instability = _selector_instability(
        target,
        candidates,
        episode_rows,
        fit_rows,
        feature_rank=rank,
        trials=instability_trials,
        split_seed=split_seed,
        minimum_fit_episodes=minimum_fit_episodes,
    )
    frozen_variant = declared_horizon_policy(horizon, policy_profile)
    stability = instability.get("stability")
    stable_selector = bool(
        stability is not None
        and float(stability) >= float(stability_threshold)
        and selector.get("selected_variant") is not None
    )
    report = {
        "artifact": "horizon_conditioned_transfer_sufficiency",
        "status": "available",
        "protocol": {
            "version": "horizon_conditioned_hybrid_memory.v1",
            "history_length": int(history_length),
            "horizon": int(horizon),
            "policy_profile": policy_profile,
            "state_rank": int(state_rank),
            "feature_rank": int(rank),
            "fit_only": True,
            "episode_disjoint": True,
            "carrier": "fixed_fourier",
            "residual": "spectral_phase gated by causal event_score",
            "selection_is_diagnostic": True,
            "stability_threshold": float(stability_threshold),
            "claim_boundary": (
                "held-out action-to-task transfer-sufficiency diagnostic only; no causal, "
                "contact-mode, pose-closure, or online-planning claim"
            ),
        },
        "metadata": arrays.get("metadata", {}),
        "projection": projection,
        "split": split,
        "support": {
            "fit_episodes": len(fit_episodes),
            "outcome_episodes": len(outcome_episodes),
            "fit_rows": int(np.sum(fit_rows)),
            "outcome_rows": int(np.sum(outcome_rows)),
        },
        "candidate_reports": reports,
        "fit_only_selector": selector,
        "selection_instability": instability,
        "frozen_horizon_policy": {
            "variant": frozen_variant,
            "outcome_report": reports[frozen_variant],
            "selection_uses_outcome": False,
        },
        "selector_diagnostic": {
            "stable": stable_selector,
            "selected_variant": selector.get("selected_variant"),
            "selected_outcome_report": (
                reports.get(str(selector["selected_variant"]))
                if selector.get("selected_variant") is not None else None
            ),
            "selection_is_not_deployed": True,
        },
        "deployment_policy": {
            "variant": frozen_variant,
            "outcome_report": reports[frozen_variant],
            "selector_used": False,
            "fallback_variant": frozen_variant,
            "fail_closed": True,
            "decision_rule": (
                "never deploy the fit-only selector; use the frozen horizon policy and "
                "report selection instability as a transfer-risk diagnostic"
            ),
        },
        "gates": {
            "selector_stable": stable_selector,
            "selector_uses_outcome": False,
            "policy_is_not_test_tuned": True,
            "task_level_utilization": "not_established",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--feature-rank", type=int, default=8)
    parser.add_argument(
        "--policy-profile",
        choices=("development_v1", "carrier_plus_phase_for_multistep_v2"),
        default="development_v1",
    )
    parser.add_argument("--instability-trials", type=int, default=16)
    args = parser.parse_args()
    arrays = load_enriched_arrays(args.source, horizon=args.horizon)
    report = build_report(
        arrays,
        split_seed=args.split_seed,
        history_length=args.history_length,
        horizon=args.horizon,
        state_rank=args.state_rank,
        feature_rank=args.feature_rank,
        policy_profile=args.policy_profile,
        instability_trials=args.instability_trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "horizon": report.get("protocol", {}).get("horizon"),
        "frozen_variant": report.get("frozen_horizon_policy", {}).get("variant"),
        "deployment_variant": report.get("deployment_policy", {}).get("variant"),
        "selector_stable": report.get("gates", {}).get("selector_stable"),
    }, indent=2))


if __name__ == "__main__":
    main()
