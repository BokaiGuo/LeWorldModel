"""Build sealed PointMaze temporal-memory panels with explicit action support.

V6 separates the temporal-memory estimand from the V5 amplitude stress test:
train and held candidate banks use the same five-by-four pulse construction
and the same ``[0.28, 0.78]`` pulse-norm envelope.  ``numeric_ood`` keeps
their action hashes disjoint.  ``shared_candidate_in_support`` instead makes
every held candidate available during source training (alongside additional
training actions), isolating root/history generalisation from candidate-action
extrapolation.  All supplied prior-panel roots and action hashes remain
disjoint.  The builder writes train targets and calibration labels, but never
computes or materializes outcome labels.

This is a renderer-backed frozen-observation control.  It is not a live RGB
planning result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import temporal_template_fingerprint
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v3_panel import _sha256
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _load_records, _simulate_sequence
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer,
    _load_jepa_model,
    _make_context,
    _make_gdd_source_contract_context,
    _pool_visual,
    fit_visual_calibration,
    green_center,
    load_episode,
)


# v2 makes action-support declaration and identity-bound calibration labels a
# mandatory contract for every new panel.  Keep v1 artifacts read-only: they
# are historical evidence, not inputs to a new confirmation run.
PANEL_SCHEMA = "pointmaze_temporal_ood_v6_matched_panel.v3"
SOURCE_SCHEMA = "pointmaze_temporal_ood_v6_matched_source.v1"
MANIFEST_SCHEMA = "pointmaze_temporal_ood_v6_matched_manifest.v1"
FROZEN_PREFLIGHT_SCHEMA = "pointmaze_counterfactual_endpoint_v6_static_preflight.v1"
PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA = "pointmaze_planning_shaped_advantage_v7_static_preflight.v1"
SHARED_FUTURE_V8_PREFLIGHT_SCHEMA = "pointmaze_shared_future_v8_static_preflight.v1"
OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA = "pointmaze_planning_shaped_opportunity_router_v9_static_preflight.v1"
ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA = "pointmaze_endpoint_repairability_router_v10_static_preflight.v1"
PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v11_static_preflight.v1"
PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v12_static_preflight.v1"
PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v14_velocity_alias_static_preflight.v1"
PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v15_radial_velocity_alias_static_preflight.v1"
PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v16_visible_radial_velocity_alias_static_preflight.v1"
PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA = "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_confirmation_static_preflight.v1"
MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA = "pointmaze_mixed_velocity_repairability_v19_static_preflight.v1"
MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA = "pointmaze_mixed_velocity_repairability_v20_budgeted_static_preflight.v1"
C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA = "pointmaze_c3_calibrated_reliability_v21_static_preflight.v1"
C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA = "pointmaze_c3_action_supported_v22_static_preflight.v1"
C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA = "pointmaze_c3_regret_budget_v23_static_preflight.v1"
C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA = "pointmaze_c3_stratified_repair_v24_static_preflight.v1"
V7_POST_EXECUTION_ALIASING_CONTRACT = {
    "representation_key": "source_history_model",
    "radius_quantile": 0.25,
    "contrast_gap_threshold": 0.05,
    "max_reported_pairs": 64,
}
SUPPORTED_FROZEN_PREFLIGHT_SCHEMAS = {
    FROZEN_PREFLIGHT_SCHEMA,
    PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
    SHARED_FUTURE_V8_PREFLIGHT_SCHEMA,
    OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
    ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
    MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
    MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
    C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
    C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
    C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
    C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
}
EXACT_SHARED_SCORER_PREFLIGHT_SCHEMAS = {
    PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
    SHARED_FUTURE_V8_PREFLIGHT_SCHEMA,
    OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
    ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
    PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
    MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
    MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
    C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
    C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
    C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
    C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
}
V6_ACTION_FAMILY = "world_fixed_random_five_pulse_matched_held_v6"
V6_TRAIN_FAMILY = "world_fixed_random_five_pulse_matched_train_v6"
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202610260, 202610261, 202610262
SPLIT_COUNTS = {"train": 64, "calibration": 48, "outcome": 32}
TRAIN_CANDIDATES, HELD_CANDIDATES = 64, 16
PULSE_COUNT, PULSE_WIDTH, TOTAL_STEPS = 5, 4, 20
PULSE_MIN, PULSE_MAX = 0.28, 0.78
HISTORY_FRAMES, FRAME_STRIDE, SOURCE_FRAME = 8, 5, 40
# Keep the legacy target named for audit/reproduction, but do not use it as
# the default for new panels: its success disk intersects the wall boundary.
LEGACY_GOAL = np.asarray([1.2, 1.2], dtype=np.float64)
DEFAULT_GOAL_QPOS = np.asarray([0.8, 0.8], dtype=np.float64)
CANDIDATE_GENERALIZATION_REGIMES = ("numeric_ood", "shared_candidate_in_support")
CANDIDATE_DESIGNS = (
    "random_matched",
    "state_interaction_pairs",
    "goal_balanced_interaction_pairs",
    "robust_goal_balanced_interaction_pairs",
    "tail_robust_goal_balanced_interaction_pairs",
    "expanded_tail_robust_goal_balanced_interaction_pairs",
    "ultra_tail_robust_goal_balanced_interaction_pairs",
    "directional_tail_robust_goal_balanced_interaction_pairs",
    "pairwise_contrast_robust_interaction_pairs",
    "pairwise_contrast_balanced_reversal_interaction_pairs",
    "pairwise_contrast_position_aliased_interaction_pairs",
    "velocity_matched_acceleration_pairs",
    "velocity_matched_braking_pairs",
)
INTERACTION_DESIGN_ROOTS, INTERACTION_PAIR_PROPOSALS = 64, 96
# The historical panels were restricted to 0..1799.  That pool is now nearly
# exhausted by prior root-disjoint panels; the backing PointMaze tensors have
# 2000 roots, so new panels may use the full state/action pool while reserving
# stored-RGB chart roots explicitly below.
STATE_ACTION_ROOT_COUNT = 2000


def validate_goal_qpos(goal_qpos: Any, *, allow_wall_sensitive_goal: bool = False) -> np.ndarray:
    """Validate an explicit task target and reject accidental legacy reuse."""
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("goal_qpos must be a finite [2] vector")
    if np.allclose(goal, LEGACY_GOAL, rtol=0.0, atol=1.0e-12) and not allow_wall_sensitive_goal:
        raise ValueError(
            "[1.2, 1.2] is the wall-sensitive legacy target; pass "
            "allow_wall_sensitive_goal=True only for explicit reproduction"
        )
    return goal.copy()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def causal_history8(frame: int, actions: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Return eight causal frames and the seven complete pre-candidate blocks."""

    start = int(frame) - (HISTORY_FRAMES - 1) * FRAME_STRIDE
    indices = list(range(start, int(frame) + 1, FRAME_STRIDE))
    history = np.asarray(actions, dtype=np.float64)[start:int(frame)]
    expected = ((HISTORY_FRAMES - 1) * FRAME_STRIDE, 2)
    if start < 0 or len(indices) != HISTORY_FRAMES or history.shape != expected:
        raise ValueError("v6 requires eight causal frames and seven complete pre-candidate action blocks")
    return indices, history.copy()


def v6_matched_action_sequences(*, seed: int, count: int) -> np.ndarray:
    """Create a numeric-OOD, support-matched bank of five four-step pulses."""

    if count < 8:
        raise ValueError("v6 requires at least eight action candidates")
    rng = np.random.default_rng(int(seed))
    values: list[np.ndarray] = []
    for _ in range(int(count)):
        angles = rng.uniform(-np.pi, np.pi, size=PULSE_COUNT)
        magnitudes = rng.uniform(PULSE_MIN, PULSE_MAX, size=PULSE_COUNT)
        pulses = np.stack((np.cos(angles), np.sin(angles)), axis=1) * magnitudes[:, None]
        value = np.repeat(pulses, PULSE_WIDTH, axis=0)
        if value.shape != (TOTAL_STEPS, 2) or float(np.abs(value).max()) > 1.0:
            raise RuntimeError("v6 action exceeds PointMaze control support")
        values.append(value)
    bank = np.stack(values).astype(np.float64)
    if len({sequence_hash(item) for item in bank}) != len(bank):
        raise RuntimeError("v6 action bank has duplicate hashes")
    return bank


def temporal_memory_candidate_banks(
    *, held_seed: int, train_seed: int, regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held and training banks under an explicitly declared support regime.

    The shared-bank regime is deliberately not an action-OOD benchmark: its
    held sequences are an exact subset of the source action templates.  It is
    the necessary control for asking whether a root-disjoint visual history
    changes the ranking of already supported candidate actions.
    """

    if regime not in CANDIDATE_GENERALIZATION_REGIMES:
        raise ValueError(f"unsupported candidate generalization regime {regime!r}")
    held = v6_matched_action_sequences(seed=int(held_seed), count=HELD_CANDIDATES)
    if regime == "numeric_ood":
        train = v6_matched_action_sequences(seed=int(train_seed), count=TRAIN_CANDIDATES)
        if set(sequence_hash(value) for value in held) & set(sequence_hash(value) for value in train):
            raise RuntimeError("numeric-OOD action bank unexpectedly overlaps source actions")
        return held, train

    # Keep the total source intervention count fixed at 64 while guaranteeing
    # exact candidate support for all 16 held actions.
    extra = v6_matched_action_sequences(seed=int(train_seed), count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(value) for value in held) & set(sequence_hash(value) for value in extra):
        raise RuntimeError("shared-bank supplemental actions unexpectedly overlap held actions")
    train = np.concatenate((held, extra), axis=0)
    if len({sequence_hash(value) for value in train}) != TRAIN_CANDIDATES:
        raise RuntimeError("shared-bank source actions are not unique")
    return held, train


def state_interaction_pair_proposals(*, seed: int, count: int = INTERACTION_PAIR_PROPOSALS) -> np.ndarray:
    """Create strict legacy-summary collisions before seeing task outcomes.

    Each pair has the same five pulse vectors, total energy, mean, variance,
    first pulse, last pulse, and horizon.  Only the three interior pulses are
    permuted.  The historical mean/std/first/last action encoder therefore
    maps both candidates to the same feature, while an ordered action encoder
    can distinguish them.
    """
    if count < HELD_CANDIDATES // 2:
        raise ValueError("interaction proposal count is too small")
    rng = np.random.default_rng(int(seed))
    pairs: list[np.ndarray] = []
    for _ in range(int(count)):
        angles = rng.uniform(-np.pi, np.pi, size=PULSE_COUNT)
        magnitudes = rng.uniform(PULSE_MIN, PULSE_MAX, size=PULSE_COUNT)
        pulses = np.stack((np.cos(angles), np.sin(angles)), axis=1) * magnitudes[:, None]
        interior = rng.permutation(np.arange(1, PULSE_COUNT - 1))
        if np.array_equal(interior, np.arange(1, PULSE_COUNT - 1)):
            interior = np.roll(interior, 1)
        permutation = np.concatenate((np.asarray([0]), interior, np.asarray([PULSE_COUNT - 1])))
        pair = np.stack((
            np.repeat(pulses, PULSE_WIDTH, axis=0),
            np.repeat(pulses[permutation], PULSE_WIDTH, axis=0),
        ))
        if pair.shape != (2, TOTAL_STEPS, 2) or float(np.abs(pair).max()) > 1.0:
            raise RuntimeError("state-interaction proposal exceeds PointMaze support")
        if not np.isclose(np.square(pair[0]).sum(), np.square(pair[1]).sum(), rtol=0.0, atol=1e-12):
            raise RuntimeError("state-interaction pair is not energy matched")
        if not (
            np.allclose(pair[0].mean(axis=0), pair[1].mean(axis=0), rtol=0.0, atol=1e-12)
            and np.allclose(pair[0].std(axis=0), pair[1].std(axis=0), rtol=0.0, atol=1e-12)
            and np.allclose(pair[0, 0], pair[1, 0], rtol=0.0, atol=1e-12)
            and np.allclose(pair[0, -1], pair[1, -1], rtol=0.0, atol=1e-12)
            and not np.array_equal(pair[0], pair[1])
        ):
            raise RuntimeError("state-interaction pair must collide under the legacy action summary")
        pairs.append(pair)
    result = np.stack(pairs).astype(np.float64)
    hashes = [sequence_hash(sequence) for pair in result for sequence in pair]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("state-interaction proposals contain duplicate sequences")
    return result


def directional_interaction_pair_proposals(*, seed: int, count: int = 384) -> np.ndarray:
    """Create legacy-summary collisions with stratified net action directions.

    Random interaction pairs can leave the declared goal's best two actions
    nearly tied on a non-trivial root tail.  This construction keeps the same
    five-by-four pulse support, energy matching, and interior-permutation
    collision, but stratifies each pair around a distinct net direction.
    Consequently the candidate-bank designer can trade off directional task
    headroom against within-pair temporal-order sensitivity without reading
    train, calibration, or outcome roots.
    """
    if count < HELD_CANDIDATES // 2:
        raise ValueError("directional interaction proposal count is too small")
    rng = np.random.default_rng(int(seed))
    pairs: list[np.ndarray] = []
    for index in range(int(count)):
        base_angle = 2.0 * np.pi * float(index) / float(count)
        # A bounded angular spread preserves a strong net direction while
        # retaining enough interior-order variation for a genuine interaction
        # contrast.  High in-support magnitudes avoid a weak-action bank.
        angles = base_angle + rng.uniform(-0.55, 0.55, size=PULSE_COUNT)
        magnitudes = rng.uniform(0.62, PULSE_MAX, size=PULSE_COUNT)
        pulses = np.stack((np.cos(angles), np.sin(angles)), axis=1) * magnitudes[:, None]
        interior = rng.permutation(np.arange(1, PULSE_COUNT - 1))
        if np.array_equal(interior, np.arange(1, PULSE_COUNT - 1)):
            interior = np.roll(interior, 1)
        permutation = np.concatenate((np.asarray([0]), interior, np.asarray([PULSE_COUNT - 1])))
        pair = np.stack((
            np.repeat(pulses, PULSE_WIDTH, axis=0),
            np.repeat(pulses[permutation], PULSE_WIDTH, axis=0),
        ))
        if pair.shape != (2, TOTAL_STEPS, 2) or float(np.abs(pair).max()) > 1.0:
            raise RuntimeError("directional interaction proposal exceeds PointMaze control support")
        if not (
            np.allclose(pair[0].mean(axis=0), pair[1].mean(axis=0), rtol=0.0, atol=1e-12)
            and np.allclose(pair[0].std(axis=0), pair[1].std(axis=0), rtol=0.0, atol=1e-12)
            and np.allclose(pair[0, 0], pair[1, 0], rtol=0.0, atol=1e-12)
            and np.allclose(pair[0, -1], pair[1, -1], rtol=0.0, atol=1e-12)
            and not np.array_equal(pair[0], pair[1])
        ):
            raise RuntimeError("directional interaction pair must collide under the legacy action summary")
        pairs.append(pair)
    result = np.stack(pairs).astype(np.float64)
    hashes = [sequence_hash(sequence) for pair in result for sequence in pair]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("directional interaction proposals contain duplicate sequences")
    return result


def screen_state_interaction_pairs(
    source_states: np.ndarray,
    proposals: np.ndarray,
    *,
    simulator: Any = _simulate_sequence,
    pair_count: int = HELD_CANDIDATES // 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select interaction-rich pairs using design roots and effects only.

    The screening objective never receives a goal or task cost.  It favors
    terminal-effect differences whose *mean* is small but whose variation over
    design roots is high, the required counterfactual signature for a
    state-action interaction that an action-only mean effect cannot explain.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("interaction design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("interaction proposals must be [pairs, 2, 20, 2]")
    if pair_count < 1 or pair_count > len(pairs):
        raise ValueError("invalid requested interaction pair count")
    deltas = np.empty((len(pairs), len(states), 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for state_index, state in enumerate(states):
            left, right = simulator(state, pair[0]), simulator(state, pair[1])
            deltas[pair_index, state_index] = np.asarray(left[:2]) - np.asarray(right[:2])
    mean_norm = np.linalg.norm(deltas.mean(axis=1), axis=1)
    centered = deltas - deltas.mean(axis=1, keepdims=True)
    interaction_scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2)))
    # Retain only the lower half of mean-effect magnitudes before ranking by
    # interaction scale.  The deterministic tie break binds selection to the
    # proposal index rather than any task outcome.
    low_mean_count = max(int(pair_count), len(pairs) // 2)
    eligible = sorted(range(len(pairs)), key=lambda index: (float(mean_norm[index]), int(index)))[:low_mean_count]
    mean_cutoff = float(max(mean_norm[index] for index in eligible))
    chosen = sorted(eligible, key=lambda index: (-float(interaction_scale[index]), int(index)))[:pair_count]
    selected = pairs[np.asarray(chosen, dtype=np.int64)].reshape(pair_count * 2, TOTAL_STEPS, 2)
    report = {
        "screening_role": "design-root terminal-effect interaction only; no goal or task-cost access",
        "proposal_count": int(len(pairs)), "design_root_count": int(len(states)),
        "mean_effect_norm_cutoff": mean_cutoff, "selected_pair_indices": [int(index) for index in chosen],
        "selected_pair_mean_effect_norm": [float(mean_norm[index]) for index in chosen],
        "selected_pair_interaction_scale": [float(interaction_scale[index]) for index in chosen],
        "selection_order": "ascending mean-effect eligibility then descending interaction scale, proposal-index tie break",
    }
    return selected, report


def state_interaction_candidate_banks(
    source_states: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return selected interaction actions plus an in-support source bank."""
    proposals = state_interaction_pair_proposals(seed=proposal_seed)
    held, report = screen_state_interaction_pairs(source_states, proposals, simulator=simulator)
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("interaction supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    report = {
        **report, "candidate_design": "state_interaction_pairs",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }
    return held, train, report


def screen_goal_balanced_interaction_pairs(
    source_states: np.ndarray, proposals: np.ndarray, goal_qpos: np.ndarray, *,
    simulator: Any = _simulate_sequence, pair_count: int = HELD_CANDIDATES // 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Screen a fixed-goal bank on disjoint design roots without test access.

    V6 interaction screening was deliberately task agnostic.  That can leave a
    globally dominant candidate in a fixed-goal benchmark, turning history
    into an unnecessary correction to a strong action prior.  This variant
    still requires equal-energy, state-interaction-rich pairs, but chooses the
    pair *set* to maximise the entropy of design-root oracle winners for the
    already declared task target.  It never sees train, calibration, or
    outcome roots and is therefore a candidate-bank design intervention, not
    a test-time selector.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("goal-balanced design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("goal-balanced proposals must be [pairs, 2, 20, 2]")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("goal-balanced design needs a finite declared [x, y] goal")
    if pair_count < 1 or pair_count > len(pairs):
        raise ValueError("invalid goal-balanced pair count")
    terminals = np.empty((len(pairs), len(states), 2, 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for state_index, state in enumerate(states):
            terminals[pair_index, state_index, 0] = np.asarray(simulator(state, pair[0]), dtype=np.float64)[:2]
            terminals[pair_index, state_index, 1] = np.asarray(simulator(state, pair[1]), dtype=np.float64)[:2]
    costs = np.linalg.norm(terminals - goal.reshape(1, 1, 1, 2), axis=-1)
    deltas = terminals[:, :, 0] - terminals[:, :, 1]
    mean_norm = np.linalg.norm(deltas.mean(axis=1), axis=1)
    centered = deltas - deltas.mean(axis=1, keepdims=True)
    interaction_scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2)))
    low_mean_count = max(int(pair_count), len(pairs) // 2)
    eligible = sorted(range(len(pairs)), key=lambda index: (float(mean_norm[index]), int(index)))[:low_mean_count]

    def score(indices: list[int]) -> tuple[float, float, int]:
        candidate_costs = costs[np.asarray(indices, dtype=np.int64)].transpose(1, 0, 2).reshape(len(states), -1)
        winners = np.argmin(candidate_costs, axis=1)
        counts = np.bincount(winners, minlength=candidate_costs.shape[1]).astype(np.float64)
        probability = counts[counts > 0.0] / float(len(states))
        entropy = float(-(probability * np.log(probability)).sum())
        normalized = entropy / float(np.log(candidate_costs.shape[1])) if candidate_costs.shape[1] > 1 else 0.0
        return normalized, float(counts.max() / len(states)), int(np.count_nonzero(counts))

    selected: list[int] = []
    remaining = list(eligible)
    while len(selected) < pair_count:
        chosen = max(
            remaining,
            key=lambda index: (
                score([*selected, index])[0],
                -score([*selected, index])[1],
                score([*selected, index])[2],
                float(interaction_scale[index]),
                -float(mean_norm[index]),
                -int(index),
            ),
        )
        selected.append(int(chosen)); remaining.remove(chosen)
    entropy, dominant_fraction, winner_count = score(selected)
    chosen_indices = np.asarray(selected, dtype=np.int64)
    # Entropy prevents a globally dominant winner, but it does not establish
    # that choosing the winner matters.  Measure that oracle headroom only on
    # these disjoint candidate-design roots; downstream splits stay unseen.
    candidate_costs = costs[chosen_indices].transpose(1, 0, 2).reshape(len(states), -1)
    ordered_costs = np.sort(candidate_costs, axis=1)
    top2_gap = ordered_costs[:, 1] - ordered_costs[:, 0]
    cost_spread = ordered_costs[:, -1] - ordered_costs[:, 0]
    headroom = {
        "task_cost": "terminal_euclidean_distance_to_declared_goal",
        "root_count": int(len(states)),
        "candidate_count": int(candidate_costs.shape[1]),
        "best_second_cost_gap": {
            "minimum": float(np.min(top2_gap)), "q10": float(np.quantile(top2_gap, 0.10)),
            "q25": float(np.quantile(top2_gap, 0.25)), "median": float(np.median(top2_gap)),
            "mean": float(np.mean(top2_gap)), "q75": float(np.quantile(top2_gap, 0.75)),
            "maximum": float(np.max(top2_gap)), "ge_0p003_fraction": float(np.mean(top2_gap >= 0.003)),
        },
        "candidate_cost_spread": {
            "minimum": float(np.min(cost_spread)), "q10": float(np.quantile(cost_spread, 0.10)),
            "median": float(np.median(cost_spread)), "mean": float(np.mean(cost_spread)),
            "maximum": float(np.max(cost_spread)),
        },
    }
    return pairs[chosen_indices].reshape(pair_count * 2, TOTAL_STEPS, 2), {
        "screening_role": "disjoint-design-root, declared-goal winner-balance plus terminal-effect interaction; no train/calibration/outcome access",
        "declared_goal_qpos": goal.tolist(), "proposal_count": int(len(pairs)), "eligible_pair_count": int(len(eligible)),
        "design_root_count": int(len(states)), "selected_pair_indices": [int(index) for index in selected],
        "selected_pair_mean_effect_norm": [float(mean_norm[index]) for index in selected],
        "selected_pair_interaction_scale": [float(interaction_scale[index]) for index in selected],
        "design_winner_entropy_normalized": entropy, "design_dominant_winner_fraction": dominant_fraction,
        "design_distinct_winner_count": winner_count,
        "design_oracle_headroom": headroom,
        "selection_order": "greedy max normalized oracle-winner entropy, min dominance, max interaction scale, min mean effect, proposal-index tie break",
    }


def goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build an in-support, task-balanced bank while retaining interaction pairs."""
    proposals = state_interaction_pair_proposals(seed=proposal_seed)
    held, report = screen_goal_balanced_interaction_pairs(source_states, proposals, goal_qpos, simulator=simulator)
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("goal-balanced supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "goal_balanced_interaction_pairs",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def _goal_headroom_metrics(costs: np.ndarray) -> dict[str, float | int]:
    """Summarise candidate usefulness on one disjoint design-root fold.

    The metrics deliberately use only a declared goal and simulated terminal
    states.  They are candidate-bank *design* diagnostics, never a fit-time or
    test-time action selector.
    """
    matrix = np.asarray(costs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("goal-headroom metrics need at least two roots and candidates")
    if not np.isfinite(matrix).all():
        raise ValueError("goal-headroom costs must be finite")
    winners = np.argmin(matrix, axis=1)
    counts = np.bincount(winners, minlength=matrix.shape[1]).astype(np.float64)
    probability = counts[counts > 0.0] / float(matrix.shape[0])
    entropy = float(-(probability * np.log(probability)).sum())
    normalized_entropy = entropy / float(np.log(matrix.shape[1]))
    ordered = np.sort(matrix, axis=1)
    top2_gap = ordered[:, 1] - ordered[:, 0]
    spread = ordered[:, -1] - ordered[:, 0]
    return {
        "root_count": int(matrix.shape[0]),
        "candidate_count": int(matrix.shape[1]),
        "winner_entropy_normalized": normalized_entropy,
        "dominant_winner_fraction": float(counts.max() / matrix.shape[0]),
        "distinct_winner_count": int(np.count_nonzero(counts)),
        "distinct_winner_fraction": float(np.count_nonzero(counts) / matrix.shape[1]),
        "best_second_gap_mean": float(np.mean(top2_gap)),
        "best_second_gap_q10": float(np.quantile(top2_gap, 0.10)),
        "best_second_gap_ge_0p003_fraction": float(np.mean(top2_gap >= 0.003)),
        "candidate_cost_spread_mean": float(np.mean(spread)),
        "candidate_cost_spread_q10": float(np.quantile(spread, 0.10)),
    }


def screen_robust_goal_balanced_interaction_pairs(
    source_states: np.ndarray, proposals: np.ndarray, goal_qpos: np.ndarray, *,
    simulator: Any = _simulate_sequence, pair_count: int = HELD_CANDIDATES // 2,
    prioritize_tail_gap: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a fixed bank whose declared-goal headroom holds in two design folds.

    A single design-root aggregate can accidentally select a bank with high
    winner entropy only because of idiosyncratic roots.  That is exactly the
    failure mode behind the V7 result: history-action interaction was present,
    but the held bank again had a dominant action prior.  This pre-outcome
    intervention selects pairs by the *worst* of two deterministic, disjoint
    design-root folds.  It retains the state-action interaction tie-break but
    removes the old low-mean-effect filter, which otherwise suppresses the
    directional actions needed for non-trivial goal headroom.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("robust goal-balanced design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("robust goal-balanced proposals must be [pairs, 2, 20, 2]")
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("robust goal-balanced design needs a finite declared [x, y] goal")
    if len(states) < 4:
        raise ValueError("robust goal-balanced design needs at least four disjoint roots")
    if pair_count < 1 or pair_count > len(pairs):
        raise ValueError("invalid requested robust interaction pair count")

    terminals = np.empty((len(pairs), len(states), 2, 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for state_index, state in enumerate(states):
            terminals[pair_index, state_index, 0] = np.asarray(simulator(state, pair[0]), dtype=np.float64)[:2]
            terminals[pair_index, state_index, 1] = np.asarray(simulator(state, pair[1]), dtype=np.float64)[:2]
    candidate_costs = np.linalg.norm(terminals - goal.reshape(1, 1, 1, 2), axis=-1)
    deltas = terminals[:, :, 0] - terminals[:, :, 1]
    centered = deltas - deltas.mean(axis=1, keepdims=True)
    interaction_scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2)))
    # The folds are fixed before candidate selection and contain disjoint root
    # indices.  Their order follows the already frozen interaction-root list.
    folds = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(len(states)), 2))

    def metrics(indices: list[int], fold: np.ndarray) -> dict[str, float | int]:
        costs = candidate_costs[np.asarray(indices, dtype=np.int64)][:, fold, :]
        return _goal_headroom_metrics(costs.transpose(1, 0, 2).reshape(len(fold), -1))

    def objective(indices: list[int]) -> tuple[float, float, float, float, float]:
        fold_metrics = [metrics(indices, fold) for fold in folds]
        # Lexicographic worst-fold criterion: keep action winners diverse,
        # then ensure selection matters, then retain interaction richness.
        # Some panels have a preregistered lower-tail gap gate.  They must
        # optimise that exact design quantity rather than rely on mean-gap
        # headroom that can hide a nearly tied tail.
        gap_key = "best_second_gap_q10" if prioritize_tail_gap else "best_second_gap_mean"
        return (
            min(float(item["winner_entropy_normalized"]) for item in fold_metrics),
            min(float(item["distinct_winner_fraction"]) for item in fold_metrics),
            min(float(item[gap_key]) for item in fold_metrics),
            min(float(item["candidate_cost_spread_mean"]) for item in fold_metrics),
            float(np.mean(interaction_scale[np.asarray(indices, dtype=np.int64)])),
        )

    selected: list[int] = []
    remaining = list(range(len(pairs)))
    while len(selected) < pair_count:
        # Bind an exact proposal-index tie break to the *candidate currently
        # under consideration*.  Using an aggregate statistic of ``selected``
        # would make ties depend on selection history instead of the declared
        # proposal identity.
        chosen = max(remaining, key=lambda index: (*objective([*selected, int(index)]), -int(index)))
        selected.append(int(chosen)); remaining.remove(chosen)
    per_fold = [metrics(selected, fold) for fold in folds]
    all_metrics = _goal_headroom_metrics(
        candidate_costs[np.asarray(selected, dtype=np.int64)].transpose(1, 0, 2).reshape(len(states), -1)
    )
    chosen_indices = np.asarray(selected, dtype=np.int64)
    return pairs[chosen_indices].reshape(pair_count * 2, TOTAL_STEPS, 2), {
        "screening_role": "two disjoint design-root folds, declared-goal worst-fold winner balance plus headroom and terminal-effect interaction; no train/calibration/outcome access",
        "declared_goal_qpos": goal.tolist(), "proposal_count": int(len(pairs)), "design_root_count": int(len(states)),
        "design_fold_root_counts": [int(len(fold)) for fold in folds],
        "selected_pair_indices": [int(index) for index in selected],
        "selected_pair_interaction_scale": [float(interaction_scale[index]) for index in selected],
        "worst_fold_objective": list(objective(selected)),
        "design_fold_headroom": per_fold,
        "design_all_root_headroom": all_metrics,
        "selection_order": (
            "greedy max worst-fold winner entropy, distinct-winner fraction, "
            f"{'best-second-gap-q10' if prioritize_tail_gap else 'best-second-gap-mean'}, "
            "cost spread, interaction scale, proposal-index tie break"
        ),
    }


def robust_goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a task-balanced in-support bank with two-fold design robustness."""
    proposals = state_interaction_pair_proposals(seed=proposal_seed)
    held, report = screen_robust_goal_balanced_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("robust goal-balanced supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "robust_goal_balanced_interaction_pairs",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def tail_robust_goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a robust bank whose design objective matches a q10 gap gate."""

    proposals = state_interaction_pair_proposals(seed=proposal_seed)
    held, report = screen_robust_goal_balanced_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator, prioritize_tail_gap=True,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("tail-robust supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "tail_robust_goal_balanced_interaction_pairs",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def expanded_tail_robust_goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence, proposal_count: int = 384,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use a larger pre-outcome proposal set for a fixed q10-gap design gate."""

    if int(proposal_count) < INTERACTION_PAIR_PROPOSALS:
        raise ValueError("expanded tail-robust design cannot reduce proposal coverage")
    proposals = state_interaction_pair_proposals(seed=proposal_seed, count=int(proposal_count))
    held, report = screen_robust_goal_balanced_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator, prioritize_tail_gap=True,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("expanded tail-robust supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "expanded_tail_robust_goal_balanced_interaction_pairs",
        "proposal_coverage": "384 deterministic state-interaction pairs before disjoint-design q10 selection",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def ultra_tail_robust_goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Predeclare denser proposal coverage for the fixed tail-headroom gate.

    This is not an outcome-adaptive action search: selection sees only
    disjoint interaction-design roots and the declared terminal-goal cost.
    Its sole purpose is to make the bank-construction objective match the q10
    oracle-headroom condition that the final panel must pass before model fit.
    """
    held, train, report = expanded_tail_robust_goal_balanced_interaction_candidate_banks(
        source_states, goal_qpos, proposal_seed=proposal_seed, supplemental_seed=supplemental_seed,
        simulator=simulator, proposal_count=1536,
    )
    return held, train, {
        **report,
        "candidate_design": "ultra_tail_robust_goal_balanced_interaction_pairs",
        "proposal_coverage": "1536 deterministic state-interaction pairs before disjoint-design q10 selection",
    }


def screen_pairwise_contrast_robust_interaction_pairs(
    source_states: np.ndarray, proposals: np.ndarray, goal_qpos: np.ndarray, *,
    simulator: Any = _simulate_sequence, pair_count: int = HELD_CANDIDATES // 2,
    contrast_margin: float = 0.003,
    minimum_fold_sign_balance: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select pairs by the C1 contrast, rather than an unrelated full-bank argmin.

    The former ultra-tail bank optimized a 16-way best/second cost gap.  V11
    evaluates adjacent summary-collision pairs, so that objective can retain
    individually near-tied pairs.  This selector instead enforces robust
    pairwise contrast on two disjoint design folds before any model is fit.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("pairwise contrast design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("pairwise contrast proposals must be [pairs, 2, 20, 2]")
    if (
        goal.shape != (2,) or not np.isfinite(goal).all() or float(contrast_margin) <= 0.0
        or not 0.0 <= float(minimum_fold_sign_balance) <= 0.5
    ):
        raise ValueError("pairwise contrast design needs finite goal, positive margin, and balance in [0, 0.5]")
    if len(states) < 4 or not 1 <= int(pair_count) <= len(pairs):
        raise ValueError("pairwise contrast design needs four roots and a valid pair count")

    costs = np.empty((len(pairs), len(states), 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for root_index, state in enumerate(states):
            for action_index, action in enumerate(pair):
                terminal = np.asarray(simulator(state, action), dtype=np.float64)
                costs[pair_index, root_index, action_index] = float(np.linalg.norm(terminal[:2] - goal))
    signed = costs[:, :, 0] - costs[:, :, 1]
    absolute = np.abs(signed)
    folds = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(len(states)), 2))
    fold_q10 = np.asarray(
        [[np.quantile(absolute[pair_index, fold], 0.10) for fold in folds] for pair_index in range(len(pairs))],
        dtype=np.float64,
    )
    fold_coverage = np.asarray(
        [[np.mean(absolute[pair_index, fold] >= float(contrast_margin)) for fold in folds] for pair_index in range(len(pairs))],
        dtype=np.float64,
    )
    sign_mixed = np.asarray(
        [bool(np.any(signed[index] < 0.0) and np.any(signed[index] > 0.0)) for index in range(len(pairs))],
        dtype=bool,
    )
    sign_balance = np.asarray(
        [min(float(np.mean(signed[index] < 0.0)), float(np.mean(signed[index] > 0.0))) for index in range(len(pairs))],
        dtype=np.float64,
    )
    fold_sign_balance = np.asarray(
        [
            [
                min(float(np.mean(signed[index, fold] < 0.0)), float(np.mean(signed[index, fold] > 0.0)))
                for fold in folds
            ]
            for index in range(len(pairs))
        ],
        dtype=np.float64,
    )
    feasible = np.asarray(
        [
            bool(sign_mixed[index])
            and float(np.min(fold_q10[index])) >= float(contrast_margin)
            and float(np.min(fold_coverage[index])) >= 0.80
            and float(np.min(fold_sign_balance[index])) >= float(minimum_fold_sign_balance)
            for index in range(len(pairs))
        ],
        dtype=bool,
    )
    ordering = sorted(
        range(len(pairs)),
        key=lambda index: (
            bool(sign_mixed[index]), float(np.min(fold_q10[index])),
            float(np.min(fold_coverage[index])), float(sign_balance[index]),
            float(np.mean(absolute[index])), -int(index),
        ),
        reverse=True,
    )
    selected = [int(index) for index in ordering if bool(feasible[index])][:int(pair_count)]
    if len(selected) != int(pair_count):
        raise ValueError(
            "pairwise contrast design has insufficient pairs meeting the hard two-fold "
            f"sign-balance gate: need {int(pair_count)}, found {int(np.sum(feasible))} "
            f"at minimum_fold_sign_balance={float(minimum_fold_sign_balance):.6f}"
        )
    selected_absolute = absolute[np.asarray(selected, dtype=np.int64)]
    selected_signed = signed[np.asarray(selected, dtype=np.int64)]
    per_pair = []
    for index in selected:
        per_pair.append({
            "proposal_index": int(index),
            "minimum_fold_absolute_gap_q10": float(np.min(fold_q10[index])),
            "minimum_fold_gap_ge_margin_fraction": float(np.min(fold_coverage[index])),
            "sign_mixed_across_design_roots": bool(sign_mixed[index]),
            "sign_balance": float(sign_balance[index]),
            "minimum_fold_sign_balance": float(np.min(fold_sign_balance[index])),
        })
    return pairs[np.asarray(selected, dtype=np.int64)].reshape(int(pair_count) * 2, TOTAL_STEPS, 2), {
        "screening_role": "two disjoint design-root folds; individual pairwise q10 contrast, coverage, and state-conditional sign reversal; no train/calibration/outcome access",
        "declared_goal_qpos": goal.tolist(), "proposal_count": int(len(pairs)),
        "design_root_count": int(len(states)), "design_fold_root_counts": [int(len(fold)) for fold in folds],
        "contrast_margin": float(contrast_margin), "selected_pair_indices": selected,
        "minimum_fold_sign_balance_requirement": float(minimum_fold_sign_balance),
        "feasible_pair_count": int(np.sum(feasible)),
        "selected_pair_contract": per_pair,
        "selected_pooled_absolute_gap_q10": float(np.quantile(selected_absolute.reshape(-1), 0.10)),
        "selected_gap_ge_margin_fraction": float(np.mean(selected_absolute >= float(contrast_margin))),
        "selected_sign_mixed_pair_count": int(sum(
            bool(np.any(selected_signed[index] < 0.0) and np.any(selected_signed[index] > 0.0))
            for index in range(len(selected))
        )),
        "selection_order": "hard sign-mixed/q10/coverage/two-fold-balance admission, then minimum-fold pairwise-q10, minimum-fold margin coverage, sign balance, mean contrast, proposal-index tie break",
    }


def pairwise_contrast_robust_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create an exact-support bank selected for V11's pairwise C1 endpoint."""
    proposals = state_interaction_pair_proposals(seed=proposal_seed, count=1536)
    held, report = screen_pairwise_contrast_robust_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("pairwise-contrast supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "pairwise_contrast_robust_interaction_pairs",
        "proposal_coverage": "1536 deterministic state-interaction pairs before two-fold pairwise-contrast selection",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def pairwise_contrast_balanced_reversal_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence, minimum_fold_sign_balance: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a pairwise C1 bank with hard, two-fold winner-reversal balance.

    V11 showed that merely requiring both winners to occur lets an action-only
    ranker exploit a dominant winner. This constructor rejects that shortcut
    before fitting: every selected pair must have each winner on at least the
    declared fraction of each independent design fold.
    """
    proposals = state_interaction_pair_proposals(seed=proposal_seed, count=1536)
    held, report = screen_pairwise_contrast_robust_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator,
        minimum_fold_sign_balance=float(minimum_fold_sign_balance),
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("balanced-reversal supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "pairwise_contrast_balanced_reversal_interaction_pairs",
        "proposal_coverage": "1536 deterministic interaction pairs before hard two-fold balanced-reversal admission",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def screen_pairwise_contrast_position_aliased_interaction_pairs(
    source_states: np.ndarray, proposals: np.ndarray, goal_qpos: np.ndarray, *,
    simulator: Any = _simulate_sequence, pair_count: int = HELD_CANDIDATES // 2,
    contrast_margin: float = 0.003, minimum_fold_sign_balance: float = 0.40,
    maximum_crossfold_position_only_sign_accuracy: float = 0.625,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select C1 pairs whose contrast is hidden from a position-only effect proxy.

    The V12 result showed that balancing the realised winner is insufficient:
    an action-only effect predictor still receives source position and goal at
    task-cost readout. For each design fold, this screen fits only the other
    fold's *mean terminal displacement per action*, combines it with the
    target fold's position/goal, and rejects pairs whose winner is thereby
    too predictable. It does not fit a learned visual model or read any
    calibration/outcome root.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("position-aliased design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("position-aliased proposals must be [pairs, 2, 20, 2]")
    if (
        goal.shape != (2,) or not np.isfinite(goal).all() or float(contrast_margin) <= 0.0
        or not 0.0 <= float(minimum_fold_sign_balance) <= 0.5
        or not 0.0 <= float(maximum_crossfold_position_only_sign_accuracy) <= 1.0
    ):
        raise ValueError("position-aliased design has invalid goal, margin, balance, or proxy-accuracy threshold")
    if len(states) < 4 or not 1 <= int(pair_count) <= len(pairs):
        raise ValueError("position-aliased design needs four roots and a valid pair count")

    terminals = np.empty((len(pairs), len(states), 2, 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for root_index, state in enumerate(states):
            for action_index, action in enumerate(pair):
                terminal = np.asarray(simulator(state, action), dtype=np.float64)
                terminals[pair_index, root_index, action_index] = terminal[:2]
    costs = np.linalg.norm(terminals - goal.reshape(1, 1, 1, 2), axis=-1)
    signed = costs[:, :, 0] - costs[:, :, 1]
    absolute = np.abs(signed)
    folds = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(len(states)), 2))
    fold_q10 = np.asarray(
        [[np.quantile(absolute[pair_index, fold], 0.10) for fold in folds] for pair_index in range(len(pairs))],
        dtype=np.float64,
    )
    fold_coverage = np.asarray(
        [[np.mean(absolute[pair_index, fold] >= float(contrast_margin)) for fold in folds] for pair_index in range(len(pairs))],
        dtype=np.float64,
    )
    fold_sign_balance = np.asarray(
        [
            [min(float(np.mean(signed[index, fold] < 0.0)), float(np.mean(signed[index, fold] > 0.0))) for fold in folds]
            for index in range(len(pairs))
        ],
        dtype=np.float64,
    )
    effects = terminals - states[None, :, None, :2]
    proxy_fold_accuracy = np.empty((len(pairs), len(folds)), dtype=np.float64)
    for pair_index in range(len(pairs)):
        for fold_index, target_fold in enumerate(folds):
            reference_fold = folds[1 - fold_index]
            mean_effect = effects[pair_index, reference_fold].mean(axis=0)
            proxy_cost = np.linalg.norm(
                states[target_fold, None, :2] + mean_effect[None, :, :] - goal.reshape(1, 1, 2), axis=-1,
            )
            proxy_signed = proxy_cost[:, 0] - proxy_cost[:, 1]
            truth = signed[pair_index, target_fold]
            valid = np.abs(truth) >= float(contrast_margin)
            if not np.any(valid):
                proxy_fold_accuracy[pair_index, fold_index] = 1.0
            else:
                proxy_fold_accuracy[pair_index, fold_index] = float(np.mean(np.sign(proxy_signed[valid]) == np.sign(truth[valid])))
    proxy_maximum = np.max(proxy_fold_accuracy, axis=1)
    feasible = np.asarray(
        [
            float(np.min(fold_q10[index])) >= float(contrast_margin)
            and float(np.min(fold_coverage[index])) >= 0.80
            and float(np.min(fold_sign_balance[index])) >= float(minimum_fold_sign_balance)
            and float(proxy_maximum[index]) <= float(maximum_crossfold_position_only_sign_accuracy)
            for index in range(len(pairs))
        ],
        dtype=bool,
    )
    ordering = sorted(
        range(len(pairs)),
        key=lambda index: (
            -float(proxy_maximum[index]), float(np.min(fold_q10[index])), float(np.min(fold_coverage[index])),
            float(np.min(fold_sign_balance[index])), float(np.mean(absolute[index])), -int(index),
        ),
        reverse=True,
    )
    selected = [int(index) for index in ordering if bool(feasible[index])][:int(pair_count)]
    if len(selected) != int(pair_count):
        contrast_coverage_balance = np.asarray(
            [
                float(np.min(fold_q10[index])) >= float(contrast_margin)
                and float(np.min(fold_coverage[index])) >= 0.80
                and float(np.min(fold_sign_balance[index])) >= float(minimum_fold_sign_balance)
                for index in range(len(pairs))
            ],
            dtype=bool,
        )
        proxy_admissible = proxy_maximum <= float(maximum_crossfold_position_only_sign_accuracy)
        raise ValueError(
            "position-aliased pair design has insufficient feasible pairs: "
            f"need {int(pair_count)}, found {int(np.sum(feasible))}; "
            f"minimum_fold_sign_balance={float(minimum_fold_sign_balance):.6f}, "
            f"maximum_crossfold_position_only_sign_accuracy={float(maximum_crossfold_position_only_sign_accuracy):.6f}; "
            f"contrast_coverage_balance_count={int(np.sum(contrast_coverage_balance))}; "
            f"proxy_admissible_count={int(np.sum(proxy_admissible))}; "
            f"intersection_count={int(np.sum(feasible))}"
        )
    selected_absolute = absolute[np.asarray(selected, dtype=np.int64)]
    selected_signed = signed[np.asarray(selected, dtype=np.int64)]
    contract = [
        {
            "proposal_index": int(index), "minimum_fold_absolute_gap_q10": float(np.min(fold_q10[index])),
            "minimum_fold_gap_ge_margin_fraction": float(np.min(fold_coverage[index])),
            "minimum_fold_sign_balance": float(np.min(fold_sign_balance[index])),
            "maximum_crossfold_position_only_sign_accuracy": float(proxy_maximum[index]),
            "position_only_sign_accuracy_by_target_fold": proxy_fold_accuracy[index].astype(float).tolist(),
        }
        for index in selected
    ]
    return pairs[np.asarray(selected, dtype=np.int64)].reshape(int(pair_count) * 2, TOTAL_STEPS, 2), {
        "screening_role": "two design folds; high true contrast/balance and low cross-fold position-goal-plus-mean-effect proxy accuracy; no learned fit or calibration/outcome access",
        "declared_goal_qpos": goal.tolist(), "proposal_count": int(len(pairs)),
        "design_root_count": int(len(states)), "design_fold_root_counts": [int(len(fold)) for fold in folds],
        "contrast_margin": float(contrast_margin),
        "minimum_fold_sign_balance_requirement": float(minimum_fold_sign_balance),
        "maximum_crossfold_position_only_sign_accuracy_requirement": float(maximum_crossfold_position_only_sign_accuracy),
        "feasible_pair_count": int(np.sum(feasible)), "selected_pair_indices": selected,
        "selected_pair_contract": contract,
        "selected_pooled_absolute_gap_q10": float(np.quantile(selected_absolute.reshape(-1), 0.10)),
        "selected_gap_ge_margin_fraction": float(np.mean(selected_absolute >= float(contrast_margin))),
        "selected_sign_mixed_pair_count": int(sum(np.any(value < 0.0) and np.any(value > 0.0) for value in selected_signed)),
        "selection_order": "hard q10/coverage/two-fold-balance/position-only-proxy admission, then lowest proxy accuracy, contrast, coverage, balance, mean contrast, proposal-index tie break",
    }


def pairwise_contrast_position_aliased_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct a C1 bank designed to require history beyond task geometry."""
    proposals = state_interaction_pair_proposals(seed=proposal_seed, count=1536)
    held, report = screen_pairwise_contrast_position_aliased_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("position-aliased supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "pairwise_contrast_position_aliased_interaction_pairs",
        "proposal_coverage": "1536 deterministic interaction pairs before cross-fold position-only aliasing admission",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def velocity_opposed_acceleration_pair_proposals(*, seed: int, count: int = 1536) -> np.ndarray:
    """Propose fixed acceleration-versus-braking pairs for velocity aliases.

    Unlike the V11--V13 temporal-order stress, this family changes the net
    intervention.  That is deliberate: its estimand is whether a history can
    resolve two roots with the *same source position* but opposed velocity.
    No action identity is hidden from either arm; the action-only arm simply
    does not receive the source velocity encoded in visual history.
    """
    if int(count) < HELD_CANDIDATES // 2:
        raise ValueError("velocity-alias proposal count is too small")
    rng = np.random.default_rng(int(seed))
    pairs: list[np.ndarray] = []
    for _ in range(int(count)):
        angle = float(rng.uniform(-np.pi, np.pi))
        magnitude = float(rng.uniform(0.35, 0.85))
        direction = magnitude * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        # Two equal-duration, opposed accelerations make the dependency on
        # incoming velocity mechanically explicit while staying in support.
        pair = np.stack((
            np.repeat(direction[None, :], TOTAL_STEPS, axis=0),
            np.repeat((-direction)[None, :], TOTAL_STEPS, axis=0),
        ))
        if pair.shape != (2, TOTAL_STEPS, 2) or float(np.max(np.abs(pair))) > 1.0:
            raise RuntimeError("velocity-alias acceleration proposal exceeds PointMaze control support")
        pairs.append(pair)
    result = np.stack(pairs).astype(np.float64)
    hashes = [sequence_hash(sequence) for pair in result for sequence in pair]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("velocity-alias acceleration proposals contain duplicate sequences")
    return result


def velocity_braking_acceleration_pair_proposals(*, seed: int, count: int = 1536) -> np.ndarray:
    """Propose weak-versus-strong acceleration along the same world direction.

    At a fixed source position, the useful *magnitude* of inward control can
    reverse with incoming velocity: a root moving toward the goal may need to
    brake, while its position-matched counterpart moving away may need a
    stronger correction.  This removes the V14 failure mode where an inward
    versus outward pair was usually settled by position geometry alone.
    """
    if int(count) < HELD_CANDIDATES // 2:
        raise ValueError("velocity-braking proposal count is too small")
    rng = np.random.default_rng(int(seed))
    pairs: list[np.ndarray] = []
    for _ in range(int(count)):
        angle = float(rng.uniform(-np.pi, np.pi))
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        weak = float(rng.uniform(0.04, 0.22)) * direction
        strong = float(rng.uniform(0.60, 0.92)) * direction
        pair = np.stack((
            np.repeat(weak[None, :], TOTAL_STEPS, axis=0),
            np.repeat(strong[None, :], TOTAL_STEPS, axis=0),
        ))
        if pair.shape != (2, TOTAL_STEPS, 2) or float(np.max(np.abs(pair))) > 1.0:
            raise RuntimeError("velocity-braking proposal exceeds PointMaze control support")
        pairs.append(pair)
    result = np.stack(pairs).astype(np.float64)
    hashes = [sequence_hash(sequence) for pair in result for sequence in pair]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("velocity-braking proposals contain duplicate sequences")
    return result


def _velocity_matched_design_folds(root_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Split complete +/− velocity groups into two independent design folds."""
    if int(root_count) < 4 or int(root_count) % 4:
        raise ValueError("velocity-matched design needs an even number of pairs in two folds")
    groups = np.arange(int(root_count), dtype=np.int64).reshape(-1, 2)
    first, second = np.array_split(groups, 2)
    return tuple(np.sort(block.reshape(-1)) for block in (first, second))  # type: ignore[return-value]


def screen_velocity_matched_acceleration_pairs(
    source_states: np.ndarray, proposals: np.ndarray, goal_qpos: np.ndarray, *,
    simulator: Any = _simulate_sequence, pair_count: int = HELD_CANDIDATES // 2,
    contrast_margin: float = 0.003, minimum_fold_reversal_fraction: float = 0.75,
    maximum_crossfold_position_only_sign_accuracy: float = 0.625,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Admit pairs only when opposed incoming velocity changes their outcome.

    ``source_states`` must be ordered as complete (positive-velocity,
    negative-velocity) groups.  We keep each group intact in a fold, so no
    static proxy is trained on the exact position-matched counterpart of a
    target root.  The gate requires contrast, cross-fold rejection of a
    position/goal/global-effect chart, and direct within-group winner reversal.
    """
    states = np.asarray(source_states, dtype=np.float64)
    pairs = np.asarray(proposals, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("velocity-alias design states must be finite [roots, 4]")
    if pairs.ndim != 4 or pairs.shape[1:] != (2, TOTAL_STEPS, 2):
        raise ValueError("velocity-alias proposals must be [pairs, 2, 20, 2]")
    if (
        goal.shape != (2,) or not np.isfinite(goal).all() or float(contrast_margin) <= 0.0
        or not 0.0 <= float(minimum_fold_reversal_fraction) <= 1.0
        or not 0.0 <= float(maximum_crossfold_position_only_sign_accuracy) <= 1.0
    ):
        raise ValueError("velocity-alias design has invalid goal, margin, reversal, or proxy threshold")
    if not 1 <= int(pair_count) <= len(pairs):
        raise ValueError("velocity-alias design needs a valid selected pair count")
    folds = _velocity_matched_design_folds(len(states))
    terminals = np.empty((len(pairs), len(states), 2, 2), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        for root_index, state in enumerate(states):
            for action_index, action in enumerate(pair):
                terminals[pair_index, root_index, action_index] = np.asarray(simulator(state, action), dtype=np.float64)[:2]
    costs = np.linalg.norm(terminals - goal.reshape(1, 1, 1, 2), axis=-1)
    signed = costs[:, :, 0] - costs[:, :, 1]
    absolute = np.abs(signed)
    fold_q10 = np.asarray(
        [[np.quantile(absolute[item, fold], 0.10) for fold in folds] for item in range(len(pairs))], dtype=np.float64,
    )
    fold_coverage = np.asarray(
        [[np.mean(absolute[item, fold] >= float(contrast_margin)) for fold in folds] for item in range(len(pairs))], dtype=np.float64,
    )
    reversal_fraction = np.empty((len(pairs), len(folds)), dtype=np.float64)
    for item in range(len(pairs)):
        for fold_index, fold in enumerate(folds):
            signs = signed[item, fold].reshape(-1, 2)
            valid = (np.abs(signs[:, 0]) >= float(contrast_margin)) & (np.abs(signs[:, 1]) >= float(contrast_margin))
            reversal_fraction[item, fold_index] = (
                float(np.mean(signs[valid, 0] * signs[valid, 1] < 0.0)) if np.any(valid) else 0.0
            )
    effects = terminals - states[None, :, None, :2]
    proxy_accuracy = np.empty((len(pairs), len(folds)), dtype=np.float64)
    for item in range(len(pairs)):
        for fold_index, target_fold in enumerate(folds):
            reference_fold = folds[1 - fold_index]
            mean_effect = effects[item, reference_fold].mean(axis=0)
            proxy_cost = np.linalg.norm(
                states[target_fold, None, :2] + mean_effect[None, :, :] - goal.reshape(1, 1, 2), axis=-1,
            )
            truth = signed[item, target_fold]
            valid = np.abs(truth) >= float(contrast_margin)
            proxy_signed = proxy_cost[:, 0] - proxy_cost[:, 1]
            proxy_accuracy[item, fold_index] = float(np.mean(np.sign(proxy_signed[valid]) == np.sign(truth[valid]))) if np.any(valid) else 1.0
    proxy_maximum = np.max(proxy_accuracy, axis=1)
    feasible = np.asarray(
        [
            float(np.min(fold_q10[item])) >= float(contrast_margin)
            and float(np.min(fold_coverage[item])) >= 0.80
            and float(np.min(reversal_fraction[item])) >= float(minimum_fold_reversal_fraction)
            and float(proxy_maximum[item]) <= float(maximum_crossfold_position_only_sign_accuracy)
            for item in range(len(pairs))
        ], dtype=bool,
    )
    ordering = sorted(
        range(len(pairs)),
        key=lambda item: (
            -float(proxy_maximum[item]), float(np.min(reversal_fraction[item])), float(np.min(fold_q10[item])),
            float(np.min(fold_coverage[item])), float(np.mean(absolute[item])), -int(item),
        ), reverse=True,
    )
    selected = [int(item) for item in ordering if bool(feasible[item])][:int(pair_count)]
    if len(selected) != int(pair_count):
        contrast_reversal = np.asarray(
            [
                float(np.min(fold_q10[item])) >= float(contrast_margin)
                and float(np.min(fold_coverage[item])) >= 0.80
                and float(np.min(reversal_fraction[item])) >= float(minimum_fold_reversal_fraction)
                for item in range(len(pairs))
            ], dtype=bool,
        )
        proxy_admissible = proxy_maximum <= float(maximum_crossfold_position_only_sign_accuracy)
        raise ValueError(
            "velocity-aliased pair design has insufficient feasible pairs: "
            f"need {int(pair_count)}, found {int(np.sum(feasible))}; "
            f"contrast_coverage_reversal_count={int(np.sum(contrast_reversal))}; "
            f"proxy_admissible_count={int(np.sum(proxy_admissible))}; intersection_count={int(np.sum(feasible))}"
        )
    selected_absolute = absolute[np.asarray(selected, dtype=np.int64)]
    selected_signed = signed[np.asarray(selected, dtype=np.int64)]
    contract = [
        {
            "proposal_index": int(item), "minimum_fold_absolute_gap_q10": float(np.min(fold_q10[item])),
            "minimum_fold_gap_ge_margin_fraction": float(np.min(fold_coverage[item])),
            "minimum_fold_velocity_reversal_fraction": float(np.min(reversal_fraction[item])),
            "maximum_crossfold_position_only_sign_accuracy": float(proxy_maximum[item]),
            "position_only_sign_accuracy_by_target_fold": proxy_accuracy[item].astype(float).tolist(),
        }
        for item in selected
    ]
    return pairs[np.asarray(selected, dtype=np.int64)].reshape(int(pair_count) * 2, TOTAL_STEPS, 2), {
        "screening_role": "paired velocity-alias folds; contrast, true opposed-velocity winner reversal, and position-goal-global-effect proxy rejection before learned fitting",
        "declared_goal_qpos": goal.tolist(), "proposal_count": int(len(pairs)),
        "design_root_count": int(len(states)), "design_fold_root_counts": [int(len(fold)) for fold in folds],
        "contrast_margin": float(contrast_margin), "minimum_fold_velocity_reversal_requirement": float(minimum_fold_reversal_fraction),
        "maximum_crossfold_position_only_sign_accuracy_requirement": float(maximum_crossfold_position_only_sign_accuracy),
        "feasible_pair_count": int(np.sum(feasible)), "selected_pair_indices": selected,
        "selected_pair_contract": contract,
        "selected_pooled_absolute_gap_q10": float(np.quantile(selected_absolute.reshape(-1), 0.10)),
        "selected_gap_ge_margin_fraction": float(np.mean(selected_absolute >= float(contrast_margin))),
        "selected_sign_mixed_pair_count": int(sum(np.any(item < 0.0) and np.any(item > 0.0) for item in selected_signed)),
        "selection_order": "hard contrast/coverage/opposed-velocity-reversal/position-proxy admission, then lowest proxy accuracy, reversal, contrast, coverage, mean contrast, proposal-index tie break",
    }


def velocity_matched_acceleration_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct a held-exact bank whose winner must use velocity history."""
    proposals = velocity_opposed_acceleration_pair_proposals(seed=proposal_seed, count=1536)
    held, report = screen_velocity_matched_acceleration_pairs(source_states, proposals, goal_qpos, simulator=simulator)
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("velocity-alias supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "velocity_matched_acceleration_pairs",
        "proposal_coverage": "1536 fixed opposed-acceleration pairs before paired velocity-alias admission",
        "action_representation_stress": "action identity visible; only source velocity is withheld from the action-only arm",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def velocity_matched_braking_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct V15's exact-support weak-versus-strong velocity bank."""
    proposals = velocity_braking_acceleration_pair_proposals(seed=proposal_seed, count=1536)
    held, report = screen_velocity_matched_acceleration_pairs(source_states, proposals, goal_qpos, simulator=simulator)
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("velocity-braking supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report, "candidate_design": "velocity_matched_braking_pairs",
        "proposal_coverage": "1536 fixed weak-versus-strong acceleration pairs before paired velocity-alias admission",
        "action_representation_stress": "action identity visible; source velocity determines braking versus stronger correction",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def directional_tail_robust_goal_balanced_interaction_candidate_banks(
    source_states: np.ndarray, goal_qpos: np.ndarray, *, proposal_seed: int, supplemental_seed: int,
    simulator: Any = _simulate_sequence,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a directional, tail-robust shared-support candidate bank."""
    proposals = directional_interaction_pair_proposals(seed=proposal_seed, count=384)
    held, report = screen_robust_goal_balanced_interaction_pairs(
        source_states, proposals, goal_qpos, simulator=simulator, prioritize_tail_gap=True,
    )
    extra = v6_matched_action_sequences(seed=supplemental_seed, count=TRAIN_CANDIDATES - HELD_CANDIDATES)
    if set(sequence_hash(item) for item in held) & set(sequence_hash(item) for item in extra):
        raise RuntimeError("directional tail-robust supplemental actions overlap selected pair actions")
    train = np.concatenate((held, extra), axis=0)
    return held, train, {
        **report,
        "candidate_design": "directional_tail_robust_goal_balanced_interaction_pairs",
        "proposal_coverage": "384 deterministic direction-stratified interaction pairs before disjoint-design q10 selection",
        "action_representation_stress": "legacy_mean_std_first_last_collision_with_interior_pulse_permutation",
        "proposal_seed": int(proposal_seed), "supplemental_seed": int(supplemental_seed),
        "held_pair_hashes": [sequence_hash(item) for item in held],
    }


def _all_prior_identities(contexts: list[Path]) -> tuple[set[int], set[str], set[str], dict[str, str]]:
    """Read all supplied contexts, not only their previous outcome roots."""

    roots, hashes, templates, context_sha256 = set(), set(), set(), {}
    if not contexts:
        raise ValueError("v6 requires at least one prior context for isolation")
    for path in contexts:
        metadata, records = _load_records(path)
        if not metadata.get("candidate_family"):
            raise ValueError(f"prior context lacks candidate family: {path}")
        context_sha256[str(path.resolve())] = _sha256(path)
        design_roots = metadata.get("interaction_design_roots", [])
        if not isinstance(design_roots, list) or any(not isinstance(root, (int, np.integer)) for root in design_roots):
            raise ValueError(f"prior context has invalid interaction_design_roots: {path}")
        roots.update(int(root) for root in design_roots)
        for row_index, row in enumerate(records):
            required = ("start_episode_id", "candidate_action_hash", "candidate_template_fingerprint")
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(
                    f"prior context record lacks candidate identity fields at {path}[{row_index}]: {missing}"
                )
            roots.add(int(row["start_episode_id"]))
            hashes.add(str(row["candidate_action_hash"]))
            templates.add(str(row["candidate_template_fingerprint"]))
    if not roots or not hashes:
        raise ValueError("prior contexts must contain roots and candidate hashes")
    return roots, hashes, templates, context_sha256


def frozen_preflight_provenance(
    ledger_path: Path | None, *, dataset_root: Path, prior_context_sha256: dict[str, str],
    split_counts: dict[str, int], candidate_design: str, goal_qpos: np.ndarray,
    construction: dict[str, int | None] | None = None,
) -> dict[str, Any] | None:
    """Bind a formal V6 build to the exact pre-outcome static ledger.

    The launcher's old live ``find results`` discovery was convenient, but it
    could silently change the historical-exclusion set between preflight and a
    later CUDA launch.  A V6 result is interpretable only if its build uses the
    same context identities and bytes that passed static preflight.
    """

    if ledger_path is None:
        return None
    path = Path(ledger_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen V6 preflight ledger: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen V6 preflight ledger must be a JSON object")
    checks = value.get("checks")
    expected_contexts = value.get("prior_context_sha256")
    expected_counts = value.get("root_counts")
    candidate = value.get("candidate_bank")
    if value.get("schema") not in SUPPORTED_FROZEN_PREFLIGHT_SCHEMAS or value.get("status") != "passed_static_preflight":
        raise ValueError("frozen panel preflight ledger is not a supported passed static-preflight artifact")
    if not isinstance(checks, dict) or not checks or not all(flag is True for flag in checks.values()):
        raise ValueError("frozen V6 preflight ledger contains a failed or malformed check")
    if not isinstance(expected_contexts, dict) or expected_contexts != prior_context_sha256:
        raise ValueError("supplied V6 contexts do not exactly match the frozen preflight ledger")
    if value.get("dataset_states_sha256") != _sha256(dataset_root / "states.pth"):
        raise ValueError("PointMaze states.pth differs from the frozen V6 preflight ledger")
    if not isinstance(expected_counts, dict) or expected_counts != split_counts:
        raise ValueError("declared V6 root counts differ from the frozen preflight ledger")
    if not isinstance(candidate, dict):
        raise ValueError("frozen V6 preflight ledger lacks candidate-bank evidence")
    if candidate.get("candidate_design") != candidate_design:
        raise ValueError("V6 candidate design differs from the frozen preflight ledger")
    ledger_goal = np.asarray(candidate.get("goal_qpos"), dtype=np.float64).reshape(-1)
    if ledger_goal.shape != (2,) or not np.array_equal(ledger_goal, np.asarray(goal_qpos, dtype=np.float64)):
        raise ValueError("V6 declared goal differs from the frozen preflight ledger")
    if candidate.get("held_shape") != [HELD_CANDIDATES, TOTAL_STEPS, 2] or candidate.get("train_shape") != [TRAIN_CANDIDATES, TOTAL_STEPS, 2]:
        raise ValueError("frozen V6 preflight ledger has incompatible candidate-bank dimensions")
    # V7/V8 bind not just available roots, but exact split/action-bank seeds.
    # Historical V6 ledgers predate this field and remain readable only as
    # historical evidence.
    if value.get("schema") in {
        PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
        SHARED_FUTURE_V8_PREFLIGHT_SCHEMA,
        OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
        C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
        C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
        C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
        C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
    }:
        expected_construction = value.get("construction")
        runner_keys = (
            "root_seed", "held_action_seed", "train_action_seed",
            "root_id_min", "root_id_max_exclusive",
        )
        if (
            construction is None or not isinstance(expected_construction, dict)
            or any(expected_construction.get(key) != construction.get(key) for key in runner_keys)
        ):
            raise ValueError("construction seeds or declared root range differ from the frozen preflight ledger")
        # Extension generation occurs before this panel builder and is bound by
        # the immutable preflight's dataset hash.  When a newer ledger also
        # declares its extension seed, bind it to the on-disk manifest here
        # rather than requiring an otherwise unused CLI parameter.
        if "extension_seed" in expected_construction:
            manifest_path = dataset_root / "extension_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("frozen preflight extension seed requires a readable extension manifest") from error
            if manifest.get("seed") != expected_construction["extension_seed"]:
                raise ValueError("extension manifest seed differs from the frozen preflight ledger")
        if "velocity_matched_pool_seed" in expected_construction:
            manifest_path = dataset_root / "velocity_matched_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("frozen velocity-matched preflight requires a readable pool manifest") from error
            if manifest.get("seed") != expected_construction["velocity_matched_pool_seed"]:
                raise ValueError("velocity-matched pool seed differs from the frozen preflight ledger")
        pairwise_preflight = value.get("schema") in {
            PAIRWISE_CONTRAST_V11_PREFLIGHT_SCHEMA,
            PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA,
            PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
            PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
            PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
            PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
            MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
            MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
            C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
            C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
            C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
        }
        if (
            candidate.get("shared_candidate_in_support") is not True
            or candidate.get("held_train_action_hash_overlap_count") != candidate.get("held_action_hash_count")
            or candidate.get("held_action_hash_count") != HELD_CANDIDATES
            or candidate.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank"
            or candidate.get("scorer_training_action_hash_count") != HELD_CANDIDATES
        ):
            raise ValueError("frozen preflight lacks exact support or oracle-headroom evidence")
        if pairwise_preflight:
            allowed_pair_contracts = {
                "state_conditional_summary_collision_pairs_v1",
                "velocity_matched_state_alias_pairs_v1",
                "mixed_velocity_alias_direct_route_v1",
            }
            if (
                candidate.get("pairwise_contrast_contract") not in allowed_pair_contracts
                or candidate.get("pairwise_absolute_cost_gap_q10", -1.0) < 0.003
                or candidate.get("pairwise_gap_ge_0p003_fraction", -1.0) < 0.80
                or candidate.get("pairwise_sign_mixed_pair_count") != HELD_CANDIDATES // 2
            ):
                raise ValueError("frozen preflight lacks the declared pairwise contrast-support evidence")
            if value.get("schema") == PAIRWISE_CONTRAST_V12_PREFLIGHT_SCHEMA and (
                candidate.get("pairwise_minimum_fold_sign_balance_requirement") != 0.25
                or candidate.get("pairwise_minimum_fold_sign_balance", -1.0) < 0.25
            ):
                raise ValueError("frozen V12 pairwise preflight lacks hard balanced-reversal evidence")
            if value.get("schema") in {PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA, PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA, PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA, PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA, MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA, MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA, C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA} and (
                candidate.get("pairwise_contrast_contract") not in {"velocity_matched_state_alias_pairs_v1", "mixed_velocity_alias_direct_route_v1"}
                or candidate.get("minimum_fold_velocity_reversal_requirement") != 0.75
                or candidate.get("minimum_fold_velocity_reversal", -1.0) < 0.75
                or candidate.get("maximum_crossfold_position_only_sign_accuracy_requirement") != 0.625
                or candidate.get("maximum_crossfold_position_only_sign_accuracy", 1.0) > 0.625
            ):
                raise ValueError("frozen V14 preflight lacks paired velocity-alias evidence")
        elif value.get("schema") not in {
            C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
            C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
            C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
        } and (
            candidate.get("oracle_headroom_contract") != "design_root_terminal_goal_cost_v1"
            or candidate.get("design_best_second_cost_gap_q10", -1.0) < 0.001
            or candidate.get("design_best_second_cost_gap_ge_0p003_fraction", -1.0) < 0.50
            or candidate.get("design_candidate_cost_spread_q10", -1.0) < 0.05
        ):
            raise ValueError("frozen preflight lacks exact support or oracle-headroom evidence")
        if value.get("schema") == C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA:
            hashes = candidate.get("v20_held_action_hashes")
            if (
                candidate.get("v20_action_support_contract")
                != "exact_held_candidate_hash_set_reused_from_frozen_v20_context"
                or not isinstance(hashes, list)
                or len(hashes) != HELD_CANDIDATES
                or len(set(map(str, hashes))) != HELD_CANDIDATES
                or candidate.get("v20_held_action_hash_count") != HELD_CANDIDATES
            ):
                raise ValueError("V22 preflight lacks exact frozen-model action-support evidence")
    if value.get("schema") in {
        PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
        OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
    }:
        is_v9_or_v10 = value.get("schema") in {
            OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
            ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
        }
        expected_contract = (
            "opportunity_compact_v1_predicted_score_action_delta"
            if is_v9_or_v10 else "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1"
        )
        expected_dimension = 15 if is_v9_or_v10 else 95
        if (
            candidate.get("router_feature_contract") != expected_contract
            or candidate.get("router_feature_dimension") != expected_dimension
            or candidate.get("router_train_root_count") != expected_counts.get("train")
            or candidate.get("minimum_router_train_root_surplus", -1) < 16
            or candidate.get("router_train_root_count", 0) < candidate.get("router_feature_dimension", 0) + candidate.get("minimum_router_train_root_surplus", 0)
            # Bootstrap-lower routing preserves V7's train-root-only blocked
            # ridge selection, then applies a pre-outcome uncertainty
            # transform.  It shares the same support/capacity interface.
            or candidate.get("router_kind") not in {"advantage_ridge", "advantage_bootstrap_lower"}
            or candidate.get("router_action_representation") != "full_ordered_20x2"
            or candidate.get("router_regularization_selection") != "train_root_only_four_fold_blocked_advantage_mse"
            or value.get("post_execution_aliasing_contract") != V7_POST_EXECUTION_ALIASING_CONTRACT
        ):
            raise ValueError("advantage-router frozen preflight lacks capacity, execution-interface, or aliasing-contract evidence")
    if value.get("schema") in {
        OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
    }:
        expected_opportunity_gate = (
            "cost_endpoint_candidate_disagreement"
            if value.get("schema") == ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA
            else "cost_planning_candidate_disagreement"
        )
        if (
            candidate.get("router_training_scope") != "scorer_disagreement_only"
            or candidate.get("opportunity_gate") != expected_opportunity_gate
            or candidate.get("minimum_router_opportunity_root_count") != 31
            or candidate.get("minimum_calibration_opportunity_roots_per_role") != 8
        ):
            raise ValueError("opportunity-router frozen preflight lacks the declared opportunity-gating contract")
    exact_split_schemas = {
        ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
        C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
        C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
        C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
        C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
    }
    velocity_alias_schemas = {
        PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V16_PREFLIGHT_SCHEMA,
        PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA,
        MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA,
        C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA,
        C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA,
        C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA,
        C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA,
    }
    if value.get("schema") == ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA:
        if candidate.get("repair_arm") != "endpoint_coupled_counterfactual":
            raise ValueError("V10 frozen preflight lacks the endpoint-coupled repair-arm contract")
    if value.get("schema") in exact_split_schemas:
        declared_roots = value.get("split_roots")
        if not isinstance(declared_roots, dict) or set(declared_roots) != set(split_counts):
            raise ValueError("frozen preflight lacks exact declared split-root identities")
        try:
            normalized_roots = {
                name: sorted(int(root) for root in declared_roots[name])
                for name in split_counts
            }
        except (TypeError, ValueError) as error:
            raise ValueError("frozen preflight split-root identities must be integer lists") from error
        flattened = [root for roots in normalized_roots.values() for root in roots]
        if (
            any(len(normalized_roots[name]) != int(split_counts[name]) for name in split_counts)
            or len(set(flattened)) != len(flattened)
        ):
            raise ValueError("frozen preflight split-root identities are incomplete or overlapping")
    if value.get("schema") in velocity_alias_schemas:
        declared_design_roots = value.get("interaction_design_roots")
        if (
            not isinstance(declared_design_roots, list)
            or len(declared_design_roots) != INTERACTION_DESIGN_ROOTS
            or any(not isinstance(root, (int, np.integer)) for root in declared_design_roots)
            or len(set(int(root) for root in declared_design_roots)) != INTERACTION_DESIGN_ROOTS
            or set(int(root) for root in declared_design_roots) & set(flattened)
        ):
            raise ValueError("frozen velocity-alias preflight lacks disjoint paired design-root identities")
    return {
        "schema": str(value["schema"]),
        "status": "passed_static_preflight",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "context_count": len(prior_context_sha256),
        "dataset_states_sha256": value["dataset_states_sha256"],
        "root_counts": dict(expected_counts),
        # The build stage must retain candidate-bank evidence from the sealed
        # ledger.  V22 consumes its exact V20 action hashes here; omitting this
        # field made a valid static ledger look like an empty declaration only
        # after expensive backbone initialisation.
        "candidate_bank": dict(candidate),
        "candidate_design": candidate_design,
        "goal_qpos": ledger_goal.tolist(),
        "construction": None if value.get("schema") == FROZEN_PREFLIGHT_SCHEMA else dict(construction or {}),
        "post_execution_aliasing_contract": (
            None if value.get("schema") not in {
                PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
                OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
                ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
            }
            else dict(V7_POST_EXECUTION_ALIASING_CONTRACT)
        ),
        "split_roots": (
            {name: list(normalized_roots[name]) for name in split_counts}
            if value.get("schema") in exact_split_schemas
            else None
        ),
        "interaction_design_roots": (
            [int(root) for root in declared_design_roots]
            if value.get("schema") in velocity_alias_schemas
            else None
        ),
    }


def _validated_split_counts(split_counts: dict[str, int] | None) -> dict[str, int]:
    """Validate a declared panel size without silently weakening V6 defaults."""
    value = dict(SPLIT_COUNTS if split_counts is None else split_counts)
    if set(value) != set(SPLIT_COUNTS) or any(int(value[name]) != value[name] or int(value[name]) <= 0 for name in SPLIT_COUNTS):
        raise ValueError("split counts must be positive integer train/calibration/outcome counts")
    if int(value["calibration"]) < 4 or int(value["calibration"]) % 2:
        raise ValueError("calibration root count must be even and at least four for split authorization")
    return {name: int(value[name]) for name in SPLIT_COUNTS}


def _eligible_roots(
    *, root_count: int, root_id_min: int | None = None, root_id_max_exclusive: int | None = None,
) -> np.ndarray:
    """Return a declared root range, rejecting ambiguous or empty ranges."""

    lower = 0 if root_id_min is None else int(root_id_min)
    upper = int(root_count) if root_id_max_exclusive is None else int(root_id_max_exclusive)
    if lower < 0 or upper > int(root_count) or lower >= upper:
        raise ValueError("root range must satisfy 0 <= min < max <= root_count")
    return np.arange(lower, upper, dtype=np.int64)


def _split_roots_v6(
    *, seed: int, forbidden: set[int], root_count: int = STATE_ACTION_ROOT_COUNT,
    split_counts: dict[str, int] | None = None, root_id_min: int | None = None,
    root_id_max_exclusive: int | None = None,
) -> dict[int, str]:
    split_counts = _validated_split_counts(split_counts)
    total = sum(split_counts.values())
    eligible = np.asarray(
        [episode for episode in _eligible_roots(
            root_count=int(root_count), root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
        ) if int(episode) not in forbidden],
        dtype=np.int64,
    )
    if len(eligible) < total:
        raise ValueError("insufficient fresh PointMaze state/action roots after prior-context exclusion")
    ordered = eligible[np.random.default_rng(int(seed)).permutation(len(eligible))[:total]]
    split: dict[int, str] = {}
    offset = 0
    for name in ("train", "calibration", "outcome"):
        width = split_counts[name]
        split.update({int(root): name for root in ordered[offset : offset + width]})
        offset += width
    return split


def _interaction_design_roots(
    *, seed: int, forbidden: set[int], evaluation_roots: set[int], root_count: int = STATE_ACTION_ROOT_COUNT,
    root_id_min: int | None = None, root_id_max_exclusive: int | None = None,
) -> list[int]:
    """Choose disjoint effect-screening roots without looking at task costs."""
    eligible = np.asarray(
        [episode for episode in _eligible_roots(
            root_count=int(root_count), root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
        ) if int(episode) not in forbidden and int(episode) not in evaluation_roots],
        dtype=np.int64,
    )
    if len(eligible) < INTERACTION_DESIGN_ROOTS:
        raise ValueError("insufficient fresh PointMaze roots for interaction pair design")
    return sorted(np.random.default_rng(int(seed) + 9173).permutation(eligible)[:INTERACTION_DESIGN_ROOTS].astype(int).tolist())


def _requires_exact_shared_scorer_bank(preflight_provenance: dict[str, Any] | None) -> bool:
    """Whether the frozen ledger binds scorer training to the held action bank."""

    return bool(
        isinstance(preflight_provenance, dict)
        and preflight_provenance.get("schema") in EXACT_SHARED_SCORER_PREFLIGHT_SCHEMAS
    )


def _renderer_chart_roots_v6(dataset_root: Path, v6_split_roots: set[int]) -> list[int]:
    """Select 40 stored-RGB chart roots, excluding only V6 evaluation roots.

    Historical panels may share these fixed renderer-calibration images. They
    are not V6 candidate roots, supervision, calibration labels, or outcome
    labels; excluding every historical context here can exhaust the 100 stored
    RGB episodes and is neither needed nor possible for the renderer chart.
    """

    available = [episode for episode in range(1800, 1900) if (dataset_root / "obses" / f"episode_{episode:03d}.pth").exists()]
    selected = [episode for episode in available if episode not in v6_split_roots]
    if len(selected) < 40:
        raise ValueError("V6 needs 40 stored-RGB roots outside its own train/calibration/outcome split for renderer calibration")
    return selected[:40]


def _renderer_chart_provenance(chart_roots: list[int], v6_split_roots: set[int], historical_roots: set[int]) -> dict[str, Any]:
    overlap = sorted(set(chart_roots) & historical_roots)
    if set(chart_roots) & v6_split_roots:
        raise RuntimeError("V6 renderer chart overlaps a V6 evaluation root")
    return {
        "role": "fixed renderer calibration RGB plus simulator-state regression targets only; never V6 candidate supervision, calibration labels, or outcome labels",
        "chart_roots": list(chart_roots),
        "chart_root_count": len(chart_roots),
        "v6_split_roots_excluded": sorted(v6_split_roots),
        "v6_split_overlap": [],
        "historical_context_root_overlap": overlap,
        "historical_context_root_overlap_count": len(overlap),
        "historical_overlap_permitted_only_for_fixed_renderer_chart": True,
        "uses_simulator_state_targets_for_renderer_calibration": True,
        "v6_outcome_labels_materialized_from_chart": False,
    }


def _pulse_features(bank: np.ndarray) -> np.ndarray:
    value = np.asarray(bank, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (TOTAL_STEPS, 2):
        raise ValueError("expected a [candidate, 20, 2] bank")
    pulses = value.reshape(len(value), PULSE_COUNT, PULSE_WIDTH, 2)
    if not np.allclose(pulses, pulses.mean(axis=2, keepdims=True), rtol=0.0, atol=1.0e-12):
        raise ValueError("v6 candidates must be constant within each four-step pulse")
    return pulses.mean(axis=2)


def _quantiles(value: np.ndarray) -> dict[str, float]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return {f"q{int(q * 100):02d}": float(np.quantile(flat, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}


def support_distance_diagnostics(train: np.ndarray, held: np.ndarray) -> dict[str, Any]:
    """Pre-label diagnostics proving matched envelope and numeric OOD distance."""

    train_pulses, held_pulses = _pulse_features(train), _pulse_features(held)
    train_norms, held_norms = np.linalg.norm(train_pulses, axis=-1), np.linalg.norm(held_pulses, axis=-1)
    train_vectors, held_vectors = train_pulses.reshape(len(train), -1), held_pulses.reshape(len(held), -1)
    cross = np.linalg.norm(held_vectors[:, None, :] - train_vectors[None, :, :], axis=-1)
    leave_one_out = np.linalg.norm(train_vectors[:, None, :] - train_vectors[None, :, :], axis=-1)
    np.fill_diagonal(leave_one_out, np.inf)
    schedule = {"pulse_count": PULSE_COUNT, "pulse_width": PULSE_WIDTH, "total_steps": TOTAL_STEPS}
    train_hashes, held_hashes = {sequence_hash(value) for value in train}, {sequence_hash(value) for value in held}
    shared_hashes = train_hashes & held_hashes
    return {
        "amplitude_envelope": {"minimum": PULSE_MIN, "maximum": PULSE_MAX},
        "pulse_schedule": schedule,
        "train_pulse_norm_quantiles": _quantiles(train_norms),
        "held_pulse_norm_quantiles": _quantiles(held_norms),
        "train_within_envelope": bool(np.all((train_norms >= PULSE_MIN) & (train_norms <= PULSE_MAX))),
        "held_within_envelope": bool(np.all((held_norms >= PULSE_MIN) & (held_norms <= PULSE_MAX))),
        "held_fraction_pulses_above_train_max": float(np.mean(held_norms > float(train_norms.max()))),
        "train_energy_l2_quantiles": _quantiles(np.linalg.norm(train.reshape(len(train), -1), axis=1)),
        "held_energy_l2_quantiles": _quantiles(np.linalg.norm(held.reshape(len(held), -1), axis=1)),
        "held_to_nearest_train_pulse_l2_quantiles": _quantiles(cross.min(axis=1)),
        "held_to_nearest_train_pulse_l2_mean": float(cross.min(axis=1).mean()),
        "train_leave_one_out_nearest_pulse_l2_quantiles": _quantiles(leave_one_out.min(axis=1)),
        "train_leave_one_out_nearest_pulse_l2_mean": float(leave_one_out.min(axis=1).mean()),
        "candidate_hash_overlap_count": len(shared_hashes),
        "held_exact_train_match_fraction": float(sum(sequence_hash(value) in train_hashes for value in held) / len(held)),
    }


def _encode_history8(
    model: Any, transform: Any, mean: np.ndarray, basis: np.ndarray, renderer: DatasetMatchedRenderer,
    calibration: Any, states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames = causal_history8(SOURCE_FRAME, np.zeros((SOURCE_FRAME, 2), dtype=np.float64))[0]
    values, geometry = [], []
    with torch.inference_mode():
        for frame in frames:
            current, previous = renderer.render(states[frame]), renderer.render(states[frame - FRAME_STRIDE])
            center, old_center = green_center(current), green_center(previous)
            proprio = np.r_[calibration.position(center), calibration.velocity((center[0] - old_center[0], center[1] - old_center[1]))]
            context = _make_context([current], [proprio])
            encoded = model.encode(_make_gdd_source_contract_context(context, transform))
            pooled = _pool_visual(encoded["visual"][:, -1]).detach().cpu().numpy()[0]
            values.append((np.asarray(pooled, dtype=np.float64) - mean) @ basis.T)
            geometry.append(np.asarray(proprio, dtype=np.float64))
    history = np.stack(values)
    visual_geometry = np.stack(geometry)
    if history.shape[0] != HISTORY_FRAMES or visual_geometry.shape != (HISTORY_FRAMES, 4):
        raise RuntimeError("v6 visual history encoder did not return eight causal frames")
    return history, visual_geometry


def _paths(output: Path) -> dict[str, Path]:
    return {
        "source_dataset": output / "temporal_ood_v6_source_dataset.json",
        "source_arrays": output / "temporal_ood_v6_source_arrays.npz",
        "context": output / "temporal_ood_v6_context.json",
        "calibration_outcomes": output / "temporal_ood_v6_calibration_outcomes.json",
        "outcome_labels_reserved": output / "temporal_ood_v6_outcome_labels.json",
        "manifest": output / "temporal_ood_v6_manifest.json",
        "build_report": output / "build_v6_matched_report.json",
    }


def _write_panel_artifacts(
    output: Path, *, source_records: list[dict[str, Any]], source_arrays: dict[str, np.ndarray],
    context_records: list[dict[str, Any]], calibration_labels: dict[str, Any], context_metadata: dict[str, Any],
    manifest_base: dict[str, Any],
) -> dict[str, Path]:
    """Write allowed artifacts and fail closed if an outcome label file exists."""

    paths = _paths(output)
    if paths["outcome_labels_reserved"].exists():
        raise RuntimeError("refusing to overwrite or consume a V6 outcome-label file during panel build")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty V6 output directory: {output}; choose a fresh artifact path"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths["source_dataset"].write_text(json.dumps({"metadata": {"schema": SOURCE_SCHEMA, "history_frames": HISTORY_FRAMES}, "records": source_records}, indent=2, sort_keys=True) + "\n")
    np.savez(paths["source_arrays"], **source_arrays)
    paths["context"].write_text(json.dumps({"metadata": context_metadata, "records": context_records}, indent=2, sort_keys=True) + "\n")
    paths["calibration_outcomes"].write_text(json.dumps(calibration_labels, indent=2, sort_keys=True) + "\n")
    if paths["outcome_labels_reserved"].exists():
        raise RuntimeError("V6 build unexpectedly materialized outcome labels")
    manifest = {
        **manifest_base,
        "schema": MANIFEST_SCHEMA,
        "artifacts": {
            "source_dataset": {"path": str(paths["source_dataset"].resolve()), "sha256": _sha256(paths["source_dataset"])},
            "source_arrays": {"path": str(paths["source_arrays"].resolve()), "sha256": _sha256(paths["source_arrays"])},
            "context": {"path": str(paths["context"].resolve()), "sha256": _sha256(paths["context"])},
            "calibration_outcomes": {"path": str(paths["calibration_outcomes"].resolve()), "sha256": _sha256(paths["calibration_outcomes"])},
            "outcome_labels": {"path": str(paths["outcome_labels_reserved"].resolve()), "materialized": False, "sha256": None},
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths


def build_panel(
    dataset_root: Path, contexts: list[Path], output: Path, *, encoder_checkpoint: Path,
    pca_checkpoint: Path, dinov2_source: Path, device: str = "cuda",
    root_seed: int = ROOT_SEED, held_action_seed: int = HELD_ACTION_SEED,
    train_action_seed: int = TRAIN_ACTION_SEED, candidate_generalization_regime: str = "numeric_ood",
    candidate_design: str = "random_matched", goal_qpos: Any = None,
    allow_wall_sensitive_goal: bool = False, split_counts: dict[str, int] | None = None,
    root_id_min: int | None = None, root_id_max_exclusive: int | None = None,
    frozen_preflight_ledger: Path | None = None,
    fixed_held_actions: np.ndarray | None = None,
    allowed_prior_action_hashes: set[str] | None = None,
) -> dict[str, Any]:
    if candidate_design not in CANDIDATE_DESIGNS:
        raise ValueError(f"unsupported candidate design {candidate_design!r}")
    interaction_designs = {
        "state_interaction_pairs",
        "goal_balanced_interaction_pairs",
        "robust_goal_balanced_interaction_pairs",
        "tail_robust_goal_balanced_interaction_pairs",
        "expanded_tail_robust_goal_balanced_interaction_pairs",
        "ultra_tail_robust_goal_balanced_interaction_pairs",
        "directional_tail_robust_goal_balanced_interaction_pairs",
        "pairwise_contrast_robust_interaction_pairs",
        "pairwise_contrast_balanced_reversal_interaction_pairs",
        "pairwise_contrast_position_aliased_interaction_pairs",
        "velocity_matched_acceleration_pairs",
        "velocity_matched_braking_pairs",
    }
    if candidate_design in interaction_designs and candidate_generalization_regime != "shared_candidate_in_support":
        raise ValueError("state-interaction design requires shared_candidate_in_support")
    goal = validate_goal_qpos(
        DEFAULT_GOAL_QPOS if goal_qpos is None else goal_qpos,
        allow_wall_sensitive_goal=allow_wall_sensitive_goal,
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty V6 output directory: {output}; choose a fresh artifact path"
        )
    if device != "cuda":
        raise ValueError("formal V6 matched panel requires CUDA")
    old_roots, old_hashes, old_templates, prior_context_sha256 = _all_prior_identities(contexts)
    states = np.asarray(torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    actions = np.asarray(torch.load(dataset_root / "actions.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != (100, 4) or actions.shape != (states.shape[0], 100, 2):
        raise ValueError("PointMaze state/action pool must be [roots,100,4] and [roots,100,2]")
    root_count = int(states.shape[0])
    split_counts = _validated_split_counts(split_counts)
    preflight_provenance = frozen_preflight_provenance(
        frozen_preflight_ledger, dataset_root=dataset_root, prior_context_sha256=prior_context_sha256,
        split_counts=split_counts, candidate_design=candidate_design, goal_qpos=goal,
        construction={
            "root_seed": int(root_seed), "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed),
            "root_id_min": None if root_id_min is None else int(root_id_min),
            "root_id_max_exclusive": None if root_id_max_exclusive is None else int(root_id_max_exclusive),
        },
    )
    allowed_root_ids = _eligible_roots(
        root_count=root_count, root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
    )
    if candidate_design in {"velocity_matched_acceleration_pairs", "velocity_matched_braking_pairs"}:
        frozen_split_roots = preflight_provenance.get("split_roots") if isinstance(preflight_provenance, dict) else None
        if not isinstance(frozen_split_roots, dict) or set(frozen_split_roots) != set(split_counts):
            raise RuntimeError("V14 frozen preflight lacks its grouped split-root ledger")
        split = {
            int(root): str(name)
            for name, roots in frozen_split_roots.items()
            if isinstance(roots, list)
            for root in roots
        }
        expected_roots = [int(root) for roots in frozen_split_roots.values() if isinstance(roots, list) for root in roots]
        if (
            len(expected_roots) != sum(split_counts.values())
            or len(set(expected_roots)) != len(expected_roots)
            or set(split) & old_roots
            or any(root not in allowed_root_ids for root in split)
            or {name: sum(value == name for value in split.values()) for name in split_counts} != split_counts
        ):
            raise RuntimeError("V14 frozen grouped split roots violate the declared panel contract")
    else:
        split = _split_roots_v6(
            seed=int(root_seed), forbidden=old_roots, root_count=root_count, split_counts=split_counts,
            root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
        )
    if set(split) & old_roots:
        raise RuntimeError("V6 root reuse from supplied prior context")
    if (
        isinstance(preflight_provenance, dict)
        and preflight_provenance.get("schema") == ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA
    ):
        observed_split_roots = {
            name: sorted(int(root) for root, assigned in split.items() if assigned == name)
            for name in split_counts
        }
        if observed_split_roots != preflight_provenance.get("split_roots"):
            raise RuntimeError("V10 panel split roots differ from the immutable static-preflight ledger")
    visual_roots = _renderer_chart_roots_v6(dataset_root, set(split))
    chart_provenance = _renderer_chart_provenance(visual_roots, set(split), old_roots)
    # Reject an undersized declared root range before loading the frozen visual
    # backbone.  Interaction-pair designs need their screening roots to be
    # disjoint from both all evaluation roots and renderer-chart roots; the
    # latter condition was formerly checked only after the expensive model
    # initialisation below.
    interaction_roots: list[int] = []
    if fixed_held_actions is None and candidate_design in interaction_designs:
        if candidate_design in {"velocity_matched_acceleration_pairs", "velocity_matched_braking_pairs"}:
            frozen_roots = preflight_provenance.get("interaction_design_roots") if isinstance(preflight_provenance, dict) else None
            if not isinstance(frozen_roots, list) or len(frozen_roots) != INTERACTION_DESIGN_ROOTS:
                raise RuntimeError("V14 frozen preflight lacks its complete paired interaction-root ledger")
            interaction_roots = [int(root) for root in frozen_roots]
            if (
                len(set(interaction_roots)) != len(interaction_roots)
                or set(interaction_roots) & (old_roots | set(visual_roots) | set(split))
                or any(root < 0 or root >= root_count for root in interaction_roots)
            ):
                raise RuntimeError("V14 frozen interaction roots violate disjointness or dataset bounds")
        else:
            interaction_roots = _interaction_design_roots(
                seed=int(root_seed), forbidden=old_roots | set(visual_roots), evaluation_roots=set(split), root_count=root_count,
                root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
            )
    visual_calibration = fit_visual_calibration(dataset_root, visual_roots, max_frames=1600)
    reference_images, _ = load_episode(dataset_root, visual_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, visual_calibration)
    root = Path(__file__).resolve().parents[3]
    model, transform, _stats, _mean, _std = _load_jepa_model(root, dataset_root, encoder_checkpoint, dinov2_source, torch.device(device))
    pca = torch.load(pca_checkpoint, map_location="cpu", weights_only=False)
    mean, basis = np.asarray(pca["pca_mean"], dtype=np.float64), np.asarray(pca["pca_basis"], dtype=np.float64)
    interaction_report: dict[str, Any] | None = None
    # This must be checked *after* model construction as well as before it:
    # candidate construction historically lived here, and an earlier V22
    # implementation accepted a frozen bank then unconditionally overwrote it
    # in the velocity-matched branch below.  A supplied bank is an immutable
    # C3 action-support contract, not a candidate-design suggestion.
    if fixed_held_actions is not None:
        held = np.asarray(fixed_held_actions, dtype=np.float64)
        if held.shape != (HELD_CANDIDATES, TOTAL_STEPS, 2) or not np.isfinite(held).all():
            raise ValueError("fixed held candidate bank must be finite [16,20,2]")
        held = held.copy()
        train = held.copy()
        interaction_report = {
            "kind": "externally_frozen_held_bank",
            "candidate_count": HELD_CANDIDATES,
            "overrides_candidate_design": candidate_design,
        }
    elif candidate_design in interaction_designs:
        if candidate_design == "state_interaction_pairs":
            held, train, interaction_report = state_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], proposal_seed=int(held_action_seed),
                supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "goal_balanced_interaction_pairs":
            held, train, interaction_report = goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "robust_goal_balanced_interaction_pairs":
            held, train, interaction_report = robust_goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "tail_robust_goal_balanced_interaction_pairs":
            held, train, interaction_report = tail_robust_goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "expanded_tail_robust_goal_balanced_interaction_pairs":
            held, train, interaction_report = expanded_tail_robust_goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "ultra_tail_robust_goal_balanced_interaction_pairs":
            held, train, interaction_report = ultra_tail_robust_goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "pairwise_contrast_robust_interaction_pairs":
            held, train, interaction_report = pairwise_contrast_robust_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "pairwise_contrast_balanced_reversal_interaction_pairs":
            held, train, interaction_report = pairwise_contrast_balanced_reversal_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "pairwise_contrast_position_aliased_interaction_pairs":
            held, train, interaction_report = pairwise_contrast_position_aliased_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "velocity_matched_acceleration_pairs":
            held, train, interaction_report = velocity_matched_acceleration_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        elif candidate_design == "velocity_matched_braking_pairs":
            held, train, interaction_report = velocity_matched_braking_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
        else:
            held, train, interaction_report = directional_tail_robust_goal_balanced_interaction_candidate_banks(
                states[np.asarray(interaction_roots, dtype=np.int64), SOURCE_FRAME], goal,
                proposal_seed=int(held_action_seed), supplemental_seed=int(train_action_seed),
            )
    else:
        held, train = temporal_memory_candidate_banks(
            held_seed=int(held_action_seed), train_seed=int(train_action_seed), regime=candidate_generalization_regime,
        )
    held_hashes, train_hashes = [sequence_hash(value) for value in held], [sequence_hash(value) for value in train]
    held_templates, train_templates = [temporal_template_fingerprint(value) for value in held], [temporal_template_fingerprint(value) for value in train]
    shared_action_hashes = set(held_hashes) & set(train_hashes)
    if candidate_generalization_regime == "numeric_ood" and shared_action_hashes:
        raise RuntimeError("numeric-OOD held and training action hashes overlap")
    if candidate_generalization_regime == "shared_candidate_in_support" and set(held_hashes) - set(train_hashes):
        raise RuntimeError("shared-bank held actions are missing from source training support")
    permitted_prior_hashes = set() if allowed_prior_action_hashes is None else set(allowed_prior_action_hashes)
    if not permitted_prior_hashes <= old_hashes:
        raise ValueError("permitted prior action hashes must be declared by supplied contexts")
    reused = (set(held_hashes) | set(train_hashes)) & old_hashes
    if reused - permitted_prior_hashes:
        raise RuntimeError("V6 action hash reuses a supplied prior context action")
    if fixed_held_actions is not None:
        declared = preflight_provenance.get("candidate_bank", {}).get("v20_held_action_hashes") if isinstance(preflight_provenance, dict) else None
        # V22's frozen-support estimand is the *set* of executed V20 action
        # sequences.  Candidate IDs are local row labels: the learned model
        # receives the sequence itself, not an ID embedding, so an arbitrary
        # permutation cannot change action support.  Require a bijection and
        # retain the observed ordered map in the context below, rather than
        # rejecting a semantically identical bank because of row ordering.
        # This is deliberately not a subset check: one missing, substituted,
        # or duplicated action remains a hard protocol failure.
        declared_hashes = [str(value) for value in declared] if isinstance(declared, list) else []
        if (
            len(declared_hashes) != HELD_CANDIDATES
            or len(set(declared_hashes)) != HELD_CANDIDATES
            or len(set(held_hashes)) != HELD_CANDIDATES
            or set(held_hashes) != set(declared_hashes)
        ):
            raise RuntimeError(
                "fixed held action bank does not exactly match the static V22 support declaration "
                f"(observed={held_hashes}, declared={declared_hashes})"
            )
    exact_shared_candidate_bank = _requires_exact_shared_scorer_bank(preflight_provenance)
    scorer_actions = held if exact_shared_candidate_bank else train
    scorer_hashes = held_hashes if exact_shared_candidate_bank else train_hashes
    scorer_templates = held_templates if exact_shared_candidate_bank else train_templates
    scorer_generator = V6_ACTION_FAMILY if exact_shared_candidate_bank else V6_TRAIN_FAMILY
    scorer_protocol = "held_exact_shared_bank" if exact_shared_candidate_bank else "train_bank"
    cuda = require_cuda_allocation(device)
    histories = {episode: _encode_history8(model, transform, mean, basis, renderer, visual_calibration, states[episode]) for episode in sorted(split)}
    source_records: list[dict[str, Any]] = []
    source_arrays: dict[str, np.ndarray] = {}
    context_records: list[dict[str, Any]] = []
    calibration_labels: dict[str, Any] = {}
    for episode in sorted(split):
        frames, prior_actions = causal_history8(SOURCE_FRAME, actions[episode])
        state = states[episode, SOURCE_FRAME].astype(np.float64)
        history, visual_geometry = histories[episode]
        common = {
            "instance_id": f"pointmaze_v6_episode_{episode}", "split": split[episode], "environment": "PointMaze",
            "start_episode_id": str(episode), "source_frame": SOURCE_FRAME, "history_frame_indices": frames,
            "history_action_interval": [frames[0], SOURCE_FRAME], "history_actions_raw": prior_actions.tolist(),
            "source_latent_model": history[-1].tolist(), "source_history_model": history.reshape(-1).tolist(),
            "source_history_visual_state": visual_geometry.tolist(),
            "source_state_raw": state.tolist(), "goal_raw": goal.tolist(),
        }
        if split[episode] == "train":
            for index, candidate in enumerate(scorer_actions):
                key = f"terminal/v6/{episode}/{index}"
                source_arrays[key] = _simulate_sequence(state, candidate)
                source_records.append({
                    **common, "candidate_id": index, "candidate_generator": scorer_generator,
                    "candidate_action_hash": scorer_hashes[index], "candidate_template_fingerprint": scorer_templates[index],
                    "candidate_bank_hash": _json_hash(scorer_hashes), "action_sequence": candidate.tolist(),
                    "realized_terminal_state_ref": f"arrays.npz::{key}",
                })
        for index, candidate in enumerate(held):
            key = f"{common['instance_id']}/{index}"
            context_records.append({
                **common, "candidate_id": index, "candidate_generator": V6_ACTION_FAMILY,
                "candidate_action_hash": held_hashes[index], "candidate_template_fingerprint": held_templates[index],
                "candidate_bank_hash": _json_hash(held_hashes), "action_sequence": candidate.tolist(),
                "outcome_label_status": "not_materialized_during_build",
            })
            if split[episode] == "calibration":
                terminal = _simulate_sequence(state, candidate)
                calibration_labels[key] = {
                    "instance_id": common["instance_id"],
                    "candidate_id": int(index),
                    "candidate_action_hash": held_hashes[index],
                    "candidate_bank_hash": _json_hash(held_hashes),
                    "real_task_cost": float(np.linalg.norm(terminal[:2] - goal)),
                    "realized_terminal_state": terminal.tolist(),
                }
    diagnostics = support_distance_diagnostics(scorer_actions, held)
    context_metadata = {
        "schema": PANEL_SCHEMA, "candidate_family": V6_ACTION_FAMILY, "root_split_seed": int(root_seed),
        "held_action_seed": int(held_action_seed), "train_action_seed": int(train_action_seed),
        "candidate_generalization_regime": candidate_generalization_regime,
        "scorer_candidate_bank_protocol": scorer_protocol,
        "scorer_training_action_hashes": scorer_hashes,
        "candidate_design": candidate_design,
        "interaction_pair_design": interaction_report,
        "goal_qpos": goal.tolist(),
        "legacy_goal_reproduction": bool(np.allclose(goal, LEGACY_GOAL, rtol=0.0, atol=1.0e-12)),
        "interaction_design_root_count": len(interaction_roots),
        "interaction_design_roots": interaction_roots,
        "history_frames": HISTORY_FRAMES, "history_frame_stride": FRAME_STRIDE, "history_is_causal": True,
        "history_actions_are_strictly_pre_candidate": True, "outcome_labels_materialized_during_build": False,
        "calibration_label_identity_bound": True,
        "renderer_backed_frozen_observation_control": True,
        "state_action_pool_root_count": root_count,
        "declared_root_range": [int(allowed_root_ids.min()), int(allowed_root_ids.max()) + 1],
        "privileged_source_state_role": "offline paired task-cost readout only; not a live visual-planning input",
        "supplied_prior_context_roots_excluded": sorted(old_roots),
        "renderer_chart_provenance": chart_provenance,
        "encoder_checkpoint_sha256": _sha256(encoder_checkpoint), "pca_checkpoint_sha256": _sha256(pca_checkpoint),
        "frozen_preflight_ledger": preflight_provenance,
    }
    paths = _write_panel_artifacts(
        output, source_records=source_records, source_arrays=source_arrays, context_records=context_records,
        calibration_labels=calibration_labels, context_metadata=context_metadata,
        manifest_base={
            "formal_protocol": True,
            "panel_schema": PANEL_SCHEMA, "prior_context_sha256": prior_context_sha256,
            "frozen_preflight_ledger": preflight_provenance,
            "root_counts": {name: sum(value == name for value in split.values()) for name in split_counts},
            "state_action_pool_root_count": root_count,
            "declared_root_range": [int(allowed_root_ids.min()), int(allowed_root_ids.max()) + 1],
            "prior_root_overlap": sorted(set(split) & old_roots),
            "prior_action_hash_overlap": sorted((set(held_hashes) | set(train_hashes)) & old_hashes),
            "prior_template_overlap_diagnostic_only": sorted((set(held_templates) | set(train_templates)) & old_templates),
            "train_action_hashes": train_hashes, "held_action_hashes": held_hashes,
            "scorer_training_action_hashes": scorer_hashes, "scorer_candidate_bank_protocol": scorer_protocol,
            "train_candidate_bank_hash": _json_hash(train_hashes), "held_candidate_bank_hash": _json_hash(held_hashes),
            "candidate_generalization_regime": candidate_generalization_regime,
            "candidate_design": candidate_design,
            "interaction_pair_design": interaction_report,
            "interaction_design_roots": interaction_roots,
            "shared_train_held_action_hashes": sorted(shared_action_hashes),
            "support_distance_diagnostics": diagnostics,
            "renderer_chart_provenance": chart_provenance,
            "outcome_labels_materialized_during_build": False,
            "claim_boundary": "Renderer-backed offline matched-support temporal-memory panel; not live visual deployment.",
        },
    )
    report = {
        "schema": PANEL_SCHEMA, "cuda_preflight": cuda, "manifest": str(paths["manifest"].resolve()),
        "manifest_sha256": _sha256(paths["manifest"]), "context": str(paths["context"].resolve()),
        "source_dataset": str(paths["source_dataset"].resolve()), "source_arrays": str(paths["source_arrays"].resolve()),
        "calibration_outcomes": str(paths["calibration_outcomes"].resolve()),
        "outcome_labels_materialized_during_build": False, "outcome_label_path_reserved": str(paths["outcome_labels_reserved"].resolve()),
        "root_counts": {name: sum(value == name for value in split.values()) for name in split_counts},
        "state_action_pool_root_count": root_count,
        "declared_root_range": [int(allowed_root_ids.min()), int(allowed_root_ids.max()) + 1],
        "prior_root_overlap": sorted(set(split) & old_roots), "prior_action_hash_overlap": sorted((set(held_hashes) | set(train_hashes)) & old_hashes),
        "support_distance_diagnostics": diagnostics, "candidate_generalization_regime": candidate_generalization_regime,
        "scorer_candidate_bank_protocol": scorer_protocol,
        "candidate_design": candidate_design, "interaction_pair_design": interaction_report,
        "goal_qpos": goal.tolist(),
        "legacy_goal_reproduction": bool(np.allclose(goal, LEGACY_GOAL, rtol=0.0, atol=1.0e-12)),
        "interaction_design_roots": interaction_roots,
        "renderer_chart_provenance": chart_provenance,
        "frozen_preflight_ledger": preflight_provenance,
        "claim_boundary": "Renderer-backed offline matched-support temporal-memory panel; not live visual deployment.",
    }
    paths["build_report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True, help="All prior V1-V5 context files to exclude.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--pca-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument("--held-action-seed", type=int, default=HELD_ACTION_SEED)
    parser.add_argument("--train-action-seed", type=int, default=TRAIN_ACTION_SEED)
    parser.add_argument("--candidate-generalization-regime", choices=CANDIDATE_GENERALIZATION_REGIMES, default="numeric_ood")
    parser.add_argument("--candidate-design", choices=CANDIDATE_DESIGNS, default="random_matched")
    parser.add_argument("--goal-qpos", type=float, nargs=2, default=DEFAULT_GOAL_QPOS.tolist(), metavar=("X", "Y"))
    parser.add_argument("--allow-wall-sensitive-goal", action="store_true", help="explicitly reproduce legacy [1.2, 1.2] target")
    parser.add_argument("--train-root-count", type=int, default=SPLIT_COUNTS["train"])
    parser.add_argument("--calibration-root-count", type=int, default=SPLIT_COUNTS["calibration"])
    parser.add_argument("--outcome-root-count", type=int, default=SPLIT_COUNTS["outcome"])
    parser.add_argument("--root-id-min", type=int, default=None, help="inclusive lower root bound for a fresh extension")
    parser.add_argument("--root-id-max-exclusive", type=int, default=None, help="exclusive upper root bound for a fresh extension")
    parser.add_argument("--frozen-preflight-ledger", type=Path, default=None, help="passed static-preflight artifact to bind this build to a frozen context ledger")
    args = vars(parser.parse_args())
    args["split_counts"] = {"train": args.pop("train_root_count"), "calibration": args.pop("calibration_root_count"), "outcome": args.pop("outcome_root_count")}
    print(json.dumps(build_panel(**args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
