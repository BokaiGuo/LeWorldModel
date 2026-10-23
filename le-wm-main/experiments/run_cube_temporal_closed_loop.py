"""Independent RGB-only closed-loop consequence for Cube temporal ranking.

The readouts are fit on calibration roots only.  At deployment each arm gets
only the live current/previous RGB frame, rendered goal, previous action, and
the fixed candidate action bank.  A fixed, non-adaptive descent/close prelude
is shared by both arms; after the prelude each arm replans one action at a
time.  Privileged state is used only to restore identical initial conditions
between paired arms and to score final goal distance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import PyTorch before configuring MuJoCo's headless renderer.  In the
# managed CUDA runtime, exporting MUJOCO_GL before importing PyTorch can make
# the CUDA/NVML availability probe report zero devices.  Configuring EGL
# inside the process after PyTorch is loaded keeps CUDA visible while making
# RGB rendering deterministic and headless.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import stable_worldmodel  # noqa: F401 - registers environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (  # noqa: E402
    extract_color_geometry_features,
    extract_color_geometry_temporal_features,
    _target_red_centroid,
    _largest_effector_centroid,
    _goal_position,
    _make_env,
    _vec,
)
from experiments.audit_cube_temporal_action_effect import (  # noqa: E402
    build_feature_families,
    collect_temporal_frames,
)
from experiments.audit_cube_temporal_action_ranking import (  # noqa: E402
    _candidate_bank,
    _fit_readout,
    _predict,
)
from experiments.checkpoint_compat import load_compatible_pretrained  # noqa: E402
from experiments.run_live_cube_rgb_macro_ranker import (  # noqa: E402
    _fit_rgb_pixel_action_mapping,
    _rgb_approach_action,
)


def _capture_initial(seed: int, task_id: int) -> dict[str, Any]:
    env = _make_env()
    try:
        _, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
        raw = env.unwrapped
        goal_image = np.asarray(raw._cur_goal_rendered, dtype=np.uint8).copy()
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "goal": _goal_position(env)[:2].copy(),
            "goal_rgb_centroid": _target_red_centroid(goal_image)[:2].astype(np.float64).tolist(),
        }
    finally:
        env.close()


def _prelude_action(
    step: int, current: np.ndarray | None = None, *, approach_steps: int = 0,
    approach_gain: float = 1.0, pixel_action_mapping: np.ndarray | None = None,
    phase_order: str = "descent_then_align",
) -> np.ndarray:
    """Return the shared approach/descent/close prelude.

    With ``approach_steps=0`` this preserves the historical fixed prelude.
    When enabled, the historical order descends to the working plane before
    RGB alignment.  ``align_then_descend`` instead aligns while elevated,
    then descends and closes.  The phase order is shared by every controller
    arm and is fixed before outcome roots are evaluated.
    """
    action = np.zeros(5, dtype=np.float32)
    if approach_steps < 0 or not np.isfinite(float(approach_gain)):
        raise ValueError("approach_steps must be non-negative and gain finite")
    if phase_order not in {"descent_then_align", "align_then_descend"}:
        raise ValueError(f"unsupported prelude phase order: {phase_order}")
    if phase_order == "align_then_descend" and approach_steps > 0:
        if step < int(approach_steps):
            if current is not None:
                action = _rgb_approach_action(
                    current, pixel_action_mapping=pixel_action_mapping,
                    gain=float(approach_gain),
                )
            return action
        if step < int(approach_steps) + 15:
            action[2] = -1.0
            return action
        close_start = int(approach_steps) + 15
    else:
        if step < 15:
            action[2] = -1.0
            return action
        if approach_steps > 0 and step < 15 + int(approach_steps):
            if current is not None:
                action = _rgb_approach_action(
                    current, pixel_action_mapping=pixel_action_mapping,
                    gain=float(approach_gain),
                )
            return action
        close_start = 15 + int(approach_steps)
    if step < close_start + 5:
        # OGB-Cube uses relative gripper-opening actions: +1 closes.
        action[4] = 1.0
    return action


def _support_exploration_action(step: int, candidates: np.ndarray) -> np.ndarray:
    """Fixed RGB-only fallback when the candidate utility is not identifiable."""
    indices = (1, 2, 3, 4)  # x+, x-, y+, y-
    return np.asarray(candidates[indices[step % len(indices)]], dtype=np.float32)


def _has_informative_task_probe_action(action: np.ndarray) -> bool:
    """Whether a proposed task probe actually perturbs the local XY chart.

    A zero action has no observable action-effect to validate.  Treating it as
    a task-aligned probe converts a deterministic tie in the local ranker into
    a spurious contact/readout failure.  It must instead use the shared
    support fallback and leave the local chart unreleased.
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    return bool(
        values.size >= 2
        and np.isfinite(values[:2]).all()
        and float(np.linalg.norm(values[:2])) > 1.0e-8
    )


# A local action-effect chart is only local after the shared approach/close
# prelude has settled the gripper and block into their planning configuration.
# Four axis-aligned pulses provide a redundant rank-two action design before
# either the chart or its matched RGB-carrier control may exploit it.
POST_CONTACT_PROBE_STEPS = 4
POST_CONTACT_LOCAL_METHODS = {"rgb_verified_effect_post_contact"}
POST_CONTACT_RGB_CONTROL_METHODS = {"rgb_geometry_action_post_contact_probe"}
POST_CONTACT_GOAL_GATED_METHODS = {"rgb_verified_effect_post_contact_goal_gated"}
POST_CONTACT_RESIDUAL_GATED_METHODS = {"rgb_verified_effect_post_contact_residual_gated"}
POST_CONTACT_HOLDOUT_CALIBRATION_METHODS = {
    "rgb_verified_effect_post_contact_holdout_calibration"
}
POST_CONTACT_HOLDOUT_GATED_METHODS = {
    "rgb_verified_effect_post_contact_holdout_residual_gated"
}
# A support-basis holdout cannot establish that the *task-selected* action is
# identifiable under the current contact regime.  The following protocol makes
# that missing premise observable before releasing a full local pulse.  The
# scale is a predeclared control constant, not selected from outcome roots.
TASK_ALIGNED_PROBE_SCALE = 0.25
# The q=.25 probe is a useful low-risk baseline, but its observed RGB effects
# can be at the same scale as centroid quantization.  The adaptive protocol
# therefore predeclares a larger, still sub-maximal information action.  This
# is calibrated on its own roots and never tuned on its outcome panel.
ADAPTIVE_TASK_ALIGNED_PROBE_SCALE = 0.50
# Contact-rich local effects can drift after a valid probe: retaining every
# early axis pulse with equal weight makes the chart increasingly stale, while
# retaining only a short suffix can lose rank.  This predeclared decay keeps
# the full rank-two support design but gives more weight to recent observed
# effects.  It is never selected from outcome roots.
TASK_PROBE_RECENCY_DECAY = 0.70
POST_CONTACT_TASK_PROBE_CARRIER_METHODS = {
    "rgb_geometry_action_post_contact_task_probe"
}
POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS = {
    "rgb_verified_effect_post_contact_task_probe_calibration"
}
POST_CONTACT_TASK_PROBE_GATED_METHODS = {
    "rgb_verified_effect_post_contact_task_probe_gated"
}
POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS = {
    "rgb_geometry_action_post_contact_adaptive_task_probe"
}
POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS = {
    "rgb_verified_effect_post_contact_adaptive_task_probe_calibration"
}
POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS = {
    "rgb_verified_effect_post_contact_adaptive_task_probe_gated"
}


def _uses_post_contact_probe(method: str) -> bool:
    """Whether an arm pays the fixed post-contact identification budget."""
    return method in (
        POST_CONTACT_LOCAL_METHODS
        | POST_CONTACT_RGB_CONTROL_METHODS
        | POST_CONTACT_GOAL_GATED_METHODS
        | POST_CONTACT_RESIDUAL_GATED_METHODS
        | POST_CONTACT_HOLDOUT_CALIBRATION_METHODS
        | POST_CONTACT_HOLDOUT_GATED_METHODS
        | POST_CONTACT_TASK_PROBE_CARRIER_METHODS
        | POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_TASK_PROBE_GATED_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
    )


def _post_contact_study_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    """Declare optional repair studies as calibration/carrier/gated triplets.

    Keeping this plan separate from ``main`` prevents a subtle deployment
    regression: an advertised CLI switch must actually produce (i) a
    calibration-only local arm, (ii) a probe-budget-matched RGB carrier, and
    (iii) the gated held-root deployment arm.  The gate is never fitted on
    the held outcome roots.
    """
    studies: list[dict[str, str]] = []
    if bool(args.goal_gated_local):
        studies.append({
            "name": "goal_gate",
            "carrier_method": "rgb_geometry_action_post_contact_probe",
            "calibration_method": "rgb_verified_effect_post_contact",
            "gated_method": "rgb_verified_effect_post_contact_goal_gated",
            "gate_key": "goal_local_gate",
        })
    if bool(args.residual_gated_local):
        studies.append({
            "name": "residual_gate",
            "carrier_method": "rgb_geometry_action_post_contact_probe",
            "calibration_method": "rgb_verified_effect_post_contact",
            "gated_method": "rgb_verified_effect_post_contact_residual_gated",
            "gate_key": "residual_local_gate",
        })
    if bool(args.holdout_residual_gated_local):
        studies.append({
            "name": "holdout_residual_gate",
            "carrier_method": "rgb_geometry_action_post_contact_probe",
            "calibration_method": "rgb_verified_effect_post_contact_holdout_calibration",
            "gated_method": "rgb_verified_effect_post_contact_holdout_residual_gated",
            "gate_key": "residual_local_gate",
        })
    if bool(args.task_probe_gated_local):
        studies.append({
            "name": "task_probe",
            "carrier_method": "rgb_geometry_action_post_contact_task_probe",
            "calibration_method": "rgb_verified_effect_post_contact_task_probe_calibration",
            "gated_method": "rgb_verified_effect_post_contact_task_probe_gated",
            "gate_key": "task_probe_gate",
        })
    if bool(args.adaptive_task_probe_local):
        studies.append({
            "name": "adaptive_task_probe",
            "carrier_method": "rgb_geometry_action_post_contact_adaptive_task_probe",
            "calibration_method": "rgb_verified_effect_post_contact_adaptive_task_probe_calibration",
            "gated_method": "rgb_verified_effect_post_contact_adaptive_task_probe_gated",
            "gate_key": "adaptive_task_probe_gate",
        })
    return studies


def _validate_post_contact_protocol(
    studies: list[dict[str, str]], *, approach_steps: int,
    prelude_steps: int, phase_order: str,
) -> None:
    """Reject a nominally local repair that never establishes RGB action support.

    A post-contact chart with no declared approach can consist entirely of
    zero-motion rows: that is a useful abstention diagnostic but not an
    identification experiment.  Requiring an explicit approach keeps an
    enabled repair study from silently consuming a contact-free prelude and
    presenting its inevitable fallback as a local-control result.
    """
    if not studies:
        return
    if int(approach_steps) <= 0:
        raise ValueError(
            "post-contact repair studies require a positive --approach-steps "
            "to establish an RGB action-effect interface"
        )
    if int(prelude_steps) != int(approach_steps) + 20:
        raise ValueError(
            "post-contact repair studies require prelude-steps == approach-steps + 20"
        )
    if phase_order not in {"descent_then_align", "align_then_descend"}:
        raise ValueError(f"unsupported post-contact repair phase order: {phase_order}")


def _readout_family_name(method: str) -> str:
    """Map control-only variants to the calibrated feature family they use."""
    if method in {
        "rgb_verified_effect_fused", "rgb_verified_effect_gated",
        *POST_CONTACT_RGB_CONTROL_METHODS,
        *POST_CONTACT_GOAL_GATED_METHODS,
        *POST_CONTACT_RESIDUAL_GATED_METHODS,
        *POST_CONTACT_HOLDOUT_CALIBRATION_METHODS,
        *POST_CONTACT_HOLDOUT_GATED_METHODS,
        *POST_CONTACT_TASK_PROBE_CARRIER_METHODS,
        *POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS,
        *POST_CONTACT_TASK_PROBE_GATED_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS,
    }:
        return "rgb_geometry_action"
    return method


def _fit_goal_local_gate(
    calibration_initials: list[dict[str, Any]],
    local_episodes: list[dict[str, Any]],
    carrier_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze an RGB-goal local-permission gate from calibration roots only.

    The gate is intentionally conservative: a goal chart permits the local
    effect map only when its calibration mean is non-positive *and* it has no
    observed loss against the matched carrier.  Task ids are used only to
    group calibration roots; runtime resolves the chart from goal-image RGB
    centroids, never an environment task id or privileged state.
    """
    initial_by_seed = {int(row["seed"]): row for row in calibration_initials}
    local_by_seed = {int(row["seed"]): row for row in local_episodes}
    carrier_by_seed = {int(row["seed"]): row for row in carrier_episodes}
    grouped: dict[int, list[tuple[float, np.ndarray]]] = {}
    for seed, initial in initial_by_seed.items():
        if seed not in local_by_seed or seed not in carrier_by_seed:
            raise ValueError("calibration goal gate requires paired local/carrier episodes")
        goal_centroid = np.asarray(initial["goal_rgb_centroid"], dtype=np.float64).reshape(2)
        delta = (
            float(local_by_seed[seed]["final_goal_distance_m"])
            - float(carrier_by_seed[seed]["final_goal_distance_m"])
        )
        grouped.setdefault(int(initial["task_id"]), []).append((delta, goal_centroid))
    charts: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        values = np.asarray([row[0] for row in grouped[task_id]], dtype=np.float64)
        prototype = np.mean(np.stack([row[1] for row in grouped[task_id]]), axis=0)
        losses = int(np.sum(values > 0.0))
        charts.append({
            "calibration_task_id": int(task_id),
            "goal_centroid": prototype.tolist(),
            "mean_local_minus_carrier": float(values.mean()),
            "wins": int(np.sum(values < 0.0)),
            "losses": losses,
            "ties": int(np.sum(values == 0.0)),
            "permit_local": bool(values.mean() <= 0.0 and losses == 0),
        })
    return {
        "schema": "cube_goal_local_gate.v1",
        "selection": "calibration mean <= 0 and zero local losses; RGB goal nearest-prototype runtime lookup",
        "charts": charts,
    }


def _goal_gate_allows_local(goal_image: np.ndarray, gate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Resolve the frozen goal chart from RGB only and return its permission."""
    centroid = _target_red_centroid(np.asarray(goal_image, dtype=np.uint8))[:2]
    charts = list(gate.get("charts", []))
    if not charts or centroid.shape != (2,) or not np.isfinite(centroid).all():
        return False, {"status": "goal_gate_invalid_or_missing"}
    prototypes = np.asarray([row["goal_centroid"] for row in charts], dtype=np.float64)
    distances = np.linalg.norm(prototypes - centroid.reshape(1, 2), axis=1)
    index = int(np.argmin(distances))
    chart = charts[index]
    return bool(chart["permit_local"]), {
        "status": "goal_gate_local_permit" if bool(chart["permit_local"]) else "goal_gate_rgb_fallback",
        "goal_centroid": centroid.astype(np.float64).tolist(),
        "matched_calibration_task_id": int(chart["calibration_task_id"]),
        "goal_prototype_distance": float(distances[index]),
    }


def _fit_local_residual_gate(
    calibration_episodes: list[dict[str, Any]], *, alpha: float,
) -> dict[str, Any]:
    """Fit a root-level split-conformal residual gate on calibration only.

    Each root contributes its maximum normalized one-step residual after the
    fixed support probes.  This makes the calibration unit match the
    root-disjoint deployment unit rather than treating correlated controller
    steps as independent samples.  The finite-sample conformal order statistic
    is intentionally *not* clipped: if ``ceil((n + 1) (1-alpha)) > n`` there
    is no finite distribution-free threshold and the gate fails closed.
    """
    if not 0.0 < float(alpha) < 1.0 or not np.isfinite(float(alpha)):
        raise ValueError("residual gate alpha must be finite and in (0, 1)")
    scores: list[float] = []
    missing_roots = 0
    for episode in calibration_episodes:
        values = np.asarray(episode.get("verified_residual_scores", []), dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            missing_roots += 1
            scores.append(float("inf"))
        else:
            scores.append(float(np.max(values)))
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    n = int(ordered.size)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    finite = bool(n > 0 and rank <= n and np.isfinite(ordered[rank - 1]))
    threshold = float(ordered[rank - 1]) if finite else float("inf")
    return {
        "schema": "cube_local_residual_gate.v1",
        "calibration_unit": "maximum normalized one-step local-map residual per root",
        "calibration_roots": n,
        "alpha": float(alpha),
        "conformal_rank": rank,
        "root_scores": ordered.tolist(),
        "roots_without_local_residual": int(missing_roots),
        "threshold": threshold,
        "certified_finite_threshold": finite,
        "runtime_action": (
            "retain and update local chart while observed residual <= threshold; "
            "permanently fall back to matched RGB carrier after first exceedance"
            if finite else
            "no finite calibration threshold; reject local chart and use matched RGB carrier"
        ),
    }


def _fit_task_probe_gate(
    calibration_episodes: list[dict[str, Any]], *, alpha: float,
) -> dict[str, Any]:
    """Fit a root-level conformal gate for task-direction probe mismatch.

    Unlike the support-axis holdout, each score is computed from the actual
    candidate direction proposed by the local map and a same-direction,
    low-amplitude observed pulse.  A root therefore contributes the one
    pre-release decision it would make at deployment.  Missing or
    sign-inconsistent observations are rejected by the preceding observed
    positive-progress condition; the conformal quantile then has the explicit
    conditional target of calibrating mismatch among eligible probes.
    """
    if not 0.0 < float(alpha) < 1.0 or not np.isfinite(float(alpha)):
        raise ValueError("task probe gate alpha must be finite and in (0, 1)")
    scores: list[float] = []
    rejected_roots = 0
    for episode in calibration_episodes:
        values = np.asarray(episode.get("task_probe_scores", []), dtype=np.float64)
        values = values[np.isfinite(values)]
        # The preceding deployment condition is observed positive progress in
        # the same task direction.  Infinite scores encode a failed sign test,
        # not a sample from the conditional mismatch distribution.  Including
        # them as ``inf`` would silently change the gate into an unconditional
        # coverage claim while it still abstains on exactly those roots.
        if values.size == 1:
            scores.append(float(values[0]))
        else:
            rejected_roots += 1
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    n = int(ordered.size)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    finite = bool(n > 0 and rank <= n and np.isfinite(ordered[rank - 1]))
    threshold = float(ordered[rank - 1]) if finite else float("inf")
    return {
        "schema": "cube_task_aligned_probe_gate.v1",
        "calibration_unit": (
            "one normalized task-direction mismatch for the locally proposed "
            "candidate-direction probe per root"
        ),
        "calibration_roots": n,
        "all_calibration_roots": int(len(calibration_episodes)),
        "alpha": float(alpha),
        "conformal_rank": rank,
        "root_scores": ordered.tolist(),
        "roots_rejected_before_conditional_calibration": int(rejected_roots),
        "threshold": threshold,
        "certified_finite_threshold": finite,
        "runtime_action": (
            "conditional on positive observed task progress, release full local action "
            "only when task-probe mismatch <= threshold; otherwise use matched RGB carrier"
            if finite else
            "too few positive-progress calibration probes for a finite conditional "
            "threshold; use matched RGB carrier after the shared probe"
        ),
    }


def _local_prediction_residual(
    mapping: np.ndarray, action_xy: np.ndarray, observed_delta: np.ndarray,
    *, response_rms: float, drift_delta: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure a scale-normalized RGB effect residual for one released pulse."""
    map_array = np.asarray(mapping, dtype=np.float64)
    action_array = np.asarray(action_xy, dtype=np.float64).reshape(2)
    observed_array = np.asarray(observed_delta, dtype=np.float64).reshape(2)
    if (
        map_array.shape != (2, 2)
        or not np.isfinite(map_array).all()
        or not np.isfinite(action_array).all()
        or not np.isfinite(observed_array).all()
    ):
        return {
            "status": "invalid_observed_rgb_effect",
            "normalized_residual": float("inf"),
            "raw_residual": float("inf"),
        }
    drift = (
        np.zeros(2, dtype=np.float64) if drift_delta is None
        else np.asarray(drift_delta, dtype=np.float64).reshape(2)
    )
    if not np.isfinite(drift).all():
        return {
            "status": "invalid_drift_delta",
            "normalized_residual": float("inf"),
            "raw_residual": float("inf"),
        }
    predicted = action_array @ map_array + drift
    raw = float(np.linalg.norm(observed_array - predicted))
    # The denominator uses only pre-action local-map information.  The floor
    # prevents a visually static contact from turning numerical pixel noise
    # into an apparent high-confidence response.
    scale = max(float(np.linalg.norm(predicted)), float(response_rms), 1.0e-6)
    return {
        "status": "ok",
        "predicted_delta": predicted.astype(np.float64).tolist(),
        "drift_delta": drift.astype(np.float64).tolist(),
        "observed_delta": observed_array.astype(np.float64).tolist(),
        "raw_residual": raw,
        "normalization_scale": scale,
        "normalized_residual": float(raw / scale),
    }


def _task_aligned_probe_score(
    mapping: np.ndarray, full_action_xy: np.ndarray, probe_action_xy: np.ndarray,
    desired_delta: np.ndarray, observed_delta: np.ndarray, *, response_rms: float,
    drift_delta: np.ndarray | None = None,
    probe_scale: float = TASK_ALIGNED_PROBE_SCALE,
) -> dict[str, Any]:
    """Score whether a small probe validates the local action needed by the task.

    The vector residual is insufficient for contact tasks: an action can agree
    in a transverse component while moving the object away from its goal.  The
    score instead projects both predicted and observed effects onto the RGB
    goal direction chosen *before* the probe.  Non-positive expected or
    observed progress is an explicit fail-closed condition, independent of a
    potentially permissive residual quantile.
    """
    map_array = np.asarray(mapping, dtype=np.float64)
    full = np.asarray(full_action_xy, dtype=np.float64).reshape(2)
    probe = np.asarray(probe_action_xy, dtype=np.float64).reshape(2)
    desired = np.asarray(desired_delta, dtype=np.float64).reshape(2)
    observed = np.asarray(observed_delta, dtype=np.float64).reshape(2)
    if (
        map_array.shape != (2, 2)
        or not np.isfinite(map_array).all()
        or not np.isfinite(full).all()
        or not np.isfinite(probe).all()
        or not np.isfinite(desired).all()
        or not np.isfinite(observed).all()
        or float(np.linalg.norm(desired)) <= 1.0e-8
        or not np.isfinite(float(response_rms))
        or not np.isfinite(float(probe_scale))
        or not 0.0 < float(probe_scale) <= 1.0
    ):
        return {
            "status": "invalid_task_aligned_probe",
            "normalized_task_mismatch": float("inf"),
            "task_progress_positive": False,
        }
    goal_direction = desired / np.linalg.norm(desired)
    drift = (
        np.zeros(2, dtype=np.float64) if drift_delta is None
        else np.asarray(drift_delta, dtype=np.float64).reshape(2)
    )
    if not np.isfinite(drift).all():
        return {
            "status": "invalid_drift_delta",
            "normalized_task_mismatch": float("inf"),
            "task_progress_positive": False,
        }
    predicted_full = full @ map_array + drift
    predicted_probe = probe @ map_array + drift
    predicted_progress = float(predicted_probe @ goal_direction)
    observed_progress = float(observed @ goal_direction)
    scale = max(abs(predicted_progress), float(response_rms) * float(probe_scale), 1.0e-6)
    mismatch = float(abs(observed_progress - predicted_progress) / scale)
    positive = bool(predicted_progress > 0.0 and observed_progress > 0.0)
    return {
        "status": "ok" if positive else "nonpositive_task_progress",
        "goal_direction": goal_direction.astype(np.float64).tolist(),
        "full_action_xy": full.astype(np.float64).tolist(),
        "probe_action_xy": probe.astype(np.float64).tolist(),
        "probe_scale": float(probe_scale),
        "predicted_full_delta": predicted_full.astype(np.float64).tolist(),
        "predicted_probe_delta": predicted_probe.astype(np.float64).tolist(),
        "drift_delta": drift.astype(np.float64).tolist(),
        "observed_probe_delta": observed.astype(np.float64).tolist(),
        "predicted_task_progress": predicted_progress,
        "observed_task_progress": observed_progress,
        "task_progress_positive": positive,
        "normalization_scale": scale,
        "normalized_task_mismatch": mismatch if positive else float("inf"),
    }


def _post_contact_holdout_residual(
    action_history: list[np.ndarray], delta_history: list[np.ndarray],
) -> dict[str, Any]:
    """Validate a post-contact chart on the fourth, withheld support pulse.

    The first three axis pulses span rank two.  The fourth pulse is deliberately
    excluded from the fitted map, so this is an actual post-contact prediction
    check rather than in-sample support reconstruction.  It is evaluated before
    the controller is allowed to issue any task-directed local action.
    """
    if len(action_history) != POST_CONTACT_PROBE_STEPS or len(delta_history) != POST_CONTACT_PROBE_STEPS:
        return {"status": "holdout_support_incomplete", "normalized_residual": float("inf")}
    fit_actions = np.asarray(action_history[:3], dtype=np.float64)
    fit_deltas = np.asarray(delta_history[:3], dtype=np.float64)
    heldout_action = np.asarray(action_history[3], dtype=np.float64)
    heldout_delta = np.asarray(delta_history[3], dtype=np.float64)
    if (
        fit_actions.shape != (3, 2)
        or fit_deltas.shape != (3, 2)
        or not np.isfinite(fit_actions).all()
        or not np.isfinite(fit_deltas).all()
    ):
        return {"status": "invalid_holdout_support", "normalized_residual": float("inf")}
    rank = int(np.linalg.matrix_rank(fit_actions))
    if rank < 2:
        return {"status": "holdout_rank_deficient", "rank": rank, "normalized_residual": float("inf")}
    mapping, _, _, singular_values = np.linalg.lstsq(fit_actions, fit_deltas, rcond=1.0e-6)
    if len(singular_values) < 2 or singular_values[-1] <= 1.0e-6:
        return {"status": "holdout_ill_conditioned", "rank": rank, "normalized_residual": float("inf")}
    residual = _local_prediction_residual(
        mapping, heldout_action, heldout_delta,
        response_rms=float(np.sqrt(np.mean(fit_deltas ** 2))),
    )
    return {
        **residual,
        "status": "ok" if residual["status"] == "ok" else residual["status"],
        "rank": rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "withheld_probe_index": 3,
    }


def _paired_root_summary(
    deltas: np.ndarray, *, seed: int, draws: int = 20_000,
) -> dict[str, Any]:
    """Root-level paired uncertainty summary for a predeclared comparison."""
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired root inference requires at least two finite deltas")
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    sampled_means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    wins = int(np.sum(values < 0.0))
    losses = int(np.sum(values > 0.0))
    non_ties = wins + losses
    if non_ties == 0:
        exact_two_sided_sign_p = 1.0
    else:
        lower_tail = sum(
            math.comb(non_ties, count) for count in range(min(wins, losses) + 1)
        ) / (2 ** non_ties)
        exact_two_sided_sign_p = float(min(1.0, 2.0 * lower_tail))
    return {
        "n_roots": int(values.size),
        "mean_final_distance_delta_m": float(values.mean()),
        "paired_bootstrap_ci95": [
            float(np.quantile(sampled_means, 0.025)),
            float(np.quantile(sampled_means, 0.975)),
        ],
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(values == 0.0)),
        "exact_two_sided_sign_p": exact_two_sided_sign_p,
        "unit": "paired outcome root",
    }


def _serialize_readout_fit(
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[list[Any]]:
    """Serialize one frozen ridge readout without retaining device state."""
    return [tensor.detach().cpu().tolist() for tensor in fit]


def _deserialize_readout_fit(
    payload: Any, *, device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore a frozen ridge readout and reject malformed calibration bundles."""
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("frozen calibration readout must contain mean, scale, and weight")
    fit = tuple(torch.as_tensor(value, device=device, dtype=torch.float64) for value in payload)
    if any(tensor.ndim < 1 or not torch.isfinite(tensor).all() for tensor in fit):
        raise ValueError("frozen calibration readout contains invalid values")
    return fit  # type: ignore[return-value]


def _load_frozen_calibration_bundle(
    path: Path, *, device: str, phase_order: str = "descent_then_align",
) -> dict[str, Any]:
    """Load the immutable calibration state used by a confirmation panel."""
    source = json.loads(path.read_text())
    bundle = source.get("frozen_calibration_bundle", source)
    if bundle.get("schema") != "cube_temporal_calibration_bundle.v1":
        raise ValueError("unsupported frozen calibration bundle schema")
    bundle_phase_order = str(bundle.get("phase_order", "descent_then_align"))
    if bundle_phase_order != phase_order:
        raise ValueError(
            "frozen calibration bundle phase order differs from the deployment protocol"
        )
    required = {"fits", "pixel_action_mapping", "pixel_mapping_report"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"frozen calibration bundle missing {missing}")
    fits = {
        str(name): _deserialize_readout_fit(value, device=device)
        for name, value in dict(bundle["fits"]).items()
    }
    expected_fits = {
        "latent_static", "latent_temporal", "rgb_geometry_action", "rgb_geometry_temporal_action",
    }
    if set(fits) != expected_fits:
        raise ValueError("frozen calibration bundle has an incompatible readout family set")
    mapping = np.asarray(bundle["pixel_action_mapping"], dtype=np.float32)
    if mapping.shape != (2, 2) or not np.isfinite(mapping).all():
        raise ValueError("frozen calibration bundle has an invalid pixel-action mapping")
    task_gate = bundle.get("task_probe_gate")
    if task_gate is not None:
        task_gate = dict(task_gate)
        if task_gate.get("schema") != "cube_task_aligned_probe_gate.v1":
            raise ValueError("frozen calibration bundle has an invalid task-probe gate")
    adaptive_task_gate = bundle.get("adaptive_task_probe_gate")
    if adaptive_task_gate is not None:
        adaptive_task_gate = dict(adaptive_task_gate)
        if adaptive_task_gate.get("schema") != "cube_task_aligned_probe_gate.v1":
            raise ValueError("frozen calibration bundle has an invalid adaptive task-probe gate")
    return {
        **bundle,
        "fits": fits,
        "pixel_action_mapping": mapping,
        "task_probe_gate": task_gate,
        "adaptive_task_probe_gate": adaptive_task_gate,
        "phase_order": bundle_phase_order,
        "source_path": str(path),
    }


def _rgb_block_goal_error(current: np.ndarray, goal: np.ndarray) -> tuple[float, bool]:
    """Return RGB block-to-goal error norm for a deployment-only feedback gate."""
    current_xy = _target_red_centroid(current)[:2]
    goal_xy = _target_red_centroid(goal)[:2]
    valid = bool(current_xy.shape == (2,) and goal_xy.shape == (2,))
    valid = valid and bool(np.linalg.norm(current_xy) > 0.0 or np.linalg.norm(goal_xy) > 0.0)
    if not valid:
        return 0.0, False
    return float(np.linalg.norm(goal_xy - current_xy)), True


def _rgb_block_goal_delta(current: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the RGB-only block-to-goal vector used by verified local control."""
    current_xy = _target_red_centroid(current)[:2]
    goal_xy = _target_red_centroid(goal)[:2]
    valid = bool(current_xy.shape == (2,) and goal_xy.shape == (2,))
    valid = valid and bool(np.isfinite(current_xy).all() and np.isfinite(goal_xy).all())
    if not valid:
        return np.zeros(2, dtype=np.float32), False
    return (goal_xy - current_xy).astype(np.float32), True


def _verified_rgb_action(
    action_history: list[np.ndarray], delta_history: list[np.ndarray],
    desired_delta: np.ndarray, candidates: np.ndarray,
    *, recency_decay: float | None = None, include_drift: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Select an action only from an RGB-observed local effect map.

    This is the visual analogue of the online-identification precedents in the
    literature.  It never consumes simulator state.  Until the observed action
    support has two independent directions and four rows, it returns ``None``
    so the caller must use the fixed exploratory support pulse.  A local map is
    therefore a verified update, not an unvalidated extrapolation from one
    contact direction.
    """
    actions = np.asarray(action_history, dtype=np.float64)
    deltas = np.asarray(delta_history, dtype=np.float64)
    candidate_xy = np.asarray(candidates, dtype=np.float64)[:, :2]
    if len(action_history) < 4 or actions.ndim != 2 or actions.shape[1] != 2:
        return None, {"status": "insufficient_support", "rows": int(len(action_history)), "rank": 0}
    if deltas.shape != actions.shape or not np.isfinite(actions).all() or not np.isfinite(deltas).all():
        return None, {"status": "invalid_rgb_effect_history", "rows": int(len(action_history))}
    if recency_decay is not None and (
        not np.isfinite(float(recency_decay)) or not 0.0 < float(recency_decay) <= 1.0
    ):
        raise ValueError("recency_decay must be finite and in (0, 1]")
    rank = int(np.linalg.matrix_rank(actions))
    if rank < 2:
        return None, {"status": "rank_deficient_support", "rows": int(len(action_history)), "rank": rank}
    design = (
        np.column_stack([actions, np.ones(len(actions), dtype=np.float64)])
        if include_drift else actions
    )
    if recency_decay is None:
        weights = np.ones(len(actions), dtype=np.float64)
    else:
        age = np.arange(len(actions) - 1, -1, -1, dtype=np.float64)
        weights = np.power(float(recency_decay), age)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_deltas = deltas * np.sqrt(weights)[:, None]
    coefficients, _, _, singular_values = np.linalg.lstsq(
        weighted_design, weighted_deltas, rcond=1.0e-6,
    )
    required_rank = 3 if include_drift else 2
    if len(singular_values) < required_rank or singular_values[-1] <= 1.0e-6:
        return None, {"status": "ill_conditioned_support", "rows": int(len(action_history)), "rank": rank}
    mapping = coefficients[:2]
    drift = coefficients[2] if include_drift else np.zeros(2, dtype=np.float64)
    predicted_delta = candidate_xy @ mapping
    scores = predicted_delta @ np.asarray(desired_delta, dtype=np.float64).reshape(2)
    selected = int(np.argmax(scores))
    response_rms = float(np.sqrt(np.mean(deltas ** 2)))
    antisymmetry: list[float] = []
    for first in range(len(actions)):
        for second in range(first + 1, len(actions)):
            if np.linalg.norm(actions[first] + actions[second]) <= 1.0e-6:
                denominator = (
                    np.linalg.norm(deltas[first]) + np.linalg.norm(deltas[second])
                )
                if denominator > 1.0e-8:
                    antisymmetry.append(float(
                        np.linalg.norm(deltas[first] + deltas[second]) / denominator
                    ))
    return np.asarray(candidates[selected], dtype=np.float32), {
        "status": "ok",
        "rows": int(len(action_history)),
        "rank": rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "response_rms": response_rms,
        "recency_decay": (float(recency_decay) if recency_decay is not None else None),
        "estimated_drift_delta": drift.astype(np.float64).tolist(),
        "weighted_rows": int(len(actions)),
        "opposite_action_antisymmetry": (
            float(np.mean(antisymmetry)) if antisymmetry else None
        ),
        "mapping": mapping.astype(np.float64).tolist(),
        "selected_index": selected,
        "predicted_progress": scores.astype(np.float64).tolist(),
    }


def _robust_rank_score(scores: torch.Tensor) -> torch.Tensor:
    """Normalize candidate costs without trusting either scale or variance.

    Latent and RGB readouts live on different regression scales.  A direct
    sum would therefore let one view dominate for purely numerical reasons.
    Median/IQR normalization is fixed at deployment and uses only the current
    candidate bank; it cannot inspect evaluator state or test outcomes.
    """
    if scores.ndim != 1 or scores.numel() < 2:
        raise ValueError("rank normalization expects a one-dimensional candidate vector")
    median = scores.median()
    quantiles = torch.tensor([0.25, 0.75], device=scores.device, dtype=scores.dtype)
    q25, q75 = torch.quantile(scores, quantiles)
    scale = (q75 - q25).clamp_min(1.0e-6)
    return (scores - median) / scale


def _rank_consensus_score(
    latent_scores: torch.Tensor, rgb_scores: torch.Tensor,
) -> torch.Tensor:
    """Return a fixed two-view rank-consensus cost for one candidate bank."""
    if latent_scores.shape != rgb_scores.shape:
        raise ValueError("latent and RGB candidate score shapes must match")
    return _robust_rank_score(latent_scores) + _robust_rank_score(rgb_scores)


def _run_arm(
    *, initial_conditions: list[dict[str, Any]], model: torch.nn.Module,
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    candidates: np.ndarray,
    device: str, max_steps: int, prelude_steps: int, batch_size: int,
    method: str,
    support_gate: bool = False,
    support_threshold: float = 5.0e-4,
    pixel_feedback: bool = False,
    pixel_feedback_threshold: float = 1.0e-3,
    pixel_feedback_mode: str = "persistent",
    approach_steps: int = 0,
    approach_gain: float = 1.0,
    phase_order: str = "descent_then_align",
    pixel_action_mapping: np.ndarray | None = None,
    goal_local_gate: dict[str, Any] | None = None,
    residual_local_gate: dict[str, Any] | None = None,
    task_probe_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if pixel_feedback_mode not in {"persistent", "one_step"}:
        raise ValueError(f"unsupported pixel feedback mode: {pixel_feedback_mode}")
    episodes = []
    uses_verified_local = method in {
        "rgb_verified_effect", "rgb_verified_effect_fused",
        "rgb_verified_effect_gated", *POST_CONTACT_LOCAL_METHODS,
        *POST_CONTACT_GOAL_GATED_METHODS,
        *POST_CONTACT_RESIDUAL_GATED_METHODS,
        *POST_CONTACT_HOLDOUT_CALIBRATION_METHODS,
        *POST_CONTACT_HOLDOUT_GATED_METHODS,
        *POST_CONTACT_TASK_PROBE_CARRIER_METHODS,
        *POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS,
        *POST_CONTACT_TASK_PROBE_GATED_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS,
        *POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS,
    }
    uses_post_contact_probe = _uses_post_contact_probe(method)
    uses_residual_local_gate = method in POST_CONTACT_RESIDUAL_GATED_METHODS
    uses_holdout_local_gate = method in POST_CONTACT_HOLDOUT_GATED_METHODS
    uses_holdout_calibration = method in POST_CONTACT_HOLDOUT_CALIBRATION_METHODS
    uses_task_probe = method in (
        POST_CONTACT_TASK_PROBE_CARRIER_METHODS
        | POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_TASK_PROBE_GATED_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
    )
    uses_adaptive_task_probe = method in (
        POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
    )
    uses_task_probe_local_gate = method in (
        POST_CONTACT_TASK_PROBE_GATED_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
    )
    uses_task_probe_calibration = method in (
        POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS
        | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
    )
    if uses_residual_local_gate and residual_local_gate is None:
        raise ValueError("residual-gated local method requires a calibration-only residual gate")
    if uses_holdout_local_gate and residual_local_gate is None:
        raise ValueError("holdout-gated local method requires a calibration-only residual gate")
    if uses_task_probe_local_gate and task_probe_gate is None:
        raise ValueError("task-probe-gated local method requires a calibration-only task probe gate")

    def _runtime_local_action(
        action_history: list[np.ndarray], delta_history: list[np.ndarray],
        desired_delta: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Fit the declared local chart without consulting evaluator state.

        The adaptive arm is deliberately limited to a recency-weighted affine
        RGB chart.  Its drift estimate corrects validation residuals, while the
        action ranking still uses only action-induced effects.
        """
        return _verified_rgb_action(
            action_history, delta_history, desired_delta, candidates,
            recency_decay=(TASK_PROBE_RECENCY_DECAY if uses_adaptive_task_probe else None),
            include_drift=uses_adaptive_task_probe,
        )

    for initial in initial_conditions:
        env = _make_env()
        try:
            obs, info = env.reset(
                seed=int(initial["seed"]),
                options={
                    "task_id": int(initial["task_id"]),
                    "render_goal": True,
                    "variation_values": copy.deepcopy(initial["variation_values"]),
                    "state": np.asarray(initial["state"], dtype=np.float64),
                },
            )
            goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
            current = np.asarray(obs, dtype=np.uint8).copy()
            previous = current.copy()
            previous_action = np.zeros(5, dtype=np.float32)
            actions: list[list[float]] = []
            distances: list[float] = []
            fallback_count = 0
            pixel_feedback_sign = 1.0
            pixel_feedback_reversals = 0
            last_pixel_error_norm: float | None = None
            pixel_error_trace: list[float] = []
            pixel_feedback_sign_trace: list[float] = []
            verified_actions: list[np.ndarray] = []
            verified_deltas: list[np.ndarray] = []
            verified_trace: list[dict[str, Any]] = []
            verified_model_disagreement_fallbacks = 0
            verified_residual_scores: list[float] = []
            residual_gate_fallbacks = 0
            task_probe_scores: list[float] = []
            task_probe_fallbacks = 0
            task_probe_local_permitted = False
            pending_task_probe: dict[str, Any] | None = None
            task_probe_runtime_reweighting = bool(uses_adaptive_task_probe)
            residual_local_disabled = bool(
                (uses_residual_local_gate or uses_holdout_local_gate)
                and not bool(residual_local_gate["certified_finite_threshold"])
            )
            for step in range(max_steps):
                reverse_current_pulse = False
                released_local_diagnostics: dict[str, Any] | None = None
                current_pixel_error, current_pixel_valid = _rgb_block_goal_error(current, goal_image)
                if current_pixel_valid:
                    pixel_error_trace.append(current_pixel_error)
                    if step == prelude_steps:
                        pixel_feedback_sign = 1.0
                        last_pixel_error_norm = current_pixel_error
                    elif pixel_feedback and step > prelude_steps and last_pixel_error_norm is not None:
                        if current_pixel_error > last_pixel_error_norm + pixel_feedback_threshold:
                            if pixel_feedback_mode == "persistent":
                                pixel_feedback_sign *= -1.0
                            else:
                                reverse_current_pulse = True
                            pixel_feedback_reversals += 1
                        last_pixel_error_norm = current_pixel_error
                if step < prelude_steps:
                    action = _prelude_action(
                        step, current, approach_steps=approach_steps,
                        approach_gain=approach_gain,
                        pixel_action_mapping=pixel_action_mapping,
                        phase_order=phase_order,
                    )
                else:
                    planning_step = step - prelude_steps
                    n_candidates = len(candidates)
                    current_batch = np.repeat(current[None], n_candidates, axis=0)
                    previous_batch = np.repeat(previous[None], n_candidates, axis=0)
                    goal_batch = np.repeat(goal_image[None], n_candidates, axis=0)
                    previous_actions = np.repeat(previous_action[None], n_candidates, axis=0)
                    valid = np.ones(n_candidates, dtype=np.float32)
                    families, _ = build_feature_families(
                        current_batch, previous_batch, goal_batch, candidates,
                        previous_actions, valid, device=device, model=model,
                        batch_size=batch_size,
                    )
                    if method in {
                        "rgb_geometry_action", "latent_rgb_rank_consensus",
                        "rgb_verified_effect_fused", "rgb_verified_effect_gated",
                        *POST_CONTACT_RGB_CONTROL_METHODS,
                        *POST_CONTACT_GOAL_GATED_METHODS,
                        *POST_CONTACT_RESIDUAL_GATED_METHODS,
                        *POST_CONTACT_HOLDOUT_CALIBRATION_METHODS,
                        *POST_CONTACT_HOLDOUT_GATED_METHODS,
                        *POST_CONTACT_TASK_PROBE_CARRIER_METHODS,
                        *POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS,
                        *POST_CONTACT_TASK_PROBE_GATED_METHODS,
                        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CARRIER_METHODS,
                        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS,
                        *POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS,
                    }:
                        families["rgb_geometry_action"] = extract_color_geometry_features(
                            current_batch, goal_batch, candidates, device,
                        )["color_geometry_action"]
                    elif method == "rgb_geometry_temporal_action":
                        families[method] = extract_color_geometry_temporal_features(
                            current_batch, previous_batch, goal_batch, candidates,
                            previous_actions, device,
                        )["color_geometry_temporal_action"]
                    if method in {"rgb_verified_effect", *POST_CONTACT_LOCAL_METHODS}:
                        # The verified controller deliberately ignores the
                        # frozen predictor at selection time.  Its candidate
                        # score comes from the RGB-observed local effect map
                        # below; keep a finite placeholder for shared logging.
                        predicted_scores = torch.zeros(
                            n_candidates, device=device, dtype=torch.float32,
                        )
                    elif method == "latent_rgb_rank_consensus":
                        if not isinstance(fit, dict):
                            raise TypeError("rank consensus requires both latent and RGB fits")
                        latent = _predict(families["latent_temporal"], fit["latent_temporal"])
                        rgb = _predict(families["rgb_geometry_action"], fit["rgb_geometry_action"])
                        latent_scores = torch.linalg.vector_norm(latent, dim=1)
                        rgb_scores = torch.linalg.vector_norm(rgb, dim=1)
                        predicted_scores = _rank_consensus_score(latent_scores, rgb_scores)
                    else:
                        if not isinstance(fit, tuple):
                            raise TypeError(f"single-view method {method} requires one fit")
                        family_name = _readout_family_name(method)
                        predicted = _predict(families[family_name], fit)
                        predicted_scores = torch.linalg.vector_norm(predicted, dim=1)
                    spread = float((predicted_scores.max() - predicted_scores.min()).item())
                    if uses_post_contact_probe and planning_step < POST_CONTACT_PROBE_STEPS:
                        # This is deliberately shared between the local chart
                        # and RGB carrier: a local update may not receive free
                        # state-identification actions unavailable to its
                        # comparison controller.
                        action = _support_exploration_action(planning_step, candidates)
                        fallback_count += 1
                        if method in (
                            POST_CONTACT_LOCAL_METHODS
                            | POST_CONTACT_GOAL_GATED_METHODS
                            | POST_CONTACT_RESIDUAL_GATED_METHODS
                            | POST_CONTACT_HOLDOUT_CALIBRATION_METHODS
                            | POST_CONTACT_HOLDOUT_GATED_METHODS
                            | POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS
                            | POST_CONTACT_TASK_PROBE_GATED_METHODS
                            | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
                            | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
                        ):
                            verified_trace.append({
                                "status": "post_contact_support_probe",
                                "rows": len(verified_actions),
                                "probe_step": int(planning_step),
                            })
                    elif uses_task_probe and planning_step == POST_CONTACT_PROBE_STEPS:
                        desired_delta, desired_valid = _rgb_block_goal_delta(current, goal_image)
                        proposed_action, probe_diagnostics = _runtime_local_action(
                            verified_actions, verified_deltas, desired_delta,
                        ) if desired_valid else (None, {
                            "status": "invalid_rgb_goal", "rows": len(verified_actions),
                        })
                        if proposed_action is None:
                            # No local chart means there is no task direction
                            # whose small-scale effect can be validated.  Use
                            # the same deterministic support fallback in both
                            # arms and prohibit a local release afterwards.
                            action = _support_exploration_action(planning_step, candidates)
                            fallback_count += 1
                            if uses_task_probe_local_gate:
                                task_probe_local_permitted = False
                            probe_diagnostics = dict(probe_diagnostics)
                            probe_diagnostics.update({
                                "status": "task_probe_unavailable_support_fallback",
                                "task_probe_scale": TASK_ALIGNED_PROBE_SCALE,
                            })
                            verified_trace.append(probe_diagnostics)
                        else:
                            full_action = np.asarray(proposed_action, dtype=np.float32).copy()
                            probe_diagnostics = dict(probe_diagnostics)
                            if not _has_informative_task_probe_action(full_action):
                                action = _support_exploration_action(planning_step, candidates)
                                fallback_count += 1
                                if uses_task_probe_local_gate:
                                    task_probe_local_permitted = False
                                probe_diagnostics.update({
                                    "status": "task_probe_noninformative_noop_fallback",
                                    "local_selected_full_action": full_action.tolist(),
                                    "executed_support_fallback": action.tolist(),
                                })
                            else:
                                action = full_action.copy()
                                probe_scale = (
                                    ADAPTIVE_TASK_ALIGNED_PROBE_SCALE
                                    if uses_adaptive_task_probe else TASK_ALIGNED_PROBE_SCALE
                                )
                                action[:2] *= probe_scale
                                probe_diagnostics.update({
                                    "status": "task_aligned_low_amplitude_probe",
                                    "task_probe_scale": probe_scale,
                                    "local_selected_full_action": full_action.tolist(),
                                    "executed_probe_action": action.tolist(),
                                })
                                pending_task_probe = {
                                    "mapping": probe_diagnostics["mapping"],
                                    "full_action_xy": full_action[:2].astype(np.float64).tolist(),
                                    "probe_action_xy": action[:2].astype(np.float64).tolist(),
                                    "desired_delta": np.asarray(desired_delta, dtype=np.float64).tolist(),
                                    "response_rms": float(probe_diagnostics["response_rms"]),
                                    "drift_delta": probe_diagnostics["estimated_drift_delta"],
                                    "probe_scale": probe_scale,
                                }
                            verified_trace.append(probe_diagnostics)
                    elif method in {
                        "rgb_verified_effect", *POST_CONTACT_LOCAL_METHODS,
                        *POST_CONTACT_GOAL_GATED_METHODS, *POST_CONTACT_RESIDUAL_GATED_METHODS,
                        *POST_CONTACT_HOLDOUT_CALIBRATION_METHODS,
                        *POST_CONTACT_HOLDOUT_GATED_METHODS,
                        *POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS,
                        *POST_CONTACT_TASK_PROBE_GATED_METHODS,
                        *POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS,
                        *POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS,
                    }:
                        desired_delta, desired_valid = _rgb_block_goal_delta(current, goal_image)
                        verified_action, verified_diagnostics = _runtime_local_action(
                            verified_actions, verified_deltas, desired_delta,
                        ) if desired_valid else (None, {
                            "status": "invalid_rgb_goal", "rows": len(verified_actions),
                        })
                        if verified_action is None:
                            action = _support_exploration_action(step - prelude_steps, candidates)
                            fallback_count += 1
                        else:
                            if method in POST_CONTACT_GOAL_GATED_METHODS:
                                if goal_local_gate is None:
                                    raise ValueError("goal-gated local method requires a frozen calibration gate")
                                permit_local, gate_diagnostics = _goal_gate_allows_local(
                                    goal_image, goal_local_gate,
                                )
                                verified_diagnostics = {
                                    **verified_diagnostics,
                                    **gate_diagnostics,
                                    "local_selected_index": int(verified_diagnostics["selected_index"]),
                                }
                                if permit_local:
                                    action = verified_action
                                else:
                                    model_selected = int(predicted_scores.argmin().item())
                                    action = np.asarray(candidates[model_selected], dtype=np.float32)
                                    verified_diagnostics["rgb_fallback_selected_index"] = model_selected
                                    verified_model_disagreement_fallbacks += 1
                            elif method in POST_CONTACT_RESIDUAL_GATED_METHODS:
                                if residual_local_disabled:
                                    model_selected = int(predicted_scores.argmin().item())
                                    action = np.asarray(candidates[model_selected], dtype=np.float32)
                                    verified_diagnostics.update({
                                        "status": "residual_gate_rgb_fallback",
                                        "rgb_fallback_selected_index": model_selected,
                                        "residual_gate_threshold": float(residual_local_gate["threshold"]),
                                    })
                                    residual_gate_fallbacks += 1
                                else:
                                    action = verified_action
                                    released_local_diagnostics = verified_diagnostics
                                    verified_diagnostics["residual_gate_threshold"] = float(
                                        residual_local_gate["threshold"]
                                    )
                            elif method in POST_CONTACT_HOLDOUT_GATED_METHODS:
                                if residual_local_disabled:
                                    model_selected = int(predicted_scores.argmin().item())
                                    action = np.asarray(candidates[model_selected], dtype=np.float32)
                                    verified_diagnostics.update({
                                        "status": "holdout_gate_rgb_fallback",
                                        "rgb_fallback_selected_index": model_selected,
                                        "holdout_gate_threshold": float(residual_local_gate["threshold"]),
                                    })
                                    residual_gate_fallbacks += 1
                                else:
                                    action = verified_action
                                    verified_diagnostics["holdout_gate_threshold"] = float(
                                        residual_local_gate["threshold"]
                                    )
                            elif method in (
                                POST_CONTACT_TASK_PROBE_GATED_METHODS
                                | POST_CONTACT_ADAPTIVE_TASK_PROBE_GATED_METHODS
                            ):
                                if not task_probe_local_permitted:
                                    model_selected = int(predicted_scores.argmin().item())
                                    action = np.asarray(candidates[model_selected], dtype=np.float32)
                                    verified_diagnostics.update({
                                        "status": "task_probe_gate_rgb_fallback",
                                        "rgb_fallback_selected_index": model_selected,
                                        "task_probe_gate_threshold": float(task_probe_gate["threshold"]),
                                    })
                                    task_probe_fallbacks += 1
                                else:
                                    action = verified_action
                                    verified_diagnostics["task_probe_gate_threshold"] = float(
                                        task_probe_gate["threshold"]
                                    )
                            elif method in POST_CONTACT_LOCAL_METHODS:
                                # The same pre-action map must be audited on
                                # calibration roots.  It is not a deployment
                                # gate here, but its observed residuals define
                                # the later split-conformal threshold.
                                action = verified_action
                                released_local_diagnostics = verified_diagnostics
                            elif method in POST_CONTACT_HOLDOUT_CALIBRATION_METHODS:
                                action = verified_action
                            elif method in (
                                POST_CONTACT_TASK_PROBE_CALIBRATION_METHODS
                                | POST_CONTACT_ADAPTIVE_TASK_PROBE_CALIBRATION_METHODS
                            ):
                                action = verified_action
                            else:
                                action = verified_action
                        verified_trace.append(verified_diagnostics)
                    elif method in {"rgb_verified_effect_fused", "rgb_verified_effect_gated"}:
                        desired_delta, desired_valid = _rgb_block_goal_delta(current, goal_image)
                        verified_action, verified_diagnostics = _verified_rgb_action(
                            verified_actions, verified_deltas, desired_delta, candidates,
                        ) if desired_valid else (None, {
                            "status": "invalid_rgb_goal", "rows": len(verified_actions),
                        })
                        if verified_action is None:
                            action = _support_exploration_action(step - prelude_steps, candidates)
                            fallback_count += 1
                        else:
                            local_progress = np.asarray(
                                verified_diagnostics["predicted_progress"], dtype=np.float64,
                            )
                            model_cost = predicted_scores.detach().cpu().numpy().astype(np.float64)

                            def _robust_numpy_scale(values: np.ndarray) -> np.ndarray:
                                median = float(np.median(values))
                                q25, q75 = np.quantile(values, [0.25, 0.75])
                                scale = max(float(q75 - q25), 1.0e-6)
                                return (values - median) / scale

                            verified_diagnostics = dict(verified_diagnostics)
                            model_selected = int(np.argmin(model_cost))
                            local_selected = int(verified_diagnostics["selected_index"])
                            verified_diagnostics.update({
                                "model_selected_index": model_selected,
                                "local_selected_index": local_selected,
                                "model_cost": model_cost.tolist(),
                            })
                            if method == "rgb_verified_effect_gated":
                                action = np.asarray(candidates[local_selected], dtype=np.float32)
                                agrees = model_selected == local_selected
                                verified_diagnostics["status"] = (
                                    "model_local_agreement_permit"
                                    if agrees else "model_disagreement_local_fallback"
                                )
                                verified_diagnostics["model_local_agreement"] = agrees
                                if not agrees:
                                    verified_model_disagreement_fallbacks += 1
                            else:
                                fused_score = _robust_numpy_scale(local_progress) - _robust_numpy_scale(model_cost)
                                selected = int(np.argmax(fused_score))
                                action = np.asarray(candidates[selected], dtype=np.float32)
                                verified_diagnostics.update({
                                    "fusion_selected_index": selected,
                                    "fused_score": fused_score.tolist(),
                                })
                        verified_trace.append(verified_diagnostics)
                    elif support_gate and spread < support_threshold:
                        action = _support_exploration_action(step - prelude_steps, candidates)
                        fallback_count += 1
                    else:
                        action = candidates[int(predicted_scores.argmin().item())]
                    should_reverse = (
                        pixel_feedback
                        and (
                            (pixel_feedback_mode == "persistent" and pixel_feedback_sign < 0.0)
                            or (pixel_feedback_mode == "one_step" and reverse_current_pulse)
                        )
                    )
                    if should_reverse:
                        action = np.asarray(action, dtype=np.float32).copy()
                        action[:2] *= -1.0
                    pixel_feedback_sign_trace.append(
                        -1.0 if should_reverse else 1.0
                    )
                if pixel_feedback and step < prelude_steps:
                    pixel_feedback_sign_trace.append(float(pixel_feedback_sign))
                actions.append(np.asarray(action, dtype=np.float32).tolist())
                next_obs, _, terminated, truncated, next_info = env.step(action)
                if uses_verified_local:
                    before_block = _target_red_centroid(current)[:2]
                    after_block = _target_red_centroid(np.asarray(next_obs, dtype=np.uint8))[:2]
                    observed_delta: np.ndarray | None = None
                    if (
                        np.isfinite(before_block).all() and np.isfinite(after_block).all()
                        and (np.linalg.norm(before_block) > 0.0 or np.linalg.norm(after_block) > 0.0)
                    ):
                        observed_delta = (after_block - before_block).astype(np.float64)
                        # A post-contact chart intentionally discards all
                        # approach/prelude observations.  Their geometry and
                        # contact state differ from the local planning regime.
                        if not uses_post_contact_probe or step >= prelude_steps:
                            verified_actions.append(np.asarray(action[:2], dtype=np.float64).copy())
                            verified_deltas.append(observed_delta)
                    if pending_task_probe is not None:
                        task_probe = _task_aligned_probe_score(
                            np.asarray(pending_task_probe["mapping"], dtype=np.float64),
                            np.asarray(pending_task_probe["full_action_xy"], dtype=np.float64),
                            np.asarray(pending_task_probe["probe_action_xy"], dtype=np.float64),
                            np.asarray(pending_task_probe["desired_delta"], dtype=np.float64),
                            (
                                observed_delta if observed_delta is not None
                                else np.asarray([np.nan, np.nan], dtype=np.float64)
                            ),
                            response_rms=float(pending_task_probe["response_rms"]),
                            drift_delta=np.asarray(
                                pending_task_probe["drift_delta"], dtype=np.float64,
                            ),
                            probe_scale=float(pending_task_probe["probe_scale"]),
                        )
                        task_probe_score = float(task_probe["normalized_task_mismatch"])
                        task_probe_scores.append(task_probe_score)
                        if verified_trace:
                            verified_trace[-1]["task_probe_validation"] = task_probe
                        if uses_task_probe_local_gate:
                            task_probe_local_permitted = bool(
                                task_probe_gate["certified_finite_threshold"]
                                and task_probe["task_progress_positive"]
                                and task_probe_score <= float(task_probe_gate["threshold"])
                            )
                            if verified_trace:
                                verified_trace[-1]["task_probe_gate_transition"] = (
                                    "permit_local_after_task_probe"
                                    if task_probe_local_permitted else
                                    "disable_local_after_task_probe"
                                )
                        pending_task_probe = None
                    if (
                        (uses_holdout_calibration or uses_holdout_local_gate)
                        and step == prelude_steps + POST_CONTACT_PROBE_STEPS - 1
                    ):
                        holdout = _post_contact_holdout_residual(
                            verified_actions, verified_deltas,
                        )
                        holdout_score = float(holdout["normalized_residual"])
                        verified_residual_scores.append(holdout_score)
                        if verified_trace:
                            verified_trace[-1]["holdout_validation"] = holdout
                        if uses_holdout_local_gate:
                            if holdout_score > float(residual_local_gate["threshold"]):
                                residual_local_disabled = True
                                if verified_trace:
                                    verified_trace[-1]["holdout_gate_transition"] = "disable_local_before_release"
                            elif verified_trace:
                                verified_trace[-1]["holdout_gate_transition"] = "permit_local"
                    if released_local_diagnostics is not None:
                        residual = _local_prediction_residual(
                            np.asarray(released_local_diagnostics["mapping"], dtype=np.float64),
                            np.asarray(action[:2], dtype=np.float64),
                            (
                                observed_delta if observed_delta is not None
                                else np.asarray([np.nan, np.nan], dtype=np.float64)
                            ),
                            response_rms=float(released_local_diagnostics["response_rms"]),
                        )
                        released_local_diagnostics["observed_residual"] = residual
                        score = float(residual["normalized_residual"])
                        verified_residual_scores.append(score)
                        if uses_residual_local_gate:
                            if score > float(residual_local_gate["threshold"]):
                                residual_local_disabled = True
                                released_local_diagnostics["residual_gate_transition"] = "disable_local"
                            else:
                                released_local_diagnostics["residual_gate_transition"] = "retain_local"
                block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - initial["goal"])))
                previous, current = current, np.asarray(next_obs, dtype=np.uint8).copy()
                previous_action = np.asarray(action, dtype=np.float32).copy()
                info = next_info
                if terminated or truncated:
                    break
            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            final_distance = float(np.linalg.norm(final_block - initial["goal"]))
            episodes.append({
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "steps": len(actions),
                "final_goal_distance_m": final_distance,
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
                "actions": actions,
                "support_fallback_count": int(fallback_count),
                "pixel_feedback_enabled": bool(pixel_feedback),
                "pixel_feedback_mode": pixel_feedback_mode,
                "pixel_feedback_reversal_count": int(pixel_feedback_reversals),
                "pixel_error_trace_runtime": pixel_error_trace,
                "pixel_feedback_sign_trace_runtime": pixel_feedback_sign_trace,
                "verified_rgb_effect_trace": verified_trace,
                "verified_rgb_effect_rows": len(verified_actions),
                "verified_model_disagreement_fallbacks": int(verified_model_disagreement_fallbacks),
                "verified_residual_scores": verified_residual_scores,
                "residual_local_disabled": bool(residual_local_disabled),
                "residual_gate_fallbacks": int(residual_gate_fallbacks),
                "task_probe_scores": task_probe_scores,
                "task_probe_local_permitted": bool(task_probe_local_permitted),
                "task_probe_fallbacks": int(task_probe_fallbacks),
            })
        finally:
            env.close()
    return {
        "method": method,
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([row["final_goal_distance_m"] for row in episodes])),
        "mean_best_goal_distance_m": float(np.mean([row["best_goal_distance_m"] for row in episodes])),
        "support_fallback_rate": float(np.mean([row["support_fallback_count"] for row in episodes]) / max(max_steps - prelude_steps, 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261070)
    parser.add_argument(
        "--outcome-seed", type=int, default=None,
        help=(
            "optional seed base for paired outcome roots; leaves all readout, pixel-map, "
            "and task-probe gate calibration rooted at --seed so a confirmation can use "
            "a frozen calibration protocol on fresh roots"
        ),
    )
    parser.add_argument(
        "--frozen-calibration", type=Path, default=None,
        help=(
            "path to a prior closed_loop.json containing a cube_temporal_calibration_bundle.v1; "
            "loads its readouts, RGB pixel-action chart, and task-probe gate without refitting"
        ),
    )
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--prelude-steps", type=int, default=20)
    parser.add_argument("--approach-steps", type=int, default=0)
    parser.add_argument("--approach-gain", type=float, default=1.0)
    parser.add_argument(
        "--phase-order",
        choices=("descent_then_align", "align_then_descend"),
        default="descent_then_align",
        help=(
            "fixed hybrid prelude: descend before RGB alignment (legacy), or "
            "perform RGB XY alignment at elevated height before descent"
        ),
    )
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-threshold", type=float, default=5.0e-4)
    parser.add_argument("--pixel-feedback-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--goal-gated-local", action="store_true",
        help=(
            "fit a conservative RGB-goal local-permission gate on separate "
            "calibration rollouts, then evaluate its post-contact local chart"
        ),
    )
    parser.add_argument(
        "--residual-gated-local", action="store_true",
        help=(
            "calibrate a root-level split-conformal post-contact local-map residual "
            "gate on separate rollouts, then retain local control only until its "
            "first observed RGB effect mismatch"
        ),
    )
    parser.add_argument(
        "--holdout-residual-gated-local", action="store_true",
        help=(
            "validate the post-contact local map on a fourth withheld support pulse "
            "before releasing any task-directed local action; calibrate its threshold "
            "on separate roots"
        ),
    )
    parser.add_argument(
        "--task-probe-gated-local", action="store_true",
        help=(
            "execute a predeclared low-amplitude pulse in the locally proposed "
            "task direction, then release local control only when a separate-root "
            "task-direction conformal gate permits it"
        ),
    )
    parser.add_argument(
        "--adaptive-task-probe-local", action="store_true",
        help=(
            "use a separate-root task-direction gate with a recency-weighted, "
            "affine RGB action-effect chart; the chart estimates local drift and "
            "reweights observed effects after every pulse"
        ),
    )
    parser.add_argument(
        "--residual-gate-alpha", type=float, default=0.1,
        help="target marginal exceedance level for the root-level residual gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Cube temporal closed loop but unavailable")
    if args.episodes < 4 or args.calibration_roots < 8:
        raise ValueError("need at least 4 episodes and 8 calibration roots")
    if args.prelude_steps < 2 or args.max_steps <= args.prelude_steps:
        raise ValueError("max-steps must exceed prelude-steps >= 2")
    if args.approach_steps < 0 or not np.isfinite(args.approach_gain):
        raise ValueError("approach-steps must be non-negative and approach-gain finite")
    if args.phase_order == "align_then_descend" and args.approach_steps <= 0:
        raise ValueError("align_then_descend requires a positive --approach-steps")
    if args.approach_steps > 0 and args.prelude_steps != args.approach_steps + 20:
        raise ValueError("approach mode requires prelude-steps == approach-steps + 20")
    if args.support_threshold < 0.0 or not np.isfinite(args.support_threshold):
        raise ValueError("support-threshold must be finite and non-negative")
    if args.pixel_feedback_threshold < 0.0 or not np.isfinite(args.pixel_feedback_threshold):
        raise ValueError("pixel-feedback-threshold must be finite and non-negative")
    if not 0.0 < args.residual_gate_alpha < 1.0 or not np.isfinite(args.residual_gate_alpha):
        raise ValueError("residual-gate-alpha must be finite and in (0, 1)")
    post_contact_study_plan = _post_contact_study_plan(args)
    _validate_post_contact_protocol(
        post_contact_study_plan, approach_steps=args.approach_steps,
        prelude_steps=args.prelude_steps, phase_order=args.phase_order,
    )
    outcome_seed = int(args.seed + 1000 if args.outcome_seed is None else args.outcome_seed)
    task_probe_calibration_start = int(args.seed + 500000)
    task_probe_calibration_stop = task_probe_calibration_start + int(args.calibration_roots)
    outcome_stop = outcome_seed + int(args.episodes)
    if max(outcome_seed, task_probe_calibration_start) < min(outcome_stop, task_probe_calibration_stop):
        raise ValueError("outcome root seeds must be disjoint from task-probe calibration root seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] independent RGB-only Cube closed-loop consequence; "
        f"device={args.device} episodes={args.episodes}"
    )
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    candidates = _candidate_bank()
    frozen_bundle = (
        _load_frozen_calibration_bundle(
            args.frozen_calibration, device=args.device,
            phase_order=args.phase_order,
        )
        if args.frozen_calibration is not None else None
    )
    if frozen_bundle is not None:
        declared_candidates = np.asarray(frozen_bundle.get("candidate_bank", []), dtype=np.float32)
        if declared_candidates.shape != candidates.shape or not np.allclose(declared_candidates, candidates):
            raise ValueError("frozen calibration bundle candidate bank differs from this protocol")
        fits = frozen_bundle["fits"]
        pixel_action_mapping = frozen_bundle["pixel_action_mapping"]
        pixel_mapping_report = dict(frozen_bundle["pixel_mapping_report"])
        calibration_metadata = {
            "mode": "loaded_frozen_bundle",
            "source_path": frozen_bundle["source_path"],
            "source_calibration_roots": int(frozen_bundle["calibration_roots"]),
        }
    else:
        cal_current, cal_previous, cal_goals, cal_previous_actions, cal_labels = collect_temporal_frames(
            roots=args.calibration_roots, frames_per_root=24, tasks=args.tasks,
            seed=args.seed, action_seed=args.seed + 1, collection_mode="guided_intervention",
        )
        cal_families, _ = build_feature_families(
            cal_current, cal_previous, cal_goals, cal_labels["action"], cal_previous_actions,
            cal_labels["history_valid"], device=args.device, model=model, batch_size=args.batch_size,
        )
        cal_target = torch.as_tensor(
            cal_labels["next_block_goal_delta_xy"], device=args.device, dtype=torch.float32,
        )
        cal_color = extract_color_geometry_features(
            cal_current, cal_goals, cal_labels["action"], args.device,
        )
        cal_color_temporal = extract_color_geometry_temporal_features(
            cal_current, cal_previous, cal_goals, cal_labels["action"],
            cal_previous_actions, args.device,
        )
        fits = {
            name: _fit_readout(cal_families[name], cal_target)
            for name in ("latent_static", "latent_temporal")
        }
        fits["rgb_geometry_action"] = _fit_readout(
            cal_color["color_geometry_action"], cal_target,
        )
        fits["rgb_geometry_temporal_action"] = _fit_readout(
            cal_color_temporal["color_geometry_temporal_action"], cal_target,
        )
        pixel_action_mapping, pixel_mapping_report = _fit_rgb_pixel_action_mapping(
            seed=args.seed + 100000,
            tasks=args.tasks,
            roots=args.calibration_roots,
            # The camera/action Jacobian must be calibrated in the same
            # continuous phase in which it is consumed.  An elevated
            # align-then-descend protocol therefore cannot reuse a chart
            # collected after descent.
            descent_steps=(0 if args.phase_order == "align_then_descend" else 15),
            probe_steps=4,
        )
        calibration_metadata = {
            "mode": "fit_this_run",
            "frames": int(len(cal_current)),
            "roots": int(args.calibration_roots),
            "pixel_mapping_phase": (
                "elevated_pre_descent" if args.phase_order == "align_then_descend"
                else "post_descent"
            ),
        }
    initial_conditions = [
        _capture_initial(outcome_seed + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.episodes)
    ]
    post_contact_calibration_initials = [
        _capture_initial(
            task_probe_calibration_start + index,
            int(args.tasks[index % len(args.tasks)]),
        )
        for index in range(args.calibration_roots)
    ] if post_contact_study_plan and frozen_bundle is None else []

    def _run_post_contact_arm(
        initials: list[dict[str, Any]], method: str, *,
        goal_local_gate: dict[str, Any] | None = None,
        residual_local_gate: dict[str, Any] | None = None,
        task_probe_gate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _run_arm(
            initial_conditions=initials, model=model,
            fit=fits["rgb_geometry_action"], candidates=candidates,
            device=args.device, max_steps=args.max_steps,
            prelude_steps=args.prelude_steps, batch_size=args.batch_size,
            method=method, approach_steps=args.approach_steps,
            approach_gain=args.approach_gain, phase_order=args.phase_order,
            pixel_action_mapping=pixel_action_mapping,
            goal_local_gate=goal_local_gate,
            residual_local_gate=residual_local_gate,
            task_probe_gate=task_probe_gate,
        )
    static = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_static"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size, method="latent_static",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    temporal = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size, method="latent_temporal",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    rgb_geometry = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_action"], candidates=candidates, device=args.device,
        max_steps=args.max_steps, prelude_steps=args.prelude_steps,
        batch_size=args.batch_size, method="rgb_geometry_action",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    rgb_geometry_temporal = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_temporal_action"], candidates=candidates,
        device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="rgb_geometry_temporal_action",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    verified_rgb_effect = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_action"], candidates=candidates, device=args.device,
        max_steps=args.max_steps, prelude_steps=args.prelude_steps,
        batch_size=args.batch_size, method="rgb_verified_effect",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    verified_rgb_effect_fused = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_action"], candidates=candidates, device=args.device,
        max_steps=args.max_steps, prelude_steps=args.prelude_steps,
        batch_size=args.batch_size, method="rgb_verified_effect_fused",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    verified_rgb_effect_gated = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_action"], candidates=candidates, device=args.device,
        max_steps=args.max_steps, prelude_steps=args.prelude_steps,
        batch_size=args.batch_size, method="rgb_verified_effect_gated",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    temporal_support_gate = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", support_gate=True,
        support_threshold=args.support_threshold,
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    temporal_pixel_feedback = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", pixel_feedback=True,
        pixel_feedback_threshold=args.pixel_feedback_threshold,
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    temporal_pixel_feedback_local = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", pixel_feedback=True,
        pixel_feedback_threshold=args.pixel_feedback_threshold,
        pixel_feedback_mode="one_step",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    temporal_rgb_rank_consensus = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit={
            "latent_temporal": fits["latent_temporal"],
            "rgb_geometry_action": fits["rgb_geometry_action"],
        }, candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_rgb_rank_consensus",
        approach_steps=args.approach_steps, approach_gain=args.approach_gain,
        pixel_action_mapping=pixel_action_mapping,
        phase_order=args.phase_order,
    )
    post_contact_repair_studies: dict[str, dict[str, Any]] = {}
    frozen_gate_sources: dict[str, Any] = {}
    for study_index, study in enumerate(post_contact_study_plan):
        gate_key = study["gate_key"]
        if frozen_bundle is not None:
            gate = frozen_bundle.get(gate_key)
            if not isinstance(gate, dict):
                raise ValueError(
                    f"frozen calibration bundle lacks required {gate_key} for "
                    f"{study['name']}"
                )
            calibration_carrier = None
            calibration_local = None
            frozen_gate_sources[gate_key] = "loaded_frozen_bundle"
        else:
            calibration_carrier = _run_post_contact_arm(
                post_contact_calibration_initials, study["carrier_method"],
            )
            calibration_local = _run_post_contact_arm(
                post_contact_calibration_initials, study["calibration_method"],
            )
            if gate_key == "goal_local_gate":
                gate = _fit_goal_local_gate(
                    post_contact_calibration_initials,
                    calibration_local["episodes"],
                    calibration_carrier["episodes"],
                )
            elif gate_key == "residual_local_gate":
                gate = _fit_local_residual_gate(
                    calibration_local["episodes"], alpha=args.residual_gate_alpha,
                )
            elif gate_key in {"task_probe_gate", "adaptive_task_probe_gate"}:
                gate = _fit_task_probe_gate(
                    calibration_local["episodes"], alpha=args.residual_gate_alpha,
                )
            else:
                raise AssertionError(f"unsupported post-contact gate: {gate_key}")
            frozen_gate_sources[gate_key] = "fit_separate_calibration_roots"
        gate_argument = (
            "task_probe_gate"
            if gate_key == "adaptive_task_probe_gate"
            else gate_key
        )
        gate_kwargs: dict[str, dict[str, Any]] = {gate_argument: gate}
        carrier = _run_post_contact_arm(initial_conditions, study["carrier_method"])
        gated = _run_post_contact_arm(
            initial_conditions, study["gated_method"], **gate_kwargs,
        )
        carrier_by_seed = {int(row["seed"]): row for row in carrier["episodes"]}
        gated_by_seed = {int(row["seed"]): row for row in gated["episodes"]}
        deltas = np.asarray([
            gated_by_seed[seed]["final_goal_distance_m"]
            - carrier_by_seed[seed]["final_goal_distance_m"]
            for seed in carrier_by_seed
        ], dtype=np.float64)
        post_contact_repair_studies[study["name"]] = {
            "study": study,
            "calibration_initial_seeds": [
                int(row["seed"]) for row in post_contact_calibration_initials
            ] if calibration_carrier is not None else None,
            "calibration_carrier": calibration_carrier,
            "calibration_local": calibration_local,
            "gate": gate,
            "carrier": carrier,
            "gated": gated,
            "paired_gated_minus_matched_carrier": _paired_root_summary(
                deltas, seed=args.seed + 700000 + study_index,
            ),
        }
    calibrated_post_contact_gates = {
        str(row["study"]["gate_key"]): row["gate"]
        for row in post_contact_repair_studies.values()
    }
    static_by_seed = {row["seed"]: row for row in static["episodes"]}
    temporal_by_seed = {row["seed"]: row for row in temporal["episodes"]}
    rgb_by_seed = {row["seed"]: row for row in rgb_geometry["episodes"]}
    rgb_temporal_by_seed = {
        row["seed"]: row for row in rgb_geometry_temporal["episodes"]
    }
    verified_by_seed = {
        row["seed"]: row for row in verified_rgb_effect["episodes"]
    }
    fused_by_seed = {
        row["seed"]: row for row in verified_rgb_effect_fused["episodes"]
    }
    gated_by_seed = {
        row["seed"]: row for row in verified_rgb_effect_gated["episodes"]
    }
    gate_by_seed = {row["seed"]: row for row in temporal_support_gate["episodes"]}
    pixel_feedback_by_seed = {row["seed"]: row for row in temporal_pixel_feedback["episodes"]}
    deltas = np.asarray([
        temporal_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    gate_deltas = np.asarray([
        gate_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    rgb_deltas = np.asarray([
        rgb_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    rgb_temporal_deltas = np.asarray([
        rgb_temporal_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    verified_deltas = np.asarray([
        verified_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    fused_deltas = np.asarray([
        fused_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    gated_deltas = np.asarray([
        gated_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    pixel_feedback_deltas = np.asarray([
        pixel_feedback_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    pixel_feedback_local_by_seed = {
        row["seed"]: row for row in temporal_pixel_feedback_local["episodes"]
    }
    consensus_by_seed = {
        row["seed"]: row for row in temporal_rgb_rank_consensus["episodes"]
    }
    pixel_feedback_local_deltas = np.asarray([
        pixel_feedback_local_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    consensus_deltas = np.asarray([
        consensus_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    report = {
        "schema": "cube_temporal_closed_loop.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_episodes": args.episodes,
            "root_disjoint": True,
            "same_initial_state_pairing": True,
            "runtime_input": "live current/previous RGB, rendered goal, previous action, fixed candidate bank",
            "runtime_policy_state_access": False,
            "fixed_prelude": (
                "RGB pixel-action approach for "
                f"{args.approach_steps} steps, then 15 descent actions, then 5 gripper-close actions"
                if args.phase_order == "align_then_descend"
                else "15 descent actions, RGB pixel-action approach for "
                f"{args.approach_steps} steps, then 5 gripper-close actions"
                if args.approach_steps > 0
                else "15 descent actions then 5 gripper-close actions"
            ),
            "approach_steps": int(args.approach_steps),
            "approach_gain": float(args.approach_gain),
            "phase_order": args.phase_order,
            "support_gate": "if predicted candidate-score spread < fixed threshold, cycle x+/x-/y+/y-",
            "support_threshold_m": float(args.support_threshold),
            "pixel_feedback": "after the fixed prelude, reverse the next latent-temporal XY pulse when RGB block-goal error increases",
            "pixel_feedback_threshold": float(args.pixel_feedback_threshold),
            "pixel_feedback_mode": "persistent",
            "additional_feedback_arm": "one_step",
            "verified_rgb_effect_arm": (
                "RGB-only local action-effect fit after >=4 observed rows, rank-two support, "
                "and finite condition number; fixed four-direction fallback otherwise"
            ),
            "verified_rgb_effect_fused_arm": (
                "rank-normalized fusion of the verified RGB local map and frozen RGB geometry "
                "prediction under the same candidate bank and action budget"
            ),
            "verified_rgb_effect_gated_arm": (
                "permit only when local observed-effect and frozen RGB geometry select the same "
                "candidate; otherwise retain the local action and log a model-disagreement fallback"
            ),
            "optional_post_contact_repair_studies": [
                study["name"] for study in post_contact_study_plan
            ],
            "post_contact_repair_protocol": (
                "each enabled study fits its permission gate on disjoint roots, then "
                "compares the gated local controller with a probe-budget-matched RGB carrier "
                "on paired outcome roots"
            ),
            "candidate_bank": candidates.tolist(),
            "readout_fit": "calibration roots only; no test outcomes",
            "evaluator_state_role": "initial-state pairing and final goal-distance scoring only",
        },
        "calibration": calibration_metadata,
        "frozen_calibration_bundle": {
            "schema": "cube_temporal_calibration_bundle.v1",
            "checkpoint": args.policy,
            "phase_order": args.phase_order,
            "calibration_roots": int(args.calibration_roots),
            "candidate_bank": candidates.tolist(),
            "fits": {
                name: _serialize_readout_fit(fit)
                for name, fit in fits.items()
            },
            "pixel_action_mapping": np.asarray(
                pixel_action_mapping, dtype=np.float64,
            ).tolist(),
            "pixel_mapping_report": pixel_mapping_report,
            "goal_local_gate": calibrated_post_contact_gates.get("goal_local_gate"),
            "residual_local_gate": calibrated_post_contact_gates.get("residual_local_gate"),
            "task_probe_gate": calibrated_post_contact_gates.get("task_probe_gate"),
            "adaptive_task_probe_gate": calibrated_post_contact_gates.get(
                "adaptive_task_probe_gate"
            ),
        },
        "pixel_action_mapping": {
            "matrix": np.asarray(pixel_action_mapping, dtype=np.float64).tolist(),
            "fit_report": pixel_mapping_report,
            "fit_roots": int(args.calibration_roots),
        },
        "static": static,
        "temporal": temporal,
        "rgb_geometry_action": rgb_geometry,
        "rgb_geometry_temporal_action": rgb_geometry_temporal,
        "rgb_verified_effect": verified_rgb_effect,
        "rgb_verified_effect_fused": verified_rgb_effect_fused,
        "rgb_verified_effect_gated": verified_rgb_effect_gated,
        "temporal_support_gate": temporal_support_gate,
        "temporal_pixel_feedback": temporal_pixel_feedback,
        "temporal_pixel_feedback_local": temporal_pixel_feedback_local,
        "temporal_rgb_rank_consensus": temporal_rgb_rank_consensus,
        "post_contact_repair_studies": post_contact_repair_studies,
        "post_contact_gate_sources": frozen_gate_sources,
        "paired": {
            "seed_order": list(static_by_seed),
            "final_distance_delta_temporal_minus_static_m": deltas.tolist(),
            "mean_final_distance_delta_m": float(deltas.mean()),
            "temporal_wins": int(np.sum(deltas < 0.0)),
            "static_wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "support_gate_final_distance_delta_vs_static_m": gate_deltas.tolist(),
            "support_gate_mean_final_distance_delta_m": float(gate_deltas.mean()),
            "support_gate_wins": int(np.sum(gate_deltas < 0.0)),
            "support_gate_losses": int(np.sum(gate_deltas > 0.0)),
            "support_gate_ties": int(np.sum(gate_deltas == 0.0)),
            "rgb_geometry_final_distance_delta_vs_static_m": rgb_deltas.tolist(),
            "rgb_geometry_mean_final_distance_delta_m": float(rgb_deltas.mean()),
            "rgb_geometry_wins": int(np.sum(rgb_deltas < 0.0)),
            "rgb_geometry_losses": int(np.sum(rgb_deltas > 0.0)),
            "rgb_geometry_ties": int(np.sum(rgb_deltas == 0.0)),
            "rgb_geometry_temporal_final_distance_delta_vs_static_m": rgb_temporal_deltas.tolist(),
            "rgb_geometry_temporal_mean_final_distance_delta_m": float(rgb_temporal_deltas.mean()),
            "rgb_geometry_temporal_wins": int(np.sum(rgb_temporal_deltas < 0.0)),
            "rgb_geometry_temporal_losses": int(np.sum(rgb_temporal_deltas > 0.0)),
            "rgb_geometry_temporal_ties": int(np.sum(rgb_temporal_deltas == 0.0)),
            "verified_rgb_effect_final_distance_delta_vs_static_m": verified_deltas.tolist(),
            "verified_rgb_effect_mean_final_distance_delta_m": float(verified_deltas.mean()),
            "verified_rgb_effect_wins": int(np.sum(verified_deltas < 0.0)),
            "verified_rgb_effect_losses": int(np.sum(verified_deltas > 0.0)),
            "verified_rgb_effect_ties": int(np.sum(verified_deltas == 0.0)),
            "verified_rgb_effect_fused_final_distance_delta_vs_static_m": fused_deltas.tolist(),
            "verified_rgb_effect_fused_mean_final_distance_delta_m": float(fused_deltas.mean()),
            "verified_rgb_effect_fused_wins": int(np.sum(fused_deltas < 0.0)),
            "verified_rgb_effect_fused_losses": int(np.sum(fused_deltas > 0.0)),
            "verified_rgb_effect_fused_ties": int(np.sum(fused_deltas == 0.0)),
            "verified_rgb_effect_gated_final_distance_delta_vs_static_m": gated_deltas.tolist(),
            "verified_rgb_effect_gated_mean_final_distance_delta_m": float(gated_deltas.mean()),
            "verified_rgb_effect_gated_wins": int(np.sum(gated_deltas < 0.0)),
            "verified_rgb_effect_gated_losses": int(np.sum(gated_deltas > 0.0)),
            "verified_rgb_effect_gated_ties": int(np.sum(gated_deltas == 0.0)),
            "pixel_feedback_final_distance_delta_vs_static_m": pixel_feedback_deltas.tolist(),
            "pixel_feedback_mean_final_distance_delta_m": float(pixel_feedback_deltas.mean()),
            "pixel_feedback_wins": int(np.sum(pixel_feedback_deltas < 0.0)),
            "pixel_feedback_losses": int(np.sum(pixel_feedback_deltas > 0.0)),
            "pixel_feedback_ties": int(np.sum(pixel_feedback_deltas == 0.0)),
            "pixel_feedback_local_final_distance_delta_vs_static_m": pixel_feedback_local_deltas.tolist(),
            "pixel_feedback_local_mean_final_distance_delta_m": float(pixel_feedback_local_deltas.mean()),
            "pixel_feedback_local_wins": int(np.sum(pixel_feedback_local_deltas < 0.0)),
            "pixel_feedback_local_losses": int(np.sum(pixel_feedback_local_deltas > 0.0)),
            "pixel_feedback_local_ties": int(np.sum(pixel_feedback_local_deltas == 0.0)),
            "rank_consensus_final_distance_delta_vs_static_m": consensus_deltas.tolist(),
            "rank_consensus_mean_final_distance_delta_m": float(consensus_deltas.mean()),
            "rank_consensus_wins": int(np.sum(consensus_deltas < 0.0)),
            "rank_consensus_losses": int(np.sum(consensus_deltas > 0.0)),
            "rank_consensus_ties": int(np.sum(consensus_deltas == 0.0)),
        },
        "claim_boundary": (
            "Independent RGB-only closed-loop consequence for a fixed-prelude "
            "candidate ranker; it does not establish universal world-model "
            "transfer, causal latent dynamics, or stable pose closure."
        ),
    }
    (args.output / "closed_loop.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
