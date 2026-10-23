"""Probe--reobserve--select closed-loop evaluation for PushT.

This is a deliberately narrow, standalone repair experiment for the
within-contact action-identifiability bottleneck.  Both paired arms execute
the same fixed RGB probe before choosing one action from a globally fixed
``+y/-y`` bank:

    reset frozen root -> fixed probe -> reobserve RGB -> select/fallback

The local arm fits an action-conditioned chart from *past rendered RGB
effects* on root-disjoint training roots.  It ranks the two declared actions
by predicted RGB-effect energy.  The comparison is intentionally
probe-budget-matched: an abstention executes exactly the fallback action, so
it cannot acquire an artificial advantage by doing less.

At decision time, functions in this module receive rendered RGB frames and
declared actions only.  The reset fixture is an offline experimental
constructor; its simulator state is never supplied to the policy.  The
terminal simulator state and frozen goal are read only after the final action
to score the paired evaluation.  In particular, no state, goal coordinate,
contact label, true successor, reward, or task cost is used to fit or deploy
the RGB chart.

This evaluator is not a learned world model, task-conditioned visual policy,
or pose-closure claim.  It tests whether a fixed local RGB-effect selection
can beat its matched fixed-action fallback on fresh contact roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_pusht_mode_aware_shared_candidate import (  # noqa: E402
    angle_error,
    make_boundary_contact_roots,
)


RUNTIME_BOUNDARY = {
    "runtime_input": "rendered RGB history plus the predeclared action bank only",
    "runtime_privileged_state_access": False,
    "runtime_successor_access": False,
    "runtime_task_cost_access": False,
    "offline_privileged_terminal_scoring_only": True,
}


@dataclass(frozen=True)
class FrozenRoot:
    """Offline root fixture.  State/goal configure reset, never the policy."""

    root_id: int
    reset_state: np.ndarray
    reset_goal: np.ndarray


@dataclass(frozen=True)
class RuntimeEpisode:
    """Policy trace containing RGB-derived values and declared actions only."""

    root_id: int
    probe_action: np.ndarray
    selected_action: np.ndarray
    selected_index: int
    released: bool
    decision_reason: str
    support_distance: float | None
    predicted_contrast: float | None
    terminal_state_for_offline_scoring: np.ndarray


def fixed_action_bank(pulse_magnitude: float) -> np.ndarray:
    """Return the globally fixed, ordered contact bank before root collection."""

    if not 0.0 < float(pulse_magnitude) <= 1.0:
        raise ValueError("pulse magnitude must lie in (0, 1]")
    return np.asarray(((0.0, float(pulse_magnitude)), (0.0, -float(pulse_magnitude))), dtype=np.float32)


def action_bank_sha256(actions: np.ndarray) -> str:
    """Fingerprint exact order and values; a changed bank is a protocol error."""

    values = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    if values.shape != (2, 2):
        raise ValueError("probe-reobserve-select requires exactly the fixed +y/-y action bank")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _rgb_pair_descriptor(previous: np.ndarray, current: np.ndarray, image_size: int) -> np.ndarray:
    """Runtime context from two RGB frames only."""

    previous = np.asarray(previous, dtype=np.uint8)
    current = np.asarray(current, dtype=np.uint8)
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[-1] != 3:
        raise ValueError("RGB history must be equal-shaped HWC frames")
    frames = torch.as_tensor(np.stack((previous, current)), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    frames = F.interpolate(frames, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)
    return torch.cat((frames[1], frames[0], frames[1] - frames[0]), dim=0).flatten().numpy().astype(np.float32)


def _rgb_effect(before: np.ndarray, after: np.ndarray, image_size: int) -> np.ndarray:
    """Observed target: rendered successor minus rendered predecessor."""

    before = np.asarray(before, dtype=np.uint8)
    after = np.asarray(after, dtype=np.uint8)
    if before.shape != after.shape:
        raise ValueError("RGB effect frames must share shape")
    frames = torch.as_tensor(np.stack((before, after)), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    frames = F.interpolate(frames, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)
    return (frames[1] - frames[0]).flatten().numpy().astype(np.float32)


def fit_rgb_effect_chart(contexts: np.ndarray, action_indices: np.ndarray, effects: np.ndarray) -> dict[str, np.ndarray]:
    """Fit a train-root-only KNN RGB-effect chart with one head per action."""

    contexts = np.asarray(contexts, dtype=np.float64)
    action_indices = np.asarray(action_indices, dtype=np.int64).reshape(-1)
    effects = np.asarray(effects, dtype=np.float64)
    if contexts.ndim != 2 or effects.ndim != 2 or len(contexts) != len(effects) or len(contexts) != len(action_indices):
        raise ValueError("chart context/action/effect rows must be aligned two-dimensional arrays")
    if len(contexts) < 2 or not np.isfinite(contexts).all() or not np.isfinite(effects).all():
        raise ValueError("chart requires at least two finite RGB-effect rows")
    if set(np.unique(action_indices).tolist()) != {0, 1}:
        raise ValueError("both fixed action identities require train-root RGB effects")
    mean = contexts.mean(axis=0, keepdims=True)
    scale = np.maximum(contexts.std(axis=0, keepdims=True), 5.0e-2)
    return {
        "contexts": (contexts - mean) / scale,
        "action_indices": action_indices,
        "effects": effects,
        "mean": mean,
        "scale": scale,
    }


def predict_rgb_effects(chart: dict[str, np.ndarray], context: np.ndarray, *, neighbors: int) -> dict[str, Any]:
    """Predict both fixed candidate effects using RGB context and action IDs only."""

    if int(neighbors) < 1:
        raise ValueError("neighbors must be positive")
    query = np.asarray(context, dtype=np.float64).reshape(1, -1)
    normalized = (query - chart["mean"]) / chart["scale"]
    contexts = np.asarray(chart["contexts"], dtype=np.float64)
    actions = np.asarray(chart["action_indices"], dtype=np.int64)
    effects = np.asarray(chart["effects"], dtype=np.float64)
    energies: list[float] = []
    support: list[float] = []
    for action_index in (0, 1):
        rows = np.flatnonzero(actions == action_index)
        distances = np.linalg.norm(contexts[rows] - normalized, axis=1) / math.sqrt(float(contexts.shape[1]))
        chosen = rows[np.argsort(distances, kind="stable")[: min(int(neighbors), len(rows))]]
        chosen_distances = np.linalg.norm(contexts[chosen] - normalized, axis=1) / math.sqrt(float(contexts.shape[1]))
        weights = 1.0 / np.maximum(chosen_distances, 1.0e-6)
        weights /= float(weights.sum())
        prediction = np.sum(effects[chosen] * weights[:, None], axis=0)
        energies.append(float(np.linalg.norm(prediction)))
        support.append(float(chosen_distances[0]))
    return {
        "selected_index": int(np.argmax(energies)),
        "predicted_effect_energy": energies,
        "predicted_contrast": float(abs(energies[0] - energies[1])),
        "support_distance": float(max(support)),
        **RUNTIME_BOUNDARY,
    }


def _finite_upper_conformal(values: Iterable[float], alpha: float) -> float | None:
    """Finite-sample split-conformal upper threshold, otherwise fail closed."""

    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - float(alpha))))
    return None if len(ordered) == 0 or rank > len(ordered) else float(ordered[rank - 1])


def calibrate_gate(predictions: Iterable[dict[str, Any]], *, alpha: float, minimum_roots: int) -> dict[str, Any]:
    """Freeze support and contrast gates from calibration roots alone."""

    values = list(predictions)
    if len(values) < int(minimum_roots):
        return {"status": "insufficient_calibration", "eligible_roots": len(values), "support_threshold": None, "contrast_floor": None}
    support = np.asarray([float(row["support_distance"]) for row in values])
    contrast = np.asarray([float(row["predicted_contrast"]) for row in values])
    threshold = _finite_upper_conformal(support, alpha)
    if threshold is None:
        return {"status": "insufficient_calibration", "eligible_roots": len(values), "support_threshold": None, "contrast_floor": None}
    lower = int(math.floor(float(alpha) * len(contrast)))
    return {
        "status": "calibrated",
        "eligible_roots": int(len(values)),
        "alpha": float(alpha),
        "support_threshold": float(threshold),
        "contrast_floor": float(np.sort(contrast)[min(lower, len(contrast) - 1)]),
        "contrast_note": "calibration-only predicted RGB-effect contrast floor; not a task-success certificate",
    }


def select_or_abstain(prediction: dict[str, Any], gate: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
    """Release a local selection only when every frozen gate condition passes."""

    if gate.get("status") != "calibrated":
        return {"selected_index": int(fallback_index), "release": False, "reason": "insufficient_calibration"}
    if float(prediction["support_distance"]) > float(gate["support_threshold"]):
        return {"selected_index": int(fallback_index), "release": False, "reason": "insufficient_support"}
    if float(prediction["predicted_contrast"]) < float(gate["contrast_floor"]):
        return {"selected_index": int(fallback_index), "release": False, "reason": "insufficient_rgb_effect_contrast"}
    return {"selected_index": int(prediction["selected_index"]), "release": True, "reason": "calibrated_rgb_effect_selection"}


def _reset_and_probe(env: gym.Env, root: FrozenRoot, probe_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute the shared runtime probe without inspecting any state/reward."""

    env.reset(seed=int(root.root_id), options={"state": root.reset_state.copy(), "goal_state": root.reset_goal.copy()})
    previous = np.asarray(env.render(), dtype=np.uint8).copy()
    env.step(np.asarray(probe_action, dtype=np.float32))  # reward/state intentionally ignored
    current = np.asarray(env.render(), dtype=np.uint8).copy()
    return previous, current


def collect_chart_rows(env: gym.Env, roots: list[FrozenRoot], *, probe_action: np.ndarray, bank: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect one predeclared action-conditioned RGB effect per train root."""

    contexts: list[np.ndarray] = []
    action_indices: list[int] = []
    effects: list[np.ndarray] = []
    for offset, root in enumerate(roots):
        previous, current = _reset_and_probe(env, root, probe_action)
        action_index = int(offset % len(bank))  # predeclared alternating schedule
        env.step(np.asarray(bank[action_index], dtype=np.float32))
        successor = np.asarray(env.render(), dtype=np.uint8).copy()
        contexts.append(_rgb_pair_descriptor(previous, current, image_size))
        action_indices.append(action_index)
        effects.append(_rgb_effect(current, successor, image_size))
    return np.stack(contexts), np.asarray(action_indices), np.stack(effects)


def calibration_predictions(env: gym.Env, roots: list[FrozenRoot], chart: dict[str, np.ndarray], *, probe_action: np.ndarray, image_size: int, neighbors: int) -> list[dict[str, Any]]:
    """Produce gate scores after the shared probe, with no task labels."""

    rows: list[dict[str, Any]] = []
    for root in roots:
        previous, current = _reset_and_probe(env, root, probe_action)
        rows.append(predict_rgb_effects(chart, _rgb_pair_descriptor(previous, current, image_size), neighbors=neighbors))
    return rows


def execute_paired_root(env: gym.Env, root: FrozenRoot, *, arm: str, probe_action: np.ndarray, bank: np.ndarray, chart: dict[str, np.ndarray], gate: dict[str, Any], image_size: int, neighbors: int, fallback_index: int) -> RuntimeEpisode:
    """Execute a full matched-budget arm; policy sees RGB/action values only."""

    previous, current = _reset_and_probe(env, root, probe_action)
    prediction = predict_rgb_effects(chart, _rgb_pair_descriptor(previous, current, image_size), neighbors=neighbors)
    decision = (
        {"selected_index": int(fallback_index), "release": False, "reason": "matched_fixed_fallback"}
        if arm == "fallback"
        else select_or_abstain(prediction, gate, fallback_index=fallback_index)
    )
    observation, _reward, _terminated, _truncated, _info = env.step(np.asarray(bank[int(decision["selected_index"])], dtype=np.float32))
    # This is the sole privileged-state read on the execution path, after all
    # actions have been committed.  It is passed directly to offline scoring.
    terminal = np.asarray(observation["state"], dtype=np.float64).copy()
    return RuntimeEpisode(
        root_id=int(root.root_id), probe_action=np.asarray(probe_action, dtype=np.float32),
        selected_action=np.asarray(bank[int(decision["selected_index"])], dtype=np.float32),
        selected_index=int(decision["selected_index"]), released=bool(decision["release"]),
        decision_reason=str(decision["reason"]), support_distance=(None if arm == "fallback" else float(prediction["support_distance"])),
        predicted_contrast=(None if arm == "fallback" else float(prediction["predicted_contrast"])),
        terminal_state_for_offline_scoring=terminal,
    )


def score_terminal(episode: RuntimeEpisode, root: FrozenRoot) -> dict[str, Any]:
    """Offline-only privileged terminal score; never fed back to a decision."""

    state = np.asarray(episode.terminal_state_for_offline_scoring, dtype=np.float64)
    goal = np.asarray(root.reset_goal, dtype=np.float64)
    xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    angle = float(angle_error(float(state[4]), float(goal[4])))
    return {"final_xy": xy, "final_angle": angle, "strict_pose_success": bool(xy < 20.0 and angle < np.pi / 9.0)}


def make_frozen_roots(env: gym.Env, *, count: int, seed: int, partition: str) -> list[FrozenRoot]:
    """Create immutable offline contact fixtures with a partitioned seed band."""

    fixtures = make_boundary_contact_roots(env, int(count), int(seed))
    return [FrozenRoot(root_id=int(seed) * 10_000 + index, reset_state=np.asarray(row["state"], dtype=np.float64).copy(), reset_goal=np.asarray(row["goal"], dtype=np.float64).copy()) for index, row in enumerate(fixtures)]


def validate_root_disjoint(*splits: list[FrozenRoot]) -> None:
    """Fail closed if split construction reused a root ID."""

    identifiers = [set(root.root_id for root in split) for split in splits]
    if any(left & right for index, left in enumerate(identifiers) for right in identifiers[index + 1:]):
        raise RuntimeError("root-disjoint train/calibration/test contract violated")


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired terminal outcomes without selecting a best root."""

    deltas = np.asarray([row["local"]["final_xy"] - row["fallback"]["final_xy"] for row in rows], dtype=np.float64)
    return {
        "test_roots": int(len(rows)),
        "mean_final_xy_delta_local_minus_fallback": float(deltas.mean()) if len(deltas) else None,
        "local_wins": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "local_losses": int(np.sum(deltas > 0.0)),
        "fallback_successes": int(sum(row["fallback"]["strict_pose_success"] for row in rows)),
        "local_successes": int(sum(row["local"]["strict_pose_success"] for row in rows)),
        "local_release_count": int(sum(row["local_runtime"]["released"] for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-roots", type=int, default=48)
    parser.add_argument("--calibration-roots", type=int, default=24)
    parser.add_argument("--test-roots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026102117)
    parser.add_argument("--pulse-magnitude", type=float, default=0.55)
    parser.add_argument("--probe-magnitude", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--minimum-calibration-roots", type=int, default=8)
    args = parser.parse_args()
    if min(args.train_roots, args.calibration_roots, args.test_roots) < 1:
        parser.error("every root-disjoint partition must be nonempty")
    if args.train_roots < 2:
        parser.error("train roots must include at least one row per fixed action")
    if not 0.0 < args.probe_magnitude <= 1.0:
        parser.error("probe magnitude must lie in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    runtime = require_cuda_allocation(args.device)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing results directory: {args.output}")
    bank = fixed_action_bank(args.pulse_magnitude)
    probe = np.asarray((float(args.probe_magnitude), 0.0), dtype=np.float32)
    fallback_index = 0
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=16)
    try:
        train = make_frozen_roots(env, count=args.train_roots, seed=args.seed, partition="train")
        calibration = make_frozen_roots(env, count=args.calibration_roots, seed=args.seed + 1_000_000, partition="calibration")
        test = make_frozen_roots(env, count=args.test_roots, seed=args.seed + 2_000_000, partition="test")
        validate_root_disjoint(train, calibration, test)
        contexts, action_indices, effects = collect_chart_rows(env, train, probe_action=probe, bank=bank, image_size=args.image_size)
        chart = fit_rgb_effect_chart(contexts, action_indices, effects)
        gate = calibrate_gate(calibration_predictions(env, calibration, chart, probe_action=probe, image_size=args.image_size, neighbors=args.neighbors), alpha=args.alpha, minimum_roots=args.minimum_calibration_roots)
        rows: list[dict[str, Any]] = []
        for root in test:
            fallback_runtime = execute_paired_root(env, root, arm="fallback", probe_action=probe, bank=bank, chart=chart, gate=gate, image_size=args.image_size, neighbors=args.neighbors, fallback_index=fallback_index)
            local_runtime = execute_paired_root(env, root, arm="local", probe_action=probe, bank=bank, chart=chart, gate=gate, image_size=args.image_size, neighbors=args.neighbors, fallback_index=fallback_index)
            rows.append({
                "root_id": int(root.root_id), "fallback": score_terminal(fallback_runtime, root), "local": score_terminal(local_runtime, root),
                "fallback_runtime": {"probe_action": fallback_runtime.probe_action.tolist(), "selected_action": fallback_runtime.selected_action.tolist(), "selected_index": fallback_runtime.selected_index, "released": False, "reason": fallback_runtime.decision_reason},
                "local_runtime": {"probe_action": local_runtime.probe_action.tolist(), "selected_action": local_runtime.selected_action.tolist(), "selected_index": local_runtime.selected_index, "released": local_runtime.released, "reason": local_runtime.decision_reason, "support_distance": local_runtime.support_distance, "predicted_contrast": local_runtime.predicted_contrast},
            })
    finally:
        env.close()
    report = {
        "schema": "pusht_probe_reobserve_select.v1",
        "runtime": runtime,
        "runtime_interface": RUNTIME_BOUNDARY,
        "protocol": {
            "environment": "swm/PushT-v1", "root_protocol": "offline frozen boundary-contact fixtures; root-disjoint train/calibration/test seed bands",
            "fixed_action_bank": bank.tolist(), "action_bank_sha256": action_bank_sha256(bank), "fallback_index": fallback_index,
            "shared_probe": probe.tolist(), "matched_budget": "both arms execute exactly one probe and one selected/fallback pulse",
            "chart": "train-root action-conditioned KNN of observed rendered RGB successor-minus-current effects",
            "selection": "maximum predicted RGB-effect energy; no state/goal/reward/task-cost/true-successor input",
            "gate": "calibration-only finite conformal support upper bound plus predicted RGB-effect contrast floor",
        },
        "gate": gate,
        "paired_rows": rows,
        "summary": paired_summary(rows),
        "sanity_checks": {"root_disjoint": True, "fixed_bank_predeclared": True, "probe_budget_matched": True, "abstain_executes_exact_fallback": True, "state_not_used_by_runtime_policy": True},
        "claim_boundary": "A tiny RGB-only probe-reobserve-select mechanism evaluation. It does not establish task-conditioned action ranking, robust pose closure, frozen world-model benefit, causal dynamics, or general transfer. Terminal privileged scores are evaluation only; no result authorizes threshold or bank tuning on these held roots.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe_reobserve_select.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report["summary"]
    (args.output / "probe_reobserve_select.md").write_text("\n".join((
        "# PushT probe-reobserve-select", "", "Root-disjoint RGB-only local effect selection against a probe-budget-matched fallback.", "",
        f"- calibration gate: `{gate['status']}`", f"- local releases: `{summary['local_release_count']}/{summary['test_roots']}`", f"- mean final XY delta (local - fallback): `{summary['mean_final_xy_delta_local_minus_fallback']}`", f"- wins/ties/losses: `{summary['local_wins']}/{summary['ties']}/{summary['local_losses']}`", "", f"Claim boundary: {report['claim_boundary']}", "",
    )), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": gate, "summary": summary, "claim_boundary": report["claim_boundary"]}, indent=2))


if __name__ == "__main__":
    main()
