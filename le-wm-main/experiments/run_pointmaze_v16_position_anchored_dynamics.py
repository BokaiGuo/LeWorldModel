"""V16: position-anchored dynamics after the V15 state-readout mismatch.

V15 found a concrete mismatch: the frozen V14 estimator reduces velocity error
but degrades position relative to the direct RGB chart, while waypoint control
is position dominated.  V16 is a different, predeclared intervention rather
than a V15 rerun.  It keeps the RGB chart as the position anchor and uses the
frozen V14 estimate only for velocity in the locally calibrated dynamics map.

The protocol has two stages.  ``design`` may use a disjoint design set and a
privileged oracle only to select a navigation goal with measurable headroom
over RGB greedy; it never queries V14 or the proposed anchor.  ``evaluate``
requires that frozen design artifact and scores new roots exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v12_decoupled_belief as v12
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _legacy_pointmaze_xml
from experiments.gpu_preflight import require_cuda_recurrent
from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer,
    RGBMazeMap,
    action_bank,
    fit_visual_calibration,
    fit_visual_dynamics_calibration,
    green_center,
    load_episode,
)
from experiments.run_pointmaze_v15_reachable_rgb_geometry import _bootstrap, _estimate, _geometry, _step


DESIGN_SCHEMA = "pointmaze_v16_position_anchored_design.v2"
SCORE_SCHEMA = "pointmaze_v16_position_anchored_score.v1"
METHODS = (
    "rgb_greedy",
    "v14_full_state_dynamics_diagnostic",
    "rgb_position_anchored_velocity",
    "privileged_one_step_oracle",
)


def _oracle_headroom_selection(
    candidates: list[dict[str, Any]],
    *,
    min_lower_bound: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select only a design goal whose oracle establishes planner headroom.

    A controller comparison is informative only if the exact candidate bank,
    horizon, and terminal objective leave some attainable improvement over the
    RGB-greedy baseline.  Merely choosing the largest *mean* oracle gap can
    otherwise certify a near-zero gap caused by Monte-Carlo noise and mislabel
    every learned method as a dynamics/representation failure.
    """

    if not candidates:
        raise ValueError("oracle-headroom selection requires at least one goal candidate")
    if not np.isfinite(float(min_lower_bound)) or float(min_lower_bound) < 0.0:
        raise ValueError("minimum oracle-headroom lower bound must be finite and non-negative")
    eligible = [
        row for row in candidates
        if float(row["oracle_minus_greedy_terminal_distance_ci"]["lower"]) > float(min_lower_bound)
    ]
    decision = {
        "criterion": "paired bootstrap lower CI of privileged-oracle minus RGB-greedy terminal-distance gain exceeds declared minimum",
        "minimum_lower_bound": float(min_lower_bound),
        "eligible_goal_count": int(len(eligible)),
        "candidate_goal_count": int(len(candidates)),
    }
    if not eligible:
        decision["status"] = "rejected_insufficient_oracle_headroom"
        return None, decision
    chosen = max(
        eligible,
        key=lambda row: (
            float(row["oracle_minus_greedy_terminal_distance_ci"]["lower"]),
            float(row["oracle_minus_greedy_terminal_distance_ci"]["mean_gain"]),
            float(row["oracle_minus_greedy_success_rate"]),
            -float(row["goal_qpos"][0]),
            -float(row["goal_qpos"][1]),
        ),
    )
    decision["status"] = "passed_oracle_headroom"
    decision["selected_goal_qpos"] = list(chosen["goal_qpos"])
    decision["selected_headroom"] = dict(chosen["oracle_minus_greedy_terminal_distance_ci"])
    return chosen, decision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(
    dataset_root: Path,
    frozen: Path,
    visual_calibration_roots: list[int],
    dynamics_calibration_roots: list[int],
    *,
    device: str,
) -> tuple[Any, dict[str, np.ndarray], Any, DatasetMatchedRenderer, RGBMazeMap, np.ndarray, mujoco.MjModel]:
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    if payload.get("observation_chart") != "rgb_geometry" or not payload.get("authorization", {}).get("permitted"):
        raise ValueError("V16 requires the authorized frozen V14 RGB-geometry estimator")
    estimator, _ranker, _action, fit = v12._load(payload, frozen, device=device)
    calibration = fit_visual_calibration(dataset_root, visual_calibration_roots, max_frames=1600)
    dynamics = fit_visual_dynamics_calibration(dataset_root, dynamics_calibration_roots)
    reference_images, _ = load_episode(dataset_root, visual_calibration_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, calibration)
    maze = RGBMazeMap.from_renderer(renderer)
    states = torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy().astype(np.float64)
    return estimator, fit, dynamics, renderer, maze, states, mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())


def _direct_position(image: np.ndarray, calibration: Any) -> np.ndarray:
    return np.asarray(calibration.position(green_center(image)), dtype=np.float64)


def _position_anchored_state(estimated: np.ndarray, image: np.ndarray, calibration: Any) -> np.ndarray:
    """Fuse the two readouts without allowing velocity to perturb position."""
    value = np.asarray(estimated, dtype=np.float64).copy()
    if value.shape != (4,):
        raise ValueError("V14 state estimator must return [x, y, vx, vy]")
    value[:2] = _direct_position(image, calibration)
    return value


def _select(
    method: str,
    *,
    estimated: np.ndarray,
    image: np.ndarray,
    calibration: Any,
    current: np.ndarray,
    target: np.ndarray,
    bank: np.ndarray,
    dynamics: Any,
    model: mujoco.MjModel,
) -> int:
    if method == "rgb_greedy":
        position = _direct_position(image, calibration)
        return int(np.argmax(bank @ (target - position)))
    if method == "v14_full_state_dynamics_diagnostic":
        state = np.asarray(estimated, dtype=np.float64)
        endpoints = np.stack([state[:2] + dynamics.predict_position_delta(state[2:], action) for action in bank])
        return int(np.argmin(np.linalg.norm(endpoints - target, axis=1)))
    if method == "rgb_position_anchored_velocity":
        state = _position_anchored_state(estimated, image, calibration)
        endpoints = np.stack([state[:2] + dynamics.predict_position_delta(state[2:], action) for action in bank])
        return int(np.argmin(np.linalg.norm(endpoints - target, axis=1)))
    if method == "privileged_one_step_oracle":
        endpoints = np.stack([_step(model, current, action)[:2] for action in bank])
        return int(np.argmin(np.linalg.norm(endpoints - target, axis=1)))
    raise ValueError(f"unsupported V16 method {method!r}")


def _run_arm(
    method: str,
    *,
    initial_history: list[np.ndarray],
    model: mujoco.MjModel,
    renderer: DatasetMatchedRenderer,
    calibration: Any,
    maze: RGBMazeMap,
    estimator: Any,
    fit: dict[str, np.ndarray],
    dynamics: Any,
    goal: np.ndarray,
    blocks: int,
) -> dict[str, Any]:
    state = initial_history[-1].copy()
    images = [renderer.render(value) for value in initial_history]
    initial = float(np.linalg.norm(state[:2] - goal))
    trace: list[dict[str, Any]] = []
    for block in range(int(blocks)):
        geometry = _geometry(images[-9:], calibration)
        estimated = _estimate(estimator, fit, geometry)
        position = _direct_position(images[-1], calibration)
        route = maze.path_waypoints(position, goal)
        target = np.asarray(route[0] if route else goal, dtype=np.float64)
        bank = action_bank(position, target, directions=8, magnitudes=[.25, .50, .75, 1.0])
        chosen = _select(
            method, estimated=estimated, image=images[-1], calibration=calibration,
            current=state, target=target, bank=bank, dynamics=dynamics, model=model,
        )
        state = _step(model, state, bank[chosen])
        images.append(renderer.render(state))
        trace.append({
            "block": int(block), "selected_candidate_id": int(chosen),
            "selected_action": bank[chosen].astype(float).tolist(), "active_waypoint": target.tolist(),
            "estimated_state": estimated.astype(float).tolist(),
            "rgb_position": position.tolist(),
            "simulator_state_in_policy_input": method == "privileged_one_step_oracle",
        })
        if np.linalg.norm(state[:2] - goal) <= .25:
            break
    final = float(np.linalg.norm(state[:2] - goal))
    return {
        "method": method, "initial_distance": initial, "final_distance": final,
        "progress": initial - final, "success": bool(final <= .25),
        "blocks_executed": len(trace), "trace": trace,
        "policy_contract": {
            "runtime_input": "RGB geometry history, RGB maze map, declared dynamic candidate bank",
            "simulator_state_in_policy_input": method == "privileged_one_step_oracle",
            "position_source": "direct RGB chart" if method in ("rgb_greedy", "rgb_position_anchored_velocity") else "V14 estimator or oracle",
            "candidate_count": 32, "replanning": "5-step pulse, reobserve, replan",
        },
    }


def _validate_split(roots: list[int], *other_splits: list[int], states: np.ndarray) -> None:
    if len(roots) < 8 or len(set(roots)) != len(roots):
        raise ValueError("V16 needs at least eight unique roots")
    if min(roots) < 7290 or max(roots) >= len(states):
        raise ValueError("V16 roots must be unused IDs in [7290, state_count)")
    if set(roots) & set().union(*(set(values) for values in other_splits)):
        raise ValueError("V16 root roles must be disjoint")


def design(
    dataset_root: Path,
    frozen: Path,
    output: Path,
    *,
    roots: list[int],
    visual_calibration_roots: list[int],
    dynamics_calibration_roots: list[int],
    blocks: int = 24,
    device: str = "cuda",
    min_oracle_headroom_lower: float = 0.0,
    fixed_goal: list[float] | None = None,
) -> dict[str, Any]:
    """Freeze a goal using only disjoint design-root oracle-vs-greedy headroom."""
    if output.exists():
        raise FileExistsError("refusing to overwrite V16 design artifact")
    runtime = require_cuda_recurrent(device)
    estimator, fit, dynamics, renderer, maze, states, model = _load_runtime(
        dataset_root, frozen, visual_calibration_roots, dynamics_calibration_roots, device=device,
    )
    _validate_split(roots, visual_calibration_roots, dynamics_calibration_roots, states=states)
    # These are RGB-inferred free-cell centers.  Selection is deliberately
    # based only on oracle-minus-greedy headroom; neither V14 variant is read.
    if fixed_goal is None:
        goals = [maze.cell_positions[cell].astype(float) for cell in sorted(maze.free_cells)]
        goal_source = "rgb_maze_free_cells"
    else:
        goal = np.asarray(fixed_goal, dtype=np.float64).reshape(-1)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("fixed_goal must be a finite [x, y] RGB-chart qpos target")
        if not any(np.allclose(goal, value) for value in maze.cell_positions.values()):
            raise ValueError("fixed_goal must be a declared RGB maze free-cell position")
        goals = [goal]
        goal_source = "predeclared_fixed_rgb_maze_cell"
    candidates = []
    for goal in goals:
        greedy, oracle = [], []
        for root in roots:
            history = [states[root, frame].copy() for frame in range(0, 41, 5)]
            greedy.append(_run_arm("rgb_greedy", initial_history=history, model=model, renderer=renderer, calibration=renderer.calibration, maze=maze, estimator=estimator, fit=fit, dynamics=dynamics, goal=goal, blocks=blocks))
            oracle.append(_run_arm("privileged_one_step_oracle", initial_history=history, model=model, renderer=renderer, calibration=renderer.calibration, maze=maze, estimator=estimator, fit=fit, dynamics=dynamics, goal=goal, blocks=blocks))
        root_gains = np.asarray(
            [left["final_distance"] - right["final_distance"] for left, right in zip(greedy, oracle)],
            dtype=np.float64,
        )
        distance_headroom = float(np.mean(root_gains))
        success_headroom = float(np.mean([right["success"] for right in oracle]) - np.mean([left["success"] for left in greedy]))
        candidates.append({
            "goal_qpos": goal.tolist(),
            "oracle_minus_greedy_terminal_distance": distance_headroom,
            "oracle_minus_greedy_terminal_distance_ci": _bootstrap(root_gains),
            "oracle_minus_greedy_success_rate": success_headroom,
        })
    chosen, headroom_gate = _oracle_headroom_selection(
        candidates, min_lower_bound=min_oracle_headroom_lower,
    )
    result = {
        "schema": DESIGN_SCHEMA, "cuda_preflight": runtime, "design_roots": sorted(roots),
        "visual_calibration_roots": sorted(visual_calibration_roots), "dynamics_calibration_roots": sorted(dynamics_calibration_roots),
        "blocks": int(blocks), "goal_candidates": candidates,
        "goal_candidate_source": goal_source,
        "oracle_headroom_gate": headroom_gate,
        "status": headroom_gate["status"],
        "selected_goal_qpos": None if chosen is None else chosen["goal_qpos"],
        "selection_rule": "among goals with positive declared oracle-headroom lower bound, maximize lower CI then mean privileged-oracle minus RGB-greedy terminal-distance headroom; V14/full/anchored methods unavailable to design",
        "frozen_v14_sha256": _sha256(frozen), "dataset_states_sha256": _sha256(dataset_root / "states.pth"),
        "claim_boundary": "Design-only privileged headroom calibration. A rejected oracle-headroom gate identifies a candidate/horizon limitation, not a learned-controller failure; it is not held-out evidence and cannot be used as a deployable policy.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate(
    dataset_root: Path,
    frozen: Path,
    design_artifact: Path,
    output: Path,
    *,
    roots: list[int],
    visual_calibration_roots: list[int],
    dynamics_calibration_roots: list[int],
    blocks: int = 24,
    device: str = "cuda",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite V16 evaluation artifact")
    design_payload = json.loads(design_artifact.read_text(encoding="utf-8"))
    if design_payload.get("schema") not in {"pointmaze_v16_position_anchored_design.v1", DESIGN_SCHEMA}:
        raise ValueError("V16 evaluation requires its sealed design artifact")
    if design_payload.get("schema") == DESIGN_SCHEMA and design_payload.get("status") != "passed_oracle_headroom":
        raise ValueError("V16 evaluation requires a design artifact that passed its oracle-headroom gate")
    if design_payload.get("frozen_v14_sha256") != _sha256(frozen):
        raise ValueError("V16 design artifact binds a different frozen V14 estimator")
    runtime = require_cuda_recurrent(device)
    estimator, fit, dynamics, renderer, maze, states, model = _load_runtime(
        dataset_root, frozen, visual_calibration_roots, dynamics_calibration_roots, device=device,
    )
    _validate_split(roots, list(design_payload["design_roots"]), visual_calibration_roots, dynamics_calibration_roots, states=states)
    if design_payload.get("dataset_states_sha256") != _sha256(dataset_root / "states.pth"):
        raise ValueError("V16 design/evaluation dataset identity mismatch")
    if int(design_payload.get("blocks")) != int(blocks):
        raise ValueError("V16 requires the design-frozen planning horizon")
    goal = np.asarray(design_payload["selected_goal_qpos"], dtype=np.float64)
    rows = []
    for root in roots:
        history = [states[root, frame].copy() for frame in range(0, 41, 5)]
        for method in METHODS:
            rows.append({"episode_id": int(root), **_run_arm(method, initial_history=history, model=model, renderer=renderer, calibration=renderer.calibration, maze=maze, estimator=estimator, fit=fit, dynamics=dynamics, goal=goal, blocks=blocks)})
    by_root = {root: {row["method"]: row for row in rows if row["episode_id"] == root} for root in roots}
    def paired(method: str, baseline: str = "rgb_greedy") -> dict[str, float]:
        return _bootstrap(np.asarray([by_root[root][baseline]["final_distance"] - by_root[root][method]["final_distance"] for root in roots]))
    result = {
        "schema": SCORE_SCHEMA, "cuda_preflight": runtime, "roots": sorted(roots), "design_artifact": str(design_artifact.resolve()),
        "design_sha256": _sha256(design_artifact), "selected_goal_qpos": goal.tolist(), "blocks": int(blocks),
        "methods": list(METHODS), "rows": rows,
        "paired_position_anchored_over_rgb_greedy": paired("rgb_position_anchored_velocity"),
        "paired_full_v14_over_rgb_greedy": paired("v14_full_state_dynamics_diagnostic"),
        "paired_oracle_over_rgb_greedy": paired("privileged_one_step_oracle"),
        "aggregates": {method: {"success_rate": float(np.mean([row["success"] for row in rows if row["method"] == method])), "mean_final_distance": float(np.mean([row["final_distance"] for row in rows if row["method"] == method])), "mean_progress": float(np.mean([row["progress"] for row in rows if row["method"] == method]))} for method in METHODS},
        "claim_boundary": "Predeclared-goal RGB-only PointMaze planner evaluation. The position-anchored policy uses direct RGB position plus frozen V14 velocity and calibration-local dynamics; it is not frozen-latent-only or cross-environment evidence.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (sub.add_parser("design"), sub.add_parser("evaluate")):
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--frozen", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--roots", type=int, nargs="+", required=True)
        command.add_argument("--visual-calibration-roots", type=int, nargs="+", default=list(range(1800, 1840)))
        command.add_argument("--dynamics-calibration-roots", type=int, nargs="+", default=list(range(1840, 1880)))
        command.add_argument("--blocks", type=int, default=24)
        command.add_argument("--device", choices=("cuda",), default="cuda")
    sub.choices["design"].add_argument("--min-oracle-headroom-lower", type=float, default=0.0)
    sub.choices["design"].add_argument("--fixed-goal", type=float, nargs=2, default=None)
    evaluate_parser = sub.choices["evaluate"]
    evaluate_parser.add_argument("--design-artifact", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = design(**args) if command == "design" else evaluate(**args)
    print(json.dumps({"artifact": str(args["output"]), "schema": result["schema"], "selected_goal_qpos": result.get("selected_goal_qpos")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
