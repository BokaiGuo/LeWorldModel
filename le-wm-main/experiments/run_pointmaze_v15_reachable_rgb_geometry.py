"""V15: evaluate frozen V14 RGB geometry under a reachable RGB-MPC contract.

V14's 20-step world-fixed bank established a short-horizon ranking effect but
could not certify terminal success.  This runner changes *only the planner
contract*: RGB-inferred maze waypoints, a fresh per-observation action bank,
and one 5-step pulse before re-observation.  The V14 estimator remains frozen.

The deployable selectors receive RGB-derived geometry and declared actions
only.  Simulator state is used to advance/evaluate MuJoCo; the explicit oracle
is the sole privileged diagnostic arm.
"""
from __future__ import annotations

import argparse
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
    DatasetMatchedRenderer, RGBMazeMap, action_bank, fit_visual_calibration,
    fit_visual_dynamics_calibration, green_center, load_episode,
)

SCHEMA = "pointmaze_v15_reachable_rgb_geometry.v1"
GOAL = np.asarray([0.8, 0.8], dtype=np.float64)
METHODS = ("rgb_greedy", "v14_rgb_geometry_dynamics", "privileged_one_step_oracle")


def _bootstrap(values: np.ndarray, seed: int = 202610381) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(4000)])
    return {"mean_gain": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975))}


def _geometry(images: list[np.ndarray], calibration: Any) -> np.ndarray:
    if len(images) != 9:
        raise ValueError("V15 needs a predecessor plus eight causal RGB observations")
    values = []
    for before, current in zip(images[:-1], images[1:]):
        center, previous = green_center(current), green_center(before)
        values.append(np.r_[calibration.position(center), calibration.velocity(np.asarray(center) - np.asarray(previous))])
    return np.asarray(values, dtype=np.float32)


def _step(model: mujoco.MjModel, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(state, dtype=np.float64)[:2]
    data.qvel[:] = np.asarray(state, dtype=np.float64)[2:]
    mujoco.mj_forward(model, data)
    for _ in range(5):
        data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(model, data)
    return np.r_[data.qpos[:2], data.qvel[:2]].copy()


def _estimate(estimator: Any, fit: dict[str, np.ndarray], geometry: np.ndarray) -> np.ndarray:
    parameter = next(estimator.parameters())
    value = (geometry[None] - fit["history_mean"]) / fit["history_scale"]
    with torch.inference_mode():
        result = estimator(torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)).cpu().numpy()[0]
    return np.asarray(result * fit["state_scale"] + fit["state_mean"], dtype=np.float64)


def _select(method: str, estimated: np.ndarray, current: np.ndarray, target: np.ndarray, bank: np.ndarray, dynamics: Any, model: mujoco.MjModel) -> tuple[int, np.ndarray | None]:
    if method == "rgb_greedy":
        return int(np.argmax(bank @ (target - current[:2]))), None
    if method == "v14_rgb_geometry_dynamics":
        endpoints = np.stack([estimated[:2] + dynamics.predict_position_delta(estimated[2:], candidate) for candidate in bank])
        return int(np.argmin(np.linalg.norm(endpoints - target, axis=1))), endpoints
    if method == "privileged_one_step_oracle":
        endpoints = np.stack([_step(model, current, candidate)[:2] for candidate in bank])
        return int(np.argmin(np.linalg.norm(endpoints - target, axis=1))), endpoints
    raise ValueError(f"unsupported method: {method}")


def _run_arm(method: str, *, initial_history: list[np.ndarray], model: mujoco.MjModel, renderer: DatasetMatchedRenderer, calibration: Any, maze: RGBMazeMap, estimator: Any, fit: dict[str, np.ndarray], dynamics: Any, blocks: int) -> dict[str, Any]:
    state = initial_history[-1].copy()
    images = [renderer.render(value) for value in initial_history]
    initial = float(np.linalg.norm(state[:2] - GOAL))
    trace: list[dict[str, Any]] = []
    for block in range(blocks):
        geometry = _geometry(images[-9:], calibration)
        estimate = _estimate(estimator, fit, geometry)
        position = geometry[-1, :2]
        route = maze.path_waypoints(position, GOAL)
        target = np.asarray(route[0] if route else GOAL, dtype=np.float64)
        bank = action_bank(position, target, directions=8, magnitudes=[.25, .50, .75, 1.0])
        index, endpoints = _select(method, estimate, state, target, bank, dynamics, model)
        state = _step(model, state, bank[index])
        images.append(renderer.render(state))
        trace.append({"block": block, "selected_candidate_id": index, "selected_action": bank[index].tolist(), "active_waypoint": target.tolist(), "estimated_state": estimate.tolist(), "candidate_endpoint_source": "privileged_simulator" if method == "privileged_one_step_oracle" else "V14_estimated_state_plus_calibration_dynamics" if method == "v14_rgb_geometry_dynamics" else "rgb_goal_projection", "simulator_state_in_policy_input": method == "privileged_one_step_oracle"})
        if np.linalg.norm(state[:2] - GOAL) <= .25:
            break
    final = float(np.linalg.norm(state[:2] - GOAL))
    return {"method": method, "initial_distance": initial, "final_distance": final, "progress": initial - final, "success": bool(final <= .25), "blocks_executed": len(trace), "trace": trace, "policy_contract": {"runtime_input": "RGB geometry history, RGB maze map, declared dynamic candidate bank", "simulator_state_in_policy_input": method == "privileged_one_step_oracle", "replanning": "5-step pulse, reobserve, replan", "candidate_count": 32}}


def run(dataset_root: Path, frozen: Path, output: Path, *, roots: list[int], visual_calibration_roots: list[int], dynamics_calibration_roots: list[int], blocks: int = 24, device: str = "cuda") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite V15 artifact")
    if set(roots) & (set(visual_calibration_roots) | set(dynamics_calibration_roots)):
        raise ValueError("held-out roots overlap a V15 calibration split")
    if any(root < 7256 for root in roots):
        raise ValueError("V15 requires roots unused by the two V14 closed-loop panels")
    cuda = require_cuda_recurrent(device)
    payload = json.loads(frozen.read_text())
    if payload.get("observation_chart") != "rgb_geometry" or not payload.get("authorization", {}).get("permitted"):
        raise ValueError("V15 requires an authorized V14 RGB-geometry frozen policy")
    estimator, _ranker, _action, fit = v12._load(payload, frozen, device=device)
    calibration = fit_visual_calibration(dataset_root, visual_calibration_roots, max_frames=1600)
    dynamics = fit_visual_dynamics_calibration(dataset_root, dynamics_calibration_roots)
    reference_images, _ = load_episode(dataset_root, visual_calibration_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, calibration)
    maze = RGBMazeMap.from_renderer(renderer)
    states = torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy().astype(np.float64)
    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    rows = []
    for root in roots:
        history = [states[root, frame].copy() for frame in range(0, 41, 5)]
        for method in METHODS:
            rows.append({"episode_id": int(root), **_run_arm(method, initial_history=history, model=model, renderer=renderer, calibration=calibration, maze=maze, estimator=estimator, fit=fit, dynamics=dynamics, blocks=blocks)})
    by_root = {root: {row["method"]: row for row in rows if row["episode_id"] == root} for root in roots}
    gains = np.asarray([by_root[root]["rgb_greedy"]["final_distance"] - by_root[root]["v14_rgb_geometry_dynamics"]["final_distance"] for root in roots])
    result = {"schema": SCHEMA, "cuda_preflight": cuda, "roots": roots, "root_disjoint": True, "visual_calibration_roots": visual_calibration_roots, "dynamics_calibration_roots": dynamics_calibration_roots, "methods": list(METHODS), "blocks": blocks, "goal_qpos": GOAL.tolist(), "rows": rows, "paired_v14_dynamics_over_rgb_greedy": _bootstrap(gains), "aggregates": {method: {"success_rate": float(np.mean([row["success"] for row in rows if row["method"] == method])), "mean_final_distance": float(np.mean([row["final_distance"] for row in rows if row["method"] == method])), "mean_progress": float(np.mean([row["progress"] for row in rows if row["method"] == method]))} for method in METHODS}, "claim_boundary": "Frozen V14 RGB-state readout plus separately calibrated visual local dynamics in same-dynamics PointMaze; not frozen-latent-only, cross-environment, or universal-world-model evidence."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--frozen", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--roots", type=int, nargs="+", required=True); parser.add_argument("--visual-calibration-roots", type=int, nargs="+", default=list(range(1800, 1840))); parser.add_argument("--dynamics-calibration-roots", type=int, nargs="+", default=list(range(1840, 1880))); parser.add_argument("--blocks", type=int, default=24); parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = vars(parser.parse_args()); result = run(**args)
    print(json.dumps({"artifact": str(args["output"]), "paired_v14_dynamics_over_rgb_greedy": result["paired_v14_dynamics_over_rgb_greedy"], "aggregates": result["aggregates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
