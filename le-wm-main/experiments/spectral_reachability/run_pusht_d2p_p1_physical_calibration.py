"""CUDA-only physical calibration for a replay-bound PushT D2P-P1 admission.

This consumes, rather than reselects, the prefix witnesses in a sealed P1
receipt.  Each selected prefix and a same-root raw-zero prefix are executed
after the identical causal warmup; only then is a one-step DLS suffix compiled
from the *observed* successor RGB/action context.  Simulator state is used
solely after execution for evaluator labels and independent deterministic
replay.  This is calibration, never an outcome/efficacy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- registers PushT
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.online_planning_protocol import validate_action_protocol
from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.run_pusht_counterfactual import image_transform
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1


SCHEMA = "pusht_d2p_p1_physical_calibration.v1"


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pose_label(state: np.ndarray, goal: np.ndarray) -> dict[str, float]:
    state, goal = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    position = float(np.linalg.norm(state[2:4] - goal[2:4]))
    angle = float(abs(np.arctan2(np.sin(state[4] - goal[4]), np.cos(state[4] - goal[4]))))
    return {"xy_distance": position, "angle_distance": angle, "pose_cost": position + 20.0 * angle}


def _reset_and_warmup(env: gym.Env, root: dict[str, np.ndarray], *, history_len: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Recreate the P1 source history and leave env at its live decision state."""
    state, goal = np.asarray(root["state"], dtype=np.float64), np.asarray(root["goal"], dtype=np.float64)
    env.reset(seed=0, options={"state": goal.copy(), "goal_state": goal.copy()})
    goal_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
    env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
    actions = np.zeros((history_len, 2), dtype=np.float32)
    frames: list[np.ndarray] = []
    for action in actions:
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("PushT causal warmup terminated")
    return frames, actions, goal_rgb


def _observed_suffix(
    model: torch.nn.Module, frames: list[np.ndarray], actions: np.ndarray, goal_rgb: np.ndarray,
    *, transform: Any, device: torch.device, damping: float, sketch_dims: int, projection_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    emb, encoded_actions = p1._encode_context(model, frames, actions, transform, device)
    goal = p0._goal_latent(model, goal_rgb, transform=transform, device=device)
    projection = p1._frozen_projection(latent_dim=int(goal.numel()), sketch=sketch_dims, device=device, seed=projection_seed)
    reach = p1._reachability(model, goal, emb, encoded_actions, horizon=1, damping=damping, projection=projection)
    action = torch.tanh(reach.pop("dls_params"))[0].detach().cpu().numpy().astype(np.float32)
    receipt = {**reach, "projection_sha256": _sha256(projection.detach().cpu().numpy())}
    return action, receipt


def _execute_branch(
    env: gym.Env, root: dict[str, np.ndarray], prefix: np.ndarray, *, model: torch.nn.Module,
    transform: Any, device: torch.device, damping: float, sketch_dims: int, projection_seed: int,
) -> dict[str, Any]:
    frames, history, goal_rgb = _reset_and_warmup(env, root, history_len=3)
    root_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy()
    _obs, _reward, terminated, truncated, _info = env.step(np.asarray(prefix, dtype=np.float32))
    if terminated or truncated:
        raise RuntimeError("PushT prefix terminated")
    successor_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
    successor_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy()
    successor_frames = frames[1:] + [successor_rgb]
    successor_actions = np.concatenate((history[1:], np.asarray(prefix, dtype=np.float32).reshape(1, 2)), axis=0)
    suffix, suffix_receipt = _observed_suffix(
        model, successor_frames, successor_actions, goal_rgb, transform=transform, device=device,
        damping=damping, sketch_dims=sketch_dims, projection_seed=projection_seed,
    )
    _obs, _reward, terminated, truncated, _info = env.step(suffix[0])
    if terminated or truncated:
        raise RuntimeError("PushT recompiled suffix terminated")
    final_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
    final_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy()
    return {
        "warmup_rgb_sha256": _sha256(np.stack(frames)), "goal_rgb_sha256": _sha256(goal_rgb),
        "prefix_action": np.asarray(prefix, dtype=np.float32).tolist(), "suffix_action": suffix.tolist(),
        "root_label": _pose_label(root_state, root["goal"]),
        "after_prefix_label": _pose_label(successor_state, root["goal"]),
        "final_label": _pose_label(final_state, root["goal"]),
        "successor_rgb_sha256": _sha256(successor_rgb), "final_rgb_sha256": _sha256(final_rgb),
        "successor_state": successor_state.tolist(), "final_state": final_state.tolist(),
        "suffix_recompile": suffix_receipt,
    }


def _replay_matches(first: dict[str, Any], second: dict[str, Any], *, atol: float = 1e-8) -> bool:
    keys = ("warmup_rgb_sha256", "goal_rgb_sha256", "successor_rgb_sha256", "final_rgb_sha256")
    return bool(all(first[key] == second[key] for key in keys) and np.allclose(first["successor_state"], second["successor_state"], atol=atol, rtol=0.0) and np.allclose(first["final_state"], second["final_state"], atol=atol, rtol=0.0))


def _load_preflight(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != p1.SCHEMA or report.get("status") != "pass" or not all(report.get("gates", {}).values()):
        raise ValueError("physical calibration requires a passing replay-bound PushT P1 preflight")
    required = ("root_seed", "root_count", "prefix_horizon", "suffix_horizon", "damping", "sketch_dims")
    if any(key not in report.get("protocol", {}) for key in required):
        raise ValueError("P1 receipt lacks deterministic replay binding")
    if int(report["protocol"]["prefix_horizon"]) != 1 or int(report["protocol"]["suffix_horizon"]) != 1:
        raise ValueError("physical calibration v1 seals one action in each stage")
    return report


def _load_calibration(path: Path) -> set[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("status") != "pass" or not all(report.get("gates", {}).values()):
        raise ValueError("outcome requires a passing physical calibration receipt")
    return {str(row["root_sha256"]) for row in report.get("rows", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable physical calibration output: {args.output}")
    p1_report = _load_preflight(args.p1_evidence)
    cohort = str(p1_report["protocol"].get("cohort", "calibration"))
    if cohort == "outcome" and args.calibration_evidence is None:
        raise ValueError("outcome physical execution requires --calibration-evidence")
    calibration_roots = set() if args.calibration_evidence is None else _load_calibration(args.calibration_evidence)
    runtime = require_cuda_allocation("cuda"); device = torch.device("cuda")
    model = load_compatible_pretrained("pusht/lewm", device=device).eval(); model.requires_grad_(False)
    if validate_action_protocol(model, action_dim=2, action_block=5) != 5:
        raise RuntimeError("PushT checkpoint action block drift")
    before = p0._parameter_hash(model); protocol = p1_report["protocol"]
    env = gym.make("swm/PushT-v1", render_mode="rgb_array"); transform = image_transform(); started = time.time()
    try:
        roots = collect_nonterminal_boundary_contact_roots(env, count=int(protocol["root_count"]), seed=int(protocol["root_seed"]), oversample_factor=int(protocol.get("root_oversample_factor", 4)))
        if calibration_roots & {root_fingerprint(root) for root in roots}:
            raise RuntimeError("outcome roots overlap physical calibration roots")
        rows: list[dict[str, Any]] = []
        for index, (root, admitted) in enumerate(zip(roots, p1_report["rows"], strict=True)):
            if root_fingerprint(root) != admitted["root_sha256"]:
                raise RuntimeError("P1 root reconstruction does not match sealed receipt")
            compilation = p1.compile_reconfiguration(model, *_reset_and_warmup(env, root, history_len=3), transform=transform, device=device, proposals=int(protocol["prefix_proposals"]), prefix_horizon=1, suffix_horizon=1, damping=float(protocol["damping"]), sketch=int(protocol["sketch_dims"]), seed=int(protocol["root_seed"]) + index)
            expected = admitted["compilation"]
            same_selection = compilation["selected_proposal_index"] == expected["selected_proposal_index"]
            same_action = compilation["selected_prefix_live_action"] is None if same_selection and compilation["selected_proposal_index"] is None else np.allclose(compilation["selected_prefix_live_action"], expected["selected_prefix_live_action"], atol=1e-7, rtol=0.0)
            if not same_selection or not same_action:
                raise RuntimeError("P1 prefix reconstruction drifted from sealed model-side selection")
            model_eligible = bool(compilation["non_greedy_witness_found"])
            selected_prefix = np.asarray(compilation["selected_prefix_live_action"], dtype=np.float32)[0] if model_eligible else np.zeros(2, dtype=np.float32)
            zero_prefix = np.zeros(2, dtype=np.float32)
            selected_seed = int(protocol["root_seed"]) + index + 20_000
            selected = _execute_branch(env, root, selected_prefix, model=model, transform=transform, device=device, damping=float(protocol["damping"]), sketch_dims=int(protocol["sketch_dims"]), projection_seed=selected_seed)
            zero = _execute_branch(env, root, zero_prefix, model=model, transform=transform, device=device, damping=float(protocol["damping"]), sketch_dims=int(protocol["sketch_dims"]), projection_seed=selected_seed if not model_eligible else int(protocol["root_seed"]) + index + 30_000)
            selected_replay = _execute_branch(env, root, selected_prefix, model=model, transform=transform, device=device, damping=float(protocol["damping"]), sketch_dims=int(protocol["sketch_dims"]), projection_seed=selected_seed)
            rows.append({"root_id": index, "root_sha256": root_fingerprint(root), "model_eligible": model_eligible, "sealed_compilation": compilation, "selected": selected, "raw_zero_control": zero, "selected_independent_replay_matches": _replay_matches(selected, selected_replay)})
            print(json.dumps({"root_id": index, "selected_after_prefix": selected["after_prefix_label"]["pose_cost"], "zero_after_prefix": zero["after_prefix_label"]["pose_cost"], "replay": rows[-1]["selected_independent_replay_matches"]}))
    finally:
        env.close()
    eligible_rows = [row for row in rows if row["model_eligible"]]
    minimum_eligible = 1 if cohort == "calibration" else max(1, int(np.ceil(0.8 * len(rows))))
    gates = {"cuda_only": True, "frozen_parameter_hash_unchanged": before == p0._parameter_hash(model), "sealed_prefixes_reconstructed": True, "same_root_raw_zero_controls": True, "all_observed_successors_recompiled": all(row["selected"]["suffix_recompile"]["jacobian_frobenius_norm"] > 1e-10 and row["raw_zero_control"]["suffix_recompile"]["jacobian_frobenius_norm"] > 1e-10 for row in rows), "enough_model_eligible_roots": len(eligible_rows) >= minimum_eligible, "all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero": bool(eligible_rows) and all(row["selected"]["after_prefix_label"]["pose_cost"] >= row["raw_zero_control"]["after_prefix_label"]["pose_cost"] - 1e-8 for row in eligible_rows), "independent_replay_exact": all(row["selected_independent_replay_matches"] for row in rows), "root_disjoint_from_physical_calibration": not bool(calibration_roots & {row["root_sha256"] for row in rows}), "cohort_binding_valid": cohort == "calibration" or args.calibration_evidence is not None}
    report = {"schema": SCHEMA, "status": "pass" if all(gates.values()) else "fail", "scope": "CUDA-only P1 physical calibration" if cohort == "calibration" else "CUDA-only root-disjoint P1 outcome measurement; it records a paired mechanism contrast but does not open P2.", "cuda_host": {**runtime, "python": platform.python_version()}, "upstream_p1": {"path": str(args.p1_evidence.resolve()), "sha256": hashlib.sha256(args.p1_evidence.read_bytes()).hexdigest()}, "protocol": {"cohort": cohort, "planner_input": "RGB/action histories and rendered goal RGB only", "evaluator_only_labels": "PushT state after execution", "prefix_control": "same-root raw zero", "stages": ["sealed prefix", "observed successor RGB recompile", "one action DLS suffix"], "calibration_evidence": None if args.calibration_evidence is None else str(args.calibration_evidence.resolve()), "outcome_roots_opened": cohort == "outcome", "minimum_model_eligible_roots": minimum_eligible}, "gates": gates, "eligibility": {"model_eligible_root_count": len(eligible_rows), "model_ineligible_root_count": len(rows) - len(eligible_rows)}, "rows": rows, "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True); (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
