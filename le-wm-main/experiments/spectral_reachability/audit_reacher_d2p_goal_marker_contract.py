"""Host-CUDA audit for Reacher's visible target-marker/goal-image contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np
import torch

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat
from experiments.spectral_reachability.reacher_d2p_context import (
    collect_executed_context,
    restore_execution_snapshot,
)


def _goal_finger_xy(env: gym.Env) -> np.ndarray:
    """Read fingertip xy at the evaluator-only joint-space goal state."""

    raw = env.unwrapped
    physics = raw.env.physics
    qpos = np.asarray(physics.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(physics.data.qvel, dtype=np.float64).copy()
    try:
        target_qpos = np.asarray(raw.env.task.target_qpos, dtype=np.float64)
        physics.data.qpos[:] = target_qpos
        physics.data.qvel[:] = 0.0
        physics.forward()
        return np.asarray(physics.named.data.geom_xpos["finger"], dtype=np.float64)[:2].copy()
    finally:
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026102595)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("goal-marker audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.roots < 4 or args.atol <= 0.0 or args.output.exists():
        raise ValueError("goal-marker audit requires a new output, four roots, and positive tolerance")
    torch.empty(0, device=torch.device("cuda"))
    install_missing_field_compat()
    output = args.output.resolve()
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    rows = []
    try:
        for offset in range(args.roots):
            seed = int(args.seed + offset)
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            raw = env.unwrapped
            physics = raw.env.physics
            expected = _goal_finger_xy(env)
            marker = np.asarray(physics.named.model.geom_pos["target"], dtype=np.float64).copy()
            if not np.allclose(marker[:2], expected, rtol=0.0, atol=args.atol):
                raise RuntimeError("visible Reacher target is not aligned to the qpos goal fingertip")
            # Confirm the evaluator-side snapshot restores the marker as well
            # as qpos/qvel/full MuJoCo state after a deliberately unrelated
            # model-geometry mutation.
            physics.named.model.geom_pos["target"] = np.array([0.0, 0.0, marker[2]], dtype=np.float64)
            restored_root, restored_goal = restore_execution_snapshot(
                env, seed=seed, snapshot=context["execution_snapshot"],
            )
            restored_marker = np.asarray(physics.named.model.geom_pos["target"], dtype=np.float64)
            rows.append({
                "seed": seed,
                "goal_finger_xy": expected.tolist(), "marker_xy": marker[:2].tolist(),
                "marker_goal_error": float(np.max(np.abs(marker[:2] - expected))),
                "snapshot_marker_restored": bool(np.allclose(restored_marker, marker, rtol=0.0, atol=args.atol)),
                "root_rgb_restored": bool(np.array_equal(restored_root, context["root_frame"])),
                "goal_rgb_restored": bool(np.array_equal(restored_goal, context["goal"])),
            })
    finally:
        env.close()
    passed = all(
        row["marker_goal_error"] <= args.atol and row["snapshot_marker_restored"]
        and row["root_rgb_restored"] and row["goal_rgb_restored"]
        for row in rows
    )
    report = {
        "schema": "reacher_d2p_goal_marker_contract_audit.v1", "status": "pass" if passed else "fail",
        "scope": "host-CUDA execution/visual-contract audit; no model is loaded, trained, or selected",
        "protocol": {"device": "cuda", "roots": args.roots, "root_seeds": [row["seed"] for row in rows], "atol": args.atol},
        "gates": {"all_visible_markers_align_to_qpos_goal_fingertip": passed}, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
