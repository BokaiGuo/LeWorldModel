"""Falsify or confirm whether Reacher's target marker reaches the RGB input.

The official Reacher HDF5 stores both pixels and the reset-selected static
``target_pos``.  This host-CUDA audit restores held-out source qpos/qvel roots
and compares two renderer states: preserving that static marker, or the
formerly deployed synthetic marker glued to the current fingertip.  It also
measures their shared source-render deviation.  A target marker that is
transparent in the renderer cannot explain a frozen model's successor error,
even if its MuJoCo geometry differs.  The audit loads no world model and
cannot alter a planning outcome.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.run_live_reacher_planning import install_missing_field_compat


SCHEMA = "reacher_d2p_source_rgb_contract_audit.v1"


def source_marker_position(live_marker: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Place the two stored source coordinates into a compatible MuJoCo geom."""

    marker = np.asarray(live_marker, dtype=np.float64).reshape(-1).copy()
    target = np.asarray(target_pos, dtype=np.float64).reshape(-1)
    if marker.size < 2 or target.shape != (2,) or not np.isfinite(marker).all() or not np.isfinite(target).all():
        raise ValueError("source target marker requires finite [x,y] coordinates")
    marker[:2] = target
    return marker


def _pixel_error(rendered: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    left = np.asarray(rendered, dtype=np.int16)
    right = np.asarray(recorded, dtype=np.int16)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError("runtime and source RGB frames have incompatible shapes")
    delta = np.abs(left - right)
    return {
        "mean_abs_pixel_error": float(delta.mean()),
        "max_abs_pixel_error": int(delta.max()),
        "changed_pixel_fraction": float(np.any(delta != 0, axis=-1).mean()),
    }


def _render_source_root(
    env: gym.Env, *, qpos: np.ndarray, qvel: np.ndarray, marker: np.ndarray,
) -> np.ndarray:
    raw = env.unwrapped
    physics = raw.env.physics
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.shape != physics.data.qpos.shape or qvel.shape != physics.data.qvel.shape:
        raise ValueError("official Reacher qpos/qvel shape differs from deployed runtime")
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
    physics.named.model.geom_pos["target"] = np.asarray(marker, dtype=np.float64)
    physics.forward()
    return np.asarray(raw.render(width=224, height=224), dtype=np.uint8).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026102504)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher source RGB contract audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.samples < 8 or args.output.exists():
        raise ValueError("audit requires new output and at least eight samples")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"official Reacher source is unavailable: {dataset}")
    torch.empty(0, device=torch.device("cuda"))
    install_missing_field_compat()
    with h5py.File(dataset, "r", swmr=True) as handle:
        required = {"pixels", "qpos", "qvel", "target_pos"}
        if not required.issubset(handle):
            raise ValueError("official Reacher source lacks RGB/state/target columns")
        rng = np.random.default_rng(int(args.seed))
        chosen = np.sort(rng.choice(len(handle["pixels"]), size=args.samples, replace=False))
        records = [
            {
                "source_row": int(row), "pixels": np.asarray(handle["pixels"][row], dtype=np.uint8),
                "qpos": np.asarray(handle["qpos"][row], dtype=np.float64),
                "qvel": np.asarray(handle["qvel"][row], dtype=np.float64),
                "target_pos": np.asarray(handle["target_pos"][row], dtype=np.float64),
            }
            for row in chosen
        ]
    env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")
    rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records):
            env.reset(seed=int(args.seed + index))
            physics = env.unwrapped.env.physics
            live_marker = np.asarray(physics.named.model.geom_pos["target"], dtype=np.float64).copy()
            static_marker = source_marker_position(live_marker, record["target_pos"])
            static = _render_source_root(env, qpos=record["qpos"], qvel=record["qvel"], marker=static_marker)
            physics.data.qpos[:] = record["qpos"]
            physics.data.qvel[:] = record["qvel"]
            physics.forward()
            finger_marker = np.asarray(physics.named.data.geom_xpos["finger"], dtype=np.float64).copy()
            synthetic = _render_source_root(env, qpos=record["qpos"], qvel=record["qvel"], marker=finger_marker)
            static_error = _pixel_error(static, record["pixels"])
            synthetic_error = _pixel_error(synthetic, record["pixels"])
            marker_swap_error = _pixel_error(static, synthetic)
            rows.append({
                "source_row": record["source_row"], "source_target_pos": record["target_pos"].tolist(),
                "static_marker": static_marker.tolist(), "finger_marker": finger_marker.tolist(),
                "static_source_error": static_error, "synthetic_finger_marker_error": synthetic_error,
                "marker_swap_runtime_rgb_error": marker_swap_error,
            })
    finally:
        env.close()
    static_errors = np.asarray([row["static_source_error"]["mean_abs_pixel_error"] for row in rows])
    synthetic_errors = np.asarray([row["synthetic_finger_marker_error"]["mean_abs_pixel_error"] for row in rows])
    swap_errors = np.asarray([row["marker_swap_runtime_rgb_error"]["mean_abs_pixel_error"] for row in rows])
    marker_is_visually_inert = bool(np.all(swap_errors == 0.0))
    result = {
        "schema": SCHEMA, "status": "pass" if marker_is_visually_inert else "fail",
        "scope": "fresh host-CUDA source-to-runtime RGB renderer audit; no model is loaded, trained, or selected",
        "inputs": {"dataset": str(dataset), "dataset_bytes": int(dataset.stat().st_size)},
        "protocol": {"device": "cuda", "samples": int(args.samples), "seed": int(args.seed), "source_contract": "restore HDF5 qpos/qvel and static target_pos marker", "contrast": "move only target geometry to current fingertip"},
        "aggregate": {"max_static_mean_abs_pixel_error": float(static_errors.max()), "mean_static_mean_abs_pixel_error": float(static_errors.mean()), "mean_synthetic_mean_abs_pixel_error": float(synthetic_errors.mean()), "max_marker_swap_mean_abs_pixel_error": float(swap_errors.max())},
        "gates": {"target_marker_does_not_change_runtime_rgb_on_every_sample": marker_is_visually_inert},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": result["gates"], "aggregate": result["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
