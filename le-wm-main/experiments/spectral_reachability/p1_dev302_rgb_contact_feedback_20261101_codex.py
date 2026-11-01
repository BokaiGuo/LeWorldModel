"""Development diagnosis of RGB contact coverage on the sealed dev302 root.

The intervention was designed after the sealed dev302 open-loop plans failed to
contact the object.  It therefore cannot admit P1.  Its narrow purpose is to
test whether an RGB-only, fail-closed contact approach followed by the same
sealed D2P tail restores physical object response.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from . import codex_p1_rgb_task_score_20261030_v2 as rgb
from . import p1_dev302_first_action_common_tail_20261101_codex as source

NAME = "p1_dev302_rgb_contact_feedback_20261101_codex"
ROOT = source.ROOT
ROLES = ("rgb_contact_then_d2p_tail", "zero_then_d2p_tail", "sealed_d2p")
PREFIX_STEPS = 5
TARGET_GAP_PIXELS = 7.0
MAX_RAW_MAGNITUDE = 0.25
NATIVE_PER_PIXEL = 512.0 / 224.0
RAW_TARGET_SCALE_NATIVE = 100.0
MOVING_PUSHER_COMPONENTS_MAX = 2
MOVING_PUSHER_BBOX_MAX_PIXELS = 20
require, sha = source.require, source.sha


def _packet_sha(actions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def rgb_contact_action(current_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return one bounded raw action using only current renderer RGB.

    The action aims the pusher centroid at the nearest visible object pixel.
    Segmentation failures are fatal; no simulator-state fallback is allowed.
    """
    image = rgb._image(current_rgb)
    pusher_mask, pusher = rgb._mask(image, rgb.BLUE, 8, single=False)
    x0, y0, x1, y1 = pusher["bounds_xyxy"]
    require(
        pusher["components"] <= MOVING_PUSHER_COMPONENTS_MAX
        and x1 - x0 + 1 <= MOVING_PUSHER_BBOX_MAX_PIXELS
        and y1 - y0 + 1 <= MOVING_PUSHER_BBOX_MAX_PIXELS,
        "moving pusher mask geometry",
    )
    object_mask, obj = rgb._mask(image, rgb.BLOCK, 8, single=False)
    del pusher_mask
    py, px = np.nonzero(object_mask)
    require(len(px) >= 8, "object pixels")
    pusher_xy = np.asarray(pusher["centroid_xy"], np.float64)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    delta = object_pixels - pusher_xy
    distances = np.linalg.norm(delta, axis=1)
    index = int(np.argmin(distances))
    gap = float(distances[index])
    nearest = object_pixels[index]
    if not math.isfinite(gap):
        raise ValueError("non-finite RGB contact gap")
    desired_pixels = max(0.0, gap - TARGET_GAP_PIXELS)
    if desired_pixels == 0.0:
        action = np.zeros(2, dtype=np.float32)
    else:
        direction = (nearest - pusher_xy) / gap
        magnitude = min(
            MAX_RAW_MAGNITUDE,
            desired_pixels * NATIVE_PER_PIXEL / RAW_TARGET_SCALE_NATIVE,
        )
        action = np.asarray(direction * magnitude, dtype=np.float32)
    require(
        action.shape == (2,)
        and np.isfinite(action).all()
        and float(np.linalg.norm(action)) <= MAX_RAW_MAGNITUDE + 1e-7,
        "bounded RGB contact action",
    )
    return action, {
        "pusher_centroid_xy": pusher_xy.tolist(),
        "nearest_object_pixel_xy": nearest.tolist(),
        "object_centroid_xy": obj["centroid_xy"],
        "centroid_to_nearest_object_gap_pixels": gap,
        "target_gap_pixels": TARGET_GAP_PIXELS,
        "raw_action": action.tolist(),
        "simulator_state_used": False,
    }


def material():
    audit, receipt, saved, variants = source.material()
    d2p = variants["d2p_first"]
    return audit, receipt, saved, d2p


def build():
    audit, receipt, saved, d2p = material()
    paths = [
        Path(__file__).resolve(),
        Path(rgb.__file__).resolve(),
        Path(source.__file__).resolve(),
        source.QUEUE,
        source.source_directory() / "report.json",
        source.source_directory() / "source_receipt.json",
        source.source_directory() / "reset_receipt.json",
        source.source_directory() / "goal.png",
        *(source.source_directory() / f"rgb_{step:02d}.png" for step in (0, 5, 10)),
        source.PLANNER_DIR / "report.json",
        source.PLANNER_DIR / "events.jsonl.gz",
        source.PLANNER_SAVED,
        *(source.PLANNER_DIR / f"root_0_{arm}.plan.json" for arm in source.base.ARMS[:2]),
    ]
    return {
        "schema": NAME + ".contract.v1",
        "seed": source.SEED,
        "roles": list(ROLES),
        "prefix_steps": PREFIX_STEPS,
        "sealed_d2p_actions": d2p.tolist(),
        "sealed_d2p_actions_sha256": _packet_sha(d2p),
        "common_tail": d2p[PREFIX_STEPS:].tolist(),
        "common_tail_sha256": _packet_sha(d2p[PREFIX_STEPS:]),
        "rgb_policy": {
            "colors": {"pusher": list(rgb.BLUE), "object": list(rgb.BLOCK)},
            "target_gap_pixels": TARGET_GAP_PIXELS,
            "max_raw_magnitude": MAX_RAW_MAGNITUDE,
            "native_per_pixel": NATIVE_PER_PIXEL,
            "raw_target_scale_native": RAW_TARGET_SCALE_NATIVE,
            "moving_pusher_components_max": MOVING_PUSHER_COMPONENTS_MAX,
            "moving_pusher_bbox_max_pixels": MOVING_PUSHER_BBOX_MAX_PIXELS,
            "fresh_rgb_before_each_prefix_action": True,
            "segmentation_failure": "fail_closed",
            "simulator_state_input": False,
        },
        "source_history_sha256": receipt["history_sha256"],
        "planner_task_id": saved["rows"][0]["task_id"],
        "designed_after_dev302_open_loop_outcome": True,
        "development_only": True,
        "equal_physical_action_cap": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "training_membership": audit["training_membership"],
        "bindings": {str(path): sha(path) for path in paths},
    }


def _fixed_actions(contract: dict, role: str) -> np.ndarray | None:
    d2p = np.asarray(contract["sealed_d2p_actions"], np.float32)
    tail = np.asarray(contract["common_tail"], np.float32)
    require(d2p.shape == (10, 2) and tail.shape == (5, 2), "sealed action shapes")
    if role == "sealed_d2p":
        return d2p
    if role == "zero_then_d2p_tail":
        return np.concatenate((np.zeros((5, 2), np.float32), tail))
    require(role == "rgb_contact_then_d2p_tail", "role")
    return None


def execute(prepared, prepared_sha, output):
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha and source.base.read(prepared) == build(), "prepared")
    contract = source.base.read(prepared)
    store = source.base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    history = source.base.read(source.source_directory() / "source_receipt.json")["history"]
    reset = source.base.read(source.source_directory() / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    goal_rgb = source.base.image(source.source_directory() / history["goal_rgb"]["file"])
    rows, errors = [], []
    started = time.monotonic()
    store.emit("run_started", prepare_sha256=prepared_sha)
    for role in ROLES:
        fixed = _fixed_actions(contract, role)
        row = {
            "role": role,
            "status": "running",
            "warmup_returned": 0,
            "plan_returned": 0,
            "transitions": [],
            "observations": [],
            "rgb_decisions": [],
            "error": None,
        }
        rows.append(row)
        env = None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == source.base.native.ENV_SHA, "environment")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            store.emit("branch_started", arm=role, seed=source.SEED)
            obs, info = env.reset(seed=source.SEED)
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
                and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                "reset",
            )
            row["observations"].append(source.base.capture(store, env, role, "warmup", 0, checkpoints[0]))
            for step in range(1, 11):
                action = np.zeros(2, np.float32)
                store.emit("action_dispatch", arm=role, phase="warmup", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
                row["warmup_returned"] += 1
                store.emit(
                    "action_return",
                    arm=role,
                    phase="warmup",
                    step=step,
                    action=action.tolist(),
                    state=np.asarray(obs["state"]).tolist(),
                    reward=float(reward),
                    n_contacts=int(info["n_contacts"]),
                    terminated=False,
                    truncated=False,
                )
                if step in checkpoints:
                    row["observations"].append(
                        source.base.capture(store, env, role, "warmup", step, checkpoints[step])
                    )
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(history["final_state"], np.float64)),
                "root",
            )
            row["initial_metrics"] = source.base.metrics(env, goal, obs["state"])
            executed = []
            for step in range(1, 11):
                if fixed is not None:
                    action = fixed[step - 1].copy()
                elif step <= PREFIX_STEPS:
                    current_rgb = np.asarray(env.render(), np.uint8)
                    action, decision = rgb_contact_action(current_rgb)
                    decision["step"] = step
                    decision["pixel_sha256"] = source.base.pixel_sha(current_rgb)
                    row["rgb_decisions"].append(decision)
                else:
                    action = np.asarray(contract["common_tail"][step - PREFIX_STEPS - 1], np.float32)
                executed.append(action.copy())
                store.emit("action_dispatch", arm=role, phase="plan", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                state = np.asarray(obs["state"], np.float64)
                require(
                    np.array_equal(np.asarray(env.latest_action), action)
                    and np.array_equal(np.asarray(info["goal_state"]), goal)
                    and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                    "plan",
                )
                transition = {
                    "step": step,
                    "action": action.tolist(),
                    "state": state.tolist(),
                    "reward": float(reward),
                    "n_contacts": int(info["n_contacts"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "metrics": source.base.metrics(env, goal, state),
                }
                row["plan_returned"] += 1
                row["transitions"].append(transition)
                store.emit("action_return", arm=role, phase="plan", **transition)
                row["observations"].append(source.base.capture(store, env, role, "plan", step))
                if step == PREFIX_STEPS:
                    row["prefix_metrics"] = deepcopy(transition["metrics"])
                if terminated or truncated:
                    row["status"] = (
                        "native_success" if transition["metrics"]["native_success"] else "native_stopped"
                    )
                    break
            else:
                row["status"] = "horizon_completed"
            actions = np.asarray(executed, dtype="<f4")
            row["executed_actions_sha256"] = _packet_sha(actions)
            row["final_metrics"] = deepcopy(row["transitions"][-1]["metrics"])
        except BaseException as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"role": role, **row["error"]})
        finally:
            if env is not None:
                env.close()
            store.json(f"{role}.row.json", row)
            store.emit("branch_closed", arm=role, status=row["status"])
    try:
        require(all(sha(path) == digest for path, digest in contract["bindings"].items()), "closure")
    except BaseException as exc:
        errors.append({"stage": "closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    status = (
        "recorded_development_diagnosis"
        if not errors
        and all(row["status"] in ("horizon_completed", "native_success", "native_stopped") for row in rows)
        else "partial_or_failed"
    )
    store.emit("run_finished", status=status, errors=errors)
    report = {
        "schema": NAME + ".report.v1",
        "status": status,
        "prepare_sha256": prepared_sha,
        "rows": rows,
        "errors": errors,
        "journal_sha256": sha(store.path / "journal.jsonl"),
        "journal_tail": store.events[-1]["sha256"],
        "wall_seconds": time.monotonic() - started,
        "designed_after_dev302_open_loop_outcome": True,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True)
    run.add_argument("--prepare-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--authorize-development-physics", required=True, action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        digest = source.base.write_new(args.output, build())
        print(json.dumps({"prepared": str(Path(args.output).absolute()), "sha256": digest}, sort_keys=True))
        return
    with source.base.locking.gpu_lock(source.base.LOCK):
        report = execute(args.prepared, args.prepare_sha256, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": sha(Path(args.output) / "report.json"),
                "rows": [
                    {
                        "role": row["role"],
                        "status": row["status"],
                        "native_success": row.get("final_metrics", {}).get("native_success"),
                        "object_position_error": row.get("final_metrics", {}).get(
                            "object_position_error"
                        ),
                        "contact_steps": [
                            item["step"] for item in row["transitions"] if item["n_contacts"] > 0
                        ],
                    }
                    for row in report["rows"]
                ],
                "errors": report["errors"],
            },
            sort_keys=True,
        )
    )
    if report["status"] != "recorded_development_diagnosis":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
