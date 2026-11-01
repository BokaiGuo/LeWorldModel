"""Presealed development-only dev303 RGB chunk versus zero diagnostic.

Preparation is CPU-only. Physical execution requires a separately supplied
contract SHA and explicit CLI flag. This module is not holdout confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_blind_adapter_20261101_codex as source
from . import p1_dev303_physical_prefix_bridge_20261101_exclusive as bridge
from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_rgb_chunk_approach_interface_20261102_exclusive as control

NAME = "p1_dev303_rgb_chunk_diagnostic_20261102_exclusive"
ROOT = source.ROOT
SEED = bridge.SEED
ARMS = ("rgb_chunk", "zero_prefix")
HORIZON = control.MAX_CHUNKS
CONTROL_SHA = "ab0433279828738db5d62298efd1671cf5972dd389cc6bbe838918be529bbec6"
SOURCE_DIR = (source.QUEUE.parent / (source.QUEUE.name + ".native_reset")
              / f"{source.SLOT}_{source.ATTEMPT:04d}.warmup")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_data():
    arrays, receipt_sha, audit = source.source()
    require(receipt_sha == bridge.RECEIPT_SHA and audit["seed"] == SEED
            and audit["P1_admission"] is False, "sealed dev303 source")
    receipt = base.read(SOURCE_DIR / "source_receipt.json")
    reset = base.read(SOURCE_DIR / "reset_receipt.json")
    history = receipt["history"]
    require([row["step"] for row in history["frames"]] == [0, 5, 10]
            and history["actions"] == [[0.0, 0.0]] * 10
            and receipt["P1_admission"] is False
            and receipt["independent_confirmation"] is False, "sealed warmup")
    require(digest((SOURCE_DIR / "reset_receipt.json").read_bytes())
            == history["reset_receipt_sha256"], "reset receipt binding")
    return arrays, receipt, reset


def build():
    require(base.sha(control.__file__) == CONTROL_SHA, "frozen single-component controller SHA")
    arrays, receipt, reset = source_data()
    paths = [
        Path(__file__), Path(control.__file__), Path(control.rgb.__file__),
        Path(control.geometry.__file__), Path(control.motion.__file__),
        Path(control.prior.__file__), Path(base.__file__), Path(base.native.__file__),
        Path(source.__file__), Path(source.core.__file__), Path(source.core.saved.__file__),
        Path(bridge.__file__),
        source.QUEUE, SOURCE_DIR / "source_receipt.json",
        SOURCE_DIR / "reset_receipt.json",
    ]
    history = receipt["history"]
    for entry in [*history["frames"], history["goal_rgb"]]:
        path = SOURCE_DIR / entry["file"]
        require(path.resolve().parent == SOURCE_DIR.resolve(), "source image path")
        paths.append(path)
    bindings = {str(p.resolve()): base.sha(p) for p in paths}
    return {
        "schema": NAME + ".contract.v1",
        "classification": "development_only_not_original_holdout_confirmation",
        "seed": SEED,
        "source_receipt_sha256": bridge.RECEIPT_SHA,
        "reset_receipt_sha256": history["reset_receipt_sha256"],
        "initial_state": reset["state"],
        "root_state": history["final_state"],
        "goal_state": history["goal"],
        "warmup_rgb_sha256": [digest(frame.tobytes()) for frame in arrays["rgb_history"]],
        "goal_rgb_sha256": digest(arrays["goal_rgb"].tobytes()),
        "arms_in_fixed_order": list(ARMS),
        "warmup_actions": 10,
        "max_control_actions_per_arm": HORIZON,
        "rgb_controller": "RGBChunkApproach, one action then actual rendered RGB",
        "rgb_controller_sha256": CONTROL_SHA,
        "zero_control": "up to five zero actions; early termination not padded",
        "early_stop": "record and stop arm; no retry, replacement, or padding",
        "selection_uses_state_goal_reward_contact_or_outcome": False,
        "state_and_goal_use": "reset/goal consistency and outcome logging only",
        "P1_admission": False,
        "independent_confirmation": False,
        "physical_execution_authorized_by_contract": False,
        "bindings": bindings,
    }


def write_new(path, obj):
    raw = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def _check_return(env, obs, info, goal, goal_rgb, action):
    actual = np.asarray(env.latest_action)
    require(actual.shape == (2,) and np.array_equal(actual, action),
            "actual action mismatch")
    require(np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), goal_rgb), "goal drift")
    require(np.isfinite(np.asarray(obs["state"], np.float64)).all(), "state nonfinite")
    return actual


def run_branch(env, store, arm, history, reset, goal_rgb, *, capture=base.capture, row=None):
    """Execute one arm; caller owns errors, branch logging, and environment."""
    require(arm in ARMS, "fixed arm")
    goal = np.asarray(history["goal"], np.float64)
    checkpoints = {row["step"]: row for row in history["frames"]}
    if row is None:
        row = {"arm": arm, "status": "running", "warmup_returned": 0,
               "control_returned": 0, "transitions": [], "observations": []}
    obs, info = env.reset(seed=SEED)
    require(np.array_equal(np.asarray(obs["state"], np.float64),
                           np.asarray(reset["state"], np.float64))
            and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), goal_rgb), "reset/goal mismatch")
    row["observations"].append(capture(store, env, arm, "warmup", 0, checkpoints[0]))
    for step in range(1, 11):
        action = np.zeros(2, dtype="<f4")
        store.emit("action_dispatch", arm=arm, phase="warmup", step=step, action=action.tolist())
        obs, reward, terminated, truncated, info = env.step(action.copy())
        store.emit("action_observed", arm=arm, phase="warmup", step=step,
                   action=np.asarray(env.latest_action).tolist(),
                   terminated=bool(terminated), truncated=bool(truncated))
        actual = _check_return(env, obs, info, goal, goal_rgb, action)
        store.emit("action_return", arm=arm, phase="warmup", step=step,
                   action=actual.tolist(), terminated=bool(terminated), truncated=bool(truncated))
        row["warmup_returned"] += 1
        require(not terminated and not truncated, "warmup stopped")
        if step in checkpoints:
            row["observations"].append(capture(store, env, arm, "warmup", step, checkpoints[step]))
    require(np.array_equal(np.asarray(obs["state"], np.float64),
                           np.asarray(history["final_state"], np.float64)), "warmup root mismatch")
    controller = control.RGBChunkApproach(initial_previous_action_is_zero=True) if arm == ARMS[0] else None
    previous_sha = None
    for step in range(1, HORIZON + 1):
        store.check()
        if controller is not None:
            current = base.image(store.path / row["observations"][-1]["file"])
            proposal = controller.observe(current, frame_id=step - 1,
                                          executed_action_sha256=previous_sha)
            store.emit("controller_decision", arm=arm, step=step, proposal=proposal)
            if "action" not in proposal:
                row["status"] = proposal["status"]
                break
            action = np.asarray(proposal["action"], dtype="<f4")
            require(digest(action.tobytes()) == proposal["action_sha256"], "proposal action hash")
        else:
            action = np.zeros(2, dtype="<f4")
        store.emit("action_dispatch", arm=arm, phase="control", step=step,
                   action=action.tolist(), action_sha256=digest(action.tobytes()))
        obs, reward, terminated, truncated, info = env.step(action.copy())
        store.emit("action_observed", arm=arm, phase="control", step=step,
                   action=np.asarray(env.latest_action).tolist(),
                   terminated=bool(terminated), truncated=bool(truncated))
        actual = _check_return(env, obs, info, goal, goal_rgb, action)
        previous_sha = digest(np.asarray(actual, dtype="<f4").tobytes())
        transition = {"step": step, "action": actual.tolist(), "action_sha256": previous_sha,
                      "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                      "n_contacts": int(info["n_contacts"]), "terminated": bool(terminated),
                      "truncated": bool(truncated)}
        row["transitions"].append(transition)
        row["control_returned"] += 1
        store.emit("action_return", arm=arm, phase="control", **transition)
        row["observations"].append(capture(store, env, arm, "control", step))
        if terminated or truncated:
            row["status"] = "native_early_stop"
            break
    else:
        row["status"] = "fixed_horizon_completed"
    return row


def execute(prepared, prepared_sha, output):
    prepared = Path(prepared)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == build(),
            "presealed contract or bindings drift")
    arrays, receipt, reset = source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared.resolve()),
                                          "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    for arm in ARMS:
        env = None
        row = {"arm": arm, "status": "running", "warmup_returned": 0,
               "control_returned": 0, "transitions": [], "observations": []}
        try:
            store.emit("branch_started", arm=arm)
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA,
                    "native environment source SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            run_branch(env, store, arm, receipt["history"], reset, arrays["goal_rgb"], row=row)
        except Exception as exc:
            row.update(status="failed",
                       error={"type": type(exc).__name__, "message": str(exc)[:1000]})
            errors.append(row["error"] | {"arm": arm})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    errors.append({"arm": arm, "stage": "close", "type": type(exc).__name__,
                                   "message": str(exc)[:1000]})
        rows.append(row)
        store.json(arm + ".row.json", row)
        store.emit("branch_closed", arm=arm, status=row["status"])
    try:
        require(build() == base.read(prepared), "post-run source closure")
    except Exception as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__,
                       "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "status": "recorded" if not errors else "partial_or_failed",
              "prepared_sha256": prepared_sha, "rows": rows, "errors": errors,
              "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"],
              "P1_admission": False, "independent_confirmation": False,
              "classification": "development_only_not_original_holdout_confirmation"}
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-development-physics", action="store_true")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.prepare_sha256 and not args.authorize_development_physics
                and not os.path.lexists(output), "prepare only")
        sha = write_new(prepared, build())
        print(json.dumps({"prepared": str(prepared), "sha256": sha}, sort_keys=True))
    else:
        require(args.authorize_development_physics and args.prepare_sha256,
                "explicit development-physics authorization and contract SHA required")
        report = execute(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "status": report["status"],
                          "report_sha256": base.sha(output / "report.json")}, sort_keys=True))
        return int(bool(report["errors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
