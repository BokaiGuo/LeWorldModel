"""Prospective dev303 development replay/tail runner. Not P1 confirmation."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_tail_handoff_20261102_exclusive as handoff
from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as prior
from . import p1_dev303_cp5_tail_physical_saved_20261102_exclusive as verifier

NAME = "p1_dev303_cp5_tail_physical_20261102_exclusive"
ROOT = prior.ROOT
base = prior.base
require = prior.require


def build():
    source = handoff.build()
    paths = [Path(__file__), Path(verifier.__file__), Path(handoff.__file__),
             Path(prior.__file__), Path(base.__file__), Path(base.native.__file__),
             Path(handoff.export.__file__), Path(handoff.saved.__file__),
             Path(verifier.saved.__file__)]
    return {
        "schema": NAME + ".contract.v1",
        "classification": "development_only_after_opened_prefix_not_holdout",
        "source": source,
        "implementation_sha256": {str(p.resolve()): base.sha(p) for p in paths},
        "seed": prior.SEED,
        "arms_in_fixed_order": list(handoff.ARMS),
        "reset_and_goal": "exact original reset, goal state and goal RGB",
        "warmup": "10 zero actions, exact original frames and snapshots at 0, 5, 10",
        "prefix": "five actual actions; exact state, snapshot and RGB at every step",
        "tail": "ten frozen model-selected actions per arm; no replanning or replacement",
        "early_stop": "stop arm on mismatch, exception, termination or truncation; no padding",
        "failure": "record failure; attempt other fixed arm with fresh environment",
        "state_use": "replay consistency guard and logging only, never selection",
        "physical_actions_authorized_by_contract": False,
        "P1_admission": False, "independent_confirmation": False,
    }


def branch(env, store, role, history, reset, goal_rgb, row, capture=base.capture):
    arm = role["arm"]
    goal = np.asarray(history["goal"], np.float64)
    checkpoints = {p["step"]: p for p in history["frames"]}
    gate = handoff.ReplayGate(role)
    obs, info = env.reset(seed=prior.SEED)
    require(np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
            and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), goal_rgb), "reset/goal mismatch")
    store.emit("reset_verified", arm=arm)
    row["observations"].append(capture(store, env, arm, "warmup", 0, checkpoints[0]))

    def step(phase, number, action):
        store.check()
        action = np.asarray(action, dtype="<f4")
        store.emit("action_dispatch", arm=arm, phase=phase, step=number, action=action.tolist())
        obs, reward, terminated, truncated, info = env.step(action.copy())
        observed = np.asarray(env.latest_action)
        store.emit("action_observed", arm=arm, phase=phase, step=number,
                   action=observed.tolist(), terminated=bool(terminated), truncated=bool(truncated))
        actual = prior._check_return(env, obs, info, goal, goal_rgb, action)
        require(np.isfinite(float(reward)), "nonfinite reward")
        transition = {"phase": phase, "step": number, "action": actual.tolist(),
                      "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                      "terminated": bool(terminated), "truncated": bool(truncated)}
        row["transitions"].append(transition)
        store.emit("action_return", arm=arm, **transition)
        return bool(terminated), bool(truncated)

    for i in range(1, 11):
        stopped = step("warmup", i, np.zeros(2, dtype="<f4"))
        require(not any(stopped), "warmup stopped")
        if i in checkpoints:
            row["observations"].append(capture(store, env, arm, "warmup", i, checkpoints[i]))
    for i, action in enumerate(role["prefix"], 1):
        stopped = step("prefix", i, action)
        require(not any(stopped), "prefix stopped")
        point = capture(store, env, arm, "prefix", i, role["prefix_checkpoints"][i - 1])
        row["observations"].append(point)
        gate.accept_prefix(action, point)
    require(row["observations"][-1]["pixel_sha256"] == role["cp5_pixel_sha256"],
            "cp5 RGB mismatch")
    tail = gate.fixed_tail()
    store.emit("cp5_gate_passed", arm=arm, pixel_sha256=role["cp5_pixel_sha256"])
    for i, action in enumerate(tail, 1):
        stopped = step("tail", i, action)
        point = capture(store, env, arm, "tail", i)
        row["observations"].append(point)
        if any(stopped):
            row["status"] = "native_early_stop"
            return
    row["status"] = "fixed_horizon_completed"


def execute(prepared, prepared_sha, output):
    prepared = Path(prepared).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == build(),
            "presealed contract or source drift")
    arrays, receipt, reset = prior.source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    for role in base.read(prepared)["source"]["roles"]:
        arm = role["arm"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=arm)
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native environment SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            branch(env, store, role, receipt["history"], reset, arrays["goal_rgb"], row)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": arm, **row["error"]})
            store.emit("branch_error", arm=arm, error=row["error"])
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    error = {"arm": arm, "stage": "close", "type": type(exc).__name__,
                             "message": str(exc)[:1000]}
                    errors.append(error)
                    store.emit("close_error", **error)
        rows.append(row)
        store.json(arm + ".row.json", row)
        store.emit("branch_closed", arm=arm, status=row["status"])
    try:
        require(base.read(prepared) == build(), "postrun source closure")
    except Exception as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__,
                       "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "prepare_sha256": prepared_sha,
              "status": "recorded" if not errors else "partial_or_failed",
              "rows": rows, "errors": errors, "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "classification": "development_only"}
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
        sha = prior.write_new(prepared, build())
        print(json.dumps({"prepared": str(prepared), "sha256": sha}))
    else:
        require(args.prepare_sha256 and args.authorize_development_physics,
                "new explicit physical authorization and prepared SHA required")
        result = execute(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "report_sha256": base.sha(output / "report.json"),
                          "status": result["status"], "P1_admission": False}))


if __name__ == "__main__":
    main()
