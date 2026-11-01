"""Prospective development-holdout test of the frozen first-action rule.

For seed 301000002, the model plans were sealed before any physical execution.
The D2P, zero, and GN first actions are followed by the same D2P actions 2--10.
The intervention rule was fixed from dev301; no dev302 physical outcome enters
selection.  Training membership remains unknown, so this is not an independent
confirmation or P1 admission result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from . import p1_constructive_blind_planners_saved_audit_20261027_v1 as planner_saved
from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_queue_goal_saved_20261101_codex as source_saved

NAME = "p1_dev302_first_action_common_tail_20261101_codex"
ROOT = base.ROOT
SEED = 301000002
QUEUE = ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV02.json"
QUEUE_SHA = "467975c9c23046e430aae0163fef0450cffc03c27fdf32f8294c616724d0067d"
SOURCE_REPORT_SHA = "aa88db0c4d5f8303ca24bd1ac52b1f6ffeaf6572ef81074589f4b1a934bbf919"
SOURCE_SLOT = "dev_holdout02"
PLANNER_DIR = ROOT / "results/p1_dev302_blind_planners_20261101_codex_RUN_MAIN01"
PLANNER_REPORT_SHA = "ae01af20fe76aec4c17cc992542f6f508cd893194a12165566179e4688ce176c"
PLANNER_PREP_SHA = "7dce84ad7b26a306bc5644369916dc2837f224cc1c9c75ce6a615b14b0bbae81"
PLANNER_SAVED = ROOT / "results/p1_dev302_blind_planners_saved_20261101_codex_MAIN02.json"
PLANNER_SAVED_SHA = "ca34a31dfdd780fa8ad292d3ad78196947e99ea864d9ed286cd11b31f5b06274"
ROLES = ("d2p_first", "zero_first", "gn_first")
require, sha = base.require, base.sha


def source_directory():
    return QUEUE.parent / (QUEUE.name + ".native_reset") / f"{SOURCE_SLOT}_0000.warmup"


def material():
    audit = source_saved.verify(QUEUE, QUEUE_SHA, SOURCE_SLOT, 0, SOURCE_REPORT_SHA)
    require(audit["seed"] == SEED and audit["P1_admission"] is False, "source")
    receipt = base.read(source_directory() / "source_receipt.json")
    require(sha(PLANNER_SAVED) == PLANNER_SAVED_SHA, "planner saved SHA")
    saved = base.read(PLANNER_SAVED)
    rebuilt = planner_saved.audit(
        PLANNER_DIR / "report.json", PLANNER_REPORT_SHA, PLANNER_PREP_SHA
    )
    expected = deepcopy(rebuilt)
    expected["auditor_sha256"] = sha(Path(planner_saved.__file__))
    require(saved == expected, "planner saved reconstruction")
    plans = {}
    for arm in base.ARMS[:2]:
        plans[arm] = base.action_array(
            base.read(PLANNER_DIR / f"root_0_{arm}.plan.json")["commands"]
        )
    d2p, gn = plans["full10_conditional_D2P"], plans["line_search_GN"]
    variants = {
        "d2p_first": d2p.copy(),
        "zero_first": np.concatenate((np.zeros((1, 2), np.float32), d2p[1:])),
        "gn_first": np.concatenate((gn[:1], d2p[1:])),
    }
    return audit, receipt, saved, variants


def build():
    audit, receipt, saved, variants = material()
    paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(source_saved.__file__).resolve(),
        Path(planner_saved.__file__).resolve(),
        QUEUE,
        source_directory() / "report.json",
        source_directory() / "source_receipt.json",
        source_directory() / "reset_receipt.json",
        source_directory() / "goal.png",
        *(source_directory() / f"rgb_{step:02d}.png" for step in (0, 5, 10)),
        PLANNER_DIR / "report.json",
        PLANNER_DIR / "events.jsonl.gz",
        PLANNER_SAVED,
        *(PLANNER_DIR / f"root_0_{arm}.plan.json" for arm in base.ARMS[:2]),
    ]
    bindings = {str(path): sha(path) for path in paths}
    return {
        "schema": NAME + ".contract.v1",
        "seed": SEED,
        "roles": [
            {
                "role": role,
                "actions": variants[role].tolist(),
                "actions_sha256": hashlib.sha256(variants[role].astype("<f4").tobytes()).hexdigest(),
            }
            for role in ROLES
        ],
        "common_tail_sha256": hashlib.sha256(
            variants["d2p_first"][1:].astype("<f4").tobytes()
        ).hexdigest(),
        "source_history_sha256": receipt["history_sha256"],
        "planner_task_id": saved["rows"][0]["task_id"],
        "planner_model_terminal_cost": {
            row["arm"]: row["terminal_cost"] for row in saved["rows"]
        },
        "rule_frozen_from_dev301_before_dev302_physics": True,
        "dev302_physical_outcomes_used_for_selection": False,
        "training_membership": audit["training_membership"],
        "historical_coverage": audit["historical_coverage"],
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "equal_physical_action_budget": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared, prepared_sha, output):
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha and base.read(prepared) == build(), "prepared")
    contract = base.read(prepared)
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    history = base.read(source_directory() / "source_receipt.json")["history"]
    reset = base.read(source_directory() / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    goal_rgb = base.image(source_directory() / history["goal_rgb"]["file"])
    rows, errors = [], []
    started = time.monotonic()
    store.emit("run_started", prepare_sha256=prepared_sha)
    for spec in contract["roles"]:
        role, commands = spec["role"], base.action_array(spec["actions"])
        row = {
            "role": role, "status": "running", "warmup_returned": 0,
            "plan_returned": 0, "transitions": [], "observations": [], "error": None,
        }
        rows.append(row)
        env = None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "environment")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            store.emit("branch_started", arm=role, seed=SEED)
            obs, info = env.reset(seed=SEED)
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
                and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                "reset",
            )
            row["observations"].append(base.capture(store, env, role, "warmup", 0, checkpoints[0]))
            for step in range(1, 11):
                action = np.zeros(2, np.float32)
                store.emit("action_dispatch", arm=role, phase="warmup", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
                row["warmup_returned"] += 1
                store.emit(
                    "action_return", arm=role, phase="warmup", step=step,
                    action=action.tolist(), state=np.asarray(obs["state"]).tolist(),
                    reward=float(reward), n_contacts=int(info["n_contacts"]),
                    terminated=False, truncated=False,
                )
                if step in checkpoints:
                    row["observations"].append(base.capture(store, env, role, "warmup", step, checkpoints[step]))
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(history["final_state"], np.float64)),
                "root",
            )
            row["initial_metrics"] = base.metrics(env, goal, obs["state"])
            for step, action in enumerate(commands, 1):
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
                    "step": step, "action": action.tolist(), "state": state.tolist(),
                    "reward": float(reward), "n_contacts": int(info["n_contacts"]),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "metrics": base.metrics(env, goal, state),
                }
                row["plan_returned"] += 1
                row["transitions"].append(transition)
                store.emit("action_return", arm=role, phase="plan", **transition)
                row["observations"].append(base.capture(store, env, role, "plan", step))
                if step == 1:
                    row["prefix_metrics"] = deepcopy(transition["metrics"])
                if terminated or truncated:
                    row["status"] = "native_success" if transition["metrics"]["native_success"] else "native_stopped"
                    break
            else:
                row["status"] = "horizon_completed"
            row["final_metrics"] = deepcopy(row["transitions"][-1]["metrics"])
            row["planned_actions_sha256"] = spec["actions_sha256"]
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
        "recorded_development_holdout"
        if not errors and all(row["status"] in ("horizon_completed", "native_success", "native_stopped") for row in rows)
        else "partial_or_failed"
    )
    store.emit("run_finished", status=status, errors=errors)
    report = {
        "schema": NAME + ".report.v1", "status": status, "prepare_sha256": prepared_sha,
        "rows": rows, "errors": errors, "journal_sha256": sha(store.path / "journal.jsonl"),
        "journal_tail": store.events[-1]["sha256"], "wall_seconds": time.monotonic() - started,
        "rule_frozen_from_dev301_before_dev302_physics": True,
        "dev302_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0, "development_only": True,
        "equal_physical_action_budget": True, "independent_confirmation": False,
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
        digest = base.write_new(args.output, build())
        print(json.dumps({"prepared": str(Path(args.output).absolute()), "sha256": digest}, sort_keys=True))
        return
    with base.locking.gpu_lock(base.LOCK):
        report = execute(args.prepared, args.prepare_sha256, args.output)
    print(json.dumps({
        "status": report["status"], "report_sha256": sha(Path(args.output) / "report.json"),
        "rows": [{"role": row["role"], "status": row["status"],
                  "native_success": row.get("final_metrics", {}).get("native_success"),
                  "object_position_error": row.get("final_metrics", {}).get("object_position_error")}
                 for row in report["rows"]], "errors": report["errors"]}, sort_keys=True))
    if report["status"] != "recorded_development_holdout":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
