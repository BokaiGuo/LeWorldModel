"""Outcome-informed development diagnostic: one prefix action, common D2P tail.

The original D2P first action, a zero first action, and the GN first action are
each followed by the exact same D2P actions 2--10.  This isolates whether the
first action changes the physical effect of the common nine-action tail.
This diagnostic was designed after inspecting the prior dev301 physical run
and therefore cannot itself provide independent confirmation or P1 admission.
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

from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_dev301_blind_plan_physical_saved_20261101_codex as prior_saved

NAME = "p1_dev301_first_action_common_tail_20261101_codex"
ROOT = base.ROOT
PRIOR_PREP = ROOT / "results/p1_dev301_blind_plan_physical_20261101_codex_PREP_MAIN01.json"
PRIOR_PREP_SHA = "46c5ada195eb75a3ee39fe237762fa3711566fd45baf9bf04b4ac0dd7bcedf01"
PRIOR_RUN = ROOT / "results/p1_dev301_blind_plan_physical_20261101_codex_RUN_MAIN01"
PRIOR_REPORT_SHA = "cf129af89cad37b845d3373dff6f145a8e65b3343c9a61e81547dd67c29db508"
PRIOR_SAVED = ROOT / "results/p1_dev301_blind_plan_physical_saved_20261101_codex_MAIN01.json"
PRIOR_SAVED_SHA = "b5611cac42197160fafada72a3fb512a861461c55139bff580815421e90fde82"
ROLES = ("d2p_first", "zero_first", "gn_first")
require, sha = base.require, base.sha


def prior():
    require(sha(PRIOR_SAVED) == PRIOR_SAVED_SHA, "prior saved receipt SHA")
    verified = prior_saved.verify(PRIOR_PREP, PRIOR_PREP_SHA, PRIOR_RUN, PRIOR_REPORT_SHA)
    require(base.read(PRIOR_SAVED) == verified, "prior saved reconstruction")
    contract = base.read(PRIOR_PREP)
    actions = {row["arm"]: base.action_array(row["actions"]) for row in contract["arms"]}
    d2p, gn = actions["full10_conditional_D2P"], actions["line_search_GN"]
    variants = {
        "d2p_first": d2p.copy(),
        "zero_first": np.concatenate((np.zeros((1, 2), np.float32), d2p[1:])),
        "gn_first": np.concatenate((gn[:1], d2p[1:])),
    }
    return verified, variants


def build():
    verified, variants = prior()
    prior_report = base.read(PRIOR_RUN / "report.json")
    prior_d2p = next(
        row for row in prior_report["rows"] if row["arm"] == "full10_conditional_D2P"
    )
    require(
        prior_d2p["transitions"][0]["metrics"]["object_position_error"]
        == prior_d2p["initial_metrics"]["object_position_error"],
        "observed step1 object unchanged",
    )
    bindings = base.bindings()
    for path in (
        Path(__file__).resolve(),
        Path(prior_saved.__file__).resolve(),
        PRIOR_PREP,
        PRIOR_RUN / "report.json",
        PRIOR_SAVED,
    ):
        bindings[str(path)] = sha(path)
    return {
        "schema": NAME + ".contract.v1",
        "seed": base.SEED,
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
        "prefix_steps": 1,
        "common_tail_steps": 9,
        "prior_dev301_saved_receipt_sha256": PRIOR_SAVED_SHA,
        "prior_observation": {
            "d2p_step1_object_position_unchanged": True,
            "diagnostic_designed_after_prior_physical_outcome": True,
        },
        "physical_outcomes_used_to_design_diagnostic": True,
        "physical_outcomes_used_to_choose_within_run": False,
        "new_neural_calls": 0,
        "development_only": True,
        "equal_budget_claim": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared, prepared_sha, output):
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha and base.read(prepared) == build(), "prepared contract")
    contract = base.read(prepared)
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    source_history = base.read(base.source_directory() / "source_receipt.json")["history"]
    reset = base.read(base.source_directory() / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in source_history["frames"]}
    goal = np.asarray(source_history["goal"], np.float64)
    goal_rgb = base.image(base.source_directory() / source_history["goal_rgb"]["file"])
    started = time.monotonic()
    rows, errors = [], []
    store.emit("run_started", prepare_sha256=prepared_sha)
    for spec in contract["roles"]:
        role = spec["role"]
        commands = base.action_array(spec["actions"])
        row = {
            "role": role,
            "status": "running",
            "warmup_returned": 0,
            "plan_returned": 0,
            "transitions": [],
            "observations": [],
            "error": None,
        }
        rows.append(row)
        env = None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "environment source")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            store.emit("branch_started", arm=role, seed=base.SEED)
            obs, info = env.reset(seed=base.SEED)
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
                and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                "exact reset",
            )
            row["observations"].append(base.capture(store, env, role, "warmup", 0, checkpoints[0]))
            for step in range(1, 11):
                action = np.zeros(2, np.float32)
                store.emit("action_dispatch", arm=role, phase="warmup", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(
                    np.array_equal(np.asarray(env.latest_action), action)
                    and not terminated
                    and not truncated,
                    "warmup return",
                )
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
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(source_history["final_state"], np.float64)),
                "exact root",
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
                    "plan return",
                )
                transition = {
                    "step": step,
                    "action": action.tolist(),
                    "state": state.tolist(),
                    "reward": float(reward),
                    "n_contacts": int(info["n_contacts"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "metrics": base.metrics(env, goal, state),
                }
                row["plan_returned"] += 1
                row["transitions"].append(transition)
                store.emit("action_return", arm=role, phase="plan", **transition)
                row["observations"].append(base.capture(store, env, role, "plan", step))
                if step == 1:
                    row["prefix_metrics"] = deepcopy(transition["metrics"])
                    row["prefix_state"] = transition["state"]
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
        require(all(sha(path) == digest for path, digest in contract["bindings"].items()), "source closure")
    except BaseException as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    status = (
        "recorded_development"
        if not errors and all(row["status"] in ("horizon_completed", "native_success", "native_stopped") for row in rows)
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
        "physical_outcomes_used_to_design_diagnostic": True,
        "physical_outcomes_used_to_choose_within_run": False,
        "new_neural_calls": 0,
        "development_only": True,
        "equal_budget_claim": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        result = execute(args.prepared, args.prepare_sha256, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "report_sha256": sha(Path(args.output) / "report.json"),
                "rows": [
                    {
                        "role": row["role"],
                        "status": row["status"],
                        "native_success": row.get("final_metrics", {}).get("native_success"),
                        "object_position_error": row.get("final_metrics", {}).get("object_position_error"),
                    }
                    for row in result["rows"]
                ],
                "errors": result["errors"],
            },
            sort_keys=True,
        )
    )
    if result["status"] != "recorded_development":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
