"""Physical dev302 comparison of model-selected and seeded cp5 tails."""
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

from . import p1_dev302_rgb_cp5_cem_20261101_codex as planner

NAME = "p1_dev302_rgb_cp5_tail_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_rgb_cp5_cem_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "948d9a323777022a13228ff435e20817915d37a39c2592bd9392d41723e22583"
PLAN_RUN = ROOT / "results/p1_dev302_rgb_cp5_cem_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "f8fa4585509389630fd74bd8b2f2feb8a2312146061c2bc5943cb4b49e92cf68"
ROLES = ("model_selected", "rgb_contact_seed", "old_d2p_tail", "old_gn_tail", "zero_tail")
require, sha = planner.physical.require, planner.physical.sha
base = planner.physical.source.base


def _action_sha(actions) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(sha(PLAN_PREP) == PLAN_PREP_SHA, "planner prep SHA")
    require(sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA, "planner report SHA")
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(
        protocol == planner.protocol()
        and report["status"] == "model_only_complete"
        and report["protocol_sha256"] == PLAN_PREP_SHA
        and report["post_cp5_physical_outcomes_used"] is False,
        "planner contract",
    )
    prefix = np.asarray(protocol["actual_prefix"], np.float32)
    seeds = {
        label: np.asarray(packet, np.float32)
        for label, packet in zip(protocol["seed_labels"], protocol["seed_packets"], strict=True)
    }
    tails = {
        "model_selected": np.asarray(report["selected"]["actions"], np.float32),
        "rgb_contact_seed": seeds["rgb_contact_tail"],
        "old_d2p_tail": seeds["old_d2p_tail"],
        "old_gn_tail": seeds["old_gn_tail"],
        "zero_tail": seeds["zero_tail"],
    }
    require(prefix.shape == (5, 2) and all(tail.shape == (5, 2) for tail in tails.values()), "plans")
    plans = {
        role: np.concatenate((prefix, tails[role]), axis=0).astype(np.float32)
        for role in ROLES
    }
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__), Path(planner.__file__), Path(planner.physical.__file__),
            PLAN_PREP, PLAN_RUN / "report.json",
            planner.physical.source.QUEUE,
            planner.physical.source.source_directory() / "report.json",
            planner.physical.source.source_directory() / "source_receipt.json",
            planner.physical.source.source_directory() / "reset_receipt.json",
            planner.physical.source.source_directory() / "goal.png",
        )
    }
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": planner.physical.source.SEED,
        "roles": [
            {"role": role, "actions": plans[role].tolist(), "actions_sha256": _action_sha(plans[role])}
            for role in ROLES
        ],
        "common_prefix_sha256": _action_sha(prefix),
        "model_selection": {
            key: report["selected"][key]
            for key in ("iteration", "candidate", "contact_projection", "nominal_q", "robust_score")
        },
        "same_actual_prefix_all_arms": True,
        "post_cp5_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared: Path, prepared_sha: str, output: Path):
    require(sha(prepared) == prepared_sha and json.loads(prepared.read_bytes()) == build(), "prepared")
    contract = json.loads(prepared.read_bytes())
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    history = base.read(
        planner.physical.source.source_directory() / "source_receipt.json"
    )["history"]
    reset = base.read(planner.physical.source.source_directory() / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    goal_rgb = base.image(
        planner.physical.source.source_directory() / history["goal_rgb"]["file"]
    )
    rows, errors = [], []
    started = time.monotonic()
    store.emit("run_started", prepare_sha256=prepared_sha)
    for spec in contract["roles"]:
        role = spec["role"]
        commands = np.asarray(spec["actions"], np.float32)
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
            require(sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "environment")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            obs, info = env.reset(seed=contract["seed"])
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
                if step == 5:
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
        "recorded_development_tail_comparison"
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
        "post_cp5_physical_outcomes_used_for_selection": False,
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
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive outputs")
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "report_sha256": sha(output / "report.json"),
        "rows": [
            {
                "role": row["role"], "status": row["status"],
                "native_success": row.get("final_metrics", {}).get("native_success"),
                "object_position_error": row.get("final_metrics", {}).get("object_position_error"),
                "contact_steps": [item["step"] for item in row["transitions"] if item["n_contacts"] > 0],
            }
            for row in report["rows"]
        ],
        "errors": report["errors"],
    }, sort_keys=True), flush=True)
    return int(report["status"] != "recorded_development_tail_comparison")


if __name__ == "__main__":
    raise SystemExit(main())
