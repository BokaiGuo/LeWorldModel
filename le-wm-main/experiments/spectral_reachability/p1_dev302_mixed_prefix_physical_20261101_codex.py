"""Physical cp5 execution of selected/zero/greedy/random mixed-bank prefixes."""
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

from . import p1_dev302_mixed_bank_cuda_v2_20261101_codex as planner

NAME = "p1_dev302_mixed_prefix_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v2_20261101_codex_PREP_MAIN02.json"
PLAN_PREP_SHA = "59d0d745d7f4b91fd10a6ae532e0f9d1a1c68eeb9d26db058170b2775faffd96"
PLAN_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v2_20261101_codex_RUN_MAIN02"
PLAN_REPORT_SHA = "50562dfe86dc3f7a8a47107e99eaab71492961dc5774557b86d853253cb1fa64"
ROLES = ("selected", "zero", "same_bank_greedy", "random")
require, sha = planner.v1.cp5.physical.require, planner.v1.cp5.physical.sha
base = planner.v1.cp5.physical.source.base


def build():
    require(sha(PLAN_PREP) == PLAN_PREP_SHA and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
            "planner inputs")
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(
        protocol == planner.protocol()
        and report["status"] == "model_only_complete"
        and report["protocol_sha256"] == PLAN_PREP_SHA
        and report["physical_outcomes_used"] is False,
        "planner contract",
    )
    rows = {row["index"]: row for row in report["rows"]}
    indices = {
        "selected": report["selected_index"],
        "zero": report["zero_index"],
        "same_bank_greedy": report["same_bank_greedy_index"],
        "random": report["random_index"],
    }
    roles = []
    for role in ROLES:
        row = rows[indices[role]]
        actions = np.asarray(row["actions"], np.float32)
        roles.append({
            "role": role,
            "candidate_index": row["index"],
            "candidate_label": row["label"],
            "actions": actions.tolist(),
            "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
            "predicted_cp5_q": row["predicted_cp5_q"],
            "predicted_terminal_robust": row["tail_selected"]["robust_score"],
        })
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__), Path(planner.__file__), PLAN_PREP, PLAN_RUN / "report.json",
            planner.v1.cp5.physical.source.QUEUE,
            planner.v1.cp5.physical.source.source_directory() / "source_receipt.json",
            planner.v1.cp5.physical.source.source_directory() / "reset_receipt.json",
            planner.v1.cp5.physical.source.source_directory() / "goal.png",
        )
    }
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": planner.v1.cp5.physical.source.SEED,
        "roles": roles,
        "feedback_boundary": 5,
        "same_physical_horizon": True,
        "post_prefix_outcomes_used_for_planning": False,
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
    directory = planner.v1.cp5.physical.source.source_directory()
    history = base.read(directory / "source_receipt.json")["history"]
    reset = base.read(directory / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    goal_rgb = base.image(directory / history["goal_rgb"]["file"])
    rows, errors = [], []
    started = time.monotonic()
    store.emit("run_started", prepare_sha256=prepared_sha)
    for spec in contract["roles"]:
        role = spec["role"]
        commands = np.asarray(spec["actions"], np.float32)
        row = {
            "role": role, "candidate_index": spec["candidate_index"],
            "candidate_label": spec["candidate_label"], "status": "running",
            "warmup_returned": 0, "plan_returned": 0, "transitions": [],
            "observations": [], "error": None,
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
                    "action_return", arm=role, phase="warmup", step=step, action=action.tolist(),
                    state=np.asarray(obs["state"]).tolist(), reward=float(reward),
                    n_contacts=int(info["n_contacts"]), terminated=False, truncated=False,
                )
                if step in checkpoints:
                    row["observations"].append(base.capture(store, env, role, "warmup", step, checkpoints[step]))
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(history["final_state"], np.float64)),
                "root",
            )
            row["initial_metrics"] = base.metrics(env, goal, obs["state"])
            for step, action in enumerate(commands, 1):
                store.emit("action_dispatch", arm=role, phase="prefix", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                state = np.asarray(obs["state"], np.float64)
                require(
                    np.array_equal(np.asarray(env.latest_action), action)
                    and np.array_equal(np.asarray(info["goal_state"]), goal)
                    and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                    "prefix",
                )
                transition = {
                    "step": step, "action": action.tolist(), "state": state.tolist(),
                    "reward": float(reward), "n_contacts": int(info["n_contacts"]),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "metrics": base.metrics(env, goal, state),
                }
                row["plan_returned"] += 1
                row["transitions"].append(transition)
                store.emit("action_return", arm=role, phase="prefix", **transition)
                row["observations"].append(base.capture(store, env, role, "prefix", step))
                if terminated or truncated:
                    row["status"] = "native_success" if transition["metrics"]["native_success"] else "native_stopped"
                    break
            else:
                row["status"] = "prefix_completed"
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
        "recorded_development_prefixes"
        if not errors and all(row["status"] in ("prefix_completed", "native_success", "native_stopped") for row in rows)
        else "partial_or_failed"
    )
    store.emit("run_finished", status=status, errors=errors)
    report = {
        "schema": NAME + ".report.v1", "status": status, "prepare_sha256": prepared_sha,
        "rows": rows, "errors": errors, "journal_sha256": sha(store.path / "journal.jsonl"),
        "journal_tail": store.events[-1]["sha256"], "wall_seconds": time.monotonic() - started,
        "post_prefix_outcomes_used_for_planning": False, "new_neural_calls_during_physical": 0,
        "development_only": True, "fair_total_cost": False,
        "independent_confirmation": False, "P1_admission": False,
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
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive")
    digest = planner.v1.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({
        "status": report["status"], "output": str(output),
        "report_sha256": sha(output / "report.json"),
        "rows": [
            {
                "role": row["role"], "label": row["candidate_label"],
                "status": row["status"], "native_success": row.get("final_metrics", {}).get("native_success"),
                "object_position_error": row.get("final_metrics", {}).get("object_position_error"),
                "contact_steps": [item["step"] for item in row["transitions"] if item["n_contacts"] > 0],
            }
            for row in report["rows"]
        ], "errors": report["errors"],
    }, sort_keys=True), flush=True)
    return int(report["status"] != "recorded_development_prefixes")


if __name__ == "__main__":
    raise SystemExit(main())
