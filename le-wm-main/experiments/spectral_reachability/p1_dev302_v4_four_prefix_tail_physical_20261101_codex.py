"""Physical execution of four actual-cp5, equal-budget replanned V4 tails."""
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

from . import p1_dev302_mixed_prefix_physical_v4_20261101_codex as prefix_source
from . import p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar as planner

NAME = "p1_dev302_v4_four_prefix_tail_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = (
    ROOT / "results/p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar_PREP_MAIN01.json"
)
PLAN_PREP_SHA = "0ce7dab889d0c773b4e4cb6162ada34aa8ff1dee658fecf8b2e29f45f4a742bd"
PLAN_RUN = (
    ROOT / "results/p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar_RUN_MAIN01"
)
PLAN_REPORT_SHA = "f79f57ff20bf36e53d89bbb2f6647eefcd3b3fe0f6b10031e5cb06e2dca61a3f"
ROLES = planner.ROLES
source = prefix_source.source
base = prefix_source.base
require, sha = prefix_source.require, prefix_source.sha


def _action_sha(actions) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "replan input SHA",
    )
    protocol = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(
        protocol
        == planner.protocol(
            Path(protocol["physical_directory"]),
            protocol["physical_report_sha256"],
        )
        and report["schema"] == planner.NAME + ".report.v1"
        and report["status"] == "model_only_complete_not_admitted"
        and report["protocol_sha256"] == PLAN_PREP_SHA
        and report["physical_execution"] is False
        and report["post_cp5_physical_outcomes_used"] is False
        and report["P1_admission"] is False,
        "sealed equal-budget replan",
    )
    source_rows = {row["role"]: row for row in protocol["roles"]}
    report_rows = {row["role"]: row for row in report["rows"]}
    require(
        set(source_rows) == set(report_rows) == set(ROLES)
        and len(report_rows) == len(ROLES),
        "four-role roster",
    )
    roles = []
    for role in ROLES:
        source_row, row = source_rows[role], report_rows[role]
        prefix = np.asarray(source_row["actual_prefix"], np.float32)
        tail = np.asarray(row["selected"]["actions"], np.float32).reshape(10, 2)
        actions = np.concatenate((prefix, tail), axis=0).astype(np.float32)
        require(
            prefix.shape == (5, 2)
            and tail.shape == (10, 2)
            and actions.shape == (15, 2)
            and row["actual_prefix_sha256"] == _action_sha(prefix)
            and row["cp5_pixel_sha256"] == source_row["cp5_pixel_sha256"],
            f"{role} plan identity",
        )
        roles.append(
            {
                "role": role,
                "candidate_index": row["candidate_index"],
                "candidate_label": row["candidate_label"],
                "actions": actions.tolist(),
                "actions_sha256": _action_sha(actions),
                "prefix_sha256": _action_sha(prefix),
                "expected_cp5_pixel_sha256": row["cp5_pixel_sha256"],
                "selected_tail": row["selected"],
            }
        )
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__),
            Path(planner.__file__),
            Path(prefix_source.__file__),
            PLAN_PREP,
            PLAN_RUN / "report.json",
            source.QUEUE,
            source.source_directory() / "source_receipt.json",
            source.source_directory() / "reset_receipt.json",
            source.source_directory() / "goal.png",
        )
    }
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": source.SEED,
        "roles": roles,
        "feedback_boundary": 5,
        "physical_horizon": 15,
        "same_replanning_budget": True,
        "same_feedback_count": True,
        "post_cp5_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared: Path, prepared_sha: str, output: Path):
    require(sha(prepared) == prepared_sha and json.loads(prepared.read_bytes()) == build(), "prepared")
    contract = json.loads(prepared.read_bytes())
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    directory = source.source_directory()
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
            "role": role,
            "candidate_index": spec["candidate_index"],
            "candidate_label": spec["candidate_label"],
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
            require(
                commands.shape == (15, 2)
                and _action_sha(commands) == spec["actions_sha256"],
                "dispatch actions",
            )
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
                receipt = base.capture(store, env, role, "plan", step)
                row["observations"].append(receipt)
                if step == 5:
                    require(
                        receipt["pixel_sha256"] == spec["expected_cp5_pixel_sha256"],
                        "actual cp5 replay identity",
                    )
                    row["prefix_metrics"] = deepcopy(transition["metrics"])
                if terminated or truncated:
                    row["status"] = (
                        "native_success"
                        if transition["metrics"]["native_success"]
                        else "native_stopped"
                    )
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
        "recorded_development_equal_budget_tails"
        if not errors
        and all(
            row["status"] in ("horizon_completed", "native_success", "native_stopped")
            for row in rows
        )
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
        "same_replanning_budget": True,
        "same_feedback_count": True,
        "post_cp5_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)), "exclusive")
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "report_sha256": sha(output / "report.json"),
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
        ),
        flush=True,
    )
    return int(report["status"] != "recorded_development_equal_budget_tails")


if __name__ == "__main__":
    raise SystemExit(main())
