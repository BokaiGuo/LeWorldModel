"""Opt-in physical runner for the sealed dev302 V3 four-prefix selection.

The default CLI only writes a preparation contract.  Physical execution
requires an explicit ``--run`` and performs no neural inference.
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

from . import p1_dev302_mixed_bank_cuda_v3_20261101_codex as planner


NAME = "p1_dev302_mixed_prefix_physical_v3_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v3_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "3ff07167154c3c5e687d83a5dc7736161b8699d28469f348b01072d884257a93"
PLAN_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v3_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "310e5539165502e2cd204be111e67c85a3645d0a11956f5217f571a00d9fce5c"
SEED = 301000002
ROLES = ("selected", "zero", "same_bank_greedy", "random")
EXPECTED_LABELS = {
    "zero": "zero",
    "same_bank_greedy": "rgb_goal_support_direct",
    "selected": "rgb_goal_support_top_route",
    "random": "random_control",
}
RGB_CALIBRATION = "historical RGB pusher displacement only; no simulator state"

source = planner.runner.cp5.physical.source
base = source.base
require, sha = source.require, source.sha


def _actions_sha(actions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def _planner_material() -> tuple[dict, dict]:
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "sealed V3 planner input SHA",
    )
    protocol = base.read(PLAN_PREP)
    report = base.read(PLAN_RUN / "report.json")
    require(protocol == planner.protocol(), "V3 protocol reconstruction")
    require(
        protocol.get("schema") == planner.NAME + ".protocol.v1"
        and report.get("schema") == planner.NAME + ".report.v1"
        and report.get("status") == "model_only_complete"
        and report.get("protocol_sha256") == PLAN_PREP_SHA
        and protocol.get("physical_outcomes_used") is False
        and report.get("physical_outcomes_used") is False,
        "V3 planner contract",
    )
    return protocol, report


def _validate_rgb_only(protocol: dict, report: dict) -> None:
    bank = protocol.get("prefix_bank")
    guard = bank.get("contact_guard") if isinstance(bank, dict) else None
    require(
        isinstance(bank, dict)
        and bank.get("schema") == "p1_dev302_mixed_prefix_bank_v3_20261101_codex.v1"
        and bank.get("candidate_order_fixed") is True
        and bank.get("simulator_state_used") is False
        and bank.get("physical_outcome_used") is False
        and isinstance(guard, dict)
        and guard.get("kind") == "rgb_calibrated_action_path_to_visible_object_boundary"
        and guard.get("calibration") == RGB_CALIBRATION,
        "RGB-only V3 prefix source",
    )
    require(
        protocol["algorithm"].get("contact_filter") == guard
        and report.get("physical_outcomes_used") is False,
        "RGB-only selection binding",
    )
    current = source.source_directory() / "rgb_10.png"
    goal = source.source_directory() / "goal.png"
    bindings = protocol.get("bindings", {})
    require(
        bindings.get(str(current.resolve())) == sha(current)
        and bindings.get(str(goal.resolve())) == sha(goal),
        "RGB source bindings",
    )


def _validated_roles(protocol: dict, report: dict) -> list[dict]:
    candidates = protocol["prefix_bank"].get("candidates")
    rows = report.get("rows")
    require(
        isinstance(candidates, list)
        and isinstance(rows, list)
        and len(candidates) == len(rows) == protocol["algorithm"]["prefix_candidates"],
        "complete V3 candidate roster",
    )
    protocol_rows = {row.get("index"): row for row in candidates}
    report_rows = {row.get("index"): row for row in rows}
    require(
        len(protocol_rows) == len(candidates)
        and len(report_rows) == len(rows)
        and set(protocol_rows) == set(report_rows) == set(range(len(rows))),
        "unique V3 candidate indices",
    )
    indices = {
        "selected": report.get("selected_index"),
        "zero": report.get("zero_index"),
        "same_bank_greedy": report.get("same_bank_greedy_index"),
        "random": report.get("random_index"),
    }
    require(
        all(type(index) is int and index in report_rows for index in indices.values())
        and len(set(indices.values())) == len(ROLES)
        and report.get("selected_distinct_from_greedy") is True,
        "four distinct V3 role indices",
    )
    roles = []
    for role in ROLES:
        plan_row = protocol_rows[indices[role]]
        row = report_rows[indices[role]]
        actions = np.asarray(row.get("actions"), np.float32)
        require(
            actions.shape == (5, 2)
            and np.isfinite(actions).all()
            and float(np.max(np.abs(actions))) <= 1.0
            and row.get("label") == plan_row.get("label") == EXPECTED_LABELS[role]
            and row.get("actions") == plan_row.get("actions")
            and row.get("actions_sha256") == plan_row.get("actions_sha256")
            and _actions_sha(actions) == row.get("actions_sha256"),
            f"{role} V3 action binding",
        )
        require(
            row.get("contact_feasible_prior") == plan_row.get("contact_feasible_prior")
            and row.get("constructed_from_rgb_contact_geometry")
            == plan_row.get("constructed_from_rgb_contact_geometry"),
            f"{role} V3 RGB metadata binding",
        )
        if role in ("selected", "same_bank_greedy"):
            require(
                row["contact_feasible_prior"] is True
                and row["constructed_from_rgb_contact_geometry"] is True,
                f"{role} must be an RGB-constructed contact-feasible prefix",
            )
        elif role == "zero":
            require(np.array_equal(actions, np.zeros((5, 2), np.float32)), "zero actions")
        roles.append({
            "role": role,
            "candidate_index": row["index"],
            "candidate_label": row["label"],
            "actions": actions.tolist(),
            "actions_sha256": row["actions_sha256"],
            "contact_feasible_prior": row["contact_feasible_prior"],
            "constructed_from_rgb_contact_geometry": row[
                "constructed_from_rgb_contact_geometry"
            ],
            "predicted_cp5_q": row["predicted_cp5_q"],
            "predicted_terminal_robust": row["tail_selected"]["robust_score"],
        })
    return roles


def build() -> dict:
    protocol, report = _planner_material()
    require(source.SEED == SEED, "dev302 reset seed")
    audit, receipt, _saved, _variants = source.material()
    require(
        audit.get("seed") == SEED
        and audit.get("P1_admission") is False
        and receipt["history"]["reset_receipt_sha256"]
        == sha(source.source_directory() / "reset_receipt.json"),
        "dev302 source authority",
    )
    _validate_rgb_only(protocol, report)
    roles = _validated_roles(protocol, report)
    paths = (
        Path(__file__).resolve(),
        Path(planner.__file__).resolve(),
        PLAN_PREP,
        PLAN_RUN / "report.json",
        source.QUEUE,
        source.source_directory() / "report.json",
        source.source_directory() / "source_receipt.json",
        source.source_directory() / "reset_receipt.json",
        source.source_directory() / "goal.png",
        *(source.source_directory() / f"rgb_{step:02d}.png" for step in (0, 5, 10)),
    )
    bindings = {str(path.resolve()): sha(path.resolve()) for path in paths}
    for path, digest in protocol["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        "schema": NAME + ".contract.v1",
        "seed": SEED,
        "planner": {
            "name": planner.NAME,
            "prep": str(PLAN_PREP.resolve()),
            "prep_sha256": PLAN_PREP_SHA,
            "report": str((PLAN_RUN / "report.json").resolve()),
            "report_sha256": PLAN_REPORT_SHA,
            "physical_outcomes_used": False,
        },
        "roles": roles,
        "feedback_boundary": 5,
        "same_physical_horizon": True,
        "equal_physical_action_budget": True,
        "prefix_source": {
            "mode": "rgb_only",
            "current_rgb_sha256": sha(source.source_directory() / "rgb_10.png"),
            "goal_rgb_sha256": sha(source.source_directory() / "goal.png"),
            "simulator_state_used_for_prefix_generation": False,
            "physical_outcomes_used_for_selection": False,
            "contact_calibration": RGB_CALIBRATION,
        },
        "post_prefix_outcomes_used_for_planning": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared: Path, prepared_sha: str, output: Path) -> dict:
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha and base.read(prepared) == build(), "prepared")
    contract = base.read(prepared)
    require(
        contract["seed"] == SEED
        and [row["role"] for row in contract["roles"]] == list(ROLES)
        and len({row["actions_sha256"] for row in contract["roles"]}) == len(ROLES),
        "physical dispatch contract",
    )
    store = base.Store(output)
    store.json("execution_binding.json", {
        "prepared": str(prepared),
        "prepare_sha256": prepared_sha,
        "planner_prep_sha256": PLAN_PREP_SHA,
        "planner_report_sha256": PLAN_REPORT_SHA,
    })
    directory = source.source_directory()
    history = base.read(directory / "source_receipt.json")["history"]
    reset = base.read(directory / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in history["frames"]}
    goal = np.asarray(history["goal"], np.float64)
    goal_rgb = base.image(directory / history["goal_rgb"]["file"])
    rows, errors = [], []
    started = time.monotonic()
    store.emit("run_started", prepare_sha256=prepared_sha, seed=SEED)
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
                commands.shape == (5, 2)
                and _actions_sha(commands) == spec["actions_sha256"],
                "dispatch action bytes",
            )
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "environment")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            store.emit("branch_started", arm=role, seed=SEED)
            obs, info = env.reset(seed=SEED)
            require(
                np.array_equal(
                    np.asarray(obs["state"], np.float64),
                    np.asarray(reset["state"], np.float64),
                )
                and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                "reset",
            )
            row["observations"].append(
                base.capture(store, env, role, "warmup", 0, checkpoints[0])
            )
            for step in range(1, 11):
                action = np.zeros(2, np.float32)
                store.emit(
                    "action_dispatch",
                    arm=role,
                    phase="warmup",
                    step=step,
                    action=action.tolist(),
                )
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(
                    not terminated
                    and not truncated
                    and np.array_equal(np.asarray(env.latest_action), action),
                    "warmup",
                )
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
                        base.capture(store, env, role, "warmup", step, checkpoints[step])
                    )
            require(
                np.array_equal(
                    np.asarray(obs["state"], np.float64),
                    np.asarray(history["final_state"], np.float64),
                ),
                "root",
            )
            row["initial_metrics"] = base.metrics(env, goal, obs["state"])
            for step, action in enumerate(commands, 1):
                require(
                    _actions_sha(commands) == spec["actions_sha256"],
                    "actions changed before dispatch",
                )
                store.emit(
                    "action_dispatch",
                    arm=role,
                    phase="prefix",
                    step=step,
                    action=action.tolist(),
                )
                obs, reward, terminated, truncated, info = env.step(action.copy())
                state = np.asarray(obs["state"], np.float64)
                require(
                    np.array_equal(np.asarray(env.latest_action), action)
                    and np.array_equal(np.asarray(info["goal_state"]), goal)
                    and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                    "prefix",
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
                store.emit("action_return", arm=role, phase="prefix", **transition)
                row["observations"].append(
                    base.capture(store, env, role, "prefix", step)
                )
                if terminated or truncated:
                    row["status"] = (
                        "native_success"
                        if transition["metrics"]["native_success"]
                        else "native_stopped"
                    )
                    break
            else:
                row["status"] = "prefix_completed"
            require(row["transitions"], "at least one returned prefix action")
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
        require(
            all(sha(Path(path)) == digest for path, digest in contract["bindings"].items()),
            "closure",
        )
    except BaseException as exc:
        errors.append({
            "stage": "closure",
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        })
    status = (
        "recorded_development_v3_prefixes"
        if not errors
        and all(
            row["status"] in ("prefix_completed", "native_success", "native_stopped")
            for row in rows
        )
        else "partial_or_failed"
    )
    store.emit("run_finished", status=status, errors=errors)
    report = {
        "schema": NAME + ".report.v1",
        "status": status,
        "prepare_sha256": prepared_sha,
        "planner_prep_sha256": PLAN_PREP_SHA,
        "planner_report_sha256": PLAN_REPORT_SHA,
        "seed": SEED,
        "rows": rows,
        "errors": errors,
        "journal_sha256": sha(store.path / "journal.jsonl"),
        "journal_tail": store.events[-1]["sha256"],
        "wall_seconds": time.monotonic() - started,
        "prefix_source": contract["prefix_source"],
        "post_prefix_outcomes_used_for_planning": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    store.json("report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(
        not os.path.lexists(prepared)
        and (not args.run or not os.path.lexists(output)),
        "exclusive outputs",
    )
    digest = planner.runner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "report_sha256": sha(output / "report.json"),
        "rows": [
            {
                "role": row["role"],
                "label": row["candidate_label"],
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
    }, sort_keys=True), flush=True)
    return int(report["status"] != "recorded_development_v3_prefixes")


if __name__ == "__main__":
    raise SystemExit(main())
