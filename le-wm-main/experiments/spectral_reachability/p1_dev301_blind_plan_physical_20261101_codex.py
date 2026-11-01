"""Development-only native PushT execution of the sealed dev301 blind plans.

The two model-selected plans are consumed from the strict saved planner audit.
Each arm receives a fresh native environment, the exact sealed reset seed,
ten actual zero warmup actions, and the same native reset goal.  A zero-action
arm is included as a physical reference.  No physical outcome is used to
select, alter, retry, or pad a plan.
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
import shutil
import time

import numpy as np
from PIL import Image

from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_constructive_blind_planners_saved_audit_20261027_v1 as planner_saved
from . import p1_dev_reset_bound_20261101_codex as native
from . import p1_queue_goal_saved_20261101_codex as source_saved

NAME = "p1_dev301_blind_plan_physical_20261101_codex"
ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
QUEUE = ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json"
QUEUE_SHA = "6756ad9f3ccf058baa422520d3058dbbdc10e7ac86ff1091b53b719f5a1fe847"
SOURCE_REPORT_SHA = "05ce658129367dcad6d8bdf9a05e963758d1eec144ad760cbd05786dd7e7a354"
SOURCE_SLOT = "dev_reused"
SOURCE_ATTEMPT = 0
SEED = 301000001
PLANNER_DIR = ROOT / "results/p1_dev301_blind_planners_20261101_codex_RUN_MAIN02"
PLANNER_REPORT_SHA = "0c9430646c236efc699f45adee2422d11eaac137b90db5a44c12f752b93be2b1"
PLANNER_PREP_SHA = "c8019608a1cd85fd58800c0367a668176616aabb1fda90852973f778e4e56b1d"
PLANNER_SAVED = ROOT / "results/p1_dev301_blind_planners_saved_20261101_codex_MAIN01.json"
PLANNER_SAVED_SHA = "c2eddba78b0b922cf6fb046e868d2e7241eb85ce16221471b225781e9abdc970"
ARMS = ("line_search_GN", "full10_conditional_D2P", "zero")
CAP = 32 * 1024**2
FLOOR = 200 * 1024**2
require = native.require


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular input: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def image(path) -> np.ndarray:
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("RGB"), np.uint8).copy()
    require(value.shape == (224, 224, 3), "RGB224")
    return value


def pixel_sha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def action_array(value) -> np.ndarray:
    result = np.asarray(value)
    require(
        result.shape == (10, 2)
        and result.dtype.kind in "fi"
        and np.isfinite(result).all()
        and (np.abs(result) <= 1).all()
        and np.array_equal(result, result.astype(np.float32)),
        "exact bounded float32 10x2 actions",
    )
    return result.astype(np.float32)


def source_directory() -> Path:
    return (
        QUEUE.parent
        / (QUEUE.name + ".native_reset")
        / f"{SOURCE_SLOT}_{SOURCE_ATTEMPT:04d}.warmup"
    )


def source_material():
    audit = source_saved.verify(
        QUEUE, QUEUE_SHA, SOURCE_SLOT, SOURCE_ATTEMPT, SOURCE_REPORT_SHA
    )
    require(audit["seed"] == SEED and audit["P1_admission"] is False, "dev301 source")
    directory = source_directory()
    receipt_path = directory / "source_receipt.json"
    receipt = read(receipt_path)
    history = receipt["history"]
    require(
        receipt["steps_returned"] == 10
        and history["actions"] == [[0.0, 0.0]] * 10
        and [row["step"] for row in history["frames"]] == [0, 5, 10],
        "sealed zero warmup",
    )
    frames = []
    for row in history["frames"]:
        path = (directory / row["file"]).resolve(strict=True)
        require(
            path.parent == directory.resolve()
            and sha(path) == row["file_sha256"],
            "sealed source frame",
        )
        pixels = image(path)
        require(pixel_sha(pixels) == row["pixel_sha256"], "source frame pixels")
        frames.append(pixels)
    goal_path = (directory / history["goal_rgb"]["file"]).resolve(strict=True)
    goal_rgb = image(goal_path)
    require(
        sha(goal_path) == history["goal_rgb"]["file_sha256"]
        and pixel_sha(goal_rgb) == history["goal_rgb"]["pixel_sha256"],
        "source goal RGB",
    )
    return audit, receipt, frames, goal_rgb


def planner_material():
    require(
        sha(PLANNER_DIR / "report.json") == PLANNER_REPORT_SHA
        and sha(PLANNER_SAVED) == PLANNER_SAVED_SHA,
        "external planner receipts",
    )
    report = read(PLANNER_DIR / "report.json")
    saved = read(PLANNER_SAVED)
    rebuilt = planner_saved.audit(
        PLANNER_DIR / "report.json", PLANNER_REPORT_SHA, PLANNER_PREP_SHA
    )
    expected = deepcopy(rebuilt)
    expected["auditor_sha256"] = sha(Path(planner_saved.__file__))
    require(
        saved == expected
        and saved["status"] == "saved_model_plans_consistent_not_admitted"
        and saved["P1_admission"] is False,
        "strict saved planner audit",
    )
    plans = {}
    rows = {row["arm"]: row for row in saved["rows"]}
    for arm in ARMS[:2]:
        path = PLANNER_DIR / f"root_0_{arm}.plan.json"
        require(
            report["artifact_sha256"][path.name] == sha(path)
            and rows[arm]["selection_sha256"] == read(path)["selection_sha256"],
            "plan artifact/audit binding",
        )
        plan = read(path)
        require(
            plan["arm"] == arm
            and plan["order"] == 0
            and plan["task_id"] == rows[arm]["task_id"],
            "plan identity",
        )
        plans[arm] = action_array(plan["commands"]).tolist()
    plans["zero"] = np.zeros((10, 2), np.float32).tolist()
    return report, saved, plans


def bindings():
    paths = [
        Path(__file__).resolve(),
        Path(native.__file__).resolve(),
        Path(source_saved.__file__).resolve(),
        Path(planner_saved.__file__).resolve(),
        Path(locking.__file__).resolve(),
        QUEUE,
        source_directory() / "report.json",
        source_directory() / "source_receipt.json",
        source_directory() / "reset_receipt.json",
        source_directory() / "journal.jsonl",
        source_directory() / "goal.png",
        *(source_directory() / f"rgb_{step:02d}.png" for step in (0, 5, 10)),
        PLANNER_DIR / "report.json",
        PLANNER_DIR / "events.jsonl.gz",
        PLANNER_DIR / "binding.json",
        PLANNER_SAVED,
        *(PLANNER_DIR / f"root_0_{arm}.plan.json" for arm in ARMS[:2]),
    ]
    return {str(path): sha(path) for path in paths}


def build():
    audit, receipt, frames, goal_rgb = source_material()
    planner_report, saved, plans = planner_material()
    source = receipt["history"]
    result = {
        "schema": NAME + ".contract.v1",
        "seed": SEED,
        "source": {
            "queue": str(QUEUE),
            "queue_sha256": QUEUE_SHA,
            "source_report_sha256": SOURCE_REPORT_SHA,
            "slot": SOURCE_SLOT,
            "attempt": SOURCE_ATTEMPT,
            "history_sha256": receipt["history_sha256"],
            "initial_state": source["frames"][0]["state"],
            "root_state": source["final_state"],
            "goal": source["goal"],
            "goal_pixel_sha256": pixel_sha(goal_rgb),
            "warmup_pixel_sha256": [pixel_sha(frame) for frame in frames],
            "training_membership": audit["training_membership"],
            "historical_coverage": audit["historical_coverage"],
        },
        "planner": {
            "directory": str(PLANNER_DIR),
            "report_sha256": PLANNER_REPORT_SHA,
            "saved_audit": str(PLANNER_SAVED),
            "saved_audit_sha256": PLANNER_SAVED_SHA,
            "task_id": saved["rows"][0]["task_id"],
            "model_terminal_cost": {
                row["arm"]: row["terminal_cost"] for row in saved["rows"]
            },
            "equal_budget_claim": planner_report["equal_budget_claim"],
        },
        "arms": [
            {
                "arm": arm,
                "actions": plans[arm],
                "selection_sha256": (
                    next(row["selection_sha256"] for row in saved["rows"] if row["arm"] == arm)
                    if arm != "zero"
                    else hashlib.sha256(np.zeros((10, 2), "<f4").tobytes()).hexdigest()
                ),
            }
            for arm in ARMS
        ],
        "execution": {
            "fresh_native_instance_per_arm": True,
            "reset_seed": SEED,
            "warmup_actions": 10,
            "plan_horizon": 10,
            "early_stop_no_padding": True,
            "physical_outcomes_used_for_selection": False,
            "new_neural_calls": 0,
            "wall_seconds": 600,
        },
        "development_only": True,
        "equal_budget_claim": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings(),
    }
    require(
        result["planner"]["equal_budget_claim"] is False
        and [row["arm"] for row in result["arms"]] == list(ARMS),
        "fixed three-arm contract",
    )
    return result


def write_new(path, value):
    path = Path(path).absolute()
    raw = canonical(value) + b"\n"
    require(
        not os.path.lexists(path)
        and len(raw) < 512 * 1024
        and shutil.disk_usage(path.parent).free >= FLOOR + len(raw),
        "exclusive bounded output",
    )
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


class Store:
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(
            not os.path.lexists(self.path)
            and shutil.disk_usage(self.path.parent).free >= FLOOR + CAP,
            "exclusive output and disk floor",
        )
        self.path.mkdir()
        self.events = []

    def emit(self, kind, **fields):
        body = {
            "index": len(self.events),
            "previous": self.events[-1]["sha256"] if self.events else None,
            "kind": kind,
            **fields,
        }
        row = {**body, "sha256": hashlib.sha256(canonical(body)).hexdigest()}
        with (self.path / "journal.jsonl").open("ab") as stream:
            line = canonical(row) + b"\n"
            require(stream.write(line) == len(line), "journal write")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(row)

    def frame(self, name, pixels):
        require(Path(name).name == name, "flat frame name")
        path = self.path / name
        with path.open("xb") as stream:
            Image.fromarray(pixels).save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "file": name,
            "file_sha256": sha(path),
            "pixel_sha256": pixel_sha(pixels),
        }

    def json(self, name, value):
        path = self.path / name
        raw = canonical(value) + b"\n"
        with path.open("xb") as stream:
            require(stream.write(raw) == len(raw), "JSON write")
            stream.flush()
            os.fsync(stream.fileno())
        return sha(path)

    def check(self):
        used = sum(path.stat().st_size for path in self.path.iterdir() if path.is_file())
        require(used < CAP and shutil.disk_usage(self.path).free >= FLOOR, "output/disk cap")


def metrics(env, goal, state):
    state = np.asarray(state, np.float64)
    goal = np.asarray(goal, np.float64)
    success, distance = env.eval_state(goal, state)
    angle = float(abs(np.arctan2(np.sin(state[4] - goal[4]), np.cos(state[4] - goal[4]))))
    return {
        "native_success": bool(success),
        "native_distance": float(distance),
        "pusher_position_error": float(np.linalg.norm(state[:2] - goal[:2])),
        "object_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
        "wrapped_object_angle_error": angle,
    }


def capture(store, env, arm, phase, step, expected=None):
    before = native.snapshot(env)
    pixels = np.asarray(env.render())
    after = native.snapshot(env)
    require(
        before == after and pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8,
        "pure native render",
    )
    frame = store.frame(f"{arm}_{phase}_{step:02d}.png", pixels)
    row = {**frame, "phase": phase, "step": step, "state": after["state"], "snapshot": after}
    if expected is not None:
        require(
            row["state"] == expected["state"]
            and row["snapshot"] == expected["snapshot"]
            and row["pixel_sha256"] == expected["pixel_sha256"],
            "exact sealed source checkpoint",
        )
    store.emit("observation", arm=arm, **row)
    return row


def execute(prepared, prepared_sha, output, factory=None):
    prepared = Path(prepared).resolve(strict=True)
    require(sha(prepared) == prepared_sha and read(prepared) == build(), "prepared contract")
    contract = read(prepared)
    store = Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    source = read(source_directory() / "source_receipt.json")["history"]
    reset = read(source_directory() / "reset_receipt.json")
    checkpoints = {row["step"]: row for row in source["frames"]}
    goal = np.asarray(source["goal"], np.float64)
    goal_rgb = image(source_directory() / source["goal_rgb"]["file"])
    started = time.monotonic()
    rows, errors = [], []
    store.emit("run_started", prepare_sha256=prepared_sha)
    for spec in contract["arms"]:
        arm = spec["arm"]
        commands = action_array(spec["actions"])
        row = {
            "arm": arm,
            "status": "running",
            "selection_sha256": spec["selection_sha256"],
            "warmup_returned": 0,
            "plan_returned": 0,
            "transitions": [],
            "observations": [],
            "error": None,
        }
        rows.append(row)
        env = None
        try:
            require(time.monotonic() - started < contract["execution"]["wall_seconds"], "wall cap")
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == native.ENV_SHA, "environment source")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False) if factory is None else factory()
            store.emit("branch_started", arm=arm, seed=SEED, selection_sha256=spec["selection_sha256"])
            obs, info = env.reset(seed=SEED)
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
                and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                and np.array_equal(np.asarray(info["goal"]), goal_rgb),
                "exact reset state/goal",
            )
            row["observations"].append(capture(store, env, arm, "warmup", 0, checkpoints[0]))
            for step in range(1, 11):
                action = np.zeros(2, np.float32)
                store.emit("action_dispatch", arm=arm, phase="warmup", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                actual = np.asarray(env.latest_action)
                require(
                    np.array_equal(actual, action)
                    and np.array_equal(np.asarray(info["goal_state"]), goal)
                    and np.array_equal(np.asarray(info["goal"]), goal_rgb)
                    and not terminated
                    and not truncated,
                    "exact nonterminal warmup return",
                )
                row["warmup_returned"] += 1
                store.emit(
                    "action_return",
                    arm=arm,
                    phase="warmup",
                    step=step,
                    action=actual.tolist(),
                    state=np.asarray(obs["state"]).tolist(),
                    reward=float(reward),
                    n_contacts=int(info["n_contacts"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if step in checkpoints:
                    row["observations"].append(capture(store, env, arm, "warmup", step, checkpoints[step]))
            require(
                np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(source["final_state"], np.float64)),
                "exact warmup root",
            )
            row["initial_metrics"] = metrics(env, goal, obs["state"])
            for step, action in enumerate(commands, 1):
                store.check()
                store.emit("action_dispatch", arm=arm, phase="plan", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                actual = np.asarray(env.latest_action)
                state = np.asarray(obs["state"], np.float64)
                require(
                    np.array_equal(actual, action)
                    and np.array_equal(np.asarray(info["goal_state"]), goal)
                    and np.array_equal(np.asarray(info["goal"]), goal_rgb)
                    and type(info["n_contacts"]) is int
                    and info["n_contacts"] >= 0,
                    "exact action/goal/contact return",
                )
                result = {
                    "step": step,
                    "action": actual.tolist(),
                    "state": state.tolist(),
                    "reward": float(reward),
                    "n_contacts": info["n_contacts"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "metrics": metrics(env, goal, state),
                }
                row["plan_returned"] += 1
                row["transitions"].append(result)
                store.emit("action_return", arm=arm, phase="plan", **result)
                row["observations"].append(capture(store, env, arm, "plan", step))
                if terminated or truncated:
                    row["status"] = "native_success" if terminated and result["metrics"]["native_success"] else "native_stopped"
                    break
            else:
                row["status"] = "horizon_completed"
            row["final_metrics"] = deepcopy(row["transitions"][-1]["metrics"])
            row["full_horizon"] = row["plan_returned"] == 10
            row["planned_actions_sha256"] = hashlib.sha256(commands.astype("<f4").tobytes()).hexdigest()
        except BaseException as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": arm, **row["error"]})
        finally:
            if env is not None:
                try:
                    env.close()
                except BaseException as exc:
                    errors.append({"arm": arm, "stage": "close", "type": type(exc).__name__, "message": str(exc)[:1000]})
            store.json(f"{arm}.row.json", row)
            store.emit("branch_closed", arm=arm, status=row["status"])
    try:
        require(all(sha(path) == digest for path, digest in contract["bindings"].items()), "post-run source closure")
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
        "new_neural_calls": 0,
        "development_only": True,
        "equal_budget_claim": False,
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
        digest = write_new(args.output, build())
        print(json.dumps({"prepared": str(Path(args.output).absolute()), "sha256": digest}, sort_keys=True))
        return
    with locking.gpu_lock(LOCK):
        result = execute(args.prepared, args.prepare_sha256, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "report_sha256": sha(Path(args.output) / "report.json"),
                "rows": [
                    {
                        "arm": row["arm"],
                        "status": row["status"],
                        "returned": row["plan_returned"],
                        "native_success": row.get("final_metrics", {}).get("native_success"),
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
