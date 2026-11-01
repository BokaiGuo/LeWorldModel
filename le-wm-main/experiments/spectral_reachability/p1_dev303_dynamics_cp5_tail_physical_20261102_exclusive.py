"""Development PushT execution of the sealed eight-way cp5 V13 tails."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive as planner
from . import p1_dev303_rgb_boundary_static_20261102_exclusive as blind

ROOT = planner.ROOT
NAME = "p1_dev303_dynamics_cp5_tail_physical_20261102_exclusive"
MODEL_REPORT_SHA = "af8a7668b493797a0e2b78dd16d37706ad42238baac3269835f775902e30e0f1"
PHYSICAL_REPORT_SHA = planner.PHYS_SHA
SEED = 301000003
AUTHORITY = "DEV303_CP5_DEVELOPMENT_SIM_PHYSICS"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return base.sha(path)


def read(path, expected=None):
    raw = base.read(path)
    if expected is not None:
        require(sha(path) == expected, f"SHA: {path}")
    return raw


def source_data():
    model_report_path = ROOT / "results/p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive_RUN_DEV01_HOST01/report.json"
    physical_path = ROOT / "results/p1_dev303_dynamics_physical_20261102_main_RUN01/report.json"
    model = read(model_report_path, MODEL_REPORT_SHA)
    physical = read(physical_path, PHYSICAL_REPORT_SHA)
    require(model["schema"] == planner.NAME + ".report.v1"
            and model["status"] == "model_only_complete"
            and model["protocol_sha256"] == "22d9b9091bbc216e34bc2516f7daf805c4ffd32d497f27d3cef8794661ec65c9"
            and model["P1_admission"] is False, "sealed model report")
    require(physical["schema"] == "dev303_dynamics_physical.report.v1"
            and physical["model_report_sha256"] == planner.MODEL_SHA
            and physical["P1_admission"] is False, "sealed prefix report")
    roles = []
    expected = {(index, metric) for index, _ in planner.ROLES for metric in ("latent", "goal_tangent")}
    require(len(model["rows"]) == len(expected)
            and {(r["index"], r["metric"]) for r in model["rows"]} == expected,
            "complete eight-way model panel")
    labels = dict(planner.ROLES)
    for m in model["rows"]:
        index, label = m["index"], labels[m["index"]]
        p = [r for r in physical["rows"] if r["index"] == index and r["metric"] == "goal_tangent"]
        require(len(p) == 1 and m["label"] == p[0]["label"] == label, "fixed role")
        p = p[0]
        prefix = [t for t in p["transitions"] if t["phase"] == "prefix"]
        require([t["step"] for t in prefix] == list(range(1, 6)), "prefix order")
        actual = np.asarray([t["action"] for t in prefix], dtype="<f4")
        require(actual.shape == (5, 2) and np.isfinite(actual).all()
                and (np.abs(actual) <= 1).all()
                and hashlib.sha256(actual.tobytes()).hexdigest() == m["actual_prefix_sha256"],
                "prefix shape and actual-action binding")
        checkpoints = [o for o in p["observations"] if o["phase"] == "prefix"]
        warmup = [o for o in p["observations"] if o["phase"] == "warmup"]
        require([o["step"] for o in checkpoints] == list(range(1, 6))
                and [o["step"] for o in warmup] == [0, 5, 10], "checkpoint roster")
        tail = np.asarray(m["selected"]["actions"], dtype="<f4").reshape(10, 2)
        require(np.isfinite(tail).all() and (np.abs(tail) <= 1).all()
                and hashlib.sha256(tail.tobytes()).hexdigest() == m["tail_actions_sha256"],
                "bounded SHA-bound tail")
        require(checkpoints[-1]["pixel_sha256"] == m["cp5_pixel_sha256"],
                "planner cp5 observation binding")
        roles.append({"arm": f"candidate_{index:02d}_{m['metric']}",
                      "index": index, "label": label, "metric": m["metric"],
                      "prefix": actual.tolist(), "tail": tail.tolist(),
                      "tail_sha256": hashlib.sha256(tail.tobytes()).hexdigest(),
                      "prefix_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                      "cp5": {k: checkpoints[-1][k] for k in
                              ("file", "file_sha256", "pixel_sha256")},
                      "prefix_checkpoints": checkpoints, "warmup_checkpoints": warmup,
                      "reset_checkpoint": warmup[0]})
    with np.load(blind.NPZ, allow_pickle=False) as archive:
        require(set(archive.files) == {"rgb_history", "raw_action_history", "goal_rgb"}
                and archive["goal_rgb"].dtype == np.uint8, "goal RGB source")
        goal_rgb = archive["goal_rgb"].copy()
    return model, physical, roles, goal_rgb


def build():
    model, physical, roles, goal_rgb = source_data()
    paths = [Path(__file__), Path(planner.__file__), Path(base.__file__), Path(base.native.__file__),
             Path(blind.__file__), ROOT / "results/p1_dev303_dynamics_cp5_v13_cem_20261102_exclusive_RUN_DEV01_HOST01/report.json",
             ROOT / "results/p1_dev303_dynamics_physical_20261102_main_RUN01/report.json", blind.NPZ]
    return {"schema": NAME + ".contract.v1", "classification": "dev303_development_simulation_only",
            "seed": SEED, "model_report_sha256": MODEL_REPORT_SHA,
            "prefix_report_sha256": PHYSICAL_REPORT_SHA, "roles": roles,
            "goal_state": physical["goal_state_evaluation_only"],
            "goal_rgb_sha256": hashlib.sha256(goal_rgb.tobytes()).hexdigest(),
            "warmup": "10 zero actions with checkpoints 0,5,10",
            "prefix": "five actual actions, exact replay checkpoints at every step",
            "tail": "ten sealed V13 actions, no replanning/replacement/padding",
            "selection_uses_reward_state_or_future_outcome": False,
            "physical_scope": "development PushT simulation only; not a real robot",
            "physical_actions_authorized_by_contract": False,
            "P1_admission": False, "independent_confirmation": False,
            "bindings": {str(p.resolve()): sha(p) for p in paths}}


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def execute(contract_path, contract_sha, output, authority):
    require(authority == AUTHORITY, "explicit development-simulation authority")
    require(sha(contract_path) == contract_sha and base.read(contract_path) == build(),
            "contract/source closure")
    contract = base.read(contract_path)
    _, physical, roles, goal_rgb = source_data()
    reset_state = roles[0]["reset_checkpoint"]["state"]
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(Path(contract_path).resolve()),
                                          "prepare_sha256": contract_sha})
    store.emit("run_started", prepare_sha256=contract_sha)
    rows, errors = [], []
    for role in roles:
        row = {k: role[k] for k in ("arm", "index", "label", "metric", "prefix", "tail",
                                    "tail_sha256", "prefix_sha256", "cp5")}
        row.update(status="running", transitions=[], observations=[], error=None)
        env = None
        try:
            store.emit("branch_started", arm=role["arm"])
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native env SHA")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            obs, info = env.reset(seed=SEED)
            require(np.array_equal(np.asarray(obs["state"]), np.asarray(reset_state))
                    and np.array_equal(np.asarray(info["goal_state"]), np.asarray(contract["goal_state"]))
                    and hashlib.sha256(np.asarray(info["goal"]).tobytes()).hexdigest()
                    == contract["goal_rgb_sha256"], "reset/goal binding")
            row["observations"].append(base.capture(store, env, role["arm"], "warmup", 0,
                                                     role["warmup_checkpoints"][0]))
            actions = [np.zeros(2, dtype="<f4")] * 10
            for step, action in enumerate(actions, 1):
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(not terminated and not truncated and np.array_equal(env.latest_action, action),
                        "warmup return")
                row["transitions"].append({"phase": "warmup", "step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                if step in (5, 10):
                    expected = role["warmup_checkpoints"][[5, 10].index(step) + 1]
                    row["observations"].append(base.capture(store, env, role["arm"], "warmup", step, expected))
            for step, action in enumerate(np.asarray(role["prefix"], dtype="<f4"), 1):
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(not terminated and not truncated and np.array_equal(env.latest_action, action),
                        "prefix return")
                row["transitions"].append({"phase": "prefix", "step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.capture(store, env, role["arm"], "prefix", step,
                                                         role["prefix_checkpoints"][step - 1]))
            require(row["observations"][-1]["pixel_sha256"] == role["cp5"]["pixel_sha256"],
                    "cp5 RGB replay mismatch")
            store.emit("cp5_gate_passed", arm=role["arm"], pixel_sha256=role["cp5"]["pixel_sha256"])
            for step, action in enumerate(np.asarray(role["tail"], dtype="<f4"), 1):
                obs, reward, terminated, truncated, info = env.step(action.copy())
                require(np.array_equal(env.latest_action, action), "tail return")
                row["transitions"].append({"phase": "tail", "step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.capture(store, env, role["arm"], "tail", step))
                if terminated or truncated:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": role["arm"], **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(role["arm"] + ".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "status": "recorded" if not errors else "partial_or_failed",
              "prepare_sha256": contract_sha, "rows": rows, "errors": errors,
              "journal_sha256": sha(store.path / "journal.jsonl"), "P1_admission": False,
              "independent_confirmation": False, "classification": "development_simulation_only"}
    store.json("report.json", report)
    return sha(store.path / "report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not os.path.lexists(prepared), "exclusive contract")
        print(json.dumps({"contract": str(prepared), "sha256": write_new(prepared, build())}))
    else:
        require(args.sha256 and args.authority, "run binding")
        print(json.dumps({"report_sha256": execute(prepared, args.sha256, output, args.authority)}))


if __name__ == "__main__":
    main()
