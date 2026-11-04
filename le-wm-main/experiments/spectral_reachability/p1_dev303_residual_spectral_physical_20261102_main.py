"""Execute the complete sealed candidate/metric panel, without outcome selection."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as prior
from . import p1_dev303_residual_spectral_candidates_20261102_main as planner

base, require, ROOT = prior.base, prior.require, prior.ROOT
REPORT = ROOT / "results/p1_dev303_residual_spectral_20261102_main_run01/report.json"
REPORT_SHA = "555eb599aa26de8d524849405e059a2366f10b40a5300acf14c57f5ffaeb3443"
AUTHORITY = "DEV303_RESIDUAL_SPECTRAL_FULL_PANEL_PHYSICS"


def specs_from_report(report):
    require(report["status"] == "model_only_complete" and report["P1_admission"] is False,
            "complete development model report")
    specs = []
    require([r["index"] for r in report["rows"]] == list(range(len(report["rows"]))), "candidate roster")
    for row in report["rows"]:
        for metric in ("latent", "goal_tangent"):
            prefix = np.asarray(row["prefix"], dtype=np.float64)
            tail = np.asarray(row["metrics"][metric]["tail"], dtype=np.float64)
            require(prefix.shape == (5, 2) and tail.shape == (10, 2), "action lengths")
            for actions in (prefix, tail):
                require(np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                        and np.array_equal(actions.astype("<f4").astype(np.float64), actions),
                        "exact bounded float32 actions")
            specs.append({"arm": f"candidate_{row['index']:02d}_{metric}",
                          "index": row["index"], "label": row["label"], "metric": metric,
                          "selected_before_physics": report["selections"][metric] == row["index"],
                          "prefix": prefix.tolist(), "tail": tail.tolist()})
    return specs


def build():
    require(base.sha(REPORT) == REPORT_SHA, "external model report SHA")
    report = base.read(REPORT)
    protocol = base.read(REPORT.parent / "protocol.json")
    require(base.sha(REPORT.parent / "protocol.json") == report["protocol_sha256"], "model protocol")
    require(all(base.sha(path) == digest for path, digest in protocol["source_sha256"].items()), "model sources")
    return {"schema": "dev303_residual_spectral_physical.protocol.v1", "model_report_sha256": REPORT_SHA,
            "specs": specs_from_report(report), "seed": prior.SEED,
            "sources": {str(Path(p).resolve()): base.sha(p)
                        for p in (__file__, prior.__file__, base.__file__)},
            "failure": "stop failed or terminated branch, no retries or padding; execute all other arms",
            "feedback": "open_loop_15_actions_not_cp5_replanning",
            "P1_admission": False, "independent_confirmation": False,
            "full_cost_certified": False}


def branch(env, store, spec, history, reset, goal_rgb, row):
    arm = spec["arm"]
    goal = np.asarray(history["goal"], np.float64)
    checkpoints = {p["step"]: p for p in history["frames"]}
    obs, info = env.reset(seed=prior.SEED)
    require(np.array_equal(obs["state"], reset["state"])
            and np.array_equal(info["goal_state"], goal)
            and np.array_equal(info["goal"], goal_rgb), "reset and goal binding")
    store.emit("reset_verified", arm=arm)
    row["observations"].append(base.capture(store, env, arm, "warmup", 0, checkpoints[0]))
    for phase, actions in (("warmup", [[0., 0.]] * 10), ("prefix", spec["prefix"]), ("tail", spec["tail"])):
        for step, values in enumerate(actions, 1):
            store.check()
            action = np.asarray(values, dtype="<f4")
            store.emit("action_dispatch", arm=arm, phase=phase, step=step, action=action.tolist())
            obs, reward, terminated, truncated, info = env.step(action.copy())
            actual = prior._check_return(env, obs, info, goal, goal_rgb, action)
            require(np.isfinite(float(reward)), "finite physical reward")
            transition = {"phase": phase, "step": step, "action": actual.tolist(),
                          "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                          "terminated": bool(terminated), "truncated": bool(truncated)}
            row["transitions"].append(transition)
            store.emit("action_return", arm=arm, **transition)
            if phase != "warmup" or step in checkpoints:
                row["observations"].append(base.capture(
                    store, env, arm, phase, step, checkpoints.get(step) if phase == "warmup" else None))
            if terminated or truncated:
                require(phase != "warmup", "warmup terminated")
                row["status"] = "native_early_stop"
                return
        if phase == "warmup":
            require(np.array_equal(obs["state"], history["final_state"]), "exact warmup root")
    row["status"] = "fixed_horizon_completed"


def execute(prepared, sha, output, authority):
    require(authority == AUTHORITY and base.sha(prepared) == sha, "authority/contract SHA")
    contract = base.read(prepared)
    require(contract == build(), "contract/source closure")
    arrays, receipt, reset = prior.source_data()
    from stable_worldmodel.envs.pusht.env import PushT
    require(base.sha(inspect.getsourcefile(PushT)) == base.native.ENV_SHA, "native source")
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(Path(prepared).resolve()), "sha256": sha})
    store.emit("started", prepared_sha256=sha)
    rows, errors = [], []
    for spec in contract["specs"]:
        row = {**spec, "status": "running", "transitions": [], "observations": [], "error": None}
        env = None
        store.emit("branch_started", arm=spec["arm"])
        try:
            env = PushT(render_mode="rgb_array", resolution=224, with_target=False)
            branch(env, store, spec, receipt["history"], reset, arrays["goal_rgb"], row)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            errors.append({"arm": spec["arm"], **row["error"]})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    errors.append({"arm": spec["arm"], "stage": "close", "message": str(exc)})
        rows.append(row)
        store.json(spec["arm"] + ".json", row)
        store.emit("branch_finished", arm=spec["arm"], status=row["status"])
    require(contract == build(), "postrun source closure")
    store.emit("finished", errors=errors)
    result = {"schema": "dev303_residual_spectral_physical.report.v1", "rows": rows, "errors": errors,
              "prepared_sha256": sha, "model_report_sha256": REPORT_SHA,
              "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "goal_state_evaluation_only": receipt["history"]["goal"],
              "status": "partial_or_failed" if errors else "recorded_development",
              "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False}
    store.json("report.json", result)
    return {"status": result["status"], "branches": len(rows),
            "report_sha256": base.sha(store.path / "report.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.sha is None and args.output is None and args.authority is None, "prepare only")
        print(json.dumps({"sha256": prior.write_new(args.contract, build())}))
    else:
        require(args.sha and args.output, "run binding")
        print(json.dumps(execute(args.contract, args.sha, args.output, args.authority)))


if __name__ == "__main__":
    main()
