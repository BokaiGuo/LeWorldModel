"""Execute both preselected dev304 feedback tails on exact replayed prefixes."""
import argparse
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np

from . import p1_dev304_diverse_feedback_20261102 as feedback

run, saved, base = feedback.run, feedback.saved, feedback.base
require = run.require
ROOT = run.ROOT
NAME = "p1_dev304_diverse_feedback_physical_20261102"
CONTRACT_SHA = "bac1bfa83f606addf6efff378aace462cc9377d9d962edc9e1dc24952424bb27"
PLAN_SHA = "a74d3835c26a557c0a724f9117c8ceb93061a1addbadb999f4dd8873ec7e5645"
AUTHORITY = "DEV304_DIVERSE_FEEDBACK_PHYSICAL"


def execute(output, authority):
    require(authority == AUTHORITY, "physical authority")
    contract_path = ROOT / f"results/{feedback.NAME}_PREP_DEV01.json"
    plan_dir = ROOT / f"results/{feedback.NAME}_PLAN_DEV01"
    spec = base.read(contract_path, CONTRACT_SHA)
    require(spec == feedback.protocol(), "feedback contract")
    report = base.read(plan_dir / "report.json", PLAN_SHA)
    require(report["contract_sha256"] == CONTRACT_SHA
            and report["model_before"] == report["model_after"] == spec["model_hash"]
            and [r["arm"] for r in report["rows"]] == [r["arm"] for r in spec["roles"]],
            "feedback plan binding")
    original = base.read(saved.ROOT / f"{saved.NAME}_EXEC_DEV01/report.json", saved.EXEC_SHA)
    store = base.base.Store(output)
    store.emit("execute_started", plan_sha256=PLAN_SHA, contract_sha256=CONTRACT_SHA)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    require(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment identity")
    rows, errors = [], []
    for role, plan_row, prior in zip(spec["roles"], report["rows"], original["rows"], strict=True):
        arm = role["arm"]
        row = {"arm": arm, "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            require(base.read(plan_dir / (arm + ".json")) == plan_row
                    and plan_row["prefix_sha256"] == role["prefix_sha256"], "row source")
            prefix = np.asarray(role["actual_history"], dtype="<f4")[5:]
            tail = np.asarray(plan_row["selected"]["actions"], dtype="<f4")
            require(prefix.shape == (5, 2) and tail.shape == (2, 5, 2)
                    and hashlib.sha256(prefix.tobytes()).hexdigest() == role["prefix_sha256"]
                    and hashlib.sha256(tail.tobytes()).hexdigest() == plan_row["selected_actions_sha256"]
                    and np.isfinite(tail).all() and (np.abs(tail) <= 1).all(), "actions")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            env.reset(seed=run.SEED)
            row["observations"].append(base.base.capture(store, env, arm, "warmup", 0,
                                                          prior["observations"][0]))
            for step in range(1, 11):
                zero = np.zeros(2, dtype="<f4")
                _, _, terminated, truncated, _ = env.step(zero)
                require(not terminated and not truncated and np.array_equal(env.latest_action, zero), "warmup")
                if step in (5, 10):
                    row["observations"].append(base.base.capture(
                        store, env, arm, "warmup", step, prior["observations"][[5, 10].index(step)+1]))
            for step, action in enumerate(prefix, 1):
                obs, reward, terminated, truncated, _ = env.step(action.copy())
                old = prior["transitions"][step-1]
                require(np.array_equal(env.latest_action, action) and not terminated and not truncated
                        and np.array_equal(obs["state"], old["state"]), "prefix replay")
                row["transitions"].append({"phase": "prefix", "step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
            row["observations"].append(base.base.capture(store, env, arm, "prefix", 5,
                                                         prior["observations"][7]))
            for step, action in enumerate(tail[0], 1):
                obs, reward, terminated, truncated, _ = env.step(action.copy())
                require(np.array_equal(env.latest_action, action), "tail action")
                row["transitions"].append({"phase": "tail", "step": step, "action": action.tolist(),
                                           "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                           "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.base.capture(store, env, arm, "tail", step))
                if terminated or truncated:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            errors.append({"arm": arm, **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(arm + ".row.json", row)
        store.emit("branch_closed", arm=arm, status=row["status"])
        rows.append(row)
    store.emit("execute_finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".v1", "status": "recorded" if not errors else "failed",
                                       "plan_sha256": PLAN_SHA, "contract_sha256": CONTRACT_SHA,
                                       "rows": rows, "errors": errors,
                                       "journal_sha256": base.sha(store.path / "journal.jsonl"),
                                       "p1_admission": False, "independent_confirmation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    print(execute(ROOT / "results" / f"{NAME}_{args.tag}", args.authority))
