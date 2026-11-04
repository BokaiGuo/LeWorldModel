"""Execute cp10 RGB feedback plans after exactly replayed spectral actions."""
import argparse
import hashlib
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_cp15_replan_20261102 as planner

base, run = planner.base, planner.run
ROOT = run.ROOT
NAME = "p1_dev305_cp15_physical_20261102"
PREP_SHA = "8414f41349fad069b15bbf3db2dc879e2653e55f1419e285f4a6ab41cce51b97"
PLAN_SHA = "63244fe481e8a907838ad7e5f2c3761dc475be13e161b55e77193f0d2dabbf79"
AUTHORITY = "DEV305_CP15_PHYSICAL"
need = planner.require


def protocol():
    contract = base.read(ROOT/f"results/{planner.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT/f"results/{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    need(contract == planner.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"], "plan binding")
    roles = []
    for source, plan in zip(contract["roles"], report["rows"], strict=True):
        action = np.asarray(plan["selected"]["actions"], dtype="<f4")
        need(plan["arm"] == source["arm"] and action.shape == (2, 5, 2)
             and np.isfinite(action).all() and (np.abs(action) <= 1).all()
             and hashlib.sha256(action.tobytes()).hexdigest() == plan["selected_actions_sha256"],
             "selected plan")
        label, _ = source["arm"].split("_", 1)
        need(label in ("terminal", "prefix"), "source panel")
        panel = planner.terminal_saved if label == "terminal" else planner.prefix_saved
        prior = base.read(panel.ROOT/f"{panel.physical.NAME}_RUN_DEV01/report.json", panel.REPORT_SHA)
        trajectory = next(row for row in prior["rows"] if row["arm"] == source["source_arm"])
        need(source["source_physical_sha256"] == panel.REPORT_SHA, "physical source")
        earlier = base.read(panel.ROOT/f"{panel.physical.NAME}_PREP_DEV01.json", panel.CONTRACT_SHA)
        prior_role = next(r for r in earlier["roles"] if r["arm"] == source["source_arm"])
        history = prior_role["history"] + prior_role["tail"]
        states = prior_role["states"] + [t["state"] for t in trajectory["transitions"]]
        need(len(history) == len(states) == 15
             and np.array_equal(np.asarray(history[-10:], dtype="<f4"),
                                np.asarray(source["actual_history"], dtype="<f4")), "full history")
        roles.append({"arm": source["arm"], "history": history,
                      "checkpoint": next(o for o in trajectory["observations"]
                                         if o["phase"] == "tail" and o["step"] == 5),
                      "states": states,
                      "tail": action[0].tolist(), "plan_sha256": plan["selected_actions_sha256"]})
    return {"schema": NAME+".protocol.v1", "code_sha256": base.sha(__file__),
            "plan_sha256": PLAN_SHA, "roles": roles, "seed": run.SEED,
            "classification": "known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False}


def execute(contract_path, digest, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, digest) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(contract_path, digest)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment")
    store = base.base.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for role in spec["roles"]:
        row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            env.reset(seed=spec["seed"])
            for action in np.zeros((10, 2), dtype="<f4"):
                _, _, terminated, truncated, _ = env.step(action.copy())
                need(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
            for i, action in enumerate(np.asarray(role["history"], dtype="<f4")):
                obs, _, terminated, truncated, _ = env.step(action.copy())
                need(not terminated and not truncated and np.array_equal(env.latest_action, action)
                     and np.array_equal(obs["state"], role["states"][i]), "history replay")
            row["observations"].append(base.base.capture(store, env, role["arm"], "cp15", 0, role["checkpoint"]))
            for step, action in enumerate(np.asarray(role["tail"], dtype="<f4"), 1):
                obs, reward, terminated, truncated, _ = env.step(action.copy())
                need(np.array_equal(env.latest_action, action), "tail action")
                row["transitions"].append({"step": step, "action": action.tolist(), "state": obs["state"].tolist(),
                                           "reward": float(reward), "terminated": bool(terminated),
                                           "truncated": bool(truncated)})
                row["observations"].append(base.base.capture(store, env, role["arm"], "tail", step))
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
        store.json(role["arm"]+".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME+".report.v1", "contract_sha256": digest,
        "status": "recorded" if not errors else "failed", "errors": errors, "rows": rows,
        "journal_sha256": base.sha(store.path/"journal.jsonl"), "p1_admission": False,
        "independent_confirmation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    contract = ROOT/"results"/f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        print(execute(contract, args.contract_sha, ROOT/"results"/f"{NAME}_RUN_{args.tag}", args.authority))
