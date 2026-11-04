"""Execute same-budget feedback plans from exactly replayed real RGB midpoints."""
import argparse
import hashlib
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_triprefix_midfeedback_20261102 as planner

base, run, ROOT, need = planner.base, planner.run, planner.ROOT, planner.need
NAME = "p1_dev305_triprefix_midfeedback_physical_20261102"
AUTHORITY = "DEV305_TRIPREFIX_MIDFEEDBACK_PHYSICAL"
PREP_SHA = "c376753bcfdf2778decca26be207b5dc653e944c83c24276178298ce34eb10be"
PLAN_SHA = "a4044405c0e562658b6468ff9ecb98ec1165289a1ead08d2a262a8777619065b"


def protocol():
    contract = base.read(ROOT / f"results/{planner.NAME}_PREP_DEV01.json", PREP_SHA)
    plan = base.read(ROOT / f"results/{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    need(contract == planner.protocol() and plan["contract_sha256"] == PREP_SHA
         and plan["model_before"] == plan["model_after"] == contract["model_hash"], "plan")
    replay = planner.prior.replay
    replay_spec = base.read(ROOT / f"results/{replay.NAME}_PREP_DEV01.json", planner.prior.REPLAY_CONTRACT_SHA)
    replay_report = base.read(ROOT / f"results/{replay.NAME}_RUN_DEV01/report.json", planner.prior.REPLAY_REPORT_SHA)
    old = base.read(ROOT / f"results/{replay.compare.physical.source.NAME}_RUN_DEV01/report.json",
                    replay.compare.physical.REPORT_SHA)
    baseline = base.read(ROOT / f"results/{run.NAME}_EXEC_DEV01/report.json",
                         replay.compare.compass.source.prior.previous.EXEC_SHA)
    shared = next(x for x in baseline["rows"] if x["arm"] == "original")
    roles = []
    for source, result in zip(contract["roles"], plan["rows"], strict=True):
        action = np.asarray(result["selected"]["actions"], dtype="<f4")
        need(result["arm"] == source["arm"] and action.shape == (2, 5, 2)
             and np.isfinite(action).all() and (np.abs(action) <= 1).all()
             and hashlib.sha256(action.tobytes()).hexdigest() == result["selected_actions_sha256"], "plan actions")
        replay_role = next(x for x in replay_spec["roles"] if x["arm"] == source["arm"])
        branch = next(x for x in replay_role["branches"] if x["name"] == "cem_selected")
        replay_row = next(x for x in replay_report["rows"]
                          if x["arm"] == source["arm"] and x["name"] == "cem_selected")
        old_row = next(x for x in old["rows"] if x["arm"] == source["arm"])
        history = replay_spec["shared_actions"] + branch["actions"][:10]
        states = [x["state"] for x in shared["transitions"]] + branch["states"][:10]
        need(len(history) == len(states) == 15 and
             np.array_equal(np.asarray(history[-10:], dtype="<f4"),
                            np.asarray(source["actual_history"], dtype="<f4")), "history binding")
        roles.append({"arm": source["arm"], "history": history, "states": states,
                      "checkpoint": replay_row["midpoint"], "tail": action[0].tolist(),
                      "plan_sha256": result["selected_actions_sha256"],
                      "old_tail_final_state_evaluation_only": old_row["transitions"][-1]["state"],
                      "old_tail_final_reward_evaluation_only": old_row["transitions"][-1]["reward"]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "plan_sha256": PLAN_SHA, "seed": run.SEED, "roles": roles,
            "classification": "known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    native = planner.prior.replay.compare.compass.source.base.base
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "environment sha")
    store = native.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for role in spec["roles"]:
        row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": []}
        env = None
        try:
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            env.reset(seed=spec["seed"])
            for action in np.zeros((10, 2), dtype="<f4"):
                _, _, term, trunc, _ = env.step(action.copy())
                need(not term and not trunc and np.array_equal(env.latest_action, action), "warmup")
            for i, action in enumerate(np.asarray(role["history"], dtype="<f4")):
                obs, _, term, trunc, _ = env.step(action.copy())
                need(not term and not trunc and np.array_equal(env.latest_action, action)
                     and np.array_equal(obs["state"], role["states"][i]), "history replay")
            row["observations"].append(native.capture(store, env, role["arm"], "checkpoint", 0, role["checkpoint"]))
            for step, action in enumerate(np.asarray(role["tail"], dtype="<f4"), 1):
                obs, reward, term, trunc, _ = env.step(action.copy())
                need(np.array_equal(env.latest_action, action), "tail action")
                row["transitions"].append({"step": step, "action": action.tolist(), "state": obs["state"].tolist(),
                                           "reward": float(reward), "terminated": bool(term), "truncated": bool(trunc)})
                row["observations"].append(native.capture(store, env, role["arm"], "tail", step))
                if term or trunc:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
        except Exception as exc:
            row.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}})
            errors.append({"arm": role["arm"], **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(role["arm"] + ".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
        "status": "recorded" if not errors else "failed", "errors": errors, "rows": rows,
        "journal_sha256": base.sha(store.path / "journal.jsonl"),
        "p1_admission": False, "independent_confirmation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        print(execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
