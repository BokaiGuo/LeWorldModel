"""Execute a fixed common-suffix panel from a shared actual dev305 cp5 state."""
import argparse
import hashlib
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_goalrank_physical_saved_20261102 as prior
from . import p1_dev305_spectral_cp10_saved_20261102 as feedback
from . import p1_dev305_three_prefix_replan_20261102 as planner
from . import p1_dev305_common_suffix_saved_20261102 as common_saved

base, physical = prior.base, prior.physical
run = physical.run
ROOT = run.ROOT
NAME = "p1_dev305_three_prefix_physical_20261102"
AUTHORITY = "DEV305_THREE_PREFIX_PHYSICAL"
PLAN_SHA = "1d51f63ae9f68f5efafb391143b64b672d7967a8bc1708bbcd1009a0e3fc22a3"
PLAN_CONTRACT_SHA = "ed41d6f4e3aef8a136b98bf3ba6622592da4b895408cac182d03f532760402a7"
need = run.require


def protocol():
    common_saved.analyze()
    source = base.read(ROOT/f"results/{common_saved.source.NAME}_PREP_DEV01.json", common_saved.CONTRACT_SHA)
    planner_spec = base.read(ROOT/f"results/{planner.NAME}_PREP_DEV01.json", PLAN_CONTRACT_SHA)
    report = base.read(ROOT/f"results/{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    need(planner_spec == planner.protocol() and report["contract_sha256"] == PLAN_CONTRACT_SHA
         and report["model_before"] == report["model_after"] == planner_spec["model_hash"], "plan source")
    arms = []
    for role, row in zip(planner_spec["roles"], report["rows"], strict=True):
        prefix = next(p for p in source["prefixes"] if p["name"] == role["arm"])
        actions = np.asarray(row["selected"]["actions"], dtype="<f4").reshape(10, 2)
        need(row["arm"] == role["arm"] and row["prefix_sha256"] == role["prefix_sha256"]
             and hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_actions_sha256"], "plan binding")
        for label in ("predictor", "action_encoder"):
            need(report["ledger"][role["arm"]+":"+label] == {"calls": 10, "samples": 3200}, "budget")
        arms.append({"name": role["arm"], "prefix": prefix["actions"], "tail": actions.tolist()})
    need([a["name"] for a in arms] == ["candidate_02", "candidate_00", "zero"], "fixed roster")
    return {"schema": NAME+".protocol.v1", "source_code_sha256": base.sha(__file__),
            "plan_sha256": PLAN_SHA, "plan_contract_sha256": PLAN_CONTRACT_SHA,
            "initial_physical_sha256": prior.previous.EXEC_SHA, "seed": run.SEED,
            "shared_actions": source["shared_actions"], "arms": arms,
            "classification": "post_outcome_development_no_confirmatory_inference",
            "stop_rule": "native termination/truncation; no padding",
            "selection_limit": "candidate and tails fixed from earlier development runs; not a greedy baseline",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(path, digest)
    old = base.read(ROOT/f"results/{run.NAME}_EXEC_DEV01/report.json", prior.previous.EXEC_SHA)
    baseline = next(row for row in old["rows"] if row["arm"] == "original")
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment SHA")
    store = base.base.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    source_report = base.read(ROOT/f"results/{common_saved.source.NAME}_RUN_DEV01/report.json", common_saved.REPORT_SHA)
    for arm in spec["arms"]:
        for _ in (0,):
            name = arm["name"]
            source_row = next(r for r in source_report["rows"] if r["arm"] == name+"__zero")
            row = {"arm": name, "prefix": name, "tail": "own_optimized",
                   "status": "running", "transitions": [], "observations": []}
            env = None
            try:
                env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                env.reset(seed=spec["seed"])
                row["observations"].append(base.base.capture(store, env, name, "warmup", 0,
                                                              baseline["observations"][0]))
                for step in range(1, 11):
                    action = np.zeros(2, dtype="<f4")
                    _, _, term, trunc, _ = env.step(action.copy())
                    need(not term and not trunc and np.array_equal(env.latest_action, action), "warmup")
                    if step in (5, 10):
                        row["observations"].append(base.base.capture(
                            store, env, name, "warmup", step, baseline["observations"][[5, 10].index(step)+1]))
                for step, action in enumerate(np.asarray(spec["shared_actions"], dtype="<f4"), 1):
                    obs, _, term, trunc, _ = env.step(action.copy())
                    need(not term and not trunc and np.array_equal(env.latest_action, action)
                         and np.array_equal(obs["state"], baseline["transitions"][step-1]["state"]), "shared replay")
                row["observations"].append(base.base.capture(store, env, name, "shared", 5,
                                                             baseline["observations"][7]))
                for phase, actions in (("prefix", arm["prefix"]), ("tail", arm["tail"])):
                    for step, action in enumerate(np.asarray(actions, dtype="<f4"), 1):
                        obs, reward, term, trunc, _ = env.step(action.copy())
                        need(np.array_equal(env.latest_action, action), "action dispatch")
                        if phase == "prefix":
                            need(np.array_equal(obs["state"], source_row["transitions"][step-1]["state"]),
                                 "prefix replay state")
                        row["transitions"].append({"phase": phase, "step": step, "action": action.tolist(),
                                                   "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                                   "terminated": bool(term), "truncated": bool(trunc)})
                        if step == len(actions) or term or trunc:
                            reference = source_row["observations"][4] if phase == "prefix" and step == 5 else None
                            row["observations"].append(base.base.capture(store, env, name, phase, step, reference))
                        if term or trunc:
                            row["status"] = "native_early_stop"
                            break
                    if row["status"] == "native_early_stop":
                        break
                else:
                    row["status"] = "fixed_horizon_completed"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                errors.append({"arm": name, **row["error"]})
            finally:
                if env is not None:
                    env.close()
            store.json(name+".row.json", row)
            store.emit("branch_closed", arm=name, status=row["status"])
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
