"""Execute a fixed common-suffix panel from a shared actual dev305 cp5 state."""
import argparse
import hashlib
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_goalrank_physical_saved_20261102 as prior
from . import p1_dev305_spectral_cp10_saved_20261102 as feedback

base, physical = prior.base, prior.physical
run = physical.run
ROOT = run.ROOT
NAME = "p1_dev305_common_suffix_20261102"
AUTHORITY = "DEV305_COMMON_SUFFIX_PHYSICAL"
need = run.require


def protocol():
    prior.analyze()
    feedback.analyze()
    model = base.read(ROOT/f"results/{physical.model_run.NAME}_RUN_DEV01/report.json", physical.MODEL_SHA)
    old = base.read(ROOT/f"results/{run.NAME}_EXEC_DEV01/report.json", prior.previous.EXEC_SHA)
    role = next(row for row in model["rows"] if row["arm"] == "original")
    initial = next(row for row in old["rows"] if row["arm"] == "original")
    shared = np.asarray([x["action"] for x in initial["transitions"]], dtype="<f4")
    need(shared.shape == (5, 2) and initial["status"] == "fixed_horizon_completed", "shared prefix")
    prefixes = []
    for index, candidate in enumerate(role["candidates"]):
        full = np.asarray(candidate["actions"], dtype="<f4")
        need(full.shape == (2, 5, 2)
             and hashlib.sha256(full.tobytes()).hexdigest() == candidate["actions_sha256"], "candidate")
        packet = full[0]
        prefixes.append({"name": f"candidate_{index:02d}", "kind": candidate["kind"],
                         "selected": index == role["selected_index"], "actions": packet.tolist(),
                         "sha256": hashlib.sha256(packet.tobytes()).hexdigest()})
    zero = np.zeros((5, 2), dtype="<f4")
    prefixes.append({"name": "zero", "kind": "zero", "selected": False, "actions": zero.tolist(),
                     "sha256": hashlib.sha256(zero.tobytes()).hexdigest()})
    feedback_report = base.read(ROOT/f"results/{feedback.planner.NAME}_PLAN_DEV01/report.json",
                                feedback.physical.PLAN_SHA)
    tails = []
    for row in feedback_report["rows"]:
        actions = np.asarray(row["selected"]["actions"], dtype="<f4").reshape(10, 2)
        need(hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_actions_sha256"], "tail")
        tails.append({"name": row["arm"], "actions": actions.tolist(), "sha256": row["selected_actions_sha256"]})
    zero_tail = np.zeros((10, 2), dtype="<f4")
    tails.append({"name": "zero", "actions": zero_tail.tolist(),
                  "sha256": hashlib.sha256(zero_tail.tobytes()).hexdigest()})
    need(len(prefixes) == 10 and len(tails) == 3 and sum(p["selected"] for p in prefixes) == 1,
         "fixed roster")
    return {"schema": NAME+".protocol.v1", "source_code_sha256": base.sha(__file__),
            "model_sha256": physical.MODEL_SHA, "feedback_plan_sha256": feedback.physical.PLAN_SHA,
            "initial_physical_sha256": prior.previous.EXEC_SHA, "seed": run.SEED,
            "shared_actions": shared.tolist(), "prefixes": prefixes, "tails": tails,
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
    for prefix in spec["prefixes"]:
        for tail in spec["tails"]:
            name = prefix["name"]+"__"+tail["name"]
            row = {"arm": name, "prefix": prefix["name"], "tail": tail["name"],
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
                for phase, actions in (("prefix", prefix["actions"]), ("tail", tail["actions"])):
                    for step, action in enumerate(np.asarray(actions, dtype="<f4"), 1):
                        obs, reward, term, trunc, _ = env.step(action.copy())
                        need(np.array_equal(env.latest_action, action), "action dispatch")
                        row["transitions"].append({"phase": phase, "step": step, "action": action.tolist(),
                                                   "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                                   "terminated": bool(term), "truncated": bool(trunc)})
                        if step == len(actions) or term or trunc:
                            row["observations"].append(base.base.capture(store, env, name, phase, step))
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
