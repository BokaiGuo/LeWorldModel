"""Execute the sealed next-packet bank with RGB checks at every physical step."""
import argparse
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_contact_second_boundary_20261102 as planner
from . import p1_dev305_triprefix_midfeedback_physical_20261102 as engine

base, need, ROOT = planner.base, planner.need, planner.ROOT
NAME = "p1_dev305_contact_second_boundary_physical_20261102"
AUTHORITY = "DEV305_CONTACT_SECOND_BOUNDARY_PHYSICAL"
PREP_SHA = "92c482359a71619992f699b04f044e4e09fa4e46c19d62cc8c2e4545a7ca09d5"
PLAN_SHA = "756f8233062c24bdd5e8e3c82b97bb432ac314a4ec948c35621695bf24cfe256"


def protocol():
    spec = base.read(ROOT / "results" / f"{planner.NAME}_PREP_DEV01.json", PREP_SHA)
    plan = base.read(ROOT / "results" / f"{planner.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    prior = base.read(ROOT / "results" / f"{planner.previous.real.source.NAME}_PREP_DEV01.json",
                      planner.previous.real.PREP_SHA)
    physical = base.read(ROOT / "results" / f"{planner.previous.real.source.NAME}_RUN_DEV01/report.json",
                         planner.previous.real.REPORT_SHA)
    need(spec == planner.protocol() and plan["contract_sha256"] == PREP_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"], "source")
    roles = []
    for source, prediction in zip(spec["roles"], plan["rows"], strict=True):
        need(source["arm"] == prediction["arm"], "arm")
        delta = np.asarray(prediction["predicted_goal_delta"], dtype=np.float32)
        scores = np.asarray(prediction["scores"])
        need(delta.shape == (4, spec["latent_dim"]) and np.isfinite(delta).all()
             and np.allclose((delta**2).mean(axis=1), scores, atol=2e-6, rtol=0)
             and int(np.argmin(scores)) == prediction["selected_index"], "selection")
        key = source["arm"] + "__" + source["previous_candidate"]
        old_spec = next(row for row in prior["roles"] if row["arm"] == key)
        old = next(row for row in physical["rows"] if row["arm"] == key)
        history = old_spec["history"] + [row["action"] for row in old["transitions"]]
        states = old_spec["states"] + [row["state"] for row in old["transitions"]]
        need(len(history) == len(states) == 20
             and np.array_equal(np.asarray(history[-10:], dtype="<f4"),
                                np.asarray(source["actual_history"], dtype="<f4")), "replay history")
        for i, label in enumerate(spec["bank_order"]):
            action = np.asarray(source["actions"][i], dtype="<f4")
            need(action.shape == (5, 2) and np.isfinite(action).all()
                 and (np.abs(action) <= 1).all(), "bounded actions")
            roles.append({"arm": source["arm"] + "__" + label, "history": history, "states": states,
                          "checkpoint": old["observations"][-1], "tail": action.tolist(),
                          "selected": i == prediction["selected_index"]})
    need(len(roles) == 12 and sum(x["selected"] for x in roles) == 3, "full candidate bank")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_contract_sha256": PREP_SHA, "plan_sha256": PLAN_SHA,
            "seed": prior["seed"], "roles": roles,
            "guard": "inspect current RGB before each action; after invalid final frame record censored, never backfill",
            "scope": "known-root development; fixed five-step plan, no new outcome in selector; not independent",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    native = engine.planner.prior.replay.compare.compass.source.base.base
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "environment")
    store = native.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for role in spec["roles"]:
        row = {"arm": role["arm"], "selected": role["selected"], "status": "running",
               "transitions": [], "observations": []}
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
                if not planner.previous.source.visibility.inspect(np.asarray(env.render()))["eligible"]:
                    row["status"] = "visibility_guard_stop"
                    break
                obs, reward, term, trunc, _ = env.step(action.copy())
                need(np.array_equal(env.latest_action, action), "action dispatch")
                row["transitions"].append({"step": step, "action": action.tolist(), "state": obs["state"].tolist(),
                                           "reward": float(reward), "terminated": bool(term), "truncated": bool(trunc)})
                row["observations"].append(native.capture(store, env, role["arm"], "tail", step))
                if term or trunc:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
            row["final_visibility"] = planner.previous.source.visibility.inspect(np.asarray(env.render()))
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
