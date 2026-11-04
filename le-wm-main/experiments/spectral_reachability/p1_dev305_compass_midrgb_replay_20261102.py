"""Replay sealed dev305 action branches and capture actual tail midpoint RGB."""
import argparse
import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_compass_model_compare_20261102 as compare
from . import p1_dev305_compass_model_compare_saved_20261102 as checked

base, need, run, ROOT = compare.base, compare.need, compare.run, compare.ROOT
NAME = "p1_dev305_compass_midrgb_replay_20261102"
AUTHORITY = "DEV305_COMPASS_MIDRGB_REPLAY_PHYSICAL"


def protocol():
    checked.analyze()
    contract = base.read(ROOT / f"results/{compare.NAME}_PREP_DEV01.json", checked.CONTRACT_SHA)
    planner = base.read(ROOT / f"results/{compare.planner.NAME}_PREP_DEV01.json", compare.PLANNER_SHA)
    compassdir = compare.compass.ROOT / f"{compare.compass.source.NAME}_RUN_DEV01"
    physdir = compare.physical.ROOT / f"{compare.physical.source.NAME}_RUN_DEV01"
    compass_report = base.read(compassdir / "report.json", compare.compass.REPORT_SHA)
    physical_report = base.read(physdir / "report.json", compare.physical.REPORT_SHA)
    roles = []
    for role in contract["roles"]:
        source_role = next(x for x in planner["roles"] if x["arm"] == role["arm"])
        original = next(x for x in physical_report["rows"] if x["arm"] == role["arm"])
        branches = []
        for item in role["rows"]:
            is_cem = item["name"] == "cem_selected"
            old = original if is_cem else next(x for x in compass_report["rows"]
                                                if x["arm"] == role["arm"] + "__" + item["name"])
            folder = physdir if is_cem else compassdir
            need(old["status"] == "fixed_horizon_completed" and len(old["transitions"]) == 15, "prior")
            # Exactly the five selected-prefix actions and the ten fixed suffix actions.
            actions = [x["action"] for x in old["transitions"]]
            need(np.array_equal(np.asarray(actions[5:], dtype="<f4"),
                                np.asarray(item["actions"], dtype="<f4")), "suffix")
            final = next(x for x in old["observations"] if x["phase"] == "tail" and x["step"] == 10)
            branches.append({"name": item["name"], "actions": actions,
                             "states": [x["state"] for x in old["transitions"]],
                             "final_pixel_sha256": final["pixel_sha256"],
                             "final_file_sha256": final["file_sha256"],
                             "old_final_path": str((folder / final["file"]).resolve())})
        roles.append({"arm": role["arm"], "source_images": role["images"],
                      "actual_history": role["actual_history"], "branches": branches})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "model_comparison_contract_sha256": checked.CONTRACT_SHA,
            "model_comparison_report_sha256": checked.REPORT_SHA,
            "seed": compare.compass.source.run.SEED, "roles": roles,
            "shared_actions": compare.compass.source.protocol()["shared_actions"],
            "classification": "post_outcome_development_replay_not_independent",
            "source_role_count": len(planner["roles"]), "p1_admission": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    old = base.read(ROOT / f"results/{compare.compass.source.run.NAME}_EXEC_DEV01/report.json",
                    compare.compass.source.prior.previous.EXEC_SHA)
    baseline = next(x for x in old["rows"] if x["arm"] == "original")
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == compare.compass.source.base.base.native.ENV_SHA, "env sha")
    store = compare.compass.source.base.base.Store(output)
    rows, errors = [], []
    for role in spec["roles"]:
        for branch in role["branches"]:
            name = role["arm"] + "__" + branch["name"]
            row = {"arm": role["arm"], "name": branch["name"], "status": "running"}
            env = None
            try:
                env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                env.reset(seed=spec["seed"])
                for _ in range(10):
                    _, _, term, trunc, _ = env.step(np.zeros(2, dtype="<f4"))
                    need(not term and not trunc, "warmup")
                for i, action in enumerate(np.asarray(spec["shared_actions"], dtype="<f4")):
                    obs, _, term, trunc, _ = env.step(action.copy())
                    need(not term and not trunc and np.array_equal(obs["state"], baseline["transitions"][i]["state"]),
                         "shared state")
                for i, action in enumerate(np.asarray(branch["actions"], dtype="<f4")):
                    obs, _, term, trunc, _ = env.step(action.copy())
                    need(not term and not trunc and np.array_equal(env.latest_action, action)
                         and np.array_equal(obs["state"], branch["states"][i]), "action/state replay")
                    if i == 9:
                        row["midpoint"] = compare.compass.source.base.base.capture(
                            store, env, name, "tail", 5)
                final = compare.compass.source.base.base.capture(store, env, name, "tail", 10)
                need(final["pixel_sha256"] == branch["final_pixel_sha256"], "final pixel replay")
                row.update({"final": final, "status": "fixed_horizon_completed"})
            except Exception as exc:
                row.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}})
                errors.append({"arm": name, **row["error"]})
            finally:
                if env is not None:
                    env.close()
            rows.append(row)
            print(name, row["status"], flush=True)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
                                      "rows": rows, "errors": errors,
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
