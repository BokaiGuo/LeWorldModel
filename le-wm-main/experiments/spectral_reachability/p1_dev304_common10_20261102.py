"""Known-root paired common-ten-action physical intervention, not confirmation."""
import argparse
import hashlib
import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from . import p1_dev304_diverse_feedback_saved_20261102 as prior

physical, f, saved, base, run = prior.physical, prior.f, prior.saved, prior.base, prior.run
ROOT = run.ROOT
NAME = "p1_dev304_common10_20261102"
AUTHORITY = "DEV304_COMMON10_PHYSICAL"
need = run.require


def protocol():
    prior.analyze()
    contract = base.read(ROOT / f"results/{f.NAME}_PREP_DEV01.json", physical.CONTRACT_SHA)
    plans = base.read(ROOT / f"results/{f.NAME}_PLAN_DEV01/report.json", physical.PLAN_SHA)
    arms = []
    for role in contract["roles"]:
        prefix = np.asarray(role["actual_history"], dtype="<f4")[5:]
        need(hashlib.sha256(prefix.tobytes()).hexdigest() == role["prefix_sha256"], "prefix hash")
        arms.append({"name": role["arm"], "prefix": prefix.tolist(), "prefix_sha256": role["prefix_sha256"]})
    tails = []
    for row in plans["rows"]:
        actions = np.asarray(row["selected"]["actions"], dtype="<f4").reshape(10, 2)
        need(hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_actions_sha256"], "tail hash")
        tails.append({"name": row["arm"], "actions": actions.tolist(),
                      "sha256": row["selected_actions_sha256"]})
    zero = np.zeros((10, 2), dtype="<f4")
    tails.append({"name": "zero", "actions": zero.tolist(), "sha256": hashlib.sha256(zero.tobytes()).hexdigest()})
    need([x["name"] for x in arms] == ["original", "coherent"]
         and [x["name"] for x in tails] == ["original", "coherent", "zero"], "fixed roster")
    return {"schema": NAME + ".protocol.v1", "classification": "post_outcome_known_root_development",
            "source_code_sha256": base.sha(__file__), "source_plan_sha256": physical.PLAN_SHA,
            "source_physical_sha256": saved.EXEC_SHA, "feedback_physical_sha256": prior.RESULT_SHA,
            "seed": run.SEED, "prefixes": arms, "tails": tails,
            "stop_rule": "stop at native termination or truncation; no padding",
            "p1_admission": False, "independent_confirmation": False}


def execute(contract_path, contract_sha, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, contract_sha) == protocol()
         and not os.path.lexists(output), "contract or authority")
    spec = base.read(contract_path, contract_sha)
    old = base.read(saved.ROOT / f"{saved.NAME}_EXEC_DEV01/report.json", saved.EXEC_SHA)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment source")
    store = base.base.Store(output)
    store.emit("start", contract_sha256=contract_sha)
    rows, errors = [], []
    for prefix, baseline in zip(spec["prefixes"], old["rows"], strict=True):
        for tail in spec["tails"]:
            arm = prefix["name"] + "__" + tail["name"]
            row = {"arm": arm, "prefix": prefix["name"], "tail": tail["name"],
                   "status": "running", "transitions": [], "observations": []}
            env = None
            try:
                env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                env.reset(seed=spec["seed"])
                row["observations"].append(base.base.capture(store, env, arm, "warmup", 0,
                                                              baseline["observations"][0]))
                for step in range(1, 11):
                    action = np.zeros(2, dtype="<f4")
                    _, _, terminated, truncated, _ = env.step(action)
                    need(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
                    if step in (5, 10):
                        row["observations"].append(base.base.capture(store, env, arm, "warmup", step,
                                                                      baseline["observations"][[5, 10].index(step)+1]))
                for step, action in enumerate(np.asarray(prefix["prefix"], dtype="<f4"), 1):
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
                    old_step = baseline["transitions"][step-1]
                    need(np.array_equal(env.latest_action, action) and not terminated and not truncated
                         and np.array_equal(obs["state"], old_step["state"]), "prefix replay")
                    row["transitions"].append({"phase": "prefix", "step": step, "action": action.tolist(),
                                               "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                               "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.base.capture(store, env, arm, "prefix", 5,
                                                             baseline["observations"][7]))
                for step, action in enumerate(np.asarray(tail["actions"], dtype="<f4"), 1):
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
                    need(np.array_equal(env.latest_action, action), "tail action")
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
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": contract_sha,
                                       "status": "recorded" if not errors else "failed", "errors": errors,
                                       "rows": rows, "journal_sha256": base.sha(store.path / "journal.jsonl"),
                                       "p1_admission": False, "independent_confirmation": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "execute"))
    p.add_argument("--tag", required=True)
    p.add_argument("--contract-sha")
    p.add_argument("--authority")
    a = p.parse_args()
    need(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{a.tag}.json"
    if a.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        print(execute(contract, a.contract_sha, ROOT/"results"/f"{NAME}_RUN_{a.tag}", a.authority))
