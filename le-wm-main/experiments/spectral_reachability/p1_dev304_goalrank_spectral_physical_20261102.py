"""Execute every fixed spectral and random candidate on its replayed RGB prefix."""
import argparse
import hashlib
import importlib
import inspect
import os

import numpy as np

from . import p1_dev304_goalrank_spectral_20261102 as model_run

base, run, previous = model_run.base, model_run.run, model_run.previous
ROOT = run.ROOT
NAME = "p1_dev304_goalrank_spectral_physical_20261102"
MODEL_SHA = "d5832908ddcc9d298b4980dd7ec1da313c92d2002e04dbcc649c00a449d4b323"
CONTRACT_SHA = "48017771016077c1d7860759c7104f46581919471f63fd169ec0fec3e94da402"
AUTHORITY = "DEV304_GOALRANK_SPECTRAL_PHYSICAL"
need = run.require


def protocol():
    model_spec = base.read(ROOT/f"results/{model_run.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    model = base.read(ROOT/f"results/{model_run.NAME}_RUN_DEV01/report.json", MODEL_SHA)
    need(model_spec == model_run.protocol() and model["contract_sha256"] == CONTRACT_SHA
         and model["model_before"] == model["model_after"] == model_spec["model_hash"], "model bindings")
    roles = []
    for role, row in zip(model_spec["roles"], model["rows"], strict=True):
        need(role["arm"] == row["arm"] and len(row["candidates"]) == 9
             and base.read(ROOT/f"results/{model_run.NAME}_RUN_DEV01/{row['arm']}.json") == row, "model roster")
        candidates = []
        for index, item in enumerate(row["candidates"]):
            action = np.asarray(item["actions"], dtype="<f4")
            need(action.shape == (2, 5, 2) and np.isfinite(action).all() and (np.abs(action) <= 1).all()
                 and hashlib.sha256(action.tobytes()).hexdigest() == item["actions_sha256"], "candidate action")
            candidates.append({"index": index, "kind": item["kind"], "actions": action.tolist(),
                               "actions_sha256": item["actions_sha256"]})
        roles.append({"arm": role["arm"], "prefix_sha256": role["prefix_sha256"],
                      "prefix": role["actual_history"][5:], "candidates": candidates,
                      "model_selected_index": row["selected_index"]})
    return {"schema": NAME+".protocol.v1", "classification": "post_outcome_known_root_development",
            "source_code_sha256": base.sha(__file__), "model_sha256": MODEL_SHA,
            "roles": roles, "stop_rule": "native termination/truncation; no padding",
            "p1_admission": False, "independent_confirmation": False}


def execute(contract_path, digest, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, digest) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(contract_path, digest)
    old = base.read(previous.saved.ROOT/f"{previous.saved.NAME}_EXEC_DEV01/report.json", previous.saved.EXEC_SHA)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == base.base.native.ENV_SHA, "environment")
    store = base.base.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for role, prior in zip(spec["roles"], old["rows"], strict=True):
        for candidate in role["candidates"]:
            arm = f"{role['arm']}_{candidate['index']:02d}"
            row = {"arm": arm, "prefix": role["arm"], "candidate": candidate["index"],
                   "status": "running", "transitions": [], "observations": []}
            env = None
            try:
                env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                env.reset(seed=run.SEED)
                row["observations"].append(base.base.capture(store, env, arm, "warmup", 0, prior["observations"][0]))
                for step in range(1, 11):
                    action = np.zeros(2, dtype="<f4")
                    _, _, terminated, truncated, _ = env.step(action)
                    need(not terminated and not truncated and np.array_equal(env.latest_action, action), "warmup")
                    if step in (5, 10):
                        row["observations"].append(base.base.capture(store, env, arm, "warmup", step,
                                                                      prior["observations"][[5, 10].index(step)+1]))
                prefix = np.asarray(role["prefix"], dtype="<f4")
                need(hashlib.sha256(prefix.tobytes()).hexdigest() == role["prefix_sha256"], "prefix bytes")
                for step, action in enumerate(prefix, 1):
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
                    need(not terminated and not truncated and np.array_equal(env.latest_action, action)
                         and np.array_equal(obs["state"], prior["transitions"][step-1]["state"]), "prefix replay")
                    row["transitions"].append({"phase": "prefix", "step": step, "action": action.tolist(),
                                               "state": np.asarray(obs["state"]).tolist(), "reward": float(reward),
                                               "terminated": bool(terminated), "truncated": bool(truncated)})
                row["observations"].append(base.base.capture(store, env, arm, "prefix", 5, prior["observations"][7]))
                for step, action in enumerate(np.asarray(candidate["actions"], dtype="<f4").reshape(10, 2), 1):
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
                    need(np.array_equal(env.latest_action, action), "candidate dispatch")
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
            store.json(arm+".row.json", row)
            store.emit("branch_closed", arm=arm, status=row["status"])
            rows.append(row)
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME+".report.v1", "contract_sha256": digest,
        "status": "recorded" if not errors else "failed", "errors": errors, "rows": rows,
        "journal_sha256": base.sha(store.path/"journal.jsonl"), "p1_admission": False,
        "independent_confirmation": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "execute"))
    p.add_argument("--tag", required=True)
    p.add_argument("--contract-sha")
    p.add_argument("--authority")
    a = p.parse_args()
    need(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT/"results"/f"{NAME}_PREP_{a.tag}.json"
    if a.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        print(execute(contract, a.contract_sha, ROOT/"results"/f"{NAME}_RUN_{a.tag}", a.authority))
