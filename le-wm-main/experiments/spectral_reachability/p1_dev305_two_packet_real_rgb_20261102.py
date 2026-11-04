"""Replay all sealed physical candidates and retain their actual final RGB."""
import argparse
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_two_packet_oracle_saved_20261102 as saved

source, base, ROOT, need = saved.source, saved.base, saved.ROOT, saved.need
NAME = "p1_dev305_two_packet_real_rgb_20261102"
AUTHORITY = "DEV305_TWO_PACKET_REAL_RGB_REPLAY"


def protocol():
    saved.analyze()
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", saved.CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", saved.REPORT_SHA)
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_contract_sha256": saved.CONTRACT_SHA, "source_report_sha256": saved.REPORT_SHA,
            "seed": contract["seed"], "shared_actions": contract["shared_actions"],
            "prefixes": contract["prefixes"], "bank": contract["bank"],
            "expected_rows": len(report["rows"]),
            "classification": "post_outcome_same_action_replay_development",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    old = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", saved.REPORT_SHA)
    baseline = base.read(ROOT / f"{source.source.run.NAME}_EXEC_DEV01/report.json",
                         source.source.prior.previous.EXEC_SHA)
    shared = next(x for x in baseline["rows"] if x["arm"] == "original")
    native = base.base
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "environment sha")
    store = native.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for prefix in spec["prefixes"]:
        for i, first in enumerate(spec["bank"]):
            for j, second in enumerate(spec["bank"]):
                name = f"{prefix['name']}__{i:02d}_{j:02d}"
                row = {"arm": prefix["name"], "first": i, "second": j, "status": "running"}
                env = None
                try:
                    prior = next(x for x in old["rows"]
                                 if x["prefix"] == prefix["name"] and x["first"] == i and x["second"] == j)
                    need(prior["status"] == "fixed_horizon_completed", "prior")
                    env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                    env.reset(seed=spec["seed"])
                    for _ in range(10):
                        _, _, term, trunc, _ = env.step(np.zeros(2, dtype="<f4"))
                        need(not term and not trunc, "warmup")
                    for k, action in enumerate(np.asarray(spec["shared_actions"], dtype="<f4")):
                        obs, _, term, trunc, _ = env.step(action.copy())
                        need(not term and not trunc and np.array_equal(obs["state"], shared["transitions"][k]["state"]),
                             "shared")
                    for k, action in enumerate(np.asarray(prefix["actions"], dtype="<f4")):
                        obs, _, term, trunc, _ = env.step(action.copy())
                        need(not term and not trunc and np.array_equal(obs["state"], prefix["states"][k]), "prefix")
                    for k, raw in enumerate((first["action"],) * 5 + (second["action"],) * 5):
                        action = np.asarray(raw, dtype="<f4")
                        obs, reward, term, trunc, _ = env.step(action.copy())
                        expected = prior["transitions"][k]
                        need(np.array_equal(env.latest_action, action) and np.array_equal(obs["state"], expected["state"])
                             and reward == expected["reward"] and bool(term) == expected["terminated"]
                             and bool(trunc) == expected["truncated"], "candidate transition")
                    row["final"] = native.capture(store, env, name, "tail", 10)
                    row["status"] = "fixed_horizon_completed"
                except Exception as exc:
                    row.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
                    errors.append({"arm": name, **row["error"]})
                finally:
                    if env is not None:
                        env.close()
                rows.append(row)
            print(f"{prefix['name']} first {i+1}/17 complete", flush=True)
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
            "rows": rows, "errors": errors, "journal_sha256": base.sha(store.path / "journal.jsonl"),
            "p1_admission": False, "independent_confirmation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        print(execute(path, args.contract_sha, ROOT / f"{NAME}_RUN_{args.tag}", args.authority))
