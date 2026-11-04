"""Privileged physical candidate ceiling for a fixed two-packet PushT bank."""
import argparse
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_compass_suffix_saved_20261102 as earlier

source, base, ROOT = earlier.source, earlier.base, earlier.ROOT
need = source.need
NAME = "p1_dev305_two_packet_physical_oracle_20261102"
AUTHORITY = "DEV305_TWO_PACKET_PHYSICAL_ORACLE"


def protocol():
    earlier.analyze()
    old = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", earlier.CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", earlier.REPORT_SHA)
    prefixes = []
    for name in ("candidate_02", "candidate_00", "zero"):
        prefix = next(x for x in old["prefixes"] if x["name"] == name)
        row = next(x for x in report["rows"] if x["arm"] == name + "__zero")
        need(row["status"] == "fixed_horizon_completed", "source prefix")
        prefixes.append({"name": name, "actions": prefix["actions"],
                         "states": [x["state"] for x in row["transitions"][:5]],
                         "state_after_prefix": row["observations"][4]["state"]})
    # Fixed before outcomes: eight evenly spaced headings at two magnitudes plus zero.
    bank = []
    for magnitude in (0.5, 1.0):
        for direction in range(8):
            angle = direction * np.pi / 4
            action = np.asarray([np.cos(angle), np.sin(angle)], dtype="<f4") * np.float32(magnitude)
            bank.append({"name": f"m{int(magnitude*10):02d}_d{direction}", "action": action.tolist()})
    bank.append({"name": "zero", "action": [0.0, 0.0]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_contract_sha256": earlier.CONTRACT_SHA, "source_report_sha256": earlier.REPORT_SHA,
            "seed": old["seed"], "shared_actions": old["shared_actions"], "prefixes": prefixes,
            "bank": bank, "packet_length": 5, "candidate_pairs_per_prefix": 289,
            "oracle_cost": "native reward at step ten, maximized, completed trajectories only",
            "greedy": "maximize native reward at step five, then maximize step-ten native reward within that first packet",
            "termination": "record early termination or truncation; never impute terminal score",
            "classification": "privileged offline physical ceiling, known root, post-outcome development",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output),
         "contract/authority")
    spec = base.read(path, digest)
    native = base.base
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "environment SHA")
    store = native.Store(output)
    store.emit("started", contract_sha256=digest)
    baseline = base.read(ROOT / f"{source.run.NAME}_EXEC_DEV01/report.json", source.prior.previous.EXEC_SHA)
    old_shared = next(x for x in baseline["rows"] if x["arm"] == "original")
    rows, errors = [], []
    for prefix in spec["prefixes"]:
        for i, first in enumerate(spec["bank"]):
            for j, second in enumerate(spec["bank"]):
                row = {"prefix": prefix["name"], "first": i, "second": j, "status": "running",
                       "transitions": []}
                env = None
                try:
                    env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                    env.reset(seed=spec["seed"])
                    for _ in range(10):
                        action = np.zeros(2, dtype="<f4")
                        _, _, term, trunc, _ = env.step(action)
                        need(not term and not trunc, "warmup")
                    for k, action in enumerate(np.asarray(spec["shared_actions"], dtype="<f4")):
                        obs, _, term, trunc, _ = env.step(action.copy())
                        need(not term and not trunc and np.array_equal(obs["state"], old_shared["transitions"][k]["state"]),
                             "shared replay")
                    for k, action in enumerate(np.asarray(prefix["actions"], dtype="<f4")):
                        obs, _, term, trunc, _ = env.step(action.copy())
                        need(not term and not trunc and np.array_equal(obs["state"], prefix["states"][k]),
                             "prefix replay")
                    need(np.array_equal(obs["state"], prefix["state_after_prefix"]), "checkpoint")
                    for t, raw in enumerate((first["action"],) * 5 + (second["action"],) * 5, 1):
                        action = np.asarray(raw, dtype="<f4")
                        obs, reward, term, trunc, _ = env.step(action.copy())
                        need(np.array_equal(env.latest_action, action), "dispatch")
                        row["transitions"].append({"step": t, "state": obs["state"].tolist(),
                                                   "reward": float(reward), "terminated": bool(term),
                                                   "truncated": bool(trunc)})
                        if term or trunc:
                            row["status"] = "native_early_stop"
                            break
                    else:
                        row["status"] = "fixed_horizon_completed"
                except Exception as exc:
                    row.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
                    errors.append({"prefix": prefix["name"], "first": i, "second": j, **row["error"]})
                finally:
                    if env is not None:
                        env.close()
                rows.append(row)
            print(f"{prefix['name']} first {i+1}/17 complete", flush=True)
        store.json(prefix["name"] + ".rows.json", [r for r in rows if r["prefix"] == prefix["name"]])
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
        "rows": rows, "errors": errors, "p1_admission": False, "independent_confirmation": False,
        "journal_sha256": base.sha(store.path / "journal.jsonl")})


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
