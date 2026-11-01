"""Development common-tail intervention after the fixed factorial experiment."""
import argparse
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_dynamics_cp5_tail_physical_20261102_exclusive as physical
from .p1_task_response_experiment import sha, write
from .p1_task_response_saved import measures


def prepare(source, output):
    plans = json.loads((source / "plans.json").read_text())
    analysis = json.loads((source / "analysis.json").read_text())
    if analysis["plans_sha256"] != sha(source / "plans.json"):
        raise ValueError("factorial verification binding")
    _, parent, originals, _ = physical.source_data()
    originals = {r["index"]: r for r in originals}
    tails = {"zero_tail": np.zeros((10, 2)).tolist()}
    prefixes = []
    for index in (1, 8, 9, 10):
        row, = [r for r in plans["rows"] if r["arm"] == f"{index:02d}_latent_coherent"]
        tails[row["arm"]] = np.asarray(row["selected"]["actions"]).reshape(10, 2).tolist()
        prefixes.append(dict(index=index, source=originals[index], prefix=row["prefix"]))
    protocol = dict(classification="posthoc_development_matched_intervention_not_confirmation",
                    tails=tails, prefixes=prefixes, goal=parent["goal_state_evaluation_only"],
                    source_plans=str(source / "plans.json"),
                    source_plans_sha256=sha(source / "plans.json"),
                    source_analysis_sha256=sha(source / "analysis.json"),
                    runner_sha256=sha(__file__), physical_helper_sha256=sha(physical.__file__),
                    comparisons="all 4 prefixes x 5 shared tails; own zero-tail changes and absolute terminal",
                    P1_admission=False)
    output.mkdir()
    write(output / "protocol.json", protocol)
    print(sha(output / "protocol.json"), flush=True)


def run(output, expected):
    p = json.loads((output / "protocol.json").read_text())
    if (sha(output / "protocol.json") != expected or sha(__file__) != p["runner_sha256"]
            or sha(physical.__file__) != p["physical_helper_sha256"]
            or sha(p["source_plans"]) != p["source_plans_sha256"]):
        raise ValueError("sealed intervention")
    base = physical.base
    store = base.Store(output / "physics")
    rows = []
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    if sha(inspect.getsourcefile(cls)) != base.native.ENV_SHA:
        raise ValueError("simulator mismatch")
    for prefix in p["prefixes"]:
        old = prefix["source"]
        for label, tail in p["tails"].items():
            arm = f"prefix_{prefix['index']:02d}__{label}"
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            row = dict(arm=arm, index=prefix["index"], suffix=label, transitions=[], error=None)
            try:
                obs, info = env.reset(seed=physical.SEED)
                if not np.array_equal(obs["state"], old["reset_checkpoint"]["state"]) or not np.array_equal(info["goal_state"], p["goal"]):
                    raise ValueError("reset/goal")
                for phase, actions in (("warmup", np.zeros((10, 2))), ("prefix", prefix["prefix"]), ("tail", tail)):
                    for step, action in enumerate(np.asarray(actions, dtype=np.float32), 1):
                        obs, reward, terminated, truncated, _ = env.step(action.copy())
                        if not np.array_equal(env.latest_action, action):
                            raise ValueError("action identity")
                        row["transitions"].append(dict(phase=phase, step=step, action=action.tolist(),
                            state=np.asarray(obs["state"]).tolist(), reward=float(reward),
                            terminated=bool(terminated), truncated=bool(truncated)))
                        if phase == "prefix":
                            base.capture(store, env, arm, phase, step, old["prefix_checkpoints"][step-1])
                        if phase == "tail":
                            base.capture(store, env, arm, phase, step)
                        if terminated or truncated:
                            if phase != "tail":
                                raise ValueError("stop before tail")
                            break
                end = row["transitions"][-1]
                row["terminal"] = measures(end["state"], p["goal"])
                row["tail_steps"] = len(row["transitions"])-15
                row["native_success"] = end["terminated"]
                row["native_reward"] = end["reward"]
            except Exception as exc:
                row["error"] = str(exc)
            finally:
                env.close()
            store.json(arm + ".json", row)
            rows.append(row)
            print(arm, row.get("native_reward"), row["error"], flush=True)
    if any(r["error"] for r in rows):
        store.json("failed_report.json", dict(rows=rows, P1_admission=False))
        raise ValueError("incomplete intervention")
    for row in rows:
        baseline, = [r for r in rows if r["index"] == row["index"] and r["suffix"] == "zero_tail"]
        # No terminal comparison across unequal stopping horizons.
        row["matched_horizon"] = row["tail_steps"] == baseline["tail_steps"]
        row["relative_zero_tail"] = (
            {k: row["terminal"][k]-baseline["terminal"][k] for k in row["terminal"]}
            if row["matched_horizon"] else None)
    store.json("report.json", dict(rows=rows, protocol_sha256=expected,
               journal_sha256=sha(store.path / "journal.jsonl"), P1_admission=False,
               independent_confirmation=False, mechanism_intervention_only=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    else:
        run(args.output.resolve(), args.sha256)
