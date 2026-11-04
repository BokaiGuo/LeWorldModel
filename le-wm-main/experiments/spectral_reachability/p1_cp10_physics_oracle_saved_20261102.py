"""Replay oracle winners and original model tails from exact cp10 snapshots."""
import argparse
import importlib
import inspect
from pathlib import Path

import numpy as np

from . import p1_cp10_physics_oracle_20261102 as oracle

ROOT = oracle.ROOT
RUN = ROOT / "results/p1_cp10_physics_oracle_20261102_RUN_DEV01"
CONTRACT = ROOT / "results/p1_cp10_physics_oracle_20261102_PREP_DEV01.json"
CONTRACT_SHA = "9c9cf2b6204c54a80c2699a699951f7d8f61ff3a71d1e79a31c5a20ac44fe5a9"
REPORT_SHA = "6eef4bfd89b0931c7fd3392294451a65b48a6d387808c58b4e6bf069af9e6332"
need = oracle.require


def analyze():
    prev = oracle.saved.physical.previous
    contract = prev.read(CONTRACT, CONTRACT_SHA)
    need(contract == oracle.protocol(), "contract closure")
    report = prev.read(RUN / "report.json", REPORT_SHA)
    need(report["contract_sha256"] == CONTRACT_SHA
         and [r["arm"] for r in report["rows"]] == list(oracle.ARMS), "report binding")
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    need(prev.sha(inspect.getsourcefile(cls)) == contract["environment_source_sha256"], "env source")
    summaries = []
    for arm, row in zip(contract["arms"], report["rows"], strict=True):
        need(prev.read(RUN / (arm["arm"] + ".json")) == row and len(row["rows"]) == oracle.COUNT,
             "candidate records")
        need([r["index"] for r in row["rows"]] == list(range(oracle.COUNT)), "candidate order")
        best = min(row["rows"], key=lambda x: (x["terminal"]["score"], x["index"]))["index"]
        need(best == row["oracle_selected_index"], "winner")
        for index in sorted({0, 1, best}):
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            try:
                _, info = env.reset(seed=contract["seed"])
                need(np.array_equal(info["goal_state"], contract["goal_state"]), "goal")
                replay = np.concatenate((np.zeros((10, 2), np.float32), np.asarray(arm["prefix"], np.float32),
                                         np.asarray(arm["previous_first_packet"], np.float32)))
                for action in replay:
                    _, _, terminated, truncated, _ = env.step(action.copy())
                    need(not terminated and not truncated, "replay termination")
                need(prev.base.native.snapshot(env) == arm["cp10_snapshot"], "cp10 snapshot")
                for step, action in enumerate(np.asarray(arm["candidate_actions"][index], np.float32), 1):
                    obs, reward, terminated, truncated, _ = env.step(action.copy())
                    need(np.array_equal(env.latest_action, action) and not truncated, "action")
                    if terminated:
                        break
                result = row["rows"][index]
                measured = oracle.objective(obs["state"], contract["goal_state"])
                need(result["steps"] == step and result["success"] == bool(terminated)
                     and np.isclose(result["reward"], reward, rtol=0, atol=1e-10)
                     and all(np.isclose(result["terminal"][k], measured[k], rtol=0, atol=1e-10)
                             for k in measured), "physical result replay")
            finally:
                env.close()
        summaries.append({"arm": arm["arm"], "oracle_index": best, "old": row["rows"][0],
                          "new": row["rows"][1], "oracle": row["rows"][best],
                          "candidate_successes": sum(r["success"] for r in row["rows"])})
    return {"schema": "p1_cp10_physics_oracle_saved.v1", "rows": summaries,
            "source_report_sha256": REPORT_SHA, "classification": "privileged_development_oracle",
            "independent_confirmation": False, "P1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(oracle.saved.physical.previous.write_new(args.output, analyze()))
