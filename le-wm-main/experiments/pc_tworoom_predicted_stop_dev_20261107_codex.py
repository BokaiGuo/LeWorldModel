"""Seal prediction-only selection, then replay actual fixed-duration actions."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_block_b_temporal_alignment_20261107_codex as prior
from experiments.pc_input_permissions import file_hash

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_PREDICTED_STOP_DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_predicted_stop_dev_20261107_codex"
PLAN = Path(str(PREFIX) + "_plan.json")
RESULT = Path(str(PREFIX) + "_result.json")


def choose(cost):
    cost = np.asarray(cost)
    if cost.shape != (64, 5) or not np.isfinite(cost).all():
        raise ValueError("expected finite 64 x 5 predicted costs")
    final = int(np.argmin(cost[:, -1]))
    flat = int(np.argmin(cost))
    candidate, time = np.unravel_index(flat, cost.shape)
    return {"final25": (final, 25), "min25": (int(candidate), 25),
            "min_stop": (int(candidate), 5 * (int(time) + 1))}


def action_hash(actions):
    a = np.ascontiguousarray(actions, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def seal():
    if PLAN.exists() or RESULT.exists():
        raise FileExistsError("output exists")
    meta = json.loads(prior.PRED_META.read_text())
    result = json.loads(prior.RESULT.read_text())
    if (meta["pred_sha256"] != file_hash(prior.PRED)
            or result["pred_sha256"] != file_hash(prior.PRED)
            or result["features_sha256"] != file_hash(prior.FEATURES)):
        raise ValueError("prediction artifacts changed")
    # Decode predicted embedding and target features, never read physical labels.
    source_plan = json.loads(prior.base.prior.PLAN.read_text())
    if source_plan["pool_sha256"] != file_hash(prior.base.prior.POOL):
        raise ValueError("candidate pool changed")
    rows = []
    with (np.load(prior.PRED, allow_pickle=False) as pred,
          np.load(prior.FEATURES, allow_pickle=False) as features,
          np.load(prior.base.prior.POOL, allow_pickle=False) as pool):
        for root in range(10):
            predicted = pred[f"r{root}_predicted"][:, 1:]
            goal = features[f"r{root}_goal"]
            cost = np.square(predicted - goal).sum(-1)
            np.testing.assert_array_equal(cost, features[f"r{root}_predicted_cost"])
            actions = pool[f"r{root}_b_actions"].reshape(64, 25, 2)
            choices = {}
            for name, (index, steps) in choose(cost).items():
                choices[name] = {"index": index, "steps": steps,
                                 "action_sha256": action_hash(actions[index, :steps])}
            rows.append({"root": root, "row": source_plan["rows"][root]["row"],
                         "choices": choices})
    record = {"rows": rows, "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "pred_sha256": file_hash(prior.PRED),
              "features_sha256": file_hash(prior.FEATURES),
              "pool_sha256": file_hash(prior.base.prior.POOL),
              "source_plan_sha256": file_hash(prior.base.prior.PLAN),
              "scope": "exposed-development, selection uses prediction only"}
    with PLAN.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("pred_sha256", prior.PRED), ("features_sha256", prior.FEATURES),
                      ("pool_sha256", prior.base.prior.POOL),
                      ("source_plan_sha256", prior.base.prior.PLAN)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    old = json.loads(prior.base.prior.PHYSICAL.read_text())
    if old["pool_sha256"] != plan["pool_sha256"]:
        raise ValueError("physical source pool mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(prior.base.prior.POOL, allow_pickle=False) as pool,
              h5py.File(prior.base.prior.source.block.prior.common.DATA, "r") as data):
            for item in plan["rows"]:
                root, task = item["root"], item["row"]
                actions = pool[f"r{root}_b_actions"].reshape(64, 25, 2)
                outcomes = {}
                for name in ("final25", "min25", "min_stop"):
                    choice = item["choices"][name]
                    chain = actions[choice["index"], :choice["steps"]]
                    if action_hash(chain) != choice["action_sha256"]:
                        raise ValueError("action fingerprint mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    expected = old["rows"][root]["outcomes"][choice["index"]]["trace"][:len(chain)]
                    if trace != expected:
                        raise ValueError(f"physical mismatch root={root} arm={name}")
                    outcomes[name] = {"trace": trace, "steps": len(trace),
                                      "ever": any(t["terminated"] for t in trace),
                                      "final": trace[-1]["terminated"]}
                if outcomes["min_stop"]["trace"] != outcomes["min25"]["trace"][:outcomes["min_stop"]["steps"]]:
                    raise ValueError("truncated and full executions differ")
                rows.append({"root": root, "row": task, "outcomes": outcomes})
                print(f"root={root} replayed", flush=True)
    finally:
        env.close()
    arms = ("final25", "min25", "min_stop")
    summary = {arm: {"ever": sum(r["outcomes"][arm]["ever"] for r in rows),
                     "final": sum(r["outcomes"][arm]["final"] for r in rows),
                     "steps": sum(r["outcomes"][arm]["steps"] for r in rows)}
               for arm in arms}
    for arm in ("min25", "min_stop"):
        summary[arm]["wins"] = sum(r["outcomes"][arm]["ever"] and not r["outcomes"]["final25"]["ever"] for r in rows)
        summary[arm]["losses"] = sum(not r["outcomes"][arm]["ever"] and r["outcomes"]["final25"]["ever"] for r in rows)
    summary["truncation_lost_ever"] = sum(r["outcomes"]["min25"]["ever"] and not r["outcomes"]["min_stop"]["ever"] for r in rows)
    summary["stop_rule_met"] = (summary["min_stop"]["ever"] > summary["final25"]["ever"]
                                and summary["truncation_lost_ever"] == 0)
    with RESULT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "plan_sha256": file_hash(PLAN),
                   "old_physical_sha256": file_hash(prior.base.prior.PHYSICAL),
                   "scope": "exposed development, not a gate or independent result"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "replay"):
        raise SystemExit("expected seal|replay")
    {"seal": seal, "replay": replay}[sys.argv[1]]()
