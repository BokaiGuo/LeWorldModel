"""Reconsider already-scored full-chain incumbents; seal before physical replay."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_VECTOR_INCUMBENT_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_vector_incumbent_20261107_codex"
PLAN = Path(str(PREFIX) + "_plan.json")
POOL = Path(str(PREFIX) + "_pool.npz")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")
METHODS = prior.METHODS


def choose(values, method_index):
    values = np.asarray(values)
    if values.shape != (6,) or not np.isfinite(values).all() or method_index not in range(4):
        raise ValueError("invalid scores")
    indices = [method_index, 4, 5]
    return indices[int(np.argmin(values[indices]))]


def action_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def seal():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed artifact exists")
    old = json.loads(prior.PLAN.read_text())
    source = json.loads(prior.base.source.PLAN.read_text())
    if (old["pool_sha256"] != file_hash(prior.POOL)
            or old["source_pool_sha256"] != file_hash(prior.base.source.POOL)
            or source["pool_sha256"] != file_hash(prior.base.source.POOL)
            or old["input_sha256"] != file_hash(prior.base.source.block.prior.inputs.INPUT)):
        raise ValueError("source integrity failure")
    common = prior.base.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.PLAN, prior.POOL,
                         prior.base.source.POOL, prior.base.source.block.prior.inputs.INPUT,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    output, rows = {}, []
    with (np.load(prior.POOL, allow_pickle=False) as pool,
          np.load(prior.base.source.POOL, allow_pickle=False) as source_pool,
          np.load(prior.base.source.block.prior.inputs.INPUT, allow_pickle=False) as data,
          torch.inference_mode()):
        for record in old["rows"]:
            root = record["root"]
            actions = []
            for method in METHODS:
                decision = record["decisions"][method]
                action = pool[f"r{root}_{decision['bank']}"].reshape(64, 25, 2)[decision["index"]]
                if action_hash(action) != decision["action_sha256"]:
                    raise ValueError("old action mismatch")
                actions.append(action)
            indices, historical_scores = {}, {}
            for bank in ("block_cem", "block_random"):
                scores = source_pool[f"r{root}_{bank}_scores"]
                if scores.shape != (300,) or not np.isfinite(scores).all():
                    raise ValueError("invalid source scores")
                index = int(np.argmin(scores))
                indices[bank] = index
                historical_scores[bank] = float(scores[index])
                actions.append(source_pool[f"r{root}_{bank}_actions"][index])
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (6, 25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                raise ValueError("invalid action")
            normalized = scaler.transform(actions.reshape(-1, 2)).reshape(1, 6, 5, 10)
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            scores, _ = RGBLeWM(model, audit).score(
                context, torch.tensor(normalized, dtype=torch.float32, device="cuda"))
            values = scores[0].cpu().numpy()
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("invalid rescoring")
            decisions = {method: choose(values, i) for i, method in enumerate(METHODS)}
            output[f"r{root}_actions"] = actions
            rows.append({"root": root, "row": record["row"], "scores": values.tolist(),
                         "historical_incumbent_scores": historical_scores,
                         "source_indices": indices, "decisions": decisions,
                         "action_sha256": [action_hash(a) for a in actions]})
            print(f"root={root} selected={decisions}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen model changed")
    with POOL.open("xb") as stream:
        np.savez_compressed(stream, **output)
    record = {"track": RGB, "rows": rows, "pool_sha256": file_hash(POOL),
              "old_plan_sha256": file_hash(prior.PLAN), "old_pool_sha256": file_hash(prior.POOL),
              "source_pool_sha256": file_hash(prior.base.source.POOL), "model_sha256": fingerprint,
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "extra_scored_chains_per_root": 6, "scope": "exposed development"}
    with PLAN.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("old_plan_sha256", prior.PLAN),
                      ("old_pool_sha256", prior.POOL), ("source_pool_sha256", prior.base.source.POOL),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding drift: {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(prior.base.source.block.prior.common.DATA, "r") as data):
            for rec in plan["rows"]:
                actions = pool[f"r{rec['root']}_actions"]
                outcomes = []
                for idx, action in enumerate(actions):
                    if action_hash(action) != rec["action_sha256"][idx]:
                        raise ValueError("action identity mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][rec["row"]])
                    actor._set_goal_state(data["proprio"][rec["row"] + 25])
                    trace = []
                    for value in action:
                        _, _, term, trunc, info = actor.step(value.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes.append({"trace": trace, "ever": any(x["terminated"] for x in trace),
                                     "final": trace[-1]["terminated"]})
                rows.append({"root": rec["root"], "outcomes": outcomes})
                print(f"root={rec['root']} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)},
                  stream, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, physical_data = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    original = json.loads(prior.PHYSICAL.read_text())
    if (physical_data["plan_sha256"] != file_hash(PLAN) or physical_data["pool_sha256"] != file_hash(POOL)
            or original["plan_sha256"] != file_hash(prior.PLAN)
            or original["pool_sha256"] != file_hash(prior.POOL)):
        raise ValueError("physical source integrity failure")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)):
        raise ValueError("unexpected panel")
    summary = {m: {"old": 0, "new": 0, "wins": 0, "losses": 0, "changed": 0}
               for m in METHODS}
    baselines = {"block_cem": 0, "block_random": 0}
    max_trace_delta = 0.0
    for rec, actual, old in zip(plan["rows"], physical_data["rows"], original["rows"], strict=True):
        if rec["root"] != actual["root"] or rec["root"] != old["root"]:
            raise ValueError("root mismatch")
        outcomes = actual["outcomes"]
        for i, method in enumerate(METHODS):
            previous = old["outcomes"][method]
            replay = outcomes[i]
            if (replay["ever"] != previous["ever"] or replay["final"] != previous["final"]
                    or len(replay["trace"]) != len(previous["trace"])):
                raise ValueError("old replay outcome mismatch")
            for x, y in zip(replay["trace"], previous["trace"], strict=True):
                delta = abs(x["distance"] - y["distance"])
                max_trace_delta = max(max_trace_delta, delta)
                if delta > 1e-6 or x["terminated"] != y["terminated"] or x["truncated"] != y["truncated"]:
                    raise ValueError("old replay trajectory mismatch")
            selected = rec["decisions"][method]
            new = outcomes[selected]
            value = summary[method]
            value["old"] += replay["ever"]
            value["new"] += new["ever"]
            value["wins"] += new["ever"] and not replay["ever"]
            value["losses"] += replay["ever"] and not new["ever"]
            value["changed"] += selected != i
        baselines["block_cem"] += outcomes[4]["ever"]
        baselines["block_random"] += outcomes[5]["ever"]
    paired = {}
    for method in ("vector", "shuffled"):
        wins = losses = 0
        for rec, actual in zip(plan["rows"], physical_data["rows"], strict=True):
            m = actual["outcomes"][rec["decisions"][method]]["ever"]
            baseline = actual["outcomes"][rec["decisions"]["lookahead"]]["ever"]
            wins += m and not baseline
            losses += baseline and not m
        paired[method] = {"wins_vs_lookahead": wins, "losses_vs_lookahead": losses}
    output = {"summary": summary, "incumbent_success": baselines, "paired": paired,
              "max_trace_distance_error": max_trace_delta, "plan_sha256": file_hash(PLAN),
              "physical_sha256": file_hash(PHYSICAL), "scope": "exposed development, not confirmation"}
    with RESULT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
