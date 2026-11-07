"""Controlled 50-task RGB development comparison of action parameterizations."""
import json
import sys
import time

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_block_action_coverage_20261107_codex as block

ROOT = block.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_BLOCK_CEM_50DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_block_cem_50dev_20261107_codex"
POOL = PREFIX.with_name(PREFIX.name + "_pool.npz")
PLAN = PREFIX.with_name(PREFIX.name + "_plan.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ARMS = ("full_cem", "block_random", "block_cem")


def normalized_cem_candidates(rng, mean, std, low, high, *, blockwise):
    size = (100, 5, 2) if blockwise else (100, 25, 2)
    samples = np.clip(mean + rng.normal(size=size) * std, low, high)
    samples[0] = mean
    return samples


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed plan or pool already exists")
    old = json.loads(block.PLAN.read_text())
    previous = json.loads(block.prior.PLAN.read_text())
    model, scaler, transform, action_hash = block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or action_hash != old["action_scaler_sha256"]:
        raise ValueError("model or action normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, block.prior.inputs.INPUT,
                                      block.prior.common.CHECKPOINT / "weights.pt",
                                      block.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets, each five environment steps")
    raw_low = np.asarray((-1 - scaler.mean_) / scaler.scale_)
    raw_high = np.asarray((1 - scaler.mean_) / scaler.scale_)
    normalized_zero = scaler.transform(np.zeros((1, 2)))[0]
    arrays, rows = {}, []
    with np.load(block.prior.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        if len(data["rows"]) != 50 or len(set(map(int, data["rows"]))) != 50:
            raise ValueError("expected fixed 50 distinct data rows")
        for root in range(50):
            wrapper = RGBLeWM(model, audit)
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            outcomes = {}
            for arm in ARMS:
                torch.cuda.synchronize()
                started = time.perf_counter()
                seed = (2718000 + root if arm == "full_cem" else
                        27071100 + root if arm == "block_random" else 28071100 + root)
                rng = np.random.default_rng(seed)
                candidates, scores = [], []
                shape = (25, 2) if arm == "full_cem" else (5, 2)
                mean = np.broadcast_to(normalized_zero, shape).copy()
                std = np.ones(shape)
                for iteration in range(3):
                    if arm == "block_random":
                        if iteration == 0:
                            raw_random = block.block_actions(seed)
                        raw = raw_random[100 * iteration:100 * (iteration + 1)]
                        expanded = scaler.transform(raw.reshape(-1, 2)).reshape(100, 25, 2)
                    else:
                        samples = normalized_cem_candidates(rng, mean, std, raw_low, raw_high,
                                                            blockwise=arm == "block_cem")
                        expanded = np.repeat(samples, 5, axis=1) if arm == "block_cem" else samples
                        raw = block.prior.to_physical(expanded, scaler).astype(np.float32)
                    np.testing.assert_allclose(block.prior.to_physical(expanded, scaler), raw,
                                               atol=1e-6, rtol=0)
                    values = []
                    for offset in range(0, 100, 32):
                        chunk = torch.as_tensor(expanded[offset:offset + 32].reshape(1, -1, 5, 10),
                                                dtype=torch.float32, device="cuda")
                        value, _ = wrapper.score(context, chunk)
                        values.extend(value[0].cpu().tolist())
                    if len(values) != 100 or not np.isfinite(values).all():
                        raise ValueError("invalid model scores")
                    candidates.extend(raw.tolist())
                    scores.extend(values)
                    if arm != "block_random":
                        elite = samples[np.argsort(values, kind="stable")[:10]]
                        mean = elite.mean(axis=0)
                        std = np.maximum(elite.std(axis=0), 1e-5)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                candidates = np.asarray(candidates, dtype=np.float32)
                scores = np.asarray(scores, dtype=np.float64)
                if candidates.shape != (300, 25, 2) or scores.shape != (300,):
                    raise ValueError("candidate count drift")
                choice = int(np.argmin(scores))
                if root < 10 and arm == "block_random":
                    np.testing.assert_array_equal(candidates, np.asarray(old["rows"][root]["actions"], np.float32))
                    if choice != old["rows"][root]["choice"]:
                        raise ValueError("prior block choice mismatch")
                if root < 10 and arm == "full_cem":
                    if choice != int(np.argmin(previous["rows"][root]["generation_costs"])):
                        raise ValueError("prior CEM choice mismatch")
                    old_action = block.prior.to_physical(
                        np.asarray(previous["rows"][root]["generation_actions"][choice]), scaler).astype(np.float32)
                    np.testing.assert_allclose(candidates[choice], old_action, atol=1e-6, rtol=0)
                key = f"r{root}_{arm}"
                arrays[key + "_actions"] = candidates
                arrays[key + "_scores"] = scores
                outcomes[arm] = {"seed": seed, "choice": choice, "cost": float(scores[choice]),
                                 "model_candidates": 300, "seconds": elapsed}
            rows.append({"root": root, "row": int(data["rows"][root]), "outcomes": outcomes})
            print(f"root={root + 1}/50 row={rows[-1]['row']} scored=900", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    record = {"track": RGB, "rows": rows, "model_sha256": fingerprint,
              "normalizer_sha256": action_hash, "input_sha256": file_hash(block.prior.inputs.INPUT),
              "prior_block_sha256": file_hash(block.PLAN), "prior_cem_sha256": file_hash(block.prior.PLAN),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "pool_sha256": file_hash(POOL), "scope": "exposed development only"}
    with PLAN.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    record = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("input_sha256", block.prior.inputs.INPUT), ("pool_sha256", POOL)):
        if record[key] != file_hash(path):
            raise ValueError(f"{key} drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (h5py.File(block.prior.common.DATA, "r") as data,
              np.load(POOL, allow_pickle=False) as pool):
            for row in record["rows"]:
                outcomes = {}
                for arm in ARMS:
                    actions = pool[f"r{row['root']}_{arm}_actions"][row["outcomes"][arm]["choice"]]
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("invalid physical actions")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[arm] = {"ever": any(t["terminated"] for t in trace),
                                     "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": row["root"], "row": row["row"], "outcomes": outcomes})
                print(f"root={row['root'] + 1}/50 physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)},
                  f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_record, physical_record = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if physical_record["plan_sha256"] != file_hash(PLAN) or physical_record["pool_sha256"] != file_hash(POOL):
        raise ValueError("physical evidence drift")
    rows = physical_record["rows"]
    if len(rows) != 50 or [r["root"] for r in rows] != list(range(50)):
        raise ValueError("incomplete tasks")
    summaries = {}
    for subset, items in (("all_50", rows), ("last_40", rows[10:])):
        summaries[subset] = {}
        for arm in ARMS:
            wins = sum(r["outcomes"][arm]["ever"] and not r["outcomes"]["full_cem"]["ever"] for r in items)
            losses = sum(not r["outcomes"][arm]["ever"] and r["outcomes"]["full_cem"]["ever"] for r in items)
            summaries[subset][arm] = {"ever": sum(r["outcomes"][arm]["ever"] for r in items),
                                      "final": sum(r["outcomes"][arm]["final"] for r in items),
                                      "wins": wins, "losses": losses}
    exits = {arm: (summaries["last_40"][arm]["wins"] >= 4
                   and summaries["last_40"][arm]["losses"] <= 2)
             for arm in ("block_random", "block_cem")}
    timings = {arm: float(np.mean([r["outcomes"][arm]["seconds"] for r in plan_record["rows"]]))
               for arm in ARMS}
    result = {"summary": summaries, "development_exit": exits, "mean_planning_seconds": timings,
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "scope": "exposed RGB development only; no descriptor or independent gate claim"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "summarize"):
        raise SystemExit("expected plan|physical|summarize")
    {"plan": plan, "physical": physical, "summarize": summarize}[sys.argv[1]]()
