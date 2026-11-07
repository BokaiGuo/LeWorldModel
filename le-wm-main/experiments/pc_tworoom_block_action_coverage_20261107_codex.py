"""RGB-only fixed-budget block-action generation, then separate physical audit."""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_BLOCK_ACTION_COVERAGE_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_block_action_coverage_20261107_codex"
PLAN = PREFIX.with_name(PREFIX.name + "_plan.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
ORACLE_ROWS = (62735, 86996, 168250)


def block_actions(seed, count=300):
    blocks = np.random.default_rng(seed).uniform(-1, 1, (count, 5, 2)).astype(np.float32)
    return np.repeat(blocks, 5, axis=1)


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    old = json.loads(prior.PLAN.read_text())
    if old["input_sha256"] != file_hash(prior.inputs.INPUT):
        raise ValueError("prior RGB input drift")
    model, scaler, transform, action_hash = prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or action_hash != old["normalizer_action_sha256"]:
        raise ValueError("model or action normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.inputs.INPUT,
                                      prior.common.CHECKPOINT / "weights.pt",
                                      prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="5 packets of 5 environment steps")
    rows = []
    with np.load(prior.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        for root, previous in enumerate(old["rows"]):
            if previous["row"] != int(data["rows"][root]):
                raise ValueError("root order mismatch")
            wrapper = RGBLeWM(model, audit)
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            raw = block_actions(27071100 + root)
            normalized = scaler.transform(raw.reshape(-1, 2)).reshape(300, 25, 2)
            np.testing.assert_allclose(prior.to_physical(normalized, scaler), raw, atol=1e-6, rtol=0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            scores = []
            for offset in range(0, 300, 32):
                chunk = torch.tensor(normalized[offset:offset + 32].reshape(1, -1, 5, 10),
                                     device="cuda", dtype=torch.float32)
                value, _ = wrapper.score(context, chunk)
                scores.extend(value[0].cpu().tolist())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if len(scores) != 300 or not np.isfinite(scores).all():
                raise ValueError("invalid candidate costs")
            old_index = int(np.argmin(previous["generation_costs"]))
            old_action = prior.to_physical(np.asarray(previous["generation_actions"][old_index]), scaler)
            rows.append({"root": root, "row": previous["row"], "seed": 27071100 + root,
                         "actions": raw.tolist(), "scores": scores, "choice": int(np.argmin(scores)),
                         "baseline_actions": old_action.tolist(), "baseline_index": old_index,
                         "baseline_cost": previous["generation_costs"][old_index],
                         "model_candidates": 300, "selection_seconds": elapsed})
            print(f"root={root} row={previous['row']} scored=300", flush=True)
    if model_fingerprint(model) != fingerprint or len(rows) != 10:
        raise ValueError("weights or panel changed")
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "model_sha256": fingerprint,
                   "input_sha256": file_hash(prior.inputs.INPUT), "source_sha256": file_hash(prior.PLAN),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "action_scaler_sha256": action_hash, "scope": "exposed RGB development"},
                  f, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    record = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("source_sha256", prior.PLAN), ("input_sha256", prior.inputs.INPUT)):
        if record[key] != file_hash(path):
            raise ValueError(f"{key} drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(prior.common.DATA, "r") as data:
            def run(row, actions):
                actions = np.asarray(actions, dtype=np.float32)
                if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                    raise ValueError("illegal actions")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                trace = []
                for action in actions:
                    _, _, terminated, truncated, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(terminated), "truncated": bool(truncated)})
                return {"action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                        "ever_success": any(x["terminated"] for x in trace),
                        "final_success": trace[-1]["terminated"], "trace": trace}

            for row in record["rows"]:
                block = np.asarray(row["actions"], dtype=np.float32)
                selected = run(row["row"], block[row["choice"]])
                baseline = run(row["row"], row["baseline_actions"])
                oracle = []
                if row["row"] in ORACLE_ROWS:
                    for i, actions in enumerate(block):
                        item = run(row["row"], actions)
                        item["candidate_index"] = i
                        oracle.append(item)
                    if selected != {k: v for k, v in oracle[row["choice"]].items() if k != "candidate_index"}:
                        raise ValueError("selected candidate replay mismatch")
                rows.append({"root": row["root"], "row": row["row"],
                             "selected": selected, "baseline": baseline, "all_candidates": oracle})
                print(f"row={row['row']} selected={selected['ever_success']} "
                      f"baseline={baseline['ever_success']} pool_success={sum(x['ever_success'] for x in oracle)}",
                      flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN),
                   "scope": "physical truth evaluation only"}, f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    physical_record = json.loads(PHYSICAL.read_text())
    if physical_record["plan_sha256"] != file_hash(PLAN):
        raise ValueError("physical plan mismatch")
    rows = physical_record["rows"]
    if len(rows) != 10 or {r["row"] for r in rows if r["all_candidates"]} != set(ORACLE_ROWS):
        raise ValueError("incomplete evaluation")
    wins = sum(r["selected"]["ever_success"] and not r["baseline"]["ever_success"] for r in rows)
    losses = sum(not r["selected"]["ever_success"] and r["baseline"]["ever_success"] for r in rows)
    pool_roots = sum(any(x["ever_success"] for x in r["all_candidates"]) for r in rows)
    summary = {"selected_success": sum(r["selected"]["ever_success"] for r in rows),
               "baseline_success": sum(r["baseline"]["ever_success"] for r in rows),
               "wins": wins, "losses": losses, "pool_success_roots": pool_roots,
               "passes_development_exit": pool_roots >= 2 and wins >= 1 and losses <= 1}
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(PHYSICAL)}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "summarize"):
        raise SystemExit("expected plan|physical|summarize")
    {"plan": plan, "physical": physical, "summarize": summarize}[sys.argv[1]]()
