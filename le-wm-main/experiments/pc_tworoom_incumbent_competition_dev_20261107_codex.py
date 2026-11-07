"""Re-score prior selected and source-incumbent chains before physical replay."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as prev
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

ROOT = prev.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_INCUMBENT_COMPETITION_DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_incumbent_competition_dev_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")
NAMES = prev.METHODS + ("block_cem", "block_random")


def select(scores, arm):
    v = np.asarray(scores, dtype=float)
    if v.shape != (6,) or not np.isfinite(v).all() or not 0 <= arm < 4:
        raise ValueError("invalid scores or arm")
    options = (arm, 4, 5)
    return options[int(np.argmin(v[list(options)]))]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def seal():
    if POOL.exists() or PLAN.exists():
        raise FileExistsError("plan exists")
    before = json.loads(prev.PLAN.read_text())
    original = json.loads(prev.base.source.PLAN.read_text())
    for expected, path in ((before["pool_sha256"], prev.POOL),
                           (before["source_plan_sha256"], prev.base.source.PLAN),
                           (before["source_pool_sha256"], prev.base.source.POOL),
                           (before["input_sha256"], prev.base.source.block.prior.inputs.INPUT),
                           (original["pool_sha256"], prev.base.source.POOL)):
        if expected != file_hash(path):
            raise ValueError("source binding mismatch")
    model, scaler, transform, normalizer = prev.base.source.block.prior.common.model_setup()
    if prev.model_fingerprint(model) != before["model_sha256"] or normalizer != before["normalizer_sha256"]:
        raise ValueError("model/normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prev.PLAN, prev.POOL,
                         prev.base.source.PLAN, prev.base.source.POOL,
                         prev.base.source.block.prior.inputs.INPUT,
                         prev.base.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    arrays, rows = {}, []
    with (np.load(prev.POOL, allow_pickle=False) as recent,
          np.load(prev.base.source.POOL, allow_pickle=False) as old,
          np.load(prev.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for item in before["rows"]:
            root = item["root"]
            candidates, old_scores, sources = [], [], []
            for name in prev.METHODS:
                decision = item["decisions"][name]
                bank, idx = decision["bank"], decision["index"]
                chain = recent[f"r{root}_{bank}"].reshape(64, 25, 2)[idx]
                if digest(chain) != decision["action_sha256"]:
                    raise ValueError("previous action hash mismatch")
                if bank == "a":
                    score = recent[f"r{root}_a_cost"].reshape(-1)[idx]
                else:
                    ids = recent[f"r{root}_b_indices"]
                    pos = int(np.searchsorted(ids, idx))
                    if pos >= len(ids) or ids[pos] != idx:
                        raise ValueError("missing selected B score")
                    score = recent[f"r{root}_b_cost"][pos]
                candidates.append(chain)
                old_scores.append(float(score))
                sources.append({"kind": "previous_arm", "name": name, "index": idx, "bank": bank})
            for name in ("block_cem", "block_random"):
                scores = old[f"r{root}_{name}_scores"]
                idx = int(np.argmin(scores))
                if original["rows"][root]["outcomes"][name]["choice"] != idx:
                    raise ValueError("source incumbent choice mismatch")
                candidates.append(old[f"r{root}_{name}_actions"][idx])
                old_scores.append(float(scores[idx]))
                sources.append({"kind": "original_arm", "name": name, "index": idx})
            candidates = np.asarray(candidates, dtype=np.float32)
            if candidates.shape != (6, 25, 2) or not np.isfinite(candidates).all() or np.abs(candidates).max() > 1:
                raise ValueError("illegal candidates")
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            norm = scaler.transform(candidates.reshape(-1, 2)).reshape(1, 6, 5, 10)
            value, _ = RGBLeWM(model, audit).score(ctx, torch.tensor(norm, dtype=torch.float32, device="cuda"))
            scores = value[0].cpu().numpy()
            choices = {name: select(scores, i) for i, name in enumerate(prev.METHODS)}
            arrays[f"r{root}_actions"] = candidates
            rows.append({"root": root, "row": item["row"], "sources": sources,
                         "action_sha256": [digest(a) for a in candidates],
                         "old_scores": old_scores, "scores": scores.tolist(), "choices": choices})
            print(f"root={root} six candidates rescored", flush=True)
    if prev.model_fingerprint(model) != before["model_sha256"]:
        raise ValueError("model weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "track": RGB, "pool_sha256": file_hash(POOL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "previous_plan_sha256": file_hash(prev.PLAN), "previous_pool_sha256": file_hash(prev.POOL),
                   "original_plan_sha256": file_hash(prev.base.source.PLAN),
                   "original_pool_sha256": file_hash(prev.base.source.POOL),
                   "model_sha256": before["model_sha256"], "normalizer_sha256": normalizer,
                   "scope": "posthoc exposed development"}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("previous_plan_sha256", prev.PLAN),
                      ("previous_pool_sha256", prev.POOL), ("original_plan_sha256", prev.base.source.PLAN),
                      ("original_pool_sha256", prev.base.source.POOL)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    old = json.loads(prev.PHYSICAL.read_text())
    original = json.loads(prev.base.source.PHYSICAL.read_text())
    if old["plan_sha256"] != file_hash(prev.PLAN) or original["plan_sha256"] != file_hash(prev.base.source.PLAN):
        raise ValueError("physical source mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(prev.base.source.block.prior.common.DATA, "r") as data):
            for item in plan["rows"]:
                root, task = item["root"], item["row"]
                outcomes = {}
                for name in prev.METHODS:
                    idx = item["choices"][name]
                    action = pool[f"r{root}_actions"][idx]
                    if digest(action) != item["action_sha256"][idx]:
                        raise ValueError("action mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace = []
                    for value in action:
                        _, _, term, trunc, info = actor.step(value.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    src = item["sources"][idx]
                    expected = (old["rows"][root-10]["outcomes"][src["name"]]["trace"]
                                if src["kind"] == "previous_arm" else
                                original["rows"][root]["outcomes"][src["name"]]["trace"])
                    if trace != expected:
                        raise ValueError(f"replay mismatch root={root} arm={name}")
                    outcomes[name] = {"trace": trace, "ever": any(t["terminated"] for t in trace),
                                      "final": trace[-1]["terminated"], "selected": idx}
                rows.append({"root": root, "row": task, "outcomes": outcomes})
                print(f"root={root} replayed", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN),
                   "previous_physical_sha256": file_hash(prev.PHYSICAL),
                   "original_physical_sha256": file_hash(prev.base.source.PHYSICAL)},
                  f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, phys, old = map(lambda p: json.loads(p.read_text()), (PLAN, PHYSICAL, prev.PHYSICAL))
    if phys["plan_sha256"] != file_hash(PLAN) or phys["previous_physical_sha256"] != file_hash(prev.PHYSICAL):
        raise ValueError("physical binding mismatch")
    if [x["root"] for x in phys["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    output = {}
    for name in prev.METHODS:
        pairs = [(r["outcomes"][name], old["rows"][i]["outcomes"][name])
                 for i, r in enumerate(phys["rows"])]
        output[name] = {"ever": sum(x["ever"] for x, _ in pairs),
                        "final": sum(x["final"] for x, _ in pairs),
                        "wins_vs_old": sum(x["ever"] and not y["ever"] for x, y in pairs),
                        "losses_vs_old": sum(not x["ever"] and y["ever"] for x, y in pairs),
                        "incumbent_selected": sum(x["selected"] >= 4 for x, _ in pairs)}
    margins = []
    for row in plan["rows"]:
        for i, name in enumerate(prev.METHODS):
            values = np.sort(np.array(row["scores"])[[i, 4, 5]])
            margins.append(float(values[1] - values[0]))
    drift = np.abs(np.asarray([r["scores"] for r in plan["rows"]]) -
                   np.asarray([r["old_scores"] for r in plan["rows"]]))
    result = {"summary": output, "max_old_score_drift": float(drift.max()),
              "near_ties_below_0_01": sum(x < 0.01 for x in margins),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "scope": "posthoc exposed development, no independent gate evidence"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
