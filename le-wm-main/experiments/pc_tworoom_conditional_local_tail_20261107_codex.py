"""Fixed-budget local tail development experiment; choices sealed before replay."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_quality_gated_retention_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

ROOT = source.ROOT
PREFIX = ROOT / "results/pc_tworoom_conditional_local_tail_20261107_codex"
PLAN, POOL = Path(str(PREFIX) + "_plan.json"), Path(str(PREFIX) + "_pool.npz")
PHYSICAL, RESULT = Path(str(PREFIX) + "_physical.json"), Path(str(PREFIX) + "_result.json")
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CONDITIONAL_LOCAL_TAIL_PROTOCOL_20261107_CODEX.md"
METHODS = ("lookahead", "gated", "shuffled")


def proposals(a, costs, root):
    a, costs = np.asarray(a), np.asarray(costs)
    if a.shape != (8, 8, 25, 2) or costs.shape != (8, 8):
        raise ValueError("invalid source shapes")
    if not np.isfinite(a).all() or not np.isfinite(costs).all() or np.abs(a).max() > 1:
        raise ValueError("nonfinite or illegal source")
    for block in (10, 15, 20):
        if not np.array_equal(a[:, :, block:block+5],
                              np.broadcast_to(a[:, :, block:block+1], (8, 8, 5, 2))):
            raise ValueError("source tail is not block-constant")
    if not np.array_equal(a[:, :, :10], np.broadcast_to(a[:, :1, :10], (8, 8, 10, 2))):
        raise ValueError("inconsistent prefixes")
    winners = np.argmin(costs, axis=1)
    result = np.empty((8, 8, 25, 2), dtype=np.float32)
    for row in range(8):
        parent = a[row, winners[row]]
        rng = np.random.default_rng(20261107000 + int(root) * 100 + row)
        base = parent[[10, 15, 20]]
        perturb = rng.normal(0, 0.2, (8, 3, 2))
        tails = np.repeat(np.clip(base[None] + perturb, -1, 1).astype(np.float32), 5, axis=1)
        result[row, :, :10] = parent[:10]
        result[row, :, 10:] = tails
    return result, winners


def pick(common_cost, costs, ids, keep):
    costs, ids, keep = np.asarray(costs), np.asarray(ids), np.asarray(keep)
    if (not np.isfinite(common_cost) or costs.ndim != 1 or costs.shape != ids.shape
            or not np.isfinite(costs).all() or keep.shape != (2,)
            or len(set(keep.tolist())) != 2 or np.any(keep < 0) or np.any(keep > 7)
            or len(set(ids.tolist())) != len(ids) or np.any(np.diff(ids) <= 0)):
        raise ValueError("invalid selection inputs")
    wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
    positions = np.searchsorted(ids, wanted)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], wanted):
        raise ValueError("missing candidate scores")
    local_index = int(np.argmin(costs[positions]))
    candidate = int(wanted[local_index])
    candidate_cost = float(costs[positions[local_index]])
    return (("local", candidate, candidate_cost) if candidate_cost < common_cost
            else ("common", -1, float(common_cost)))


def seal():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("output exists")
    old = json.loads(source.PLAN.read_text())
    for key, path in (("pool_sha256", source.POOL), ("old_pool_sha256", source.old.POOL),
                      ("input_sha256", source.old.base.source.block.prior.inputs.INPUT)):
        if old[key] != file_hash(path):
            raise ValueError("source binding mismatch")
    common = source.old.base.source.block.prior.common
    input_path = source.old.base.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.PLAN, source.POOL,
                                      source.old.POOL, input_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    rows, arrays = [], {}
    with (np.load(source.POOL, allow_pickle=False) as prev,
          np.load(source.old.POOL, allow_pickle=False) as candidates,
          np.load(input_path, allow_pickle=False) as inputs, torch.inference_mode()):
        for item in old["rows"]:
            root = item["root"]
            actions = prev[f"r{root}_actions"]
            decisions = item["decisions"]
            common_action = actions[source.METHODS.index("lookahead")]
            if any(not np.array_equal(common_action, x) for x in actions):
                raise ValueError("previous arms do not share final incumbent")
            if any(decisions[k]["bank"] not in ("a", "incumbent") for k in METHODS):
                raise ValueError("previous common includes B")
            if any(source.digest(common_action) != decisions[k]["action_sha256"] for k in METHODS):
                raise ValueError("common action binding mismatch")
            a, ac = candidates[f"r{root}_a"], prev[f"r{root}_a_cost"]
            local, winners = proposals(a, ac, root)
            keep = {m: decisions[m]["retained"] for m in METHODS}
            ids = np.asarray(sorted({p * 8 + j for m in METHODS for p in keep[m] for j in range(8)}))
            raw = np.concatenate((common_action[None], local.reshape(64, 25, 2)[ids]))
            norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 5, 10)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)
            values = []
            for offset in range(0, len(raw), 32):
                scores, _ = session.score(ctx, torch.as_tensor(norm[offset:offset+32][None],
                                                               dtype=torch.float32, device="cuda"))
                values.extend(scores[0].cpu().tolist())
            common_cost, scores = values[0], np.asarray(values[1:], dtype=np.float64)
            if abs(common_cost - decisions["lookahead"]["cost"]) > 0.005:
                raise ValueError("common score drift")
            selected, selected_actions = {}, []
            for method in METHODS:
                bank, index, cost = pick(common_cost, scores, ids, keep[method])
                chain = common_action if bank == "common" else local.reshape(64, 25, 2)[index]
                selected[method] = {"bank": bank, "index": index, "cost": cost,
                                    "retained": keep[method], "action_sha256": source.digest(chain)}
                selected_actions.append(chain)
            arrays[f"r{root}_local"] = local
            arrays[f"r{root}_ids"] = ids
            arrays[f"r{root}_scores"] = scores
            arrays[f"r{root}_actions"] = np.asarray(selected_actions, dtype=np.float32)
            rows.append({"root": root, "row": item["row"], "winners": winners.tolist(),
                         "common_cost": common_cost, "decisions": selected,
                         "physical_scored_candidates": int(1 + len(ids)), "logical_per_arm": 691})
            print(f"root={root} scored local union={len(ids)}", flush=True)
    if model_fingerprint(model) != old["model_sha256"]:
        raise ValueError("model changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "pool_sha256": file_hash(POOL),
                   "source_plan_sha256": file_hash(source.PLAN), "source_pool_sha256": file_hash(source.POOL),
                   "input_sha256": file_hash(input_path), "model_sha256": old["model_sha256"],
                   "normalizer_sha256": normalizer, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__)}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("source_plan_sha256", source.PLAN),
                      ("source_pool_sha256", source.POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__)):
        if plan[key] != file_hash(path):
            raise ValueError("plan binding mismatch")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, h5py.File(source.old.base.source.block.prior.common.DATA, "r") as data:
            for row in plan["rows"]:
                root, outcomes = row["root"], {}
                for index, method in enumerate(METHODS):
                    chain = pool[f"r{root}_actions"][index]
                    if (source.digest(chain) != row["decisions"][method]["action_sha256"]
                            or not np.isfinite(chain).all() or np.max(np.abs(chain)) > 1):
                        raise ValueError("invalid selected chain")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[method] = {"ever": any(t["terminated"] for t in trace),
                                        "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": root, "outcomes": outcomes})
                print(f"root={root} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)},
                  f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, phys, old = (json.loads(p.read_text()) for p in (PLAN, PHYSICAL, source.PHYSICAL))
    if phys["plan_sha256"] != file_hash(PLAN) or phys["pool_sha256"] != file_hash(POOL):
        raise ValueError("binding mismatch")
    if old["plan_sha256"] != file_hash(source.PLAN) or old["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("old physical binding mismatch")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    if [r["root"] for r in phys["rows"]] != [r["root"] for r in plan["rows"]] or \
       [r["root"] for r in old["rows"]] != [r["root"] for r in plan["rows"]]:
        raise ValueError("root mismatch")
    summary = {}
    gated = [r["outcomes"]["gated"]["ever"] for r in phys["rows"]]
    for m in METHODS:
        current = [r["outcomes"][m]["ever"] for r in phys["rows"]]
        previous = [r["outcomes"][m]["ever"] for r in old["rows"]]
        summary[m] = {"ever": sum(current), "final": sum(r["outcomes"][m]["final"] for r in phys["rows"]),
                      "gated_wins": sum(x and not y for x, y in zip(gated, current)),
                      "gated_losses": sum(y and not x for x, y in zip(gated, current)),
                      "own_previous_wins": sum(x and not y for x, y in zip(current, previous)),
                      "own_previous_losses": sum(y and not x for x, y in zip(current, previous)),
                      "local_selected": sum(r["decisions"][m]["bank"] == "local" for r in plan["rows"])}
    passed = (all(summary[m]["gated_wins"] >= 2 and summary[m]["gated_losses"] <= 1
                  for m in ("lookahead", "shuffled"))
              and summary["gated"]["own_previous_wins"] >= 2
              and summary["gated"]["own_previous_losses"] <= 1)
    result = {"summary": summary, "development_exit": passed, "plan_sha256": file_hash(PLAN),
              "physical_sha256": file_hash(PHYSICAL), "old_physical_sha256": file_hash(source.PHYSICAL),
              "scope": "exposed development, not independent confirmation"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
