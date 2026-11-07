"""Independent arithmetic reconstruction and same-backend GPU/physical replay."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_local_perturb_stability_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_LOCAL_PERTURB_INDEPENDENT_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_local_perturb_independent_replay_20261107_codex.json"


def reconstruct(parents, seed):
    parents = np.asarray(parents, np.float32)
    blocks = parents[:, 10::5, :]
    noise = np.random.default_rng(seed).normal(0, 0.05, (8, 4, 3, 2))
    child = np.empty((8, 4, 25, 2), np.float32)
    child[:, :, :10] = parents[:, None, :10]
    child[:, :, 10:] = np.repeat(np.clip(blocks[:, None] + noise, -1, 1).astype(np.float32), 5, axis=2)
    return child


def recompute_choices(old, new, mapping):
    parent = np.argsort(old, kind="stable")[:8]
    direct = int(np.argmin(np.r_[old, new.ravel()]))
    robust = int(parent[np.argmin((old[parent] + np.sum(new, axis=1)) / 5)])
    shuffled = int(parent[np.argmin((old[parent] + np.sum(new[mapping], axis=1)) / 5)])
    return {"parents": parent.tolist(), "direct": direct, "robust": robust, "shuffled": shuffled}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.OUT.read_text())
    for key, path in (("source_plan_sha256", source.source.PLAN), ("source_pool_sha256", source.source.POOL),
                      ("pool_sha256", source.POOL), ("protocol_sha256", source.PROTOCOL),
                      ("runner_sha256", source.__file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"source binding mismatch: {key}")
    if physical["plan_sha256"] != file_hash(source.PLAN) or physical["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("physical binding mismatch")
    common = source.source.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.PLAN, source.POOL, source.OUT,
                         source.source.POOL, source.source.source.block.prior.inputs.INPUT,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="TwoRoom RGB 224 and action scaler", clock="five packets of five steps")
    wrapper = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.source.POOL, allow_pickle=False) as old_pool,
              np.load(source.POOL, allow_pickle=False) as new_pool,
              np.load(source.source.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for row, original in zip(plan["rows"], physical["rows"], strict=True):
                root = row["root"]
                if root != original["root"]:
                    raise ValueError("root alignment")
                base = old_pool[f"r{root}_b_actions"].reshape(64, 25, 2)
                children = new_pool[f"r{root}_children"]
                if not np.array_equal(reconstruct(base[row["parents"]], 29151000 + root), children):
                    raise ValueError("perturbation reconstruction mismatch")
                actions = np.concatenate((base, children.reshape(32, 25, 2)))
                current = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                normalized = scaler.transform(actions.reshape(-1, 2)).reshape(96, 5, 10)
                costs = []
                for start in range(0, 96, 32):
                    x = torch.as_tensor(normalized[None, start:start+32], dtype=torch.float32, device="cuda")
                    scores, _ = wrapper.score(current, x)
                    costs.extend(scores[0].cpu().tolist())
                costs = np.asarray(costs)
                previous = np.concatenate((old_pool[f"r{root}_b_cost"].ravel(),
                                           new_pool[f"r{root}_child_cost"].ravel()))
                chosen = recompute_choices(costs[:64], costs[64:].reshape(8, 4), np.asarray(row["mapping"]))
                matches = {key: chosen[key] == row[key] for key in ("parents", "direct", "robust", "shuffled")}
                replay = None
                if row["direct"] >= 64:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    offset = row["row"]
                    actor._set_state(data["proprio"][offset])
                    actor._set_goal_state(data["proprio"][offset + 25])
                    trace = []
                    for action in actions[row["direct"]]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        if trunc:
                            raise ValueError("unexpected truncation")
                        trace.append((float(info["distance_to_target"]), bool(term)))
                    prior = original["arms"]["direct"]["trace"]
                    replay = {"max_distance_difference": max(abs(x[0] - y["distance"]) for x, y in zip(trace, prior, strict=True)),
                              "success_mismatches": sum(x[1] != y["success"] for x, y in zip(trace, prior, strict=True))}
                rows.append({"root": root, "max_score_difference": float(np.max(np.abs(costs - previous))),
                             "choices_match": matches, "recomputed_choices": chosen, "physical_replay": replay})
                print(f"root={root} rescored; choices={all(matches.values())}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model drift")
    result = {"rows": rows, "plan_sha256": file_hash(source.PLAN), "physical_sha256": file_hash(source.OUT),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "model_sha256": plan["model_sha256"], "rescored_chains": 960,
              "physical_replays": sum(r["physical_replay"] is not None for r in rows),
              "scope": "same backend and exposed tasks; not independent confirmation"}
    atomic_write_json(OUT, result)
    print(json.dumps({"max_score_difference": max(r["max_score_difference"] for r in rows),
                      "all_choices_match": all(all(r["choices_match"].values()) for r in rows),
                      "physical_replays": result["physical_replays"],
                      "max_distance_difference": max(r["physical_replay"]["max_distance_difference"] for r in rows if r["physical_replay"])}, indent=2))


if __name__ == "__main__":
    main()
