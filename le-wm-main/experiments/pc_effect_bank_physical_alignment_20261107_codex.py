"""Two-stage frozen RGB scoring and evaluator-only physical replay."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as prior
from experiments.pc_executable_effect_bank_20261107_codex import effect_bank
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

BASE = prior.ROOT / "results/pc_effect_bank_physical_alignment_20261107_codex"
SCORES, SEAL, PHYS, RESULT = [Path(str(BASE) + x) for x in
                              ("_scores.npz", "_seal.json", "_physical.npz", "_result.json")]
PROTOCOL = prior.ROOT.parent / "docs/PC_EFFECT_BANK_PHYSICAL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
ROOTS = list(range(10, 50))


def roots_and_chains(pool, root):
    actions = pool[f"r{root}_a"]
    if actions.shape != (8, 8, 25, 2):
        raise ValueError("invalid source action bank")
    prefixes, probes = actions[:, 0, :10], actions[0, :, 10:]
    rebuilt = np.concatenate((np.broadcast_to(prefixes[:, None], (8, 8, 10, 2)),
                              np.broadcast_to(probes[None], (8, 8, 15, 2))), axis=2)
    if not np.array_equal(rebuilt, actions):
        raise ValueError("prefix/probe identity mismatch")
    return prefixes, probes


def score():
    if SCORES.exists() or SEAL.exists():
        raise FileExistsError("score artifact exists")
    original = json.loads(prior.PLAN.read_text())
    inp = prior.base.source.block.prior.inputs.INPUT
    common = prior.base.source.block.prior.common
    if original["pool_sha256"] != file_hash(prior.POOL) or original["input_sha256"] != file_hash(inp):
        raise ValueError("source mismatch")
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.PLAN, prior.POOL, inp,
                                      common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    arrays, errors = {}, []
    with (np.load(prior.POOL, allow_pickle=False) as pool, np.load(inp, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in ROOTS:
            prefixes, probes = roots_and_chains(pool, root)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)

            def evaluate(chains):
                norm = scaler.transform(chains.reshape(-1, 2)).reshape(-1, 5, 10)
                out = []
                for start in range(0, len(norm), 32):
                    cost, _ = session.score(context, torch.as_tensor(
                        norm[start:start+32][None], dtype=torch.float32, device="cuda"))
                    out.extend(cost[0].cpu().tolist())
                return np.asarray(out)

            result = effect_bank(prefixes, probes, np.zeros((15, 2), dtype=np.float32), evaluate)
            err = float(np.max(np.abs(result["probe_costs"] - pool[f"r{root}_a_cost"])))
            if err > 0.005:
                raise ValueError("probe score drift")
            errors.append(err)
            arrays[f"r{root}_effect"] = result["effect"]
            arrays[f"r{root}_ref"] = result["reference_cost"]
            arrays[f"r{root}_probe"] = result["probe_costs"]
            print(f"root={root} scores sealed in memory", flush=True)
    if model_fingerprint(model) != original["model_sha256"]:
        raise ValueError("model changed")
    with SCORES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths = {"scores": SCORES, "source_plan": prior.PLAN, "source_pool": prior.POOL,
             "input": inp, "data": common.DATA, "runner": Path(__file__), "protocol": PROTOCOL,
             "effect_component": Path(effect_bank.__code__.co_filename)}
    with SEAL.open("x") as f:
        json.dump({"track": RGB, "roots": ROOTS, "sha256": {k: file_hash(v) for k, v in paths.items()},
                   "model_sha256": original["model_sha256"], "normalizer_sha256": normalizer,
                   "max_probe_score_error": max(errors), "full_chain_scores_per_root": 72},
                  f, indent=2, allow_nan=False)


def verify():
    seal = json.loads(SEAL.read_text())
    common = prior.base.source.block.prior.common
    paths = {"scores": SCORES, "source_plan": prior.PLAN, "source_pool": prior.POOL,
             "input": prior.base.source.block.prior.inputs.INPUT, "data": common.DATA,
             "runner": Path(__file__), "protocol": PROTOCOL,
             "effect_component": Path(effect_bank.__code__.co_filename)}
    if seal["roots"] != ROOTS or any(seal["sha256"][k] != file_hash(p) for k, p in paths.items()):
        raise ValueError("sealed data drift")


def physical():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    verify()
    if PHYS.exists():
        raise FileExistsError(PHYS)
    plan = json.loads(prior.PLAN.read_text())
    rows = {r["root"]: r["row"] for r in plan["rows"]}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays = {}
    try:
        with (np.load(prior.POOL, allow_pickle=False) as pool,
              h5py.File(prior.base.source.block.prior.common.DATA, "r") as data):
            for root in ROOTS:
                prefixes, probes = roots_and_chains(pool, root)
                tails = np.concatenate((np.zeros((1, 15, 2), dtype=np.float32), probes))
                distance = np.empty((8, 9, 25), dtype=np.float64)
                terminal = np.empty((8, 9, 25), dtype=bool)
                displacement = np.empty(8, dtype=np.float64)
                for i, prefix in enumerate(prefixes):
                    for j, tail in enumerate(tails):
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][rows[root]])
                        actor._set_goal_state(data["proprio"][rows[root] + 25])
                        at_prefix = None
                        for t, action in enumerate(np.concatenate((prefix, tail))):
                            _, _, term, _, info = actor.step(action.copy())
                            distance[i, j, t] = info["distance_to_target"]
                            terminal[i, j, t] = term
                            if t == 9:
                                at_prefix = actor.agent_position.detach().cpu().numpy().copy()
                        if j == 0:
                            end = actor.agent_position.detach().cpu().numpy()
                            displacement[i] = np.linalg.norm(end - at_prefix)
                arrays[f"r{root}_distance"], arrays[f"r{root}_term"] = distance, terminal
                arrays[f"r{root}_zero_displacement"] = displacement
                print(f"root={root} physical=72", flush=True)
    finally:
        env.close()
    with PHYS.open("xb") as f:
        np.savez_compressed(f, **arrays)


def sign_stats(predicted, actual, eligible):
    if predicted.shape != (8, 8) or actual.shape != (8, 8) or eligible.shape != (8,):
        raise ValueError("invalid shapes")
    mask = (np.abs(predicted) > 1e-6) & (np.abs(actual) > 1e-6) & eligible[:, None]
    return int(np.sum(mask & (np.sign(predicted) == np.sign(actual)))), int(mask.sum())


def summarize():
    verify()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    rows = []
    with np.load(SCORES, allow_pickle=False) as scores, np.load(PHYS, allow_pickle=False) as physical:
        for root in ROOTS:
            distance, term = physical[f"r{root}_distance"], physical[f"r{root}_term"]
            if distance.shape != (8, 9, 25) or term.shape != (8, 9, 25):
                raise ValueError("physical shape mismatch")
            if not np.array_equal(term[:, :, :10], np.broadcast_to(term[:, :1, :10], (8, 9, 10))):
                raise ValueError("prefix physical mismatch")
            actual = distance[:, :1, -1] - distance[:, 1:, -1]
            predicted = scores[f"r{root}_effect"]
            all_count = sign_stats(predicted, actual, np.ones(8, dtype=bool))
            non_success = sign_stats(predicted, actual, ~term[:, 0, :10].any(axis=1))
            rows.append({"root": root, "all": all_count, "prefix_unsuccessful": non_success,
                         "zero_max_displacement": float(physical[f"r{root}_zero_displacement"].max())})
    summary = {}
    for key in ("all", "prefix_unsuccessful"):
        valid = [r[key] for r in rows if r[key][1]]
        summary[key] = {"eligible_roots": len(valid), "matching": sum(v[0] for v in valid),
                        "pairs": sum(v[1] for v in valid),
                        "macro_accuracy": float(np.mean([a / b for a, b in valid])) if valid else None}
    summary["zero_max_displacement"] = max(r["zero_max_displacement"] for r in rows)
    summary["development_exit"] = all(summary[k]["eligible_roots"] >= 30 and
                                       summary[k]["macro_accuracy"] >= .65
                                       for k in ("all", "prefix_unsuccessful"))
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "rows": rows, "seal_sha256": file_hash(SEAL),
                   "physical_sha256": file_hash(PHYS), "scope": "exposed development, not PC1 pass"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "physical", "summarize"):
        raise SystemExit("expected score|physical|summarize")
    {"score": score, "physical": physical, "summarize": summarize}[sys.argv[1]]()
