"""Fixed-pool test of goal distance projected onto model action-effect directions."""
import json
import sys
import time

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_ACTION_SENSITIVE_METRIC_DEV_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_action_sensitive_metric_plan_20261107_codex.json"
FEATURES = source.ROOT / "results/pc_tworoom_action_sensitive_metric_features_20261107_codex.npz"
PHYSICAL = source.ROOT / "results/pc_tworoom_action_sensitive_metric_physical_20261107_codex.json"
ARMS = ("original", "action_sensitive", "random_subspace")


def action_subspace(cross, rank=8):
    cross = np.asarray(cross, dtype=np.float64)
    if cross.ndim != 3 or cross.shape[:2] != (8, 8) or not np.isfinite(cross).all():
        raise ValueError("expected finite 8x8 action-effect embeddings")
    centered = (cross - cross.mean(axis=1, keepdims=True)).reshape(64, -1)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    valid = bool(len(singular) >= rank and singular[0] > 0 and singular[rank - 1] > singular[0] * 1e-6)
    return (right[:rank] if valid else None), singular


def run_plan():
    if OUTPUT.exists() or FEATURES.exists():
        raise FileExistsError("output exists")
    old = json.loads(source.PLAN.read_text())
    if (old["runner_sha256"] != file_hash(source.__file__) or old["input_sha256"] != file_hash(source.inputs.INPUT)):
        raise ValueError("source plan drift")
    model, scaler, transform, action_hash = source.common.model_setup()
    model_id = model_fingerprint(model)
    if model_id != old["model_sha256"] or action_hash != old["normalizer_action_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [source.__file__, PROTOCOL, source.inputs.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="5 packets, 5 steps")
    rows, features = [], {}
    started = time.perf_counter()
    torch.cuda.synchronize()
    with np.load(source.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        for root, row in enumerate(old["rows"]):
            if int(data["rows"][root]) != row["row"]:
                raise ValueError("task mismatch")
            initial = candidates_for(row)
            chains = [item[1] for item in initial]
            recorded = [item[2] for item in initial]
            for ref in row["refinements"]:
                chains.extend(ref["actions"])
                recorded.extend(ref["costs"])
            chains = np.asarray(chains, dtype=np.float32)
            recorded = np.asarray(recorded)
            if chains.shape != (656, 25, 2) or recorded.shape != (656,):
                raise ValueError("candidate pool mismatch")
            source.to_physical(chains, scaler)
            wrapper = RGBLeWM(model, audit)
            pixels = transform(data["current"][root])[None, None, None].cuda()
            goal = transform(data["goal"][root])[None, None, None].cuda()
            forecast, model_cost = [], []
            for offset in range(0, len(chains), 32):
                actions = torch.tensor(chains[offset:offset + 32].reshape(1, -1, 5, 10),
                                       dtype=torch.float32, device="cuda")
                values, context = wrapper.score({"pixels": pixels, "goal": goal,
                                                 "action": torch.empty((1, 1, 0, 10), device="cuda")}, actions)
                forecast.append(context["predicted_emb"][0, :, -1].cpu().numpy())
                model_cost.extend(values[0].cpu().numpy().tolist())
            forecast, model_cost = np.concatenate(forecast), np.asarray(model_cost)
            discrepancy = float(np.max(np.abs(model_cost - recorded)))
            if discrepancy > 1e-3:
                raise ValueError(f"saved cost mismatch: {discrepancy}")
            goal_latent = model.encode({"pixels": goal[:, 0]})["emb"][0, 0].cpu().numpy()
            difference = forecast.astype(np.float64) - goal_latent.astype(np.float64)
            full_cost = np.square(difference).sum(axis=1)
            if np.max(np.abs(full_cost - model_cost)) > 1e-3:
                raise ValueError("model cost is not endpoint squared distance")
            basis, singular = action_subspace(forecast[304:368].reshape(8, 8, -1))
            rng = np.random.default_rng(2719200 + root)
            random_basis, _ = np.linalg.qr(rng.normal(size=(forecast.shape[1], 8)), mode="reduced")
            scores = {"original": full_cost,
                      "action_sensitive": full_cost if basis is None else np.square(difference @ basis.T).sum(axis=1),
                      "random_subspace": np.square(difference @ random_basis).sum(axis=1)}
            outcome = {}
            for arm in ARMS:
                index = int(np.argmin(scores[arm]))
                outcome[arm] = {"candidate_index": index, "score": float(scores[arm][index]),
                                "original_cost": float(full_cost[index]),
                                "actions": source.to_physical(chains[index], scaler).tolist()}
            rows.append({"root": root, "row": row["row"], "rank_valid": basis is not None,
                         "singular_values": singular.tolist(), "outcomes": outcome,
                         "max_saved_cost_error": discrepancy, "scored_candidates": 656})
            features[f"root_{root}_predicted"] = forecast
            features[f"root_{root}_goal"] = goal_latent
            features[f"root_{root}_action_basis"] = np.empty((0, forecast.shape[1])) if basis is None else basis
            features[f"root_{root}_random_basis"] = random_basis
            features[f"root_{root}_scores"] = np.stack([scores[arm] for arm in ARMS])
            print(f"{root + 1}/10 row={row['row']} rank_valid={basis is not None}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != model_id:
        raise ValueError("model weights changed")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "features_sha256": file_hash(FEATURES), "source_sha256": file_hash(source.PLAN),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "model_sha256": model_id, "normalizer_action_sha256": action_hash, "elapsed_seconds": elapsed,
                   "scope": "post-hoc exposed RGB fixed pool; not a deployment planner"},
                  stream, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(OUTPUT.read_text())
    if (plan["runner_sha256"] != file_hash(__file__) or plan["features_sha256"] != file_hash(FEATURES)
            or plan["protocol_sha256"] != file_hash(PROTOCOL) or plan["source_sha256"] != file_hash(source.PLAN)):
        raise ValueError("sealed plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(source.common.DATA, "r") as data:
            for row in plan["rows"]:
                outcomes = {}
                for arm in ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    actions = np.asarray(row["outcomes"][arm]["actions"], dtype=np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("illegal action")
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[arm] = {"trace": trace, "ever_success": any(x["terminated"] for x in trace),
                                     "final_distance": trace[-1]["distance"]}
                rows.append({"row": row["row"], "outcomes": outcomes})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(OUTPUT), "runner_sha256": file_hash(__file__)},
                  stream, indent=2, allow_nan=False)
    for arm in ARMS:
        print(f"{arm}: {sum(row['outcomes'][arm]['ever_success'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": run_plan, "physical": physical}[sys.argv[1]]()
