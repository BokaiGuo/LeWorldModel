"""Recompute every sealed equal-tail model score without physical outcome labels."""
import json

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_equal_tail_replan_20261107_codex as original
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = original.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EQUAL_TAIL_GPU_SCORE_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_equal_tail_gpu_score_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    saved = json.loads(original.PLAN.read_text())
    for key, path in (("protocol_sha256", original.PROTOCOL),
                      ("runner_sha256", original.__file__),
                      ("source_plan_sha256", original.base.PLAN),
                      ("pool_sha256", original.POOL)):
        if saved[key] != file_hash(path):
            raise ValueError(f"sealed input drift: {key}")
    base = original.base
    oldplan = json.loads(base.ledger.prior.source.PLAN.read_text())
    model, scaler, transform, normalizer = base.ledger.prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != saved["model_sha256"] or normalizer != oldplan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, original.PLAN, original.POOL],
                         preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="5 packets x 5 steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(original.POOL, allow_pickle=False) as pool,
              np.load(base.ledger.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(base.ledger.prior.source.block.prior.common.DATA, "r") as data,
              torch.inference_mode()):
            for row in saved["rows"]:
                root, task = row["root"], row["row"]
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                arms = {}
                for arm in base.ARMS:
                    chains = pool[f"r{root}_{arm}_chains"]
                    reference = pool[f"r{root}_{arm}_scores"]
                    if chains.shape != (48, 25, 2) or reference.shape != (48,):
                        raise ValueError("invalid sealed pool")
                    wrapper = RGBLeWM(model, audit)
                    recomputed = []
                    for offset in range(0, 48, 16):
                        normalized = scaler.transform(chains[offset:offset + 16].reshape(-1, 2))
                        normalized = normalized.reshape(1, 16, 5, 10)
                        score, _ = wrapper.score(ctx, torch.as_tensor(normalized, device="cuda", dtype=torch.float32))
                        recomputed.extend(score[0].cpu().tolist())
                    values = np.asarray(recomputed, dtype=np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError("non-finite recomputed score")
                    diff = np.abs(values - reference)
                    original_choice = int(np.argmin(reference))
                    new_choice = int(np.argmin(values))
                    arms[arm] = {
                        "max_abs_difference": float(diff.max()),
                        "above_1e_minus_5": int((diff > 1e-5).sum()),
                        "original_choice": original_choice,
                        "recomputed_choice": new_choice,
                        "scores": values.tolist(),
                        "receipts": wrapper.receipts,
                    }
                rows.append({"root": root, "arms": arms})
                print(f"root={root}: 240 scores recomputed", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    all_arms = [a for row in rows for a in row["arms"].values()]
    result = {
        "track": RGB, "protocol_sha256": file_hash(PROTOCOL),
        "runner_sha256": file_hash(__file__), "plan_sha256": file_hash(original.PLAN),
        "pool_sha256": file_hash(original.POOL), "model_sha256": fingerprint,
        "normalizer_sha256": normalizer, "rows": rows,
        "summary": {"scores": 48 * len(all_arms),
                    "max_abs_difference": max(a["max_abs_difference"] for a in all_arms),
                    "above_1e_minus_5": sum(a["above_1e_minus_5"] for a in all_arms),
                    "changed_choices": sum(a["original_choice"] != a["recomputed_choice"] for a in all_arms)},
        "scope": "same implementation and model, score replay only",
    }
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
