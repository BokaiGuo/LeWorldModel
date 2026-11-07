"""Three-root real-model engineering smoke; no scientific performance claim."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as source
from experiments import pc_fixed_prefix_compiler_20261107_codex as compiler
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUT = source.ROOT / "results/pc_fixed_prefix_compiler_smoke_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    if file_hash(source.POOL) != plan["pool_sha256"]:
        raise ValueError("source candidate pool changed")
    model, scaler, transform, normalizer = source.block.prior.common.model_setup()
    before = model_fingerprint(model)
    if before != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    deps = [__file__, compiler.__file__, source.PLAN, source.POOL,
            source.block.prior.inputs.INPUT, source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", deps, preprocessing="ImageNet 224 and TwoRoom action scaler",
                         clock="five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    records = []
    try:
        with (np.load(source.POOL, allow_pickle=False) as pool,
              np.load(source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in (10, 11, 12):
                parents = pool[f"r{root}_block_cem_actions"][:3]
                rng = np.random.default_rng(202611070 + root)
                probes = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
                tails = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
                ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                boundary = RGBLeWM(model, audit)
                result = compiler.compile_fixed_prefix(
                    boundary, ctx, parents, probes, tails, scaler=scaler,
                    prefix_steps=10, retain_count=1, strategy="lookahead", max_queries=11, seed=42)
                actions = result["actions"]
                norm = scaler.transform(actions).reshape(1, 1, 5, 10)
                check, _ = boundary.score(ctx, torch.as_tensor(norm, dtype=torch.float32, device="cuda"))
                check_cost = float(check.item())
                if abs(check_cost - result["choice"].cost) > 1e-3:
                    raise ValueError("selected cost not reproduced")
                row = int(inputs["rows"][root])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                trace = []
                for action in actions:
                    _, _, term, trunc, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                records.append({"root": root, "row": row, "actions": actions.tolist(),
                                "choice": {"source": result["choice"].source, "index": result["choice"].index,
                                           "cost": result["choice"].cost},
                                "retained_prefixes": result["retained_prefixes"],
                                "logical_model_queries": result["model_queries"],
                                "verification_extra_queries": 1, "receipts": boundary.receipts,
                                "ever": any(t["terminated"] for t in trace), "trace": trace})
    finally:
        env.close()
    if model_fingerprint(model) != before:
        raise ValueError("model changed")
    output = {"track": RGB, "audit_hash": audit.fingerprint, "source_plan_sha256": file_hash(source.PLAN),
              "compiler_sha256": file_hash(compiler.__file__), "rows": records,
              "scope": "engineering smoke only, exposed tasks, no advantage or gate claim"}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps({"roots": len(records), "ever": sum(r["ever"] for r in records),
                      "track": RGB, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
