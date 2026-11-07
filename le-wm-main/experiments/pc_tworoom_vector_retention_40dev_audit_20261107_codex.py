"""GPU score and physical replay for all paired-discordant development roots."""
import json
from pathlib import Path

import h5py
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as exp
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(exp.PLAN.read_text())
    physical = json.loads(exp.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    if (result["plan_sha256"] != file_hash(exp.PLAN)
            or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or plan["pool_sha256"] != file_hash(exp.POOL)):
        raise ValueError("artifact mismatch")
    roots = [r["root"] for r in physical["rows"]
             if r["outcomes"]["vector"]["ever"] != r["outcomes"]["lookahead"]["ever"]]
    model, scaler, transform, normalizer = exp.base.source.block.prior.common.model_setup()
    if exp.model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, exp.PLAN, exp.POOL, exp.PROTOCOL],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    checked = []
    try:
        with (np.load(exp.POOL, allow_pickle=False) as pool,
              np.load(exp.base.source.POOL, allow_pickle=False) as old,
              np.load(exp.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(exp.base.source.block.prior.common.DATA, "r") as data,
              torch.inference_mode()):
            for root in roots:
                task, truth = plan["rows"][root - 10], physical["rows"][root - 10]
                candidates = exp.base.select_prefixes(old[f"r{root}_block_cem_actions"],
                                                     old[f"r{root}_block_cem_scores"],
                                                     old[f"r{root}_block_random_actions"])
                a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
                ids = pool[f"r{root}_b_indices"]
                ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                wrapper = RGBLeWM(model, audit)

                def score(chains):
                    raw = np.asarray(chains).reshape(-1, 25, 2)
                    norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 25, 2)
                    out = []
                    for offset in range(0, len(raw), 32):
                        action = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                              dtype=torch.float32, device="cuda")
                        value, _ = wrapper.score(ctx, action)
                        out.extend(value[0].cpu().tolist())
                    return np.asarray(out)

                pp = score(candidates)
                aa = score(a).reshape(8, 8)
                bb = score(b.reshape(64, 25, 2)[ids])
                errors = [float(np.abs(pp - pool[f"r{root}_parent"]).max()),
                          float(np.abs(aa - pool[f"r{root}_a_cost"]).max()),
                          float(np.abs(bb - pool[f"r{root}_b_cost"]).max())]
                if max(errors) > 1e-3:
                    raise ValueError(f"GPU score mismatch root={root} {errors}")
                retained = exp.rule.retention(aa, pp, 29114000 + root)
                for method in exp.METHODS:
                    choice = task["decisions"][method]
                    bank, index = exp.select_from_union(aa, ids, bb, retained[method])
                    if (retained[method] != choice["retained"] or
                            (bank, index) != (choice["bank"], choice["index"])):
                        raise ValueError("selection mismatch")
                for method in ("lookahead", "vector"):
                    choice = task["decisions"][method]
                    chain = pool[f"r{root}_{choice['bank']}"].reshape(64, 25, 2)[choice["index"]]
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    if trace != truth["outcomes"][method]["trace"]:
                        raise ValueError("physical replay mismatch")
                checked.append({"root": root, "max_score_error": max(errors)})
                print(f"root={root} verified", flush=True)
    finally:
        env.close()
    if exp.model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with OUTPUT.open("x") as f:
        json.dump({"roots": checked, "result_sha256": file_hash(exp.RESULT),
                   "scope": "same model, same simulator; discordant roots only"}, f, indent=2)


if __name__ == "__main__":
    main()
