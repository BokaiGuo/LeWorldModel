"""Rescore and replay every vector-vs-CEM discordant root."""
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

from experiments import pc_tworoom_vector_budget290_dev_20261107_codex as exp
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical = json.loads(exp.PLAN.read_text()), json.loads(exp.PHYSICAL.read_text())
    baseline = json.loads(exp.prev.base.source.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    if (result["plan_sha256"] != file_hash(exp.PLAN) or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or plan["pool_sha256"] != file_hash(exp.POOL) or
            result["baseline_physical_sha256"] != file_hash(exp.prev.base.source.PHYSICAL)):
        raise ValueError("source drift")
    roots = [r["root"] for r in physical["rows"]
             if r["outcomes"]["vector"]["ever"] != baseline["rows"][r["root"]]["outcomes"]["block_cem"]["ever"]]
    model, scaler, transform, normalizer = exp.prev.base.source.block.prior.common.model_setup()
    if exp.prev.model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, exp.PLAN, exp.POOL, exp.PROTOCOL],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    checked = []
    try:
        with (np.load(exp.POOL, allow_pickle=False) as pool,
              np.load(exp.prev.base.source.POOL, allow_pickle=False) as old,
              np.load(exp.prev.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(exp.prev.base.source.block.prior.common.DATA, "r") as data,
              torch.inference_mode()):
            for root in roots:
                row = plan["rows"][root-10]
                cem = old[f"r{root}_block_cem_actions"][:100]
                rand = old[f"r{root}_block_random_actions"][:100]
                parents = exp.prev.base.select_prefixes(cem, old[f"r{root}_block_cem_scores"][:100], rand)
                inc = pool[f"r{root}_inc"]
                a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
                ids = pool[f"r{root}_b_indices"]
                ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                wrapper = RGBLeWM(model, audit)

                def score(raw):
                    raw = np.asarray(raw).reshape(-1, 25, 2)
                    norm = scaler.transform(raw.reshape(-1, 2)).reshape(-1, 25, 2)
                    values = []
                    for offset in range(0, len(raw), 32):
                        batch = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                             dtype=torch.float32, device="cuda")
                        costs, _ = wrapper.score(ctx, batch)
                        values.extend(costs[0].cpu().tolist())
                    return np.asarray(values)

                for label, chains in (("block_cem", cem), ("block_random", rand)):
                    values = score(chains)
                    original = old[f"r{root}_{label}_scores"][:100]
                    if int(np.argmin(values)) != int(np.argmin(original)):
                        raise ValueError(f"source incumbent ranking changed {root} {label}")
                p = score(np.concatenate((parents, inc)))
                ac = score(a).reshape(8, 8)
                bc = score(b.reshape(64, 25, 2)[ids])
                errors = [np.abs(p - pool[f"r{root}_parent_inc_cost"]).max(),
                          np.abs(ac - pool[f"r{root}_a_cost"]).max(),
                          np.abs(bc - pool[f"r{root}_b_cost"]).max()]
                if max(errors) > 0.001:
                    raise ValueError(f"score mismatch {root} {errors}")
                retained = exp.prev.rule.retention(ac, p[:8], 29114000 + root)
                for arm in exp.prev.METHODS:
                    decision = row["decisions"][arm]
                    bank, idx = exp.choose(ac, p[8:], ids, bc, retained[arm])
                    if retained[arm] != decision["retained"] or (bank, idx) != (decision["bank"], decision["index"]):
                        raise ValueError("selection mismatch")
                decision = row["decisions"]["vector"]
                actions = pool[f"r{root}_{decision['bank']}"].reshape(-1, 25, 2)[decision["index"]]
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row["row"]])
                actor._set_goal_state(data["proprio"][row["row"] + 25])
                trace = []
                for action in actions:
                    _, _, term, trunc, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                if trace != physical["rows"][root-10]["outcomes"]["vector"]["trace"]:
                    raise ValueError("physical trace mismatch")
                checked.append({"root": root, "max_new_score_error": float(max(errors))})
                print(f"root={root} verified", flush=True)
    finally:
        env.close()
    if exp.prev.model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with OUTPUT.open("x") as f:
        json.dump({"roots": checked, "result_sha256": file_hash(exp.RESULT),
                   "scope": "same model and simulator, discordant roots only"}, f, indent=2)


if __name__ == "__main__":
    main()
