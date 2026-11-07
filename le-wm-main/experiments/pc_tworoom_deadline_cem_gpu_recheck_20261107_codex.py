"""Post-hoc GPU score recomputation for three deadline-CEM roots."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_deadline_cem_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = run.ROOT.parent / "docs/PC_TWOROOM_DEADLINE_CEM_GPU_RECHECK_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_tworoom_deadline_cem_gpu_recheck_20261107_codex.json"
ROOTS = (14, 22, 44)


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(run.OUT.read_text())
    for key, path in (("protocol", run.PROTOCOL), ("runner", run.__file__), ("pool", run.POOL)):
        if source[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    model, scaler, transform, _ = run.prior.prior.source.block.prior.common.model_setup()
    if model_fingerprint(model) != source["model"]:
        raise ValueError("model mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, run.OUT, run.POOL,
                                      run.prior.prior.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler", clock="remaining five-step packets")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(run.POOL, allow_pickle=False) as pool,
              np.load(run.prior.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.prior.prior.source.block.prior.common.DATA, "r") as db, torch.inference_mode()):
            for root in ROOTS:
                row = next(x for x in source["rows"] if x["root"] == root)
                for arm in run.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(db["proprio"][row["row"]])
                    actor._set_goal_state(db["proprio"][row["row"] + 25])
                    goal = transform(inputs["goal"][root])[None, None, None].cuda()
                    decision_results, step = [], 0
                    for dec in row["arms"][arm]["decisions"]:
                        ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        for receipt in dec["receipts"]:
                            if (receipt["inputs"]["pixels"] != tensor_receipt(ctx["pixels"])
                                    or receipt["inputs"]["goal"] != tensor_receipt(goal)):
                                raise ValueError("observation drift")
                        normal = pool[dec["pool_key"]+"_normal"]
                        old_cost = pool[dec["pool_key"]+"_costs"]
                        groups = ((0, 1), (1, 5), (5, 9), (9, 13)) if arm == "cem13" else ((0, 100), (100, 200), (200, 300))
                        remaining = dec["remaining_steps"]
                        boundary = RGBLeWM(model, audit)
                        new_cost = []
                        for begin, end in groups:
                            for pos in range(begin, end, 32):
                                chunk = torch.as_tensor(normal[pos:min(pos+32, end), :remaining].reshape(1, -1, remaining//5, 10),
                                                        dtype=torch.float32, device="cuda")
                                value, _ = boundary.score(ctx, chunk)
                                new_cost.extend(value[0].cpu().tolist())
                        new_cost = np.asarray(new_cost)
                        if new_cost.shape != old_cost.shape or not np.isfinite(new_cost).all():
                            raise ValueError("invalid rescored costs")
                        elite_count = 2 if arm == "cem13" else 10
                        elite_groups = groups[1:] if arm == "cem13" else groups
                        elites_equal = [bool(np.array_equal(np.argsort(old_cost[a:b], kind="stable")[:elite_count],
                                                           np.argsort(new_cost[a:b], kind="stable")[:elite_count]))
                                        for a, b in elite_groups]
                        decision_results.append({"step": step, "max_abs_difference": float(np.max(np.abs(new_cost-old_cost))),
                                                 "winner_matches": bool(np.argmin(new_cost) == dec["chosen"]),
                                                 "elites_match": elites_equal, "count": len(new_cost)})
                        for action in np.asarray(dec["dispatched"], np.float32):
                            _, _, term, trunc, info = actor.step(action.copy())
                            expected = row["arms"][arm]["trace"][step]
                            if ((bool(term), bool(trunc)) != (expected["terminated"], expected["truncated"])
                                    or abs(float(info["distance_to_target"])-expected["distance"]) > 1e-6):
                                raise ValueError("physical replay drift")
                            step += 1
                    rows.append({"root": root, "arm": arm, "decisions": decision_results})
                print(f"rescored root={root}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != source["model"]:
        raise ValueError("weights changed")
    with OUT.open("x") as stream:
        json.dump({"protocol": file_hash(PROTOCOL), "runner": file_hash(__file__), "source": file_hash(run.OUT),
                   "pool": file_hash(run.POOL), "rows": rows,
                   "scope": "post-hoc same model and environment; not independent confirmation"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
