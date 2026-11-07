"""Sealed RGB response predictions followed by evaluator-only physical replay."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_RGB_DIRECTIONAL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
PLAN = source.ROOT / "results/pc_tworoom_rgb_directional_alignment_plan_20261107_codex.json"
PHYSICAL = source.ROOT / "results/pc_tworoom_rgb_directional_alignment_physical_20261107_codex.json"
SCALES = (.002, .02, .2)


def directions():
    bank = np.zeros((4, 25, 2), dtype=np.float32)
    for index, (begin, axis) in enumerate(((0, 0), (0, 1), (20, 0), (20, 1))):
        bank[index, begin:begin + 5, axis] = 1.
    return bank


def finite_difference(positive, negative, scale):
    values = np.asarray([positive, negative, scale], float)
    if not np.isfinite(values).all() or scale <= 0:
        raise ValueError("finite costs and positive scale required")
    return float((positive - negative) / (2 * scale))


def plan():
    if PLAN.exists() or PHYSICAL.exists():
        raise FileExistsError("output exists")
    old = json.loads(source.PLAN.read_text())
    if old["input_sha256"] != file_hash(source.inputs.INPUT):
        raise ValueError("input drift")
    model, scaler, transform, action_hash = source.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or action_hash != old["normalizer_action_sha256"]:
        raise ValueError("model or normalizer drift")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.inputs.INPUT, source.PLAN,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; action scaler", clock="five five-step packets")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1., physical_high=1.)
    bank = directions()
    configs = [(d, h, sign) for d in range(4) for h in SCALES for sign in (1, -1)]
    raw = np.stack([bank[d] * h * sign for d, h, sign in configs])
    normalized = scaler.transform(raw.reshape(-1, 2)).reshape(1, -1, 5, 10)
    np.testing.assert_allclose(scaler.inverse_transform(normalized.reshape(-1, 2)),
                               raw.reshape(-1, 2), atol=1e-6, rtol=0)
    center = scaler.transform(np.zeros((25, 2))).reshape(1, 1, 5, 10)
    rows = []
    with np.load(source.inputs.INPUT, allow_pickle=False) as data:
        for row in old["rows"]:
            idx = int(np.flatnonzero(data["rows"] == row["row"])[0])
            context = {"pixels": transform(data["current"][idx])[None, None, None].cuda(),
                       "goal": transform(data["goal"][idx])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            x = torch.tensor(center, device="cuda", dtype=torch.float32, requires_grad=True)
            value, _ = guard.score(context, x)
            gradient, = torch.autograd.grad(value.sum(), x)
            grad_raw = gradient.detach().cpu().numpy().reshape(25, 2) / scaler.scale_
            ad = np.sum(bank * grad_raw[None], axis=(1, 2))
            with torch.inference_mode():
                candidate = torch.tensor(normalized, device="cuda", dtype=torch.float32)
                cost, context_out = guard.score(context, candidate)
                predicted = context_out["predicted_emb"][0, :, -1].cpu().numpy()
            vals = cost[0].cpu().numpy().reshape(4, 3, 2)
            fd = [[finite_difference(vals[d, h, 0], vals[d, h, 1], SCALES[h])
                   for h in range(3)] for d in range(4)]
            rows.append({"root": row["root"], "row": row["row"], "center_cost": float(value.item()),
                         "ad": ad.tolist(), "fd": fd, "predicted_costs": vals.tolist(),
                         "predicted_features": predicted.tolist()})
            print(f"root={row['root']} AD/FD computed", flush=True)
    guard.verify_unchanged()
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "configs": configs, "raw_actions": raw.tolist(), "source_plan": file_hash(source.PLAN),
                   "input_sha256": file_hash(source.inputs.INPUT), "model_sha256": old["model_sha256"],
                   "normalizer_sha256": action_hash, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__),
                   "scope": "RGB model-only sealed directional predictions on exposed tasks"}, f, allow_nan=False)


def physical():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sealed = json.loads(PLAN.read_text())
    if (sealed["source_plan"] != file_hash(source.PLAN) or sealed["input_sha256"] != file_hash(source.inputs.INPUT)
            or sealed["protocol_sha256"] != file_hash(PROTOCOL) or sealed["runner_sha256"] != file_hash(__file__)):
        raise ValueError("sealed plan drift")
    model, scaler, transform, action_hash = source.common.model_setup()
    if model_fingerprint(model) != sealed["model_sha256"] or action_hash != sealed["normalizer_sha256"]:
        raise ValueError("model drift")
    indices = [i for i, (_, scale, _) in enumerate(sealed["configs"]) if scale in (.02, .2)]
    raw = np.asarray(sealed["raw_actions"], dtype=np.float32)
    expected = np.stack([directions()[d] * h * sign for d, h, sign in sealed["configs"]])
    np.testing.assert_array_equal(raw, expected)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(source.common.DATA, "r") as dataset, \
                np.load(source.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
            for row in sealed["rows"]:
                frames, physical = [], []
                for i in indices:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(dataset["proprio"][row["row"]])
                    actor._set_goal_state(dataset["proprio"][row["row"] + 25])
                    trace = []
                    for action in raw[i]:
                        _, _, terminated, truncated, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                    frames.append(actor.render().copy())
                    physical.append(trace)
                idx = int(np.flatnonzero(data["rows"] == row["row"])[0])
                goal = transform(data["goal"][idx])[None, None].cuda()
                target = model.encode({"pixels": goal})["emb"][0, 0]
                images = torch.stack([transform(frame) for frame in frames])[:, None].cuda()
                observed = model.encode({"pixels": images})["emb"][:, 0]
                actual_cost = (observed - target).square().sum(-1)
                rows.append({"root": row["root"], "row": row["row"], "indices": indices,
                             "true_features": observed.cpu().tolist(), "target_feature": target.cpu().tolist(),
                             "encoded_costs": actual_cost.cpu().tolist(), "traces": physical})
                print(f"root={row['root']} 16 physical chains replayed", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != sealed["model_sha256"]:
        raise ValueError("model changed")
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "rows": rows,
                   "scope": "evaluation-only physical successors; not independent task evidence"}, f, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
