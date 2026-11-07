"""Controlled RGB initialization factorial with fixed actions and physical replay."""
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_INITIAL_RGB_FACTORIAL_40DEV_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_initial_rgb_factorial_40dev_20261107_codex"
POOL = Path(str(PREFIX) + "_pool.npz")
PLAN = Path(str(PREFIX) + "_plan.json")
PHYSICAL = Path(str(PREFIX) + "_physical.json")
RESULT = Path(str(PREFIX) + "_result.json")
ARMS = ("data_data", "render_data", "data_render", "render_render")


def pixel_stats(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("invalid pixels")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max": int(d.max()), "changed_pixels": int(np.any(d != 0, axis=-1).sum()),
            "mean": float(d.mean())}


def select(scores):
    scores = np.asarray(scores)
    if scores.shape != (32,) or not np.isfinite(scores).all():
        raise ValueError("invalid scores")
    order = np.argsort(scores, kind="stable")
    return int(order[0]), float(scores[order[1]] - scores[order[0]])


def seal():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if POOL.exists() or PLAN.exists():
        raise FileExistsError("output exists")
    old = json.loads(source.base.source.PLAN.read_text())
    if old["pool_sha256"] != file_hash(source.base.source.POOL):
        raise ValueError("old pool changed")
    model, scaler, transform, normalizer = source.base.source.block.prior.common.model_setup()
    model_hash = source.model_fingerprint(model)
    if model_hash != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or scaler changed")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.base.source.PLAN,
                         source.base.source.POOL, source.base.source.block.prior.inputs.INPUT,
                         source.base.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays, rows = {}, []
    try:
        with (np.load(source.base.source.POOL, allow_pickle=False) as old_pool,
              np.load(source.base.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.base.source.block.prior.common.DATA, "r") as data,
              torch.inference_mode()):
            for root in range(10, 50):
                task = int(inputs["rows"][root])
                actions, identities = [], []
                for bank in ("block_cem", "block_random"):
                    indices = np.argsort(old_pool[f"r{root}_{bank}_scores"], kind="stable")[:16]
                    actions.extend(old_pool[f"r{root}_{bank}_actions"][indices])
                    identities.extend({"bank": bank, "index": int(i)} for i in indices)
                actions = np.asarray(actions, dtype=np.float32)
                if actions.shape != (32, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise ValueError("invalid actions")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task+25])
                # The environment exposes position tensors, not a full state getter.
                state_before = actor.agent_position.detach().cpu().numpy().copy()
                target_before = actor.target_position.detach().cpu().numpy().copy()
                render_current = actor.render().copy()
                render_goal = actor._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
                np.testing.assert_array_equal(state_before, actor.agent_position.detach().cpu().numpy())
                np.testing.assert_array_equal(target_before, actor.target_position.detach().cpu().numpy())
                current, goal = inputs["current"][root], inputs["goal"][root]
                image_set = {"data": current, "render": render_current}
                goal_set = {"data": goal, "render": render_goal}
                norm = scaler.transform(actions.reshape(-1, 2)).reshape(1, 32, 5, 10)
                batch = torch.tensor(norm, dtype=torch.float32, device="cuda")
                scores, choices, margins = {}, {}, {}
                for arm in ARMS:
                    current_key, goal_key = arm.split("_")
                    ctx = {"pixels": transform(image_set[current_key])[None, None, None].cuda(),
                           "goal": transform(goal_set[goal_key])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    cost, _ = RGBLeWM(model, audit).score(ctx, batch)
                    scores[arm] = cost[0].cpu().numpy()
                    choices[arm], margins[arm] = select(scores[arm])
                arrays[f"r{root}_actions"] = actions
                arrays[f"r{root}_scores"] = np.stack([scores[x] for x in ARMS])
                arrays[f"r{root}_images"] = np.stack([current, render_current, goal, render_goal])
                rows.append({"root": root, "row": task, "identities": identities, "choices": choices,
                             "margins": margins, "current_pixel_diff": pixel_stats(current, render_current),
                             "goal_pixel_diff": pixel_stats(goal, render_goal),
                             "choice_sha256": {arm: hashlib.sha256(actions[idx].tobytes()).hexdigest()
                                               for arm, idx in choices.items()},
                             "candidate_scores": 128})
                print(f"root={root} scored", flush=True)
    finally:
        env.close()
    if source.model_fingerprint(model) != model_hash:
        raise ValueError("weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "pool_sha256": file_hash(POOL), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "source_plan_sha256": file_hash(source.base.source.PLAN),
                   "source_pool_sha256": file_hash(source.base.source.POOL),
                   "model_sha256": model_hash, "normalizer_sha256": normalizer,
                   "scope": "exposed fixed-candidate diagnostic, goal-render arms oracle-only"},
                  f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    for key, path in (("pool_sha256", POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("source_plan_sha256", source.base.source.PLAN),
                      ("source_pool_sha256", source.base.source.POOL)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch {key}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(source.base.source.block.prior.common.DATA, "r") as data):
            for item in plan["rows"]:
                root, task = item["root"], item["row"]
                outputs = {}
                for arm in ARMS:
                    actions = pool[f"r{root}_actions"][item["choices"][arm]]
                    if hashlib.sha256(actions.tobytes()).hexdigest() != item["choice_sha256"][arm]:
                        raise ValueError("action hash mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task+25])
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outputs[arm] = {"trace": trace, "ever": any(x["terminated"] for x in trace),
                                    "final": trace[-1]["terminated"]}
                rows.append({"root": root, "row": task, "outcomes": outputs})
                print(f"root={root} replayed", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN)}, f, separators=(",", ":"), allow_nan=False)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, physics = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if physics["plan_sha256"] != file_hash(PLAN) or [r["root"] for r in physics["rows"]] != list(range(10, 50)):
        raise ValueError("physical mismatch")
    summary = {}
    for arm in ARMS:
        vals = [r["outcomes"][arm] for r in physics["rows"]]
        summary[arm] = {"ever": sum(v["ever"] for v in vals), "final": sum(v["final"] for v in vals),
                        "changed_vs_data": sum(r["choices"][arm] != r["choices"]["data_data"] for r in plan["rows"]),
                        "wins_vs_data": sum(r["outcomes"][arm]["ever"] and not r["outcomes"]["data_data"]["ever"] for r in physics["rows"]),
                        "losses_vs_data": sum(not r["outcomes"][arm]["ever"] and r["outcomes"]["data_data"]["ever"] for r in physics["rows"]),
                        "near_ties": sum(r["margins"][arm] < 0.01 for r in plan["rows"])}
    with np.load(POOL, allow_pickle=False) as data:
        scores = np.stack([data[f"r{r}_scores"] for r in range(10, 50)])
        maxdiff = {arm: float(np.abs(scores[:, idx] - scores[:, 0]).max()) for idx, arm in enumerate(ARMS)}
    result = {"summary": summary, "max_abs_score_diff": maxdiff,
              "max_current_pixel_diff": max(r["current_pixel_diff"]["max"] for r in plan["rows"]),
              "max_goal_pixel_diff": max(r["goal_pixel_diff"]["max"] for r in plan["rows"]),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL),
              "scope": "exposed fixed candidate diagnostic; rendered goal is evaluation-only"}
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "physical", "summarize"):
        raise SystemExit("expected seal|physical|summarize")
    {"seal": seal, "physical": physical, "summarize": summarize}[sys.argv[1]]()
