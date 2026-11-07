"""Sealed same-incumbent Adam versus random local refinement, exposed roots only."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_history_refinement_20261107_codex as ref
from experiments import pc_rgb_derivative_backend_20261107_codex as backend
from experiments.pc_input_permissions import file_hash

q, history = ref.q, ref.history
BASE = ref.ROOT / "results/pc_tworoom_joint_random_dev_20261107_codex"
PLAN, POOL, PHYSICAL = (Path(str(BASE) + s) for s in ("_plan.json", "_pool.npz", "_physical.json"))
PROTOCOL = ref.ROOT.parent / "docs/PC_TWOROOM_JOINT_RANDOM_DEV_PROTOCOL_20261107_CODEX.md"
ARMS = ("incumbent", "adam", "random")


def packets(blocks, mean, scale):
    return ((blocks[:, None, :].expand(5, 5, 2) - mean) / scale).reshape(1, 5, 10)


def score(model, initial, goal, blocks, mean, scale):
    actions = packets(blocks, mean, scale)
    trajectory = history.rollout_history(model, initial, actions[:, :0], actions)
    result = (trajectory[:, -1] - goal).square().sum()
    if not torch.isfinite(result):
        raise ValueError("nonfinite objective")
    return result


def optimize(model, initial, goal, initial_blocks, mean, scale, root):
    mean = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
    scale = torch.as_tensor(scale, dtype=torch.float32, device="cuda")
    x = torch.tensor(initial_blocks, dtype=torch.float32, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.05)
    trajectories, metrics = {}, {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    adam_actions, adam_scores, grads = [], [], []
    for i in range(9):
        opt.zero_grad(set_to_none=True)
        value = score(model, initial, goal, x, mean, scale)
        adam_actions.append(x.detach().cpu().numpy().copy())
        adam_scores.append(float(value.detach()))
        if i < 8:
            value.backward()
            if x.grad is None or not torch.isfinite(x.grad).all():
                raise ValueError("invalid action gradient")
            grads.append(x.grad.detach().cpu().numpy().copy())
            opt.step()
            with torch.no_grad():
                x.clamp_(-1, 1)
    torch.cuda.synchronize()
    metrics["adam_seconds"] = time.perf_counter() - start
    trajectories["adam_actions"] = np.asarray(adam_actions)
    trajectories["adam_scores"] = np.asarray(adam_scores)
    trajectories["adam_gradients"] = np.asarray(grads)
    rng = np.random.default_rng(491000 + root)
    random_actions = np.concatenate((initial_blocks[None],
        np.clip(initial_blocks[None] + rng.normal(0, 0.15, (8, 5, 2)), -1, 1).astype(np.float32)))
    torch.cuda.synchronize()
    start = time.perf_counter()
    random_scores = []
    with torch.no_grad():
        for a in random_actions:
            random_scores.append(float(score(model, initial, goal, torch.as_tensor(a, device="cuda"), mean, scale)))
    torch.cuda.synchronize()
    metrics["random_seconds"] = time.perf_counter() - start
    trajectories["random_actions"], trajectories["random_scores"] = random_actions, np.asarray(random_scores)
    metrics["adam_index"] = int(np.argmin(adam_scores))
    metrics["random_index"] = int(np.argmin(random_scores))
    return trajectories, metrics


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("output already exists")
    previous = json.loads(q.PLAN.read_text())
    input_path = q.old.base.source.block.prior.inputs.INPUT
    common = q.old.base.source.block.prior.common
    if previous["pool_sha256"] != file_hash(q.POOL) or previous["input_sha256"] != file_hash(input_path):
        raise ValueError("input drift")
    if [r["root"] for r in previous["rows"]] != list(range(10, 50)):
        raise ValueError("panel mismatch")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = ref.model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model drift")
    rows, arrays = [], {}
    with np.load(q.POOL, allow_pickle=False) as pool, np.load(input_path, allow_pickle=False) as data, \
            backend.precise_action_derivatives():
        for record in previous["rows"]:
            root = record["root"]
            original = pool[f"r{root}_actions"][0].copy()
            if q.digest(original) != record["decisions"]["lookahead"]["action_sha256"]:
                raise ValueError("initial action mismatch")
            blocks = ref.validate_initial(original)
            with torch.no_grad():
                initial = model.encode({"pixels": transform(data["current"][root])[None, None].cuda()})["emb"]
                goal = model.encode({"pixels": transform(data["goal"][root])[None, None].cuda()})["emb"][:, 0]
            trace, metrics = optimize(model, initial, goal, blocks, scaler.mean_, scaler.scale_, root)
            chosen = [original,
                      np.repeat(trace["adam_actions"][metrics["adam_index"]], 5, axis=0),
                      np.repeat(trace["random_actions"][metrics["random_index"]], 5, axis=0)]
            arrays[f"r{root}_chosen"] = np.stack(chosen)
            for k, v in trace.items():
                arrays[f"r{root}_{k}"] = v
            rows.append({"root": root, "row": record["row"], "metrics": metrics,
                         "action_sha256": {arm: q.digest(a) for arm, a in zip(ARMS, chosen)}})
            print(f"root={root} planned", flush=True)
    if ref.model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    sources = [Path(__file__), PROTOCOL, Path(history.__file__), Path(backend.__file__), q.PLAN, q.POOL,
               input_path, common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"]
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": "rgb_only", "rows": rows, "pool_sha256": file_hash(POOL),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "source_sha256": {str(p): file_hash(p) for p in sources}}, f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan_data = json.loads(PLAN.read_text())
    if file_hash(POOL) != plan_data["pool_sha256"]:
        raise ValueError("pool drift")
    for name, digest in plan_data["source_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"source drift: {name}")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, h5py.File(q.old.base.source.block.prior.common.DATA, "r") as data:
            for item in plan_data["rows"]:
                outcomes = {}
                for arm, action in zip(ARMS, pool[f"r{item['root']}_chosen"]):
                    if q.digest(action) != item["action_sha256"][arm]:
                        raise ValueError("action drift")
                    ref.validate_initial(action)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][item["row"]])
                    actor._set_goal_state(data["proprio"][item["row"] + 25])
                    trace = []
                    for u in action:
                        _, _, term, trunc, info = actor.step(u.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[arm] = {"ever": any(t["terminated"] for t in trace),
                                     "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": item["root"], "outcomes": outcomes})
                print(f"root={item['root']} evaluated", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL), "rows": rows},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
