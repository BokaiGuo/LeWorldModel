"""Exposed-root joint vs split gradient refinement and sealed physical replay."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_quality_gated_retention_20261107_codex as q
from experiments import pc_latent_history_rollout_20261107_codex as history
from experiments.pc_input_permissions import RGB, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives

ROOT = q.ROOT
BASE = ROOT / "results/pc_tworoom_history_refinement_20261107_codex"
PLAN, POOL, PHYSICAL = (Path(str(BASE) + suffix) for suffix in ("_plan.json", "_pool.npz", "_physical.json"))
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_HISTORY_REFINEMENT_PROTOCOL_20261107_CODEX.md"
ROOTS = (10, 11, 38, 39)
ARMS = ("incumbent", "joint_full", "joint_split", "fixed_prefix")


def validate_initial(actions):
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (25, 2) or not np.isfinite(array).all() or np.max(np.abs(array)) > 1:
        raise ValueError("invalid initial action")
    blocks = array.reshape(5, 5, 2)
    if not np.array_equal(blocks, np.broadcast_to(blocks[:, :1], blocks.shape)):
        raise ValueError("expected constant action packets")
    return blocks[:, 0].copy()


def refine(model, initial, goal, raw_blocks, mean, scale, mode):
    if mode not in ARMS[1:]:
        raise ValueError("unknown mode")
    params = torch.tensor(raw_blocks, dtype=torch.float32, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.05)
    mean = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
    scale = torch.as_tensor(scale, dtype=torch.float32, device="cuda")
    baseline = params.detach().clone()
    best_cost, best_action, observations = float("inf"), None, []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for iteration in range(9):
        optimizer.zero_grad(set_to_none=True)
        raw = params[:, None, :].expand(5, 5, 2)
        packets = ((raw - mean) / scale).reshape(1, 5, 10)
        if mode == "joint_full":
            trajectory = history.rollout_history(model, initial, packets[:, :0], packets)
        else:
            prefix = history.rollout_history(model, initial, packets[:, :0], packets[:, :2])
            trajectory = history.rollout_history(model, prefix, packets[:, :2], packets[:, 2:])
        cost = (trajectory[:, -1] - goal).square().sum()
        if not torch.isfinite(cost):
            raise ValueError("nonfinite model score")
        value = float(cost.detach())
        if value < best_cost:
            best_cost, best_action = value, params.detach().cpu().numpy().copy()
        if iteration == 8:
            observations.append({"iteration": iteration, "cost": value})
            break
        cost.backward()
        grad = params.grad
        if grad is None or not torch.isfinite(grad).all():
            raise ValueError("missing or nonfinite action gradient")
        if mode == "fixed_prefix":
            grad[:2].zero_()
        observations.append({"iteration": iteration, "cost": value,
                             "gradient": grad.detach().cpu().tolist()})
        optimizer.step()
        with torch.no_grad():
            params.clamp_(-1, 1)
            if mode == "fixed_prefix" and not torch.equal(params[:2], baseline[:2]):
                raise ValueError("fixed prefix changed")
    torch.cuda.synchronize()
    result = np.repeat(best_action, 5, axis=0).astype(np.float32)
    validate_initial(result)
    return result, {"best_cost": best_cost, "iterations": observations,
                    "rollouts": 9 if mode == "joint_full" else 18, "backward_calls": 8,
                    "planning_seconds": time.perf_counter() - start}


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("plan outputs exist")
    previous = json.loads(q.PLAN.read_text())
    if previous["pool_sha256"] != file_hash(q.POOL):
        raise ValueError("source pool drift")
    common = q.old.base.source.block.prior.common
    input_path = q.old.base.source.block.prior.inputs.INPUT
    if previous["input_sha256"] != file_hash(input_path):
        raise ValueError("RGB inputs drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    rows, arrays = [], {}
    with np.load(q.POOL, allow_pickle=False) as pool, np.load(input_path, allow_pickle=False) as data, \
            precise_action_derivatives():
        for root in ROOTS:
            record = next(r for r in previous["rows"] if r["root"] == root)
            original = pool[f"r{root}_actions"][0].copy()
            if q.digest(original) != record["decisions"]["lookahead"]["action_sha256"]:
                raise ValueError("incumbent identity mismatch")
            blocks = validate_initial(original)
            with torch.no_grad():
                initial = model.encode({"pixels": transform(data["current"][root])[None, None].cuda()})["emb"]
                goal = model.encode({"pixels": transform(data["goal"][root])[None, None].cuda()})["emb"][:, 0]
            candidates, metrics = [original], {}
            for mode in ARMS[1:]:
                action, m = refine(model, initial, goal, blocks, scaler.mean_, scaler.scale_, mode)
                candidates.append(action)
                metrics[mode] = m
            if not np.array_equal(candidates[-1][:10], original[:10]):
                raise ValueError("saved fixed prefix changed")
            arrays[f"r{root}_actions"] = np.stack(candidates)
            rows.append({"root": root, "row": record["row"], "metrics": metrics,
                         "action_sha256": {arm: q.digest(a) for arm, a in zip(ARMS, candidates)}})
            print(f"root={root} model refinement sealed", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    sources = [Path(__file__), PROTOCOL, Path(history.__file__), q.PLAN, q.POOL, input_path,
               common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"]
    with POOL.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with PLAN.open("x") as stream:
        json.dump({"track": RGB, "rows": rows, "pool_sha256": file_hash(POOL),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "source_sha256": {str(p): file_hash(p) for p in sources},
                   "scope": "exposed development cases, not independent performance"},
                  stream, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sealed = json.loads(PLAN.read_text())
    if sealed["pool_sha256"] != file_hash(POOL):
        raise ValueError("pool drift")
    if any(file_hash(Path(p)) != v for p, v in sealed["source_sha256"].items()):
        raise ValueError("source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, \
                h5py.File(q.old.base.source.block.prior.common.DATA, "r") as data:
            for item in sealed["rows"]:
                outcomes = {}
                for arm, action in zip(ARMS, pool[f"r{item['root']}_actions"]):
                    if q.digest(action) != item["action_sha256"][arm]:
                        raise ValueError("action identity drift")
                    validate_initial(action)
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
                print(f"root={item['root']} physical complete", flush=True)
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL),
                   "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
