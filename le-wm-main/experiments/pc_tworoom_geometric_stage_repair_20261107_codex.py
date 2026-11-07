"""Paired candidate-generator stage repair on the exposed TwoRoom development panel."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_rgb_geometric_candidates_v2_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.v1.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_GEOMETRIC_STAGE_REPAIR_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_geometric_stage_repair_20261107_codex.json"
SOURCE = ROOT / "results/pc_tworoom_geometry_pool_alignment_20261107_codex.json"


def follow_waypoints(start, points, gain, steps):
    position = np.asarray(start, dtype=float).copy()
    points = np.asarray(points, dtype=float)
    if position.shape != (2,) or points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("invalid points")
    if not np.isfinite(position).all() or not np.isfinite(points).all() or gain <= 0 or steps < 1:
        raise ValueError("invalid trajectory parameters")
    actions, stages = [], []
    stage = 0
    for _ in range(steps):
        while stage < len(points) - 1 and np.max(np.abs(position - points[stage])) <= .5:
            stage += 1
        delta = points[stage] - position
        command = np.clip(delta / 5. * gain, -1., 1.)
        command = np.sign(command) * np.minimum(np.abs(command), np.abs(delta) / 5.)
        position += 5. * command
        actions.append(command)
        stages.append(stage)
    return np.asarray(actions, np.float32), stages


def candidate_pool(current, goal):
    start, target, wall, door = prior.v1.geometry(current, goal)
    chains, stages = [np.zeros((25, 2), np.float32)], [[-1] * 25]
    for offset in (0., -3., 3., -6., 6.):
        for gain in (1., .8, 1.2):
            side = -1. if start[0] < wall else 1.
            if (start[0] - wall) * (target[0] - wall) < 0:
                safe_x = wall + side * 16.
                first_x = safe_x if side * (start[0] - safe_x) > 0 else start[0]
                points = [[first_x, door + offset], [wall - side * 16., door + offset], target]
            else:
                points = [target]
            chain, stage = follow_waypoints(start, points, gain, 25)
            chains.append(chain)
            stages.append(stage)
    pool = np.stack(chains)
    if pool.shape != (16, 25, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("invalid candidate pool")
    return pool, stages


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(SOURCE.read_text())
    if old["source_sha256"] != file_hash(prior.OUTPUT):
        raise ValueError("prior result drift")
    previous = json.loads(prior.OUTPUT.read_text())
    model, scaler, transform, normalizer = prior.v1.panel.source.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, SOURCE, prior.v1.panel.PUBLIC,
                         prior.v1.panel.source.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five physical steps per packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Paired stage-repair candidate coverage and frozen-model selection.", flush=True)
    try:
        with np.load(prior.v1.panel.PUBLIC, allow_pickle=False) as public, \
             np.load(prior.v1.panel.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            if len(old["rows"]) != 40 or len(previous["rows"]) != 40:
                raise ValueError("incomplete source panel")
            for i, seed in enumerate(public["seeds"]):
                if old["rows"][i]["seed"] != int(seed) or previous["rows"][i]["seed"] != int(seed):
                    raise ValueError("task mismatch")
                pool, stages = candidate_pool(public["current"][i], public["goal"][i])
                normal = scaler.transform(pool.reshape(-1, 2)).reshape(1, 16, 5, 10)
                ctx = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                       "goal": transform(public["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                endpoint, info = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                costs = (info["predicted_emb"][:, :, 1:] -
                         info["goal_emb"][:, -1, None, None]).square().sum(-1)
                if costs.shape != (1, 16, 5) or not torch.isfinite(costs).all():
                    raise ValueError("invalid prediction")
                if not torch.allclose(endpoint, costs[:, :, -1], atol=1e-3, rtol=0):
                    raise ValueError("endpoint mismatch")
                scores = costs.min(-1).values[0].cpu().numpy()
                chosen = int(np.argmin(scores))
                chains = []
                for actions in pool:
                    env.reset(seed=int(seed), options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace = []
                    for action in actions:
                        _, _, term, trunc, physical = actor.step(action.copy())
                        trace.append({"distance": float(physical["distance_to_target"]), "success": bool(term),
                                      "truncated": bool(trunc)})
                    chains.append(trace)
                rows.append({"seed": int(seed), "pool": pool.tolist(), "stages": stages, "scores": scores.tolist(),
                             "chosen": chosen, "queries": 16, "packet_steps": 80, "chains": chains,
                             "unique_actions": len({x.tobytes() for x in pool})})
                print(f"task={i} success={sum(any(t['success'] for t in chain) for chain in chains)}/16",
                      flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "paired exposed development; true state for physical replay only", "track": RGB,
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "source_sha256": file_hash(SOURCE), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
