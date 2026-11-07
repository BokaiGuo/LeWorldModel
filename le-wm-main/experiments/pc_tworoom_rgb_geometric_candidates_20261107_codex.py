"""RGB-only geometric candidate baseline on the exposed cross-wall panel."""
import json
import time

import numpy as np
import torch

from experiments import pc_tworoom_crosswall_mechanism_dev_20261107_codex as panel
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = panel.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_GEOMETRIC_CANDIDATE_DEV_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_rgb_geometric_candidates_20261107_codex.json"


def red_center(image):
    image = np.asarray(image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("expected uint8 RGB 224x224")
    weights = np.maximum(image[:, :, 0].astype(np.int16) - image[:, :, 1].astype(np.int16), 0)
    if weights.sum() < 1000:
        raise ValueError("no visible red agent")
    y, x = np.indices(weights.shape)
    return np.array([(weights * x).sum(), (weights * y).sum()]) / weights.sum()


def geometry(current, goal):
    start, target = red_center(current), red_center(goal)
    dark = np.max(goal, axis=2) < 64
    counts = dark[20:-20, 40:-40].sum(axis=0)
    cols = np.flatnonzero(counts >= 100) + 40
    if len(cols) < 3 or np.any(np.diff(cols) != 1):
        raise ValueError("unsupported wall")
    wall = float(cols.mean())
    open_rows = ~dark[:, int(round(wall))]
    runs = np.split(np.flatnonzero(open_rows), np.flatnonzero(np.diff(np.flatnonzero(open_rows)) != 1) + 1)
    runs = [r for r in runs if len(r) >= 10 and r[0] > 10 and r[-1] < 213]
    if len(runs) != 1:
        raise ValueError("unsupported door")
    return start, target, wall, float(runs[0].mean())


def candidate_pool(current, goal, old_tail, remaining):
    if remaining not in (5, 10, 15, 20, 25):
        raise ValueError("invalid remaining horizon")
    old = panel.source.cem.old_or_hold(old_tail)[:remaining]
    try:
        start, target, wall, door = geometry(current, goal)
    except ValueError:
        return np.concatenate((old[None], np.zeros((15, remaining, 2), np.float32))), None
    chains = []
    for offset in (0., -3., 3., -6., 6.):
        for gain in (1., .8, 1.2):
            position, stage = start.copy(), 0
            side = -1. if start[0] < wall else 1.
            crossing = (start[0] - wall) * (target[0] - wall) < 0
            points = [np.array([wall + side * 16., door + offset]),
                      np.array([wall - side * 16., door + offset]), target] if crossing else [target]
            actions = []
            for _ in range(remaining):
                while stage < len(points) - 1 and np.max(np.abs(position - points[stage])) < 1e-4:
                    stage += 1
                action = np.clip((points[stage] - position) / 5. * gain, -1., 1.)
                position = position + 5. * action
                actions.append(action)
            chains.append(actions)
    pool = np.concatenate((old[None], np.asarray(chains, dtype=np.float32)))
    if pool.shape != (16, remaining, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("illegal candidate")
    return pool, {"start": start.tolist(), "target": target.tolist(), "wall": wall, "door": door}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(panel.REFERENCE.read_text())
    for key, path in (("public_sha256", panel.PUBLIC), ("truth_sha256", panel.TRUTH),
                      ("protocol_sha256", panel.PROTOCOL), ("runner_sha256", panel.__file__)):
        if reference[key] != file_hash(path):
            raise ValueError(f"panel provenance drift: {key}")
    model, scaler, transform, normalizer = panel.source.common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, panel.PUBLIC,
                         panel.source.common.CHECKPOINT / "weights.pt",
                         panel.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five physical steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("RGB geometric candidates: same pool, model selection versus fixed geometry.", flush=True)
    try:
        with np.load(panel.PUBLIC, allow_pickle=False) as public, \
             np.load(panel.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, seed in enumerate(public["seeds"]):
                goal_image = public["goal"][i]
                goal_tensor = transform(goal_image)[None, None, None].cuda()
                outcomes = {}
                for arm in ("model", "geometry"):
                    env.reset(seed=int(seed), options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    tail, trace, decisions = None, [], []
                    task_start = time.perf_counter()
                    for iteration in range(5):
                        image = actor.render().copy()
                        remaining = 25 - len(trace)
                        torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        pool, parsed = candidate_pool(image, goal_image, tail, remaining)
                        # Evaluation only: never passed into candidate_pool or selector.
                        error = None if parsed is None else {
                            "start": float(np.linalg.norm(np.asarray(parsed["start"]) - actor.agent_position.cpu().numpy())),
                            "target": float(np.linalg.norm(np.asarray(parsed["target"]) - truth["target"][i]))}
                        normal = scaler.transform(pool.reshape(-1, 2)).reshape(16, remaining, 2)
                        actions_tensor = torch.as_tensor(normal.reshape(1, 16, remaining // 5, 10),
                                                         dtype=torch.float32, device="cuda")
                        ctx = {"pixels": transform(image)[None, None, None].cuda(), "goal": goal_tensor,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        endpoint, context = boundary.score(ctx, actions_tensor)
                        costs = (context["predicted_emb"][:, :, 1:] -
                                 context["goal_emb"][:, -1, None, None]).square().sum(-1)
                        if costs.shape != (1, 16, remaining // 5) or not torch.isfinite(costs).all():
                            raise ValueError("invalid model scores")
                        if not torch.allclose(endpoint, costs[:, :, -1], atol=1e-3, rtol=0):
                            raise ValueError("endpoint mismatch")
                        scores = costs.min(-1).values[0].cpu().numpy()
                        selected = int(np.argmin(scores)) if arm == "model" else 1
                        actions = pool[selected].copy()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start_time
                        dispatched = []
                        for action in actions[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "remaining": remaining, "parsed": parsed,
                                          "evaluation_coordinate_error": error, "pool": pool.tolist(),
                                          "scores": scores.tolist(), "chosen": selected, "dispatched": dispatched,
                                          "queries": 16, "packet_steps": 16 * (remaining // 5),
                                          "planning_seconds": elapsed, "receipts": boundary.receipts})
                        tail = actions[len(dispatched):].copy()
                        if term or trunc:
                            break
                    outcomes[arm] = {"trace": trace, "decisions": decisions,
                                     "success": any(item["terminated"] for item in trace),
                                     "wall_seconds": time.perf_counter() - task_start}
                rows.append({"seed": int(seed), "arms": outcomes})
                print(f"task={i} model={outcomes['model']['success']} geometry={outcomes['geometry']['success']}",
                      flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed cross-wall development, no process isolation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "reference_sha256": file_hash(panel.REFERENCE), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
