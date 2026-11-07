"""Gain-robust waypoint switching; sealed RGB plans then physical replay."""
import json
import sys

import numpy as np
import torch

from experiments import pc_tworoom_rgb_geometric_candidates_v2_20261107_codex as v2
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = v2.v1.ROOT
PANEL = v2.v1.panel
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_WAYPOINT_GAIN_REPAIR_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_waypoint_gain_repair_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_waypoint_gain_repair_result_20261107_codex.json"


def waypoint_actions(start, points, gain, horizon):
    position = np.asarray(start, dtype=float).copy()
    points = np.asarray(points, dtype=float)
    if (position.shape != (2,) or points.ndim != 2 or points.shape[1] != 2
            or not len(points) or not np.isfinite(np.r_[position, points.ravel(), gain]).all()
            or gain <= 0 or horizon <= 0):
        raise ValueError("invalid waypoint input")
    stage, actions, stages = 0, [], []
    for _ in range(horizon):
        while stage < len(points) - 1 and np.max(np.abs(position - points[stage])) < 1e-9:
            stage += 1
        delta = points[stage] - position
        if np.max(np.abs(delta)) <= 5. * min(gain, 1.):
            action = delta / 5.
        else:
            action = np.clip(delta * gain / 5., -1., 1.)
            action = np.sign(action) * np.minimum(np.abs(action), np.abs(delta) / 5.)
        position += 5. * action
        actions.append(action.copy())
        stages.append(stage)
    return np.asarray(actions, dtype=np.float32), stages


def candidate_pool(current, goal):
    start, target, wall, door = v2.v1.geometry(current, goal)
    side = -1. if start[0] < wall else 1.
    crossing = (start[0] - wall) * (target[0] - wall) < 0
    chains = [np.zeros((25, 2), dtype=np.float32)]
    stages = [[]]
    for offset in (0., -3., 3., -6., 6.):
        for gain in (1., .8, 1.2):
            if crossing:
                safe_x = wall + side * 16.
                first_x = safe_x if side * (start[0] - safe_x) > 0 else start[0]
                points = [[first_x, door + offset], [wall - side * 16., door + offset], target]
            else:
                points = [target]
            action, stage = waypoint_actions(start, points, gain, 25)
            chains.append(action)
            stages.append(stage)
    pool = np.stack(chains)
    if pool.shape != (16, 25, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("invalid pool")
    return pool, stages


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("public input drift")
    model, scaler, transform, normalizer = PANEL.source.common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, PANEL.PUBLIC,
                         PANEL.source.common.CHECKPOINT / "weights.pt",
                         PANEL.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five physical steps per model packet")
    wrapper = RGBLeWM(model, audit)
    rows = []
    print("Generate and score fixed RGB waypoint candidates before opening physical truth.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, seed in enumerate(public["seeds"]):
            pool, stages = candidate_pool(public["current"][i], public["goal"][i])
            normal = scaler.transform(pool.reshape(-1, 2)).reshape(1, 16, 5, 10)
            context = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                       "goal": transform(public["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            endpoint, out = wrapper.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
            scores = (out["predicted_emb"][:, :, 1:] - out["goal_emb"][:, -1, None, None]).square().sum(-1)
            if scores.shape != (1, 16, 5) or not torch.isfinite(scores).all():
                raise ValueError("invalid score shape")
            torch.testing.assert_close(endpoint, scores[:, :, -1], atol=1e-3, rtol=0)
            costs = scores[0].cpu().numpy()
            chosen = int(np.argmin(costs.min(axis=1)))
            rows.append({"seed": int(seed), "pool": pool.tolist(), "stages": stages,
                         "costs": costs.tolist(), "chosen": chosen})
            print(f"task={i} chosen={chosen}", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel or model drift")
    with PLAN.open("x") as stream:
        json.dump({"track": RGB, "public_sha256": file_hash(PANEL.PUBLIC),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists():
        raise FileExistsError(RESULT)
    sealed = json.loads(PLAN.read_text())
    if any(sealed[k] != file_hash(p) for k, p in (
            ("public_sha256", PANEL.PUBLIC), ("runner_sha256", __file__), ("protocol_sha256", PROTOCOL))):
        raise ValueError("plan drift")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay every sealed candidate; physical truth is used only for reset and evaluation.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth:
            for i, row in enumerate(sealed["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("task mismatch")
                chains = []
                for action_chain in np.asarray(row["pool"], dtype=np.float32):
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace = []
                    for action in action_chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                    chains.append(trace)
                successes = [any(step["success"] for step in chain) for chain in chains]
                rows.append({"seed": row["seed"], "chains": chains, "success": successes,
                             "selected_success": successes[row["chosen"]]})
                print(f"task={i} successes={sum(successes)}/16 selected={successes[row['chosen']]}", flush=True)
    finally:
        env.close()
    with RESULT.open("x") as stream:
        json.dump({"plan_sha256": file_hash(PLAN), "truth_sha256": file_hash(PANEL.TRUTH),
                   "scope": "exposed tasks; offline physical replay", "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "replay"):
        raise SystemExit("expected plan|replay")
    {"plan": plan, "replay": replay}[sys.argv[1]]()
