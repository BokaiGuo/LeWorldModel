"""Sealed RGB predictions and physical replay for fixed wide waypoint offsets."""
import json
import sys

import numpy as np
import torch

from experiments import pc_tworoom_waypoint_gain_repair_20261107_codex as repair
from experiments import pc_tworoom_geometry_pool_alignment_20261107_codex as alignment
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = repair.ROOT
PANEL = repair.PANEL
OFFSETS = (0., -3., 3., -6., 6., -9., 9., -12., 12., -15., 15., -18., 18.)
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_WIDE_OFFSET_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_wide_offset_plan_20261107_codex.json"
PRED = ROOT / "results/pc_tworoom_wide_offset_predicted_20261107_codex.npz"
RESULT = ROOT / "results/pc_tworoom_wide_offset_result_20261107_codex.json"
ACTUAL = ROOT / "results/pc_tworoom_wide_offset_actual_20261107_codex.npz"


def candidate_pool(current, goal):
    start, target, wall, door = repair.v2.v1.geometry(current, goal)
    side = -1. if start[0] < wall else 1.
    if (start[0] - wall) * (target[0] - wall) >= 0:
        raise ValueError("expected cross-wall task")
    safe_x = wall + side * 16.
    first_x = safe_x if side * (start[0] - safe_x) > 0 else start[0]
    result = []
    for offset in OFFSETS:
        points = [[first_x, door + offset], [wall - side * 16., door + offset], target]
        actions, _ = repair.waypoint_actions(start, points, 1., 25)
        result.append(actions)
    pool = np.stack(result)
    if pool.shape != (13, 25, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("invalid candidate pool")
    return pool


def setup():
    model, scaler, transform, normalizer = PANEL.source.common.model_setup()
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, PANEL.PUBLIC,
                         PANEL.source.common.CHECKPOINT / "weights.pt",
                         PANEL.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five physical steps per packet")
    return model, scaler, transform, normalizer, RGBLeWM(model, audit)


def plan():
    if PLAN.exists() or PRED.exists():
        raise FileExistsError("prediction artifact exists")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("public input drift")
    model, scaler, transform, normalizer, boundary = setup()
    fingerprint = model_fingerprint(model)
    rows, features = [], {}
    print("Seal fixed wide-offset candidates and RGB-model predictions before physical replay.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, seed in enumerate(public["seeds"]):
            pool = candidate_pool(public["current"][i], public["goal"][i])
            normal = scaler.transform(pool.reshape(-1, 2)).reshape(1, 13, 5, 10)
            context = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                       "goal": transform(public["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            endpoint, out = boundary.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
            pred, goal = out["predicted_emb"][0, :, 1:], out["goal_emb"][0, -1]
            cost = (pred - goal).square().sum(-1)
            if cost.shape != (13, 5) or not torch.isfinite(cost).all():
                raise ValueError("model time or finite check failed")
            torch.testing.assert_close(endpoint[0], cost[:, -1], atol=1e-3, rtol=0)
            features[f"pred_{i}"] = pred.cpu().numpy()
            features[f"goal_{i}"] = goal.cpu().numpy()
            rows.append({"seed": int(seed), "actions": pool.tolist(), "costs": cost.cpu().tolist(),
                         "selected": int(cost.min(-1).values.argmin().item())})
            print(f"task={i} selected_offset={OFFSETS[rows[-1]['selected']]}", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel or model drift")
    with PRED.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with PLAN.open("x") as stream:
        json.dump({"track": RGB, "offsets": OFFSETS, "rows": rows,
                   "public_sha256": file_hash(PANEL.PUBLIC), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "repair_sha256": file_hash(repair.__file__),
                   "pred_sha256": file_hash(PRED), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer}, stream, separators=(",", ":"), allow_nan=False)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists() or ACTUAL.exists():
        raise FileExistsError("physical artifact exists")
    seal = json.loads(PLAN.read_text())
    for key, path in (("public_sha256", PANEL.PUBLIC), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("repair_sha256", repair.__file__),
                      ("pred_sha256", PRED)):
        if seal[key] != file_hash(path):
            raise ValueError(f"sealed input drift: {key}")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    model, _, transform, normalizer, boundary = setup()
    if model_fingerprint(model) != seal["model_sha256"] or normalizer != seal["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    print("Replay sealed actions and encode real packet-time images; truth remains evaluation-only.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, \
             np.load(PANEL.TRUTH, allow_pickle=False) as truth, \
             np.load(PRED, allow_pickle=False) as predicted, torch.inference_mode():
            for i, row in enumerate(seal["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("seed mismatch")
                predicted_cost = ((predicted[f"pred_{i}"] - predicted[f"goal_{i}"]) ** 2).sum(axis=-1)
                np.testing.assert_allclose(predicted_cost, row["costs"], atol=1e-3, rtol=1e-5)
                chains, images = [], []
                for actions in np.asarray(row["actions"], np.float32):
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace = []
                    for t, action in enumerate(actions):
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if (t + 1) % 5 == 0:
                            images.append(actor.render().copy())
                    chains.append(trace)
                encoded = []
                for begin in range(0, len(images), 16):
                    batch = torch.stack([transform(im) for im in images[begin:begin + 16]]).cuda()
                    encoded.append(boundary.encode({"pixels": batch[:, None]})["emb"][:, 0])
                actual = torch.cat(encoded).reshape(13, 5, -1)
                goal = torch.as_tensor(predicted[f"goal_{i}"], device="cuda")
                cost = (actual - goal).square().sum(-1).cpu().numpy()
                model_cost = np.asarray(row["costs"]).min(axis=1)
                success = np.array([any(step["success"] for step in chain) for chain in chains])
                arrays[f"actual_{i}"] = actual.cpu().numpy()
                rows.append({"seed": row["seed"], "chains": chains, "success": success.tolist(),
                             "selected_success": bool(success[row["selected"]]), "zero_offset_success": bool(success[0]),
                             "encoded_costs": cost.tolist(),
                             "model_pair_accuracy": alignment.success_pair_accuracy(model_cost, success),
                             "encoded_pair_accuracy": alignment.success_pair_accuracy(cost.min(axis=1), success)})
                print(f"task={i} successes={success.sum()}/13 selected={success[row['selected']]}", flush=True)
    finally:
        env.close()
    if len(rows) != 40 or model_fingerprint(model) != seal["model_sha256"]:
        raise ValueError("panel or model drift")
    with ACTUAL.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with RESULT.open("x") as stream:
        json.dump({"plan_sha256": file_hash(PLAN), "truth_sha256": file_hash(PANEL.TRUTH),
                   "actual_sha256": file_hash(ACTUAL), "rows": rows,
                   "scope": "exposed tasks; physical state restricted to replay"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "replay"):
        raise SystemExit("expected plan|replay")
    {"plan": plan, "replay": replay}[sys.argv[1]]()
