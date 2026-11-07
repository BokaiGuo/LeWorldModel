"""Compare fixed candidates under dataset versus reset-rendered current RGB."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_successful_refinement_alignment_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = base.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CURRENT_PIXEL_SCORE_AUDIT_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_current_pixel_score_audit_20261107_codex.json"


def compare_order(left, right, *, atol):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (a.ndim != 1 or a.size < 2 or b.shape != a.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(atol) or atol < 0):
        raise ValueError("finite vectors of equal size and valid tolerance required")
    i, j = np.triu_indices(a.size, 1)
    da, db = a[i] - a[j], b[i] - b[j]
    near = (np.abs(da) <= atol) | (np.abs(db) <= atol)
    return {"pairs": int(len(i)), "strict_sign_changes": int(np.sum(np.sign(da) != np.sign(db))),
            "robust_reversals": int(np.sum((da * db < 0) & ~near)),
            "near_ties_either": int(np.sum(near))}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    print("Fixed candidate audit: dataset current RGB versus reset-rendered current RGB.", flush=True)
    source, common = base.source, base.source.common
    old = json.loads(base.OUTPUT.read_text())
    if (old["features_sha256"] != file_hash(base.FEATURES)
            or old["plan_sha256"] != file_hash(source.PLAN)
            or old["prior_sha256"] != file_hash(base.prior.OUTPUT)):
        raise ValueError("alignment artifacts changed")
    plan = json.loads(source.PLAN.read_text())
    prior = json.loads(base.prior.OUTPUT.read_text())
    row = next(x for x in plan["rows"] if x["row"] == base.ROW)
    matches = [x for x in next(x for x in prior["rows"] if x["row"] == base.ROW)["candidates"]
               if x["ever_success"]]
    if len(matches) != 12:
        raise ValueError("candidate set changed")
    model, scaler, transform, action_hash = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or action_hash != plan["normalizer_action_sha256"]:
        raise ValueError("checkpoint or action normalizer drift")
    actions = [source.to_physical(row["refinements"][x["prefix"]]["actions"][x["sample"]], scaler)
               for x in matches]
    actions.extend(np.asarray(row["outcomes"][arm]["actions"]) for arm in ("parent", "lookahead"))
    actions = np.asarray(actions, dtype=np.float32)
    normal = scaler.transform(actions.reshape(-1, 2)).reshape(1, 14, 5, 10)
    np.testing.assert_allclose(scaler.inverse_transform(normal.reshape(-1, 2)),
                               actions.reshape(-1, 2), rtol=0, atol=1e-6)
    with np.load(source.inputs.INPUT, allow_pickle=False) as data:
        i = int(np.flatnonzero(data["rows"] == base.ROW)[0])
        current, goal = data["current"][i].copy(), data["goal"][i].copy()
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with h5py.File(common.DATA, "r") as dataset:
            env.reset(seed=42)
            env.unwrapped._set_state(dataset["proprio"][base.ROW])
            env.unwrapped._set_goal_state(dataset["proprio"][base.ROW + 25])
            rendered = env.unwrapped.render().copy()
    finally:
        env.close()
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.inputs.INPUT,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224", clock="five five-step packets")
    costs, predictions = [], []
    with torch.inference_mode():
        target = transform(goal)[None, None, None].cuda()
        candidates = torch.as_tensor(normal, dtype=torch.float32, device="cuda")
        for frame in (current, rendered):
            pixels = transform(frame)[None, None, None].cuda()
            value, context = RGBLeWM(model, audit).score(
                {"pixels": pixels, "goal": target, "action": torch.empty((1, 1, 0, 10), device="cuda")},
                candidates)
            pred = context["predicted_emb"][0, :, -1]
            manual = (pred - context["goal_emb"][0, -1]).square().sum(-1)
            np.testing.assert_allclose(value[0].cpu().numpy(), manual.cpu().numpy(), rtol=0, atol=1e-4)
            costs.append(value[0].cpu().numpy())
            predictions.append(pred.cpu().numpy())
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with np.load(base.FEATURES, allow_pickle=False) as features:
        saved = features["predicted"]
    np.testing.assert_allclose(costs[0], [x["predicted_cost"] for x in old["rows"]], rtol=0, atol=1e-3)
    np.testing.assert_allclose(predictions[0], saved, rtol=0, atol=1e-6)
    delta = current.astype(np.int16) - rendered.astype(np.int16)
    parent_gaps = [cost[12] - cost[:12] for cost in costs]
    result = {"row": base.ROW, "pixel_components_different": int(np.count_nonzero(delta)),
              "pixel_max_abs_u8": int(np.max(np.abs(delta))),
              "prediction_max_abs_difference": float(np.max(np.abs(predictions[0] - predictions[1]))),
              "score_max_abs_difference": float(np.max(np.abs(costs[0] - costs[1]))),
              "all_pair_orders": compare_order(costs[0], costs[1], atol=1e-4),
              "parent_vs_success_gaps": [gap.tolist() for gap in parent_gaps],
              "parent_vs_success_strict_sign_changes": int(np.sum(np.sign(parent_gaps[0]) != np.sign(parent_gaps[1]))),
              "costs_dataset_and_rendered": [cost.tolist() for cost in costs],
              "model_sha256": fingerprint, "normalizer_action_sha256": action_hash,
              "alignment_sha256": file_hash(base.OUTPUT), "features_sha256": file_hash(base.FEATURES),
              "input_sha256": file_hash(source.inputs.INPUT), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(Path(__file__)),
              "scope": "exposed one-root evaluator current-frame audit, not planning or gate evidence"}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("pixel_components_different", "pixel_max_abs_u8", "prediction_max_abs_difference",
                       "score_max_abs_difference", "all_pair_orders", "parent_vs_success_strict_sign_changes")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
