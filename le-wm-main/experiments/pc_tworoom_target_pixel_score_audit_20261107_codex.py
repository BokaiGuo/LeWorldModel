"""Post-hoc target pixel equivalence and candidate score contract audit."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_successful_refinement_alignment_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = base.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_TARGET_PIXEL_SCORE_AUDIT_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_target_pixel_score_audit_20261107_codex.json"


def sign_comparison(cost):
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (14,) or not np.isfinite(cost).all():
        raise ValueError("expected twelve successes and two controls")
    # Negative means the parent control is preferred to the success candidate.
    return np.sign(cost[12] - cost[:12]).astype(int).tolist()


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
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
        current, goal_data = data["current"][i].copy(), data["goal"][i].copy()
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with h5py.File(common.DATA, "r") as dataset:
            env.reset(seed=42)
            env.unwrapped._set_state(dataset["proprio"][base.ROW])
            env.unwrapped._set_goal_state(dataset["proprio"][base.ROW + 25])
            goal_render = common.image_goal(env.unwrapped)
            initial_render = env.unwrapped.render().copy()
    finally:
        env.close()
    delta = goal_data.astype(np.int16) - goal_render.astype(np.int16)
    initial_delta = current.astype(np.int16) - initial_render.astype(np.int16)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.inputs.INPUT,
                         common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224", clock="five five-step packets")
    costs, predictions, goals = [], [], []
    with torch.inference_mode():
        pixels = transform(current)[None, None, None].cuda()
        candidates = torch.as_tensor(normal, dtype=torch.float32, device="cuda")
        for goal in (goal_data, goal_render):
            goal_tensor = transform(goal)[None, None, None].cuda()
            wrapper = RGBLeWM(model, audit)
            value, context = wrapper.score({"pixels": pixels, "goal": goal_tensor,
                                             "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                           candidates)
            pred = context["predicted_emb"][0, :, -1]
            target = context["goal_emb"][0, -1]
            manual = (pred - target).square().sum(-1)
            np.testing.assert_allclose(value[0].cpu().numpy(), manual.cpu().numpy(), rtol=0, atol=1e-4)
            costs.append(value[0].cpu().numpy())
            predictions.append(pred.cpu().numpy())
            goals.append(target.cpu().numpy())
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with np.load(base.FEATURES, allow_pickle=False) as features:
        actual = features["actual"].astype(np.float64)
        old_goal = features["goal"].astype(np.float64)
        saved_prediction = features["predicted"]
    old_cost = np.asarray([x["predicted_cost"] for x in old["rows"]])
    np.testing.assert_allclose(costs[0], old_cost, rtol=0, atol=1e-3)
    np.testing.assert_allclose(goals[0], old_goal, rtol=0, atol=1e-4)
    actual_costs = [np.square(actual - goal.astype(np.float64)).sum(axis=-1) for goal in goals]
    expected_actual = np.asarray([x["real_encoded_cost"] for x in old["rows"]])
    np.testing.assert_allclose(actual_costs[0], expected_actual, rtol=0, atol=1e-3)
    result = {"row": base.ROW, "pixels": {"different_components": int(np.count_nonzero(delta)),
                  "max_abs_u8": int(np.max(np.abs(delta))),
                  "initial_different_components": int(np.count_nonzero(initial_delta)),
                  "initial_max_abs_u8": int(np.max(np.abs(initial_delta)))},
              "goal_embedding_max_abs_diff": float(np.max(np.abs(goals[0] - goals[1]))),
              "same_actions_prediction_max_abs_diff": float(np.max(np.abs(predictions[0] - predictions[1]))),
              "saved_prediction_max_abs_diff": float(np.max(np.abs(predictions[0] - saved_prediction))),
              "prediction_cost": [x.tolist() for x in costs],
              "actual_encoded_cost": [x.tolist() for x in actual_costs],
              "prediction_parent_vs_success_signs": [sign_comparison(x) for x in costs],
              "actual_parent_vs_success_signs": [sign_comparison(x) for x in actual_costs],
              "model_fingerprint": fingerprint, "normalizer_action_sha256": action_hash,
              "alignment_sha256": file_hash(base.OUTPUT), "features_sha256": file_hash(base.FEATURES),
              "input_sha256": file_hash(source.inputs.INPUT), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(Path(__file__)),
              "scope": "post-hoc one exposed task, evaluator target rendering; not planning or gate evidence"}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("pixels", "goal_embedding_max_abs_diff", "same_actions_prediction_max_abs_diff",
                       "prediction_parent_vs_success_signs", "actual_parent_vs_success_signs")}, indent=2))


if __name__ == "__main__":
    main()
