"""Paired data-frame versus physical re-render scoring of the same action pool."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_nonlinear_selection_20261107_codex as source
from experiments.pc_tworoom_nonlinear_selection_audit_20261107_codex import image_difference

PROTOCOL = source.source.ROOT.parent / "docs/PC_TWOROOM_RENDER_INPUT_DECISION_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.source.ROOT / "results/pc_tworoom_render_input_decision_audit_20261107_codex.json"


def score_summary(left, right, n=30):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if (left.ndim != 1 or left.shape != right.shape or len(left) < n or n < 1
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("invalid candidate scores")
    a, b = np.argsort(left, kind="stable"), np.argsort(right, kind="stable")
    return {"max_abs_difference": float(np.max(np.abs(left - right))),
            "mean_abs_difference": float(np.mean(np.abs(left - right))),
            "top_n": n, "top_intersection": len(set(a[:n]) & set(b[:n])),
            "left_best": int(a[0]), "right_best": int(b[0]),
            "left_gap": float(left[a[1]] - left[a[0]]),
            "right_gap": float(right[b[1]] - right[b[0]])}


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    prior = json.loads(source.source.PLAN.read_text())
    if (plan["runner_sha256"] != file_hash(source.__file__)
            or plan["source_sha256"] != file_hash(source.source.PLAN)
            or plan["features_sha256"] != file_hash(source.comparison.FEATURES)):
        raise ValueError("source drift")
    model, scaler, transform, action_hash = source.source.common.model_setup()
    model_id = model_fingerprint(model)
    if model_id != prior["model_sha256"] or action_hash != prior["normalizer_action_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.source.inputs.INPUT,
                         source.source.common.CHECKPOINT / "weights.pt",
                         source.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="5 packets, 5 steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.source.inputs.INPUT, allow_pickle=False) as inputs,
              np.load(source.comparison.FEATURES, allow_pickle=False) as old,
              h5py.File(source.source.common.DATA, "r") as data,
              torch.inference_mode()):
            for root, (selected, row) in enumerate(zip(plan["rows"], prior["rows"], strict=True)):
                if selected["row"] != row["row"] or row["row"] != int(inputs["rows"][root]):
                    raise ValueError("root mismatch")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["proprio"][row["row"]]))
                actor._set_goal_state(np.asarray(data["proprio"][row["row"] + 25]))
                observed = (actor.render().copy(), actor._target_img.cpu().numpy().transpose(1, 2, 0).copy())
                recorded = (inputs["current"][root], inputs["goal"][root])
                actions = [item[1] for item in source.candidates_for(row)]
                for ref in row["refinements"]:
                    actions.extend(ref["actions"])
                chains = np.asarray(actions, dtype=np.float32)
                if chains.shape != (656, 25, 2):
                    raise ValueError("unexpected candidate pool")
                source.source.to_physical(chains, scaler)
                results = []
                for pixels, goal in (recorded, observed):
                    wrapper = RGBLeWM(model, audit)
                    pixel_tensor = transform(pixels)[None, None, None].cuda()
                    goal_tensor = transform(goal)[None, None, None].cuda()
                    forecast, costs = [], []
                    for offset in range(0, len(chains), 32):
                        batch = torch.tensor(chains[offset:offset + 32].reshape(1, -1, 5, 10),
                                             dtype=torch.float32, device="cuda")
                        cost, ctx = wrapper.score({"pixels": pixel_tensor, "goal": goal_tensor,
                                                   "action": torch.empty((1, 1, 0, 10), device="cuda")}, batch)
                        forecast.append(ctx["predicted_emb"][0, :, -1].cpu().numpy())
                        costs.extend(cost[0].cpu().numpy().tolist())
                    forecast, costs = np.concatenate(forecast), np.asarray(costs)
                    templates, offsets, info = source.chart.templates(pixels, goal)
                    if info["valid"]:
                        embeddings = []
                        for offset in range(0, len(templates), 32):
                            batch = torch.stack([transform(im) for im in templates[offset:offset + 32]])[:, None].cuda()
                            embeddings.append(model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy())
                        offsets_pred = source.chart.nearest(np.concatenate(embeddings), forecast, offsets)
                        index = source.select_candidate(offsets_pred, costs)
                        decoded = offsets_pred[index].tolist()
                        count = len(templates)
                        offset_list = offsets.tolist()
                    else:
                        index, decoded, count, offset_list = int(np.argmin(costs)), None, 0, None
                    results.append({"costs": costs, "chart_valid": info["valid"],
                                    "reason": info["reason"], "chart_index": index,
                                    "selected_offset": decoded, "template_count": count,
                                    "offsets": offset_list})
                first, second = results
                rows.append({"root": root, "row": row["row"],
                             "current_rgb_difference": image_difference(*[recorded[0], observed[0]]),
                             "goal_rgb_difference": image_difference(*[recorded[1], observed[1]]),
                             "score_comparison": score_summary(first["costs"], second["costs"]),
                             "recorded_vs_sealed": score_summary(first["costs"], old[f"root_{root}_scores"][0]),
                             "recorded_chart_index": first["chart_index"],
                             "rerender_chart_index": second["chart_index"],
                             "sealed_chart_index": selected["candidate_index"],
                             "recorded_chart_valid": first["chart_valid"], "rerender_chart_valid": second["chart_valid"],
                             "recorded_template_count": first["template_count"],
                             "rerender_template_count": second["template_count"],
                             "same_template_offsets": first["offsets"] == second["offsets"],
                             "recorded_selected_offset": first["selected_offset"],
                             "rerender_selected_offset": second["selected_offset"]})
                print(f"root {root + 1}/10: latent {np.argmin(first['costs'])}/{np.argmin(second['costs'])}; "
                      f"chart {first['chart_index']}/{second['chart_index']}", flush=True)
    finally:
        env.close()
    torch.cuda.synchronize()
    if model_fingerprint(model) != model_id:
        raise ValueError("model weights changed")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(source.PLAN),
                   "input_sha256": file_hash(source.source.inputs.INPUT),
                   "features_sha256": file_hash(source.comparison.FEATURES), "model_sha256": model_id,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "exposed RGB input contract audit, not independent outcome evidence"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    run()
