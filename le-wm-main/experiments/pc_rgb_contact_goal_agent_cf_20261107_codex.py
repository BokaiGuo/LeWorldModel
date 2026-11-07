"""Counterfactual goal-pusher rendering and frozen feature evaluation."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CONTACT_GOAL_AGENT_COUNTERFACTUAL_PROTOCOL_20261107_CODEX.md"
IMAGES = ROOT / "results/pc_rgb_contact_goal_agent_cf_20261107_codex_images.npz"
RENDER = ROOT / "results/pc_rgb_contact_goal_agent_cf_20261107_codex_render.json"
RESULT = ROOT / "results/pc_rgb_contact_goal_agent_cf_20261107_codex_result.json"


def render():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot

    if IMAGES.exists() or RENDER.exists():
        raise FileExistsError("render output exists")
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["images_sha256"] != source.sha(source.IMAGES):
        raise ValueError("physical source drift")
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    images, rows = {}, []
    print("Render goal-pusher counterfactuals only; preserve source goal object pose.", flush=True)
    try:
        with np.load(source.IMAGES, allow_pickle=False) as original:
            for row in physical["rows"]:
                if not row["valid"]:
                    raise ValueError("invalid source root")
                for side in (-1, 1):
                    condition = next(c for c in row["conditions"] if c["side"] == side and c["appearance"] == "white")
                    goal = np.asarray(row["goal"], dtype=np.float64)
                    alternate = goal.copy()
                    alternate[:2] = np.asarray(condition["requested_state"][:2])
                    env.reset(seed=881001 + row["root"], options={"state": goal.copy(), "goal_state": goal.copy()})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    original_goal = actor.render().copy()
                    np.testing.assert_array_equal(original_goal, original[f"r{row['root']}_white_goal"])
                    env.reset(seed=881001 + row["root"], options={"state": alternate.copy(), "goal_state": goal.copy()})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    state = np.asarray(actor._get_obs(), dtype=np.float64)
                    np.testing.assert_allclose(state, alternate, atol=1e-6, rtol=0)
                    before = snapshot(actor)
                    alternative_image = actor.render().copy()
                    np.testing.assert_array_equal(before, snapshot(actor))
                    geometry = source.geometry(actor)
                    key = f"r{row['root']}_s{side}"
                    images[key + "_original"] = original_goal
                    images[key + "_alternative"] = alternative_image
                    rows.append({"root": row["root"], "side": side, "goal": goal.tolist(),
                                 "alternate": alternate.tolist(), "geometry": geometry,
                                 "image_changed_pixels": int(np.count_nonzero(
                                     np.any(original_goal != alternative_image, axis=-1)))})
    finally:
        env.close()
    with IMAGES.open("xb") as stream:
        np.savez_compressed(stream, **images)
    with RENDER.open("x") as stream:
        json.dump({"rows": rows, "image_sha256": source.sha(IMAGES), "physical_sha256": source.sha(source.PHYSICAL),
                   "protocol_sha256": source.sha(PROTOCOL), "runner_sha256": source.sha(__file__),
                   "scope": "privileged counterfactual goal render, not planner input"}, stream, indent=2,
                  allow_nan=False)


def analyze():
    import torch
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_rgb_true_history_alignment_20261107_codex import model_and_scaler

    if RESULT.exists():
        raise FileExistsError(RESULT)
    render_record = json.loads(RENDER.read_text())
    physical, prior_model = json.loads(source.PHYSICAL.read_text()), json.loads(source.MODEL.read_text())
    if (render_record["image_sha256"] != source.sha(IMAGES)
            or render_record["physical_sha256"] != source.sha(source.PHYSICAL)
            or render_record["protocol_sha256"] != source.sha(PROTOCOL)
            or render_record["runner_sha256"] != source.sha(__file__)
            or prior_model["physical_sha256"] != source.sha(source.PHYSICAL)
            or prior_model["features_sha256"] != source.sha(source.FEATURES)):
        raise ValueError("source drift")
    model, _, transform, action_hash = model_and_scaler()
    before = model_fingerprint(model)
    if before != prior_model["model_sha256"]:
        raise ValueError("model differs")
    rows = []
    print("Encode fixed and counterfactual goals; frozen predictions are not modified.", flush=True)
    with np.load(IMAGES, allow_pickle=False) as images, np.load(source.FEATURES, allow_pickle=False) as feats, \
            torch.inference_mode():
        for r in render_record["rows"]:
            root, side = r["root"], r["side"]
            condition = next(c for c in physical["rows"][root]["conditions"]
                             if c["side"] == side and c["appearance"] == "white")
            key = f"r{root}_s{side}"
            goal_tensors = torch.stack([transform(images[key + "_" + variant]) for variant in
                                        ("original", "alternative")]).cuda()
            embeddings = model.encode({"pixels": goal_tensors[:, None]})["emb"][:, 0].cpu().numpy()
            features = np.stack([feats[f"r{root}_white_{side}_{name}"] for name in ("zero", "plus", "minus")])
            np.testing.assert_allclose(features[:, 3], np.broadcast_to(embeddings[0], features[:, 3].shape),
                                       atol=1e-5, rtol=1e-5)
            physical_cost = np.asarray([condition["actions"][name]["object_pose_cost"]
                                        for name in ("zero", "plus", "minus")])
            options = {}
            for index, variant in enumerate(("original", "alternative")):
                methods = {}
                for name, feature_index in (("predicted", 1), ("real_encoded", 2)):
                    costs = np.square(features[:, feature_index] - embeddings[index]).sum(axis=1)
                    selected = int(np.argmin(costs))
                    methods[name] = {"scores": costs.tolist(), "selected": selected,
                                     "physical_regret": float(physical_cost[selected] - physical_cost.min())}
                options[variant] = methods
            rows.append({"root": root, "side": side, "physical_costs": physical_cost.tolist(),
                         "overlap": bool(r["geometry"]["overlaps"]), "results": options})
    if model_fingerprint(model) != before:
        raise ValueError("model drift")
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "render_sha256": source.sha(RENDER), "images_sha256": source.sha(IMAGES),
                   "features_sha256": source.sha(source.FEATURES), "model_sha256": before,
                   "action_array_sha256": action_hash, "runner_sha256": source.sha(__file__),
                   "scope": "post-hoc counterfactual goal, not usable planner or confirmation"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("render", "analyze"):
        raise SystemExit("expected render|analyze")
    {"render": render, "analyze": analyze}[sys.argv[1]]()
