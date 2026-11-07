"""Evaluation-only alignment of observed successful refinements and selected failures."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments import pc_tworoom_failed_root_full_refinement_20261107_codex as prior

ROW = 82490
PROTOCOL = source.ROOT.parent / "docs/PC_TWOROOM_SUCCESSFUL_REFINEMENT_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_tworoom_successful_refinement_alignment_20261107_codex.json"
FEATURES = source.ROOT / "results/pc_tworoom_successful_refinement_alignment_features_20261107_codex.npz"


def run():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists() or FEATURES.exists():
        raise FileExistsError("outputs already exist")
    old, plan = json.loads(prior.OUTPUT.read_text()), json.loads(source.PLAN.read_text())
    if old["plan_sha256"] != file_hash(source.PLAN):
        raise ValueError("plan changed")
    row = next(x for x in plan["rows"] if x["row"] == ROW)
    prior_row = next(x for x in old["rows"] if x["row"] == ROW)
    successes = [x for x in prior_row["candidates"] if x["ever_success"]]
    if len(successes) != 12:
        raise ValueError("success set changed")
    model, scaler, transform, action_hash = source.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or action_hash != plan["normalizer_action_sha256"]:
        raise ValueError("model or normalizer drift")
    chains, labels = [], []
    for candidate in successes:
        normal = np.asarray(row["refinements"][candidate["prefix"]]["actions"][candidate["sample"]])
        chains.append(source.to_physical(normal, scaler))
        labels.append({"type": "success", "prefix": candidate["prefix"], "sample": candidate["sample"],
                       "prior_cost": candidate["model_cost"]})
    for arm in ("parent", "lookahead"):
        chains.append(np.asarray(row["outcomes"][arm]["actions"]))
        labels.append({"type": arm, "prior_cost": row["outcomes"][arm]["cost"]})
    chains = np.asarray(chains, dtype=np.float32)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    physical, frames = [], []
    try:
        with h5py.File(source.common.DATA, "r") as dataset:
            for chain in chains:
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(dataset["proprio"][ROW])
                actor._set_goal_state(dataset["proprio"][ROW + 25])
                trace = []
                for action in chain:
                    _, _, term, trunc, info = actor.step(action.copy())
                    trace.append({"distance": float(info["distance_to_target"]),
                                  "terminated": bool(term), "truncated": bool(trunc)})
                frames.append(actor.render().copy())
                physical.append(trace)
    finally:
        env.close()
    for i, candidate in enumerate(successes):
        if physical[i] != candidate["trace"]:
            raise ValueError("success trace mismatch")
    audit = freeze_audit(RGB, "lewm", [source.__file__, PROTOCOL, source.inputs.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler", clock="5 packets, 5 steps each")
    with np.load(source.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        i = int(np.flatnonzero(data["rows"] == ROW)[0])
        image = transform(data["current"][i])[None, None, None].cuda()
        goal_image = transform(data["goal"][i])[None, None, None].cuda()
        normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, -1, 5, 10)
        wrapper = RGBLeWM(model, audit)
        cost, context = wrapper.score({"pixels": image, "goal": goal_image,
                                       "action": torch.empty((1, 1, 0, 10), device="cuda")},
                                      torch.tensor(normal, dtype=torch.float32, device="cuda"))
        predicted = context["predicted_emb"][0, :, -1].cpu().numpy()
        goal = model.encode({"pixels": goal_image[:, 0]})["emb"][0, 0]
        batch = torch.stack([transform(frame) for frame in frames])
        true = model.encode({"pixels": batch[:, None].cuda()})["emb"][:, 0]
        actual_cost = (true - goal).square().sum(-1).cpu().numpy()
        true, goal = true.cpu().numpy(), goal.cpu().numpy()
    rows = []
    for label, pred_cost, latent_cost, forecast, actual, trace in zip(
            labels, cost[0].cpu().numpy(), actual_cost, predicted, true, physical, strict=True):
        rows.append({**label, "predicted_cost": float(pred_cost), "real_encoded_cost": float(latent_cost),
                     "feature_mse": float(np.square(forecast - actual).mean()),
                     "ever_success": any(x["terminated"] for x in trace),
                     "final_success": trace[-1]["terminated"], "final_distance": trace[-1]["distance"]})
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, predicted=predicted, actual=true, goal=goal)
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "features_sha256": file_hash(FEATURES), "prior_sha256": file_hash(prior.OUTPUT),
                   "plan_sha256": file_hash(source.PLAN), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)),
                   "scope": "post-hoc single exposed task; physical labels evaluation only"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run()
