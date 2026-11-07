"""Test goal tangent using validated dual-RGB background recovery."""
import json
import sys
import time

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_goal_tangent_metric_20261107_codex as old
from experiments import pc_tworoom_dual_rgb_goal_reconstruction_20261107_codex as reconstruction
from experiments import pc_tworoom_dual_rgb_renderer_audit_20261107_codex as render_audit
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for

ROOT = old.SOURCE.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_DUAL_RGB_GOAL_TANGENT_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_dual_rgb_goal_tangent_plan_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_dual_rgb_goal_tangent_features_20261107_codex.npz"
PHYSICAL = ROOT / "results/pc_tworoom_dual_rgb_goal_tangent_physical_20261107_codex.json"
ARMS = ("original", "goal_tangent", "random_tangent")


def plan():
    if PLAN.exists() or FEATURES.exists():
        raise FileExistsError("output exists")
    prior, input_audit = json.loads(old.previous.OUTPUT.read_text()), json.loads(reconstruction.OUTPUT.read_text())
    renderer = json.loads(render_audit.OUTPUT.read_text())
    original = json.loads(old.SOURCE.PLAN.read_text())
    if (prior["features_sha256"] != file_hash(old.previous.FEATURES)
            or input_audit["input_sha256"] != file_hash(old.SOURCE.inputs.INPUT)
            or input_audit["images_sha256"] != file_hash(reconstruction.IMAGES)
            or renderer["reconstruction_sha256"] != file_hash(reconstruction.OUTPUT)
            or prior["source_sha256"] != file_hash(old.SOURCE.PLAN)):
        raise ValueError("sealed inputs changed")
    if sum(x["inspection"]["valid"] for x in input_audit["rows"]) < 40:
        raise ValueError("reconstruction development gate not satisfied")
    model, scaler, transform, action_hash = old.SOURCE.common.model_setup()
    model_id = model_fingerprint(model)
    if model_id != prior["model_sha256"] or action_hash != prior["normalizer_action_sha256"]:
        raise ValueError("model or action normalizer changed")
    rows, artifacts = [], {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with np.load(reconstruction.IMAGES, allow_pickle=False) as images, \
            np.load(old.previous.FEATURES, allow_pickle=False) as saved, torch.inference_mode():
        for root, row in enumerate(original["rows"]):
            check = input_audit["rows"][root]
            if check["row"] != row["row"]:
                raise ValueError("task identity mismatch")
            forecast = saved[f"root_{root}_predicted"].astype(np.float64)
            goal = saved[f"root_{root}_goal"].astype(np.float64)
            basis, singular = None, []
            probe_embedding = np.empty((0, len(goal)))
            if check["inspection"]["valid"]:
                tensors = torch.stack([transform(image) for image in images[f"root_{root}"]])[:, None].cuda()
                probe_embedding = model.encode({"pixels": tensors})["emb"][:, 0].cpu().numpy()
                basis, values = old.tangent_basis(probe_embedding)
                singular = values.tolist()
            delta = forecast - goal
            raw = np.square(delta).sum(axis=1)
            random, _ = np.linalg.qr(np.random.default_rng(2719300 + root).normal(size=(len(goal), 2)), mode="reduced")
            scores = {"original": raw, "goal_tangent": raw if basis is None else np.square(delta @ basis).sum(axis=1),
                      "random_tangent": np.square(delta @ random).sum(axis=1)}
            chains = [item[1] for item in candidates_for(row)]
            for ref in row["refinements"]:
                chains.extend(ref["actions"])
            if len(chains) != 656:
                raise ValueError("candidate count changed")
            outcomes = {}
            for arm in ARMS:
                i = int(np.argmin(scores[arm]))
                outcomes[arm] = {"candidate_index": i, "score": float(scores[arm][i]),
                                 "actions": old.SOURCE.to_physical(np.asarray(chains[i], dtype=np.float32), scaler).tolist()}
            rows.append({"row": row["row"], "root": root, "image_valid": check["inspection"]["valid"],
                         "basis_valid": basis is not None, "singular_values": singular,
                         "extra_goal_encodings": 4 if check["inspection"]["valid"] else 0, "outcomes": outcomes})
            artifacts[f"root_{root}_scores"] = np.stack([scores[arm] for arm in ARMS])
            artifacts[f"root_{root}_probe_embedding"] = probe_embedding
            artifacts[f"root_{root}_basis"] = np.empty((len(goal), 0)) if basis is None else basis
            artifacts[f"root_{root}_random"] = random
            print(f"{root+1}/10 row={row['row']} basis_valid={basis is not None}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != model_id:
        raise ValueError("model weights changed")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **artifacts)
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "features_sha256": file_hash(FEATURES),
                   "source_sha256": file_hash(old.SOURCE.PLAN), "prediction_features_sha256": file_hash(old.previous.FEATURES),
                   "reconstruction_sha256": file_hash(reconstruction.OUTPUT), "renderer_audit_sha256": file_hash(render_audit.OUTPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "model_sha256": model_id, "elapsed_seconds": elapsed,
                   "scope": "exposed RGB-only fixed-pool development"}, stream, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    record = json.loads(PLAN.read_text())
    if (record["runner_sha256"] != file_hash(__file__) or record["features_sha256"] != file_hash(FEATURES)
            or record["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("sealed plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(old.SOURCE.common.DATA, "r") as data:
            for row in record["rows"]:
                outcomes = {}
                for arm in ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    actions = np.asarray(row["outcomes"][arm]["actions"], dtype=np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("illegal action")
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[arm] = {"trace": trace, "ever_success": any(item["terminated"] for item in trace),
                                     "final_distance": trace[-1]["distance"]}
                rows.append({"row": row["row"], "outcomes": outcomes})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "runner_sha256": file_hash(__file__)},
                  stream, indent=2, allow_nan=False)
    for arm in ARMS:
        print(f"{arm}: {sum(row['outcomes'][arm]['ever_success'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
