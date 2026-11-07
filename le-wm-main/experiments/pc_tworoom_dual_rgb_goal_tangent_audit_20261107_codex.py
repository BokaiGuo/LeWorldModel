"""Reconstruct dual-RGB tangent choices and replay each selected action chain."""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for
from experiments import pc_tworoom_dual_rgb_goal_tangent_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_tworoom_dual_rgb_goal_tangent_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical, original, image_audit = (json.loads(p.read_text()) for p in
        (run.PLAN, run.PHYSICAL, run.old.SOURCE.PLAN, run.reconstruction.OUTPUT))
    if (physical["plan_sha256"] != file_hash(run.PLAN) or plan["features_sha256"] != file_hash(run.FEATURES)
            or plan["source_sha256"] != file_hash(run.old.SOURCE.PLAN)
            or plan["prediction_features_sha256"] != file_hash(run.old.previous.FEATURES)
            or plan["reconstruction_sha256"] != file_hash(run.reconstruction.OUTPUT)
            or plan["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("artifact drift")
    with h5py.File(run.old.SOURCE.common.DATA, "r") as data:
        raw = np.asarray(data["action"][:])
    raw = raw[~np.isnan(raw).any(axis=1)]
    scaler = StandardScaler().fit(raw)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    max_score_error = max_distance_error = 0.
    try:
        with np.load(run.FEATURES, allow_pickle=False) as result, \
                np.load(run.old.previous.FEATURES, allow_pickle=False) as prediction, \
                h5py.File(run.old.SOURCE.common.DATA, "r") as data:
            for root, (row, phys, original_row) in enumerate(zip(plan["rows"], physical["rows"], original["rows"], strict=True)):
                if row["row"] != phys["row"] or row["row"] != original_row["row"]:
                    raise ValueError("root mismatch")
                valid = image_audit["rows"][root]["inspection"]["valid"]
                if valid != row["image_valid"]:
                    raise ValueError("image permission mismatch")
                basis = None
                if valid:
                    basis, singular = run.old.tangent_basis(result[f"root_{root}_probe_embedding"])
                    np.testing.assert_allclose(singular, row["singular_values"], atol=1e-10)
                if (basis is not None) != row["basis_valid"]:
                    raise ValueError("rank mismatch")
                if basis is not None:
                    saved = result[f"root_{root}_basis"]
                    np.testing.assert_allclose(basis @ basis.T, saved @ saved.T, atol=1e-10)
                forecast = prediction[f"root_{root}_predicted"].astype(np.float64)
                goal = prediction[f"root_{root}_goal"].astype(np.float64)
                delta = forecast - goal
                raw_score = np.square(delta).sum(axis=1)
                random, _ = np.linalg.qr(np.random.default_rng(2719300 + root).normal(size=(len(goal), 2)),
                                         mode="reduced")
                np.testing.assert_allclose(random, result[f"root_{root}_random"], atol=1e-12)
                scores = np.stack((raw_score, raw_score if basis is None else np.square(delta @ basis).sum(axis=1),
                                   np.square(delta @ random).sum(axis=1)))
                max_score_error = max(max_score_error, float(np.max(np.abs(scores - result[f"root_{root}_scores"]))))
                chains = [item[1] for item in candidates_for(original_row)]
                for ref in original_row["refinements"]:
                    chains.extend(ref["actions"])
                for i, arm in enumerate(run.ARMS):
                    j = int(np.argmin(scores[i]))
                    if row["outcomes"][arm]["candidate_index"] != j:
                        raise ValueError("selected index mismatch")
                    action = run.old.SOURCE.to_physical(np.asarray(chains[j], dtype=np.float32), scaler)
                    np.testing.assert_allclose(action, row["outcomes"][arm]["actions"], atol=1e-7, rtol=0)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for command, step in zip(action, phys["outcomes"][arm]["trace"], strict=True):
                        _, _, term, trunc, info = actor.step(command.astype(np.float32))
                        max_distance_error = max(max_distance_error, abs(float(info["distance_to_target"]) - step["distance"]))
                        if (bool(term), bool(trunc)) != (step["terminated"], step["truncated"]):
                            raise ValueError("physical replay mismatch")
    finally:
        env.close()
    if max_score_error > 1e-8 or max_distance_error > 1e-6:
        raise ValueError("audit tolerance exceeded")
    with OUTPUT.open("x") as stream:
        json.dump({"roots": len(plan["rows"]), "max_score_error": max_score_error,
                   "max_distance_error": max_distance_error, "plan_sha256": file_hash(run.PLAN),
                   "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "same probe embeddings and same simulator, not independent encoder"},
                  stream, indent=2)
    print(f"audited {len(plan['rows'])} roots; max score={max_score_error}, physical={max_distance_error}")


if __name__ == "__main__":
    audit()
