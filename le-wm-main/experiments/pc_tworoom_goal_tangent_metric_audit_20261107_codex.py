"""Recompute image checks, fixed-pool projections and same-backend physical trajectories."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_goal_tangent_metric_20261107_codex as run

OUTPUT = run.SOURCE.ROOT / "results/pc_tworoom_goal_tangent_metric_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    if (physical["plan_sha256"] != file_hash(run.PLAN) or plan["features_sha256"] != file_hash(run.FEATURES)
            or plan["previous_features_sha256"] != file_hash(run.previous.FEATURES)
            or plan["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("artifact drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    max_score_error = max_distance_error = 0.
    try:
        with np.load(run.SOURCE.inputs.INPUT, allow_pickle=False) as data, \
                np.load(run.previous.FEATURES, allow_pickle=False) as old, \
                np.load(run.FEATURES, allow_pickle=False) as saved, \
                h5py.File(run.SOURCE.common.DATA, "r") as truth:
            for root, (row, physical_row) in enumerate(zip(plan["rows"], physical["rows"], strict=True)):
                if row["row"] != physical_row["row"] or row["row"] != int(data["rows"][root]):
                    raise ValueError("task mismatch")
                images, inspected = run.make_goal_probes(data["goal"][root])
                basis = None
                if images is not None:
                    np.testing.assert_array_equal(images, saved[f"root_{root}_probes"])
                    basis, singular = run.tangent_basis(saved[f"root_{root}_probe_embedding"])
                    if basis is None:
                        inspected["valid"], inspected["reason"] = False, "rank_deficient"
                    else:
                        np.testing.assert_allclose(singular, row["singular_values"], atol=1e-10)
                        np.testing.assert_allclose(basis @ basis.T,
                                                   saved[f"root_{root}_basis"] @ saved[f"root_{root}_basis"].T,
                                                   atol=1e-10)
                if inspected != row["inspection"]:
                    raise ValueError("image inspection mismatch")
                delta = old[f"root_{root}_predicted"].astype(np.float64) - old[f"root_{root}_goal"].astype(np.float64)
                raw = np.square(delta).sum(axis=1)
                random, _ = np.linalg.qr(np.random.default_rng(2719300 + root).normal(size=(delta.shape[1], 2)),
                                         mode="reduced")
                np.testing.assert_allclose(random, saved[f"root_{root}_random"], atol=1e-12)
                scores = np.stack((raw, raw if basis is None else np.square(delta @ basis).sum(axis=1),
                                   np.square(delta @ random).sum(axis=1)))
                max_score_error = max(max_score_error, float(np.max(np.abs(scores - saved[f"root_{root}_scores"]))))
                for i, arm in enumerate(run.ARMS):
                    if row["outcomes"][arm]["candidate_index"] != int(np.argmin(scores[i])):
                        raise ValueError("choice mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(truth["proprio"][row["row"]])
                    actor._set_goal_state(truth["proprio"][row["row"] + 25])
                    for action, trace in zip(row["outcomes"][arm]["actions"],
                                             physical_row["outcomes"][arm]["trace"], strict=True):
                        _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                        max_distance_error = max(max_distance_error, abs(float(info["distance_to_target"]) - trace["distance"]))
                        if (bool(term), bool(trunc)) != (trace["terminated"], trace["truncated"]):
                            raise ValueError("physical mismatch")
    finally:
        env.close()
    if max_score_error > 1e-8 or max_distance_error > 1e-6:
        raise ValueError("audit tolerance exceeded")
    with OUTPUT.open("x") as stream:
        json.dump({"roots": len(plan["rows"]), "score_error": max_score_error,
                   "distance_error": max_distance_error, "plan_sha256": file_hash(run.PLAN),
                   "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "same image algorithm, saved probe features and simulator; not independent encoder"},
                  stream, indent=2)
    print(f"audited {len(plan['rows'])} roots, max score={max_score_error}, physical={max_distance_error}")


if __name__ == "__main__":
    audit()
