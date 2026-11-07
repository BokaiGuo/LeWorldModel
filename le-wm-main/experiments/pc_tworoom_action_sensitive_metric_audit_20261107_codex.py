"""Reconstruct SVD projectors, fixed-pool scores, identities and physical execution."""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_action_sensitive_metric_20261107_codex as run
from experiments.pc_tworoom_failed_root_full_pool_20261107_codex import candidates_for

OUTPUT = run.source.ROOT / "results/pc_tworoom_action_sensitive_metric_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical, old = (json.loads(p.read_text()) for p in (run.OUTPUT, run.PHYSICAL, run.source.PLAN))
    if (plan["features_sha256"] != file_hash(run.FEATURES) or plan["source_sha256"] != file_hash(run.source.PLAN)
            or physical["plan_sha256"] != file_hash(run.OUTPUT) or plan["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("artifact hash mismatch")
    with h5py.File(run.source.common.DATA, "r") as data:
        raw = np.asarray(data["action"][:])
    raw = raw[~np.isnan(raw).any(axis=1)]
    scaler = StandardScaler().fit(raw)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    max_score_err = max_distance_err = 0.
    try:
        with np.load(run.FEATURES, allow_pickle=False) as features, h5py.File(run.source.common.DATA, "r") as data:
            for root, (row, phys, original) in enumerate(zip(plan["rows"], physical["rows"], old["rows"], strict=True)):
                if row["row"] != phys["row"] or row["row"] != original["row"]:
                    raise ValueError("task mismatch")
                forecast, goal = features[f"root_{root}_predicted"], features[f"root_{root}_goal"]
                basis, singular = run.action_subspace(forecast[304:368].reshape(8, 8, -1))
                np.testing.assert_allclose(singular, row["singular_values"], atol=1e-8, rtol=1e-8)
                saved_basis = features[f"root_{root}_action_basis"]
                if basis is None or not row["rank_valid"]:
                    if basis is not None or row["rank_valid"] or saved_basis.shape[0] != 0:
                        raise ValueError("rank fallback mismatch")
                else:
                    np.testing.assert_allclose(basis.T @ basis, saved_basis.T @ saved_basis, atol=1e-8, rtol=0)
                rng = np.random.default_rng(2719200 + root)
                random_basis, _ = np.linalg.qr(rng.normal(size=(forecast.shape[1], 8)), mode="reduced")
                np.testing.assert_allclose(random_basis, features[f"root_{root}_random_basis"], atol=1e-12, rtol=0)
                diff = forecast.astype(np.float64) - goal.astype(np.float64)
                full = np.square(diff).sum(axis=1)
                scores = np.stack((full, full if basis is None else np.square(diff @ basis.T).sum(axis=1),
                                   np.square(diff @ random_basis).sum(axis=1)))
                max_score_err = max(max_score_err, float(np.max(np.abs(scores - features[f"root_{root}_scores"]))))
                chains = [item[1] for item in candidates_for(original)]
                for ref in original["refinements"]:
                    chains.extend(ref["actions"])
                for i, arm in enumerate(run.ARMS):
                    index = int(np.argmin(scores[i]))
                    if row["outcomes"][arm]["candidate_index"] != index:
                        raise ValueError("selection mismatch")
                    expected = run.source.to_physical(np.asarray(chains[index], dtype=np.float32), scaler)
                    np.testing.assert_allclose(expected, row["outcomes"][arm]["actions"], atol=1e-7, rtol=0)
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for action, trace in zip(row["outcomes"][arm]["actions"], phys["outcomes"][arm]["trace"], strict=True):
                        _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                        max_distance_err = max(max_distance_err, abs(float(info["distance_to_target"]) - trace["distance"]))
                        if (bool(term), bool(trunc)) != (trace["terminated"], trace["truncated"]):
                            raise ValueError("physical trace mismatch")
    finally:
        env.close()
    if max_score_err > 1e-8 or max_distance_err > 1e-6:
        raise ValueError("audit tolerance exceeded")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": len(plan["rows"]), "max_score_error": max_score_err,
                   "max_physical_distance_error": max_distance_err, "plan_sha256": file_hash(run.OUTPUT),
                   "physical_sha256": file_hash(run.PHYSICAL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "reconstruct from frozen predictions, same simulator; not an independent model"},
                  stream, indent=2)
    print(f"audited {len(plan['rows'])} roots; score error {max_score_err}, physical error {max_distance_err}")


if __name__ == "__main__":
    audit()
