"""Reconstruct response matrices and pairwise rankings from saved probe data."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_actual_fd_physical_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def vector_metrics(model, observed):
    model, observed = np.asarray(model), np.asarray(observed)
    mn, on = float(np.linalg.norm(model)), float(np.linalg.norm(observed))
    difference = float(np.linalg.norm(model-observed))
    return {"model_norm": mn, "observed_norm": on, "absolute_error": difference,
            "relative_to_observed": difference/on if on > 1e-8 else None,
            "cosine": float(np.sum(model*observed)/(mn*on)) if min(mn, on) > 1e-8 else None}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.live.host.bridge.sha_file
    manifest, result = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(run.MANIFEST) or result["arrays_sha256"] != sha(run.ARRAYS):
        raise ValueError("source mismatch")
    decision = json.loads(run.prev.DECISION.read_text())
    with np.load(run.ARRAYS, allow_pickle=False) as c:
        data = {k: c[k].copy() for k in c.files}
    if data["actions"].shape != (2, 21, 5, 2) or data["observed"].shape[:2] != (2, 21):
        raise ValueError("shape mismatch")
    np.testing.assert_allclose(data["chart"], decision["chart"], rtol=0, atol=1e-12)
    positions = data["observed"] @ data["chart"]
    reports = []
    for k, row in enumerate(result["rows"]):
        info = decision["result"]["first_response"][k]
        nominal = np.asarray(info["nominal"])
        axes = np.eye(10).reshape(10, 5, 2)
        requested = np.concatenate((nominal[None], nominal[None]+.05*axes, nominal[None]-.05*axes))
        np.testing.assert_array_equal(data["actions"][k], requested.astype(np.float32))
        candidates = [b for b in decision["trace"] if np.array_equal(np.asarray(b["requested"]), requested)]
        if len(candidates) != 1:
            raise ValueError("source trace mismatch")
        np.testing.assert_array_equal(data["model_positions"][k], candidates[0]["positions"])
        np.testing.assert_array_equal(row["model_goal_mse"], candidates[0]["scores"])
        observed = ((positions[k, 1:11] - positions[k, 11:21])/.1).T
        model = ((data["model_positions"][k, 1:11] - data["model_positions"][k, 11:21])/.1).T
        np.testing.assert_allclose(model, row["model_matrix"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(observed, row["observed_matrix"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.mean((data["observed"][k] - data["target"])**2, axis=1),
                                   row["observed_goal_mse"], rtol=0, atol=1e-7)
        distances = []
        for state in row["states"]:
            distance = float(run.live.p0._qpos_distance(np.asarray(state["qpos"]), np.asarray(state["target_qpos"])))
            if abs(distance-state["distance"]) > 1e-12:
                raise ValueError("physical distance mismatch")
            distances.append(distance)
        if len(distances) != 21:
            raise ValueError("physical observations missing")
        m = np.asarray(row["model_goal_mse"])
        o = np.asarray(row["observed_goal_mse"])
        q = np.asarray(distances)
        inversions = {}
        for label, a, b in (("model_vs_observed", m, o), ("model_vs_physical", m, q),
                            ("observed_vs_physical", o, q)):
            da, db = a[1:11]-a[11:21], b[1:11]-b[11:21]
            eligible = (np.abs(da) > 1e-8) & (np.abs(db) > 1e-8)
            inversions[label] = {"opposite": int(np.sum((da*db < 0) & eligible)),
                                 "eligible": int(eligible.sum())}
        _, _, vh = np.linalg.svd(model, full_matrices=False)
        reports.append({"nominal": k, "matrix": vector_metrics(model, observed),
                        "columns": [vector_metrics(model[:, i], observed[:, i]) for i in range(10)],
                        "model_svd_directions": [vector_metrics(model @ d, observed @ d) for d in vh],
                        "plus_minus_score_inversions": inversions})
    seal.verify(manifest["seal"])
    run.live.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                                 "arrays_sha256": sha(run.ARRAYS), "audit_script_sha256": sha(Path(__file__)),
                                 "reports": reports,
                                 "scope": "rebuild from recorded traces and saved latent arrays; no encoder rerun"})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
