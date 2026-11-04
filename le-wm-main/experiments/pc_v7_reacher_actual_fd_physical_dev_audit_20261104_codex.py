"""Recompute saved v7 Reacher probe statistics and bind actual request batches."""

import json
import numpy as np

from experiments import pc_v7_reacher_actual_fd_physical_dev_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.configured.live.host.bridge.sha_file
    manifest, result = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    if result["manifest_sha256"] != sha(run.MANIFEST) or result["arrays_sha256"] != sha(run.ARRAYS):
        raise ValueError("artifact binding mismatch")
    seal.verify(manifest["seal"])
    decision_path = run.configured.PREFIX.with_name(run.configured.PREFIX.name + "_response_decision2.json")
    decision = json.loads(decision_path.read_text())
    with np.load(run.ARRAYS, allow_pickle=False) as data:
        if data["actions"].shape != (2, 21, 5, 2) or data["frames"].shape[:2] != (2, 21):
            raise ValueError("incorrect recorded probe shapes")
        projected = data["observed"] @ data["chart"]
        np.testing.assert_array_equal(data["chart"], np.asarray(decision["chart"]))
        checks = []
        for i, row in enumerate(result["rows"]):
            nominal = np.asarray(decision["result"]["first_response"][i]["nominal"])
            axes = np.eye(10).reshape(10, 5, 2)
            probes = np.concatenate((nominal[None], nominal[None]+.05*axes, nominal[None]-.05*axes))
            matches = [entry for entry in decision["trace"] if np.array_equal(np.asarray(entry["requested"]), probes)]
            if len(matches) != 1:
                raise ValueError("probe batch ambiguity")
            np.testing.assert_array_equal(data["actions"][i], np.asarray(matches[0]["raw"], dtype=np.float32))
            np.testing.assert_array_equal(data["model_positions"][i], np.asarray(matches[0]["positions"]))
            observed = ((projected[i, 1:11]-projected[i, 11:21])/.1).T
            model = ((data["model_positions"][i, 1:11]-data["model_positions"][i, 11:21])/.1).T
            np.testing.assert_allclose(observed, row["observed_matrix"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(model, row["model_matrix"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.mean((data["observed"][i]-data["target"])**2, axis=1),
                                       row["observed_goal_mse"], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(np.asarray(matches[0]["scores"]), np.asarray(row["model_goal_mse"]))
            upper = np.triu_indices(21, k=1)
            a, b = np.asarray(row["model_goal_mse"]), np.asarray(row["observed_goal_mse"])
            da, db = a[:, None]-a[None, :], b[:, None]-b[None, :]
            valid = (da[upper] != 0) & (db[upper] != 0)
            agreement = float(np.mean(np.sign(da[upper][valid]) == np.sign(db[upper][valid]))) if np.any(valid) else None
            if row["pair_count"] != int(valid.sum()) or row["pairwise_order_agreement_non_ties"] != agreement:
                raise ValueError("ordering calculation mismatch")
            if not np.isclose(row["matrix_abs_frobenius_error"], np.linalg.norm(model-observed), rtol=0, atol=1e-12):
                raise ValueError("matrix difference mismatch")
            checks.append({"nominal": i, "pair_count": int(valid.sum()), "order_agreement": agreement})
    seal.verify(manifest["seal"])
    run.configured.live.host.write_new(OUT, {"result_sha256": sha(run.RESULT), "arrays_sha256": sha(run.ARRAYS),
        "decision_sha256": sha(decision_path), "checks": checks,
        "scope": "saved arrays and statistic replay; physical execution not independently repeated"})
    print("Recomputed both probe matrices and all reported ordering counts.", flush=True)


if __name__ == "__main__":
    main()
