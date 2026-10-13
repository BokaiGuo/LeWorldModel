import json

import numpy as np

from experiments.empirical_spectral_audit.run_external_dynamics_horizon_transport import run_report


def _write_fixture(tmp_path):
    vectors = tmp_path / "vectors.npz"
    targets = {}
    for horizon in (1, 2, 4, 8):
        array = np.zeros((4, 8), dtype=np.float64)
        array[:, 0] = (horizon / 2.0) * np.arange(1, 5)
        targets[f"observed/h{horizon}/target"] = array
    np.savez(vectors, **targets)

    metadata = tmp_path / "metadata.json"
    records = []
    for episode_id in (0, 1):
        for row in range(2):
            records.append(
                {
                    "episode_id": episode_id,
                    "horizon": 1,
                    "mode": "observed",
                    "mse": 0.0,
                    "start_observation": row,
                }
            )
    metadata.write_text(json.dumps({"records": records}))

    covariance = np.eye(8)[:, 2:]
    dynamics = np.eye(8)[:, :2]
    basis_paths = []
    for index in range(2):
        basis = tmp_path / f"basis_{index}.npz"
        np.savez(
            basis,
            covariance_basis=covariance,
            dynamics_basis=dynamics,
            conditional_dynamics_basis=dynamics,
        )
        basis_paths.append(basis)
    return vectors, metadata, basis_paths


def test_external_horizon_transport_separates_dynamics_from_covariance(tmp_path):
    vectors, metadata, basis_paths = _write_fixture(tmp_path)
    report = run_report(
        vectors_path=vectors,
        metadata_path=metadata,
        basis_paths=basis_paths,
        output=tmp_path / "report.json",
        ks=[2],
        random_repeats=8,
        bootstrap_repeats=20,
        seed=7,
    )
    summary = report["summary"]["dynamics_basis/h1_to_h2/k2"]
    assert summary["delta_vs_covariance_positive_all_sources"]
    assert summary["delta_vs_random_positive_all_sources"]
    assert report["claim_matrix"][0]["verdict"] == "supported as conditional latent-trajectory transport"
