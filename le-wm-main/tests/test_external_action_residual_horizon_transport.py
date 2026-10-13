import json
import pickle

import numpy as np
import torch

from experiments.empirical_spectral_audit.run_external_action_residual_horizon_transport import run_report


def _fixture(tmp_path):
    vectors = tmp_path / "vectors.npz"
    arrays = {}
    for horizon in (1, 2, 4, 8):
        target = np.zeros((4, 8), dtype=np.float64)
        target[:, 0] = horizon * np.arange(1, 5)
        arrays[f"observed/h{horizon}/target"] = target
    np.savez(vectors, **arrays)

    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "episode_id": episode,
                        "horizon": 1,
                        "mode": "observed",
                        "start_observation": window,
                    }
                    for episode in (0, 1)
                    for window in (0, 1)
                ]
            }
        )
    )

    val_root = tmp_path / "val"
    val_root.mkdir()
    torch.save(torch.zeros((2, 100, 2)), val_root / "rel_actions.pth")
    with (val_root / "seq_lengths.pkl").open("wb") as handle:
        pickle.dump([100, 100], handle)

    basis_paths = []
    for index in range(2):
        basis = tmp_path / f"basis_{index}.npz"
        np.savez(
            basis,
            covariance_basis=np.eye(8)[:, 2:],
            dynamics_basis=np.eye(8)[:, :2],
            conditional_dynamics_basis=np.eye(8)[:, :2],
        )
        basis_paths.append(basis)
    return vectors, metadata, val_root, basis_paths


def test_action_residual_transport_is_crossfit_and_finite(tmp_path):
    vectors, metadata, val_root, basis_paths = _fixture(tmp_path)
    report = run_report(
        vectors_path=vectors,
        metadata_path=metadata,
        val_root=val_root,
        basis_paths=basis_paths,
        output=tmp_path / "report.json",
        ks=[2],
        random_repeats=8,
        bootstrap_repeats=20,
        ridge=1e-3,
        seed=7,
    )
    assert report["status"] == "complete_crossfit_action_residual_transport"
    assert report["summary"]["public/dynamics_basis/h1_to_h2/k2"][
        "delta_vs_covariance_positive_all_sources"
    ]
    assert report["claim_matrix"][0]["verdict"] == "supported as action-conditioned latent diagnostic"
