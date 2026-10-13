import json

import numpy as np

from experiments.theory_spectral_transfer.run_cross_model_temporal_geometry_audit import run_audit


def _write_artifact(path, seed_offset: float) -> None:
    rng = np.random.default_rng(4)
    target = rng.normal(size=(12, 8))
    predicted = target @ (np.eye(8) * (0.8 + seed_offset)) + 0.03 * rng.normal(size=(12, 8))
    path.write_text(
        json.dumps(
            {
                "protocol": {"model": "synthetic"},
                "operator_fit_vectors": {
                    "target": {"1": target.tolist(), "2": target.tolist()},
                    "predicted": {"1": predicted.tolist(), "2": predicted.tolist()},
                },
            }
        )
    )


def test_cross_model_audit_uses_heldout_operator_residual(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_artifact(first, 0.0)
    _write_artifact(second, 0.05)
    report = run_audit({"A": first, "B": second}, [0, 1], top_k=4)
    assert len(report["records"]) == 8
    assert len(report["summary"]) == 4
    assert report["cross_model_association"]["commutator_vs_holdout_residual"]["n"] == 8
    assert all(np.isfinite(row["holdout_residual_mse"]) for row in report["records"])
