from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_geometry_error_association_frozen import run_report


def test_geometry_error_association_smoke(tmp_path: Path):
    rng = np.random.default_rng(21)
    source = rng.normal(size=(96, 6)).astype(np.float32)
    actions = rng.normal(size=(96, 2)).astype(np.float32)
    target = (source @ rng.normal(size=(6, 6)) + actions[:, :1] @ rng.normal(size=(1, 6))).astype(np.float32)
    archive = tmp_path / "latents.npz"
    np.savez(archive, latents=source, predicted_dynamics=target - source, actions=actions)
    output = tmp_path / "report.json"

    report = run_report(archive, output, sample_caps=[64], seeds=[3], bins=[2], ridge=1e-3)

    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert len(report["records"]) == 1
    assert len(report["records"][0]["group_results"]) == 2
    assert np.isfinite(report["records"][0]["group_results"][0]["heldout_residual_mse"])
