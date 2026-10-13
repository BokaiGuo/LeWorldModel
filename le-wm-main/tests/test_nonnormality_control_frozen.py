from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_nonnormality_control_frozen import run_report


def test_nonnormality_control_smoke(tmp_path: Path):
    rng = np.random.default_rng(12)
    source = rng.normal(size=(64, 6)).astype(np.float32)
    operator = np.diag(np.linspace(0.7, 1.1, 6)).astype(np.float32)
    operator[0, 1] = 0.8
    target = (source @ operator.T).astype(np.float32)
    archive = tmp_path / "latents.npz"
    np.savez(archive, latents=source, predicted_dynamics=target - source)
    output = tmp_path / "report.json"

    report = run_report(archive, output, sample_caps=[32], seeds=[3], ridge=1e-3, horizons=[1, 2, 4])

    record = report["records"][0]
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert np.isfinite(record["operator"]["henrici_nonnormality"])
    assert set(record["operator"]["operator_powers"]) == {"1", "2", "4"}
