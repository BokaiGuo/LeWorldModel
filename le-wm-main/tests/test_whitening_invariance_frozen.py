from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_whitening_invariance_frozen import run_report


def test_whitening_invariance_control_smoke(tmp_path: Path):
    rng = np.random.default_rng(4)
    source = rng.normal(size=(64, 6)).astype(np.float32)
    target = (source @ rng.normal(size=(6, 6)) + 0.01 * rng.normal(size=(64, 6))).astype(np.float32)
    archive = tmp_path / "latents.npz"
    np.savez(archive, latents=source, predicted_dynamics=target - source)
    output = tmp_path / "report.json"

    report = run_report(archive, output, sample_caps=[32], seeds=[3], ridge=1e-3)

    record = report["records"][0]
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert np.isfinite(record["original"]["commutator"]["normalized_frobenius"])
    assert record["global_whitening"]["stable_rank"] > 5.0
