from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_conditional_whitening_control_frozen import run_report


def test_conditional_whitening_control_smoke(tmp_path: Path):
    rng = np.random.default_rng(9)
    source = rng.normal(size=(96, 6)).astype(np.float32)
    actions = rng.normal(size=(96, 2)).astype(np.float32)
    target = (source @ rng.normal(size=(6, 6)) + actions[:, :1] @ rng.normal(size=(1, 6))).astype(np.float32)
    archive = tmp_path / "latents.npz"
    np.savez(archive, latents=source, predicted_dynamics=target - source, actions=actions)
    output = tmp_path / "report.json"

    report = run_report(archive, output, sample_caps=[64], seeds=[3], bins=[2], ridge=1e-3)

    record = report["records"][0]
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert record["global_whitened"]["normalized_frobenius"] < 1e-8
    assert np.isfinite(record["local_whitened_weighted"]["normalized_frobenius"])
