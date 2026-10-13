from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_conditioned_controllability_robustness import run_report


def test_conditioned_controllability_robustness_smoke(tmp_path: Path):
    rng = np.random.default_rng(7)
    n = 96
    actions = rng.normal(size=(n, 3)).astype(np.float32)
    states = rng.normal(size=(n, 4)).astype(np.float32)
    dynamics = np.concatenate(
        [actions @ rng.normal(size=(3, 5)), rng.normal(size=(n, 7))], axis=1
    ).astype(np.float32)
    source = tmp_path / "latents.npz"
    np.savez(source, actions=actions, states=states, dynamics=dynamics)
    output = tmp_path / "report.json"

    report = run_report(source, output, sample_caps=[64], clusters=[2], seeds=[3])

    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert len(report["records"]) == 1
    assert len(report["aggregates"]) == 1
    aggregate = report["aggregates"][0]
    assert aggregate["num_seeds"] == 1
    assert np.isfinite(aggregate["conditional_action_r2_mean"])
