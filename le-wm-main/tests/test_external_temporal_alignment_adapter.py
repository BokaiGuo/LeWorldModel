import numpy as np
import json
from pathlib import Path

from experiments.empirical_spectral_audit.run_external_temporal_alignment_adapter import (
    TemporalAdapterConfig,
    _sign_flip,
    _split_episodes,
)
from experiments.empirical_spectral_audit.summarize_external_temporal_alignment_adapter import summarize


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "results/pusht/lewm/external_temporal_alignment_adapter_normalization_robustness_20261004.json"


def test_temporal_adapter_episode_split_is_disjoint_and_nonempty():
    split = _split_episodes(list(range(21)), TemporalAdapterConfig(split_seed=3072))
    groups = [set(values) for values in split.values()]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set().union(*groups) == set(range(21))


def test_temporal_adapter_sign_flip_uses_the_declared_small_sample_reference():
    assert _sign_flip(np.asarray([-1.0, -2.0, -3.0])) == 0.25
    assert _sign_flip(np.arange(13, dtype=np.float64)) is None


def test_normalization_summary_has_unique_protocol_rows_and_h8_coverage():
    report = json.loads(ROBUSTNESS.read_text())
    assert len(report["normalizations"]) == 3
    assert len(set(report["normalizations"])) == 3
    assert len(report["rows"]) == 84
    assert all(row["n_split_seeds"] == 3 for row in report["rows"])
    assert sum(row["horizon"] == 8 for row in report["rows"]) == 21


def test_normalization_summary_rejects_duplicate_normalization(tmp_path):
    source = ROOT / "results/pusht/lewm/external_temporal_alignment_adapter_20261003.json"
    duplicate = tmp_path / "duplicate.json"
    payload = json.loads(source.read_text())
    duplicate.write_text(json.dumps(payload))
    try:
        summarize([source, duplicate], tmp_path / "out.json")
    except ValueError as exc:
        assert "distinct normalization" in str(exc)
    else:
        raise AssertionError("duplicate normalization should be rejected")
