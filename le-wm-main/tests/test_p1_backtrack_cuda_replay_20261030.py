from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_backtrack_cuda_replay_20261030 as replay


def test_sources_include_runner_and_unbound_helpers():
    bound = replay.files()
    assert str(Path(replay.__file__).resolve()) in bound
    assert str(Path(replay.inventory.verifier.producer.base.__file__).resolve()) in bound
    assert str(replay.PROTOCOL.resolve()) in bound
    assert all(len(h) == 64 for h in bound.values())


def test_prepare_rejects_existing_or_outside_results(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        replay.prepare(tmp_path / "p1_backtrack_cuda_replay_20261030_bad.json")
    with pytest.raises(ValueError, match="exclusive"):
        replay.prepare(replay.ROOT / "results/p1_backtrack_cuda_replay_20261030_prepare_MAIN01.json")


def test_manifest_sha_rejection_before_cuda(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        replay.run(str(manifest), "0" * 64, str(tmp_path / "out.json"))
