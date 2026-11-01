"""P0 pre-inference source wiring; no neural computation."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import _seal_p0_sources


def test_p0_archives_explicit_sources_and_args_before_run(tmp_path):
    args = SimpleNamespace(output=tmp_path / "run", device="cuda", roots=2)
    receipt = _seal_p0_sources(args)
    assert receipt["source_count"] == 15
    assert not (args.output / "report.json").exists()
    assert verify_seal(output_dir=receipt["output_dir"],
                       expected_manifest_sha256=receipt["manifest_sha256"])["status"] == "verified"
    with pytest.raises(FileExistsError):
        _seal_p0_sources(args)


def test_p0_changed_archived_args_rejected(tmp_path):
    args = SimpleNamespace(output=tmp_path / "run", device="cuda")
    receipt = _seal_p0_sources(args)
    (args.output / "prerun_source_seal" / "args.json").write_text("{}\n")
    with pytest.raises(ValueError, match="args SHA256"):
        verify_seal(output_dir=receipt["output_dir"],
                    expected_manifest_sha256=receipt["manifest_sha256"])
