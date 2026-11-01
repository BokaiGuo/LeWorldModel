import json
import sys
from pathlib import Path

import pytest

from experiments.spectral_reachability import p2_replay_full_source_preflight_20261101_codex as m


def test_source_census_excludes_outputs_and_rejects_symlinks(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "generated.py").write_text("")
    assert m.sources(tmp_path) == [str(tmp_path / "module.py")]
    (tmp_path / "alias.py").symlink_to(tmp_path / "module.py")
    with pytest.raises(ValueError, match="symlink"):
        m.sources(tmp_path)


def test_snapshot_detects_file_changes_additions_and_inventory_drift(tmp_path, monkeypatch):
    source = tmp_path / "main.py"
    source.write_text("x = 1\n")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    # Default arguments capture the real project, so explicitly bind the fixture.
    real_snapshot = m.snapshot
    monkeypatch.setattr(m, "snapshot", lambda inv, sha: real_snapshot(inv, sha, tmp_path))
    monkeypatch.setattr(m.inherited, "verify", lambda *_: {"prepared": "p", "prepared_sha256": "h"})
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps(m.snapshot(inventory, "digest")))
    digest = m.inherited.sha(snap)
    assert m.verify(snap, digest)["P2_admission"] is False
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="drift"):
        m.verify(snap, digest)
    source.write_text("x = 1\n")
    (tmp_path / "new.py").write_text("")
    with pytest.raises(ValueError, match="drift"):
        m.verify(snap, digest)
    with pytest.raises(ValueError, match="SHA"):
        m.verify(snap, "wrong")


def test_inherited_failure_is_not_suppressed(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("")
    def fail(*_):
        raise ValueError("inherited drift")
    monkeypatch.setattr(m.inherited, "verify", fail)
    with pytest.raises(ValueError, match="inherited drift"):
        m.snapshot(tmp_path / "inv", "sha", tmp_path)


def test_execute_rechecks_snapshot_at_each_launcher_verification(tmp_path, monkeypatch):
    from experiments.spectral_reachability import p2_replay_exclusive_cuda_20261101 as launcher

    inv = tmp_path / "inventory.json"
    inv.write_text("{}")
    binding = {"inherited_inventory": str(inv), "inherited_inventory_sha256": "abc"}
    calls = []
    def verify(*_):
        calls.append("snapshot")
        if len(calls) == 3:
            raise ValueError("locked drift")
        return binding
    def original(*_):
        return binding
    def execute(**kwargs):
        assert kwargs["coordinated"] is True
        launcher.seal.verify(inv, "abc")
        launcher.seal.verify(inv, "abc")
        pytest.fail("drift must stop execution")
    monkeypatch.setattr(m, "verify", verify)
    monkeypatch.setattr(m.inherited, "verify", original)
    monkeypatch.setattr(launcher, "execute", execute)
    monkeypatch.setattr(sys, "argv", ["preflight", "execute", "--snapshot", "s",
                       "--snapshot-sha256", "h", "--output", str(tmp_path / "out"),
                       "--device-uuid", "GPU-a", "--operator-coordinated"])
    with pytest.raises(ValueError, match="locked drift"):
        m.main()
    assert len(calls) == 3
    assert m.inherited.verify is original
