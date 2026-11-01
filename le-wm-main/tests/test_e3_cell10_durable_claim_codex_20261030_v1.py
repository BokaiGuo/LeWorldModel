import concurrent.futures
import json
import os

import pytest

from experiments.spectral_reachability import e3_cell10_durable_claim_codex_20261030_v1 as m


def test_concurrent_callers_have_one_durable_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DIRECTORY", tmp_path)
    def attempt(_):
        try:
            return m.claim("a" * 64)
        except FileExistsError:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(attempt, range(24)))
    assert sum(x is not None for x in outcomes) == 1
    assert next(x for x in outcomes if x)["global_fence"] is False
    record = json.loads((tmp_path / m.MARKER).read_text())
    assert record["cell"] == 10 and record["authority_sha256"] == m.AUTHORITY_SHA
    assert record["closure_sha256"] == "a" * 64


def test_write_failure_keeps_marker_and_prevents_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DIRECTORY", tmp_path)
    def fail(*args):
        raise OSError("disk failure")
    with monkeypatch.context() as patch:
        patch.setattr(m.os, "fsync", fail)
        with pytest.raises(OSError, match="disk failure"):
            m.claim("b" * 64)
    assert (tmp_path / m.MARKER).exists()
    with pytest.raises(FileExistsError):
        m.claim("b" * 64)


def test_invalid_digest_and_existing_symlink_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DIRECTORY", tmp_path)
    with pytest.raises(ValueError):
        m.claim("x" * 64)
    target = tmp_path / "preserved"
    target.write_text("untouched")
    os.symlink(target, tmp_path / m.MARKER)
    with pytest.raises(FileExistsError):
        m.claim("c" * 64)
    assert target.read_text() == "untouched"


def test_directory_fsync_failure_leaves_non_reusable_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DIRECTORY", tmp_path)
    actual_fsync = os.fsync
    count = 0
    def fail_second(fd):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("directory sync failed")
        actual_fsync(fd)
    with monkeypatch.context() as patch:
        patch.setattr(m.os, "fsync", fail_second)
        with pytest.raises(OSError, match="directory sync failed"):
            m.claim("d" * 64)
    assert (tmp_path / m.MARKER).exists()
    with pytest.raises(FileExistsError):
        m.claim("d" * 64)
