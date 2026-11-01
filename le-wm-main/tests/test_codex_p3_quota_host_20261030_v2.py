import json
import os
import pytest

from experiments.spectral_reachability import codex_p3_quota_host_20261030_v2 as host


def test_exclusive_durable_write(tmp_path):
    path = tmp_path / "x"
    sha = host.write_new(path, {"x": 1})
    assert sha == host.prior.sha(path)
    assert json.loads(path.read_bytes()) == {"x": 1}
    with pytest.raises(FileExistsError):
        host.write_new(path, {"x": 2})


def test_output_scope_rejects_existing_and_outside(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        host.output_path(str(tmp_path / (host.NAME + "_RUN_x")), "_RUN_")
    with pytest.raises(ValueError, match="exclusive"):
        host.output_path(str(host.ROOT / "results"), "_RUN_")


def test_lock_inode_check(tmp_path, monkeypatch):
    file = tmp_path / "lock"
    file.write_bytes(b"")
    monkeypatch.setattr(host, "LOCK", file)
    fd = os.open(file, os.O_RDONLY)
    try:
        assert host._same_lock(fd)
        other = tmp_path / "other"
        other.write_bytes(b"")
        monkeypatch.setattr(host, "LOCK", other)
        assert not host._same_lock(fd)
    finally:
        os.close(fd)


def test_gpu_snapshot_rejects_insufficient(monkeypatch):
    monkeypatch.setattr(host.subprocess, "check_output", lambda *a, **kw:
                        host.UUID + ", 100, 16311\n")
    with pytest.raises(ValueError, match="GPU admission"):
        host.gpu_snapshot()
