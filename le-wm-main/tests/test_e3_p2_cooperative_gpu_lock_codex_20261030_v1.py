import os
import subprocess
import sys

import pytest

from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


def test_process_competition_and_release(tmp_path):
    path = tmp_path / "gpu.lock"
    path.touch()
    code = ("from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 "
            "import gpu_lock; import sys\n"
            "with gpu_lock(sys.argv[1]): print('acquired')\n")
    with gpu_lock(path):
        result = subprocess.run([sys.executable, "-B", "-c", code, str(path)],
                                capture_output=True, text=True)
        assert result.returncode != 0
        assert "BlockingIOError" in result.stderr
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "acquired"


def test_exception_releases_and_preserves_file(tmp_path):
    path = tmp_path / "gpu.lock"
    path.touch()
    inode = path.stat().st_ino
    with pytest.raises(RuntimeError):
        with gpu_lock(path):
            raise RuntimeError("failure")
    with gpu_lock(path):
        assert path.stat().st_ino == inode


def test_rejects_missing_symlink_and_hardlink(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        with gpu_lock(missing):
            pass
    path = tmp_path / "lock"
    path.touch()
    link = tmp_path / "symlink"
    link.symlink_to(path)
    with pytest.raises(OSError):
        with gpu_lock(link):
            pass
    os.link(path, tmp_path / "hardlink")
    with pytest.raises(ValueError, match="single-link"):
        with gpu_lock(path):
            pass


def test_replacement_during_acquisition_rejected(tmp_path, monkeypatch):
    from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as m

    path = tmp_path / "lock"
    path.touch()
    original = m.fcntl.flock

    def replace(fd, operation):
        original(fd, operation)
        if operation == m.fcntl.LOCK_EX | m.fcntl.LOCK_NB:
            path.rename(tmp_path / "old")
            path.touch()

    monkeypatch.setattr(m.fcntl, "flock", replace)
    with pytest.raises(ValueError, match="replaced"):
        with gpu_lock(path):
            pytest.fail("replacement accepted")
