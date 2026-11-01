import fcntl
import os
import threading

import pytest

from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import codex_p3_queued_cost_batch_20261030_v1 as batch


def test_lock_waits_and_preserves_inode(tmp_path):
    path = tmp_path / "lock"
    path.touch()
    fd = os.open(path, os.O_RDWR)
    inode = os.fstat(fd).st_ino
    fcntl.flock(fd, fcntl.LOCK_EX)
    entered = threading.Event()
    done = threading.Event()
    values = []

    def worker():
        try:
            with lock.gpu_lock(path) as waited:
                values.append(waited)
                entered.set()
        finally:
            done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert not entered.wait(0.15)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert done.wait(5)
    thread.join()
    assert entered.is_set()
    assert values[0] >= 0.1
    assert path.stat().st_ino == inode


def test_lock_rejects_symlink_and_relative_path(tmp_path):
    path = tmp_path / "lock"
    path.touch()
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        with lock.gpu_lock(link):
            pass
    with pytest.raises(ValueError):
        with lock.gpu_lock("relative"):
            pass


def test_fixed_six_slots_and_caps():
    assert batch.SLOTS == [(20, "uniform"), (20, "signal"), (20, "reverse"),
                           (30, "uniform"), (30, "signal"), (30, "reverse")]
    axes = ("action_encoder", "predict", "predictor", "pred_proj")
    actual = {**dict.fromkeys(axes, 48), "jacobian": 2, "vjp_directions": 384,
              "grad_api": 384, "raw_backward_hook": 384, "svd": 2, "metric": 24}
    row = {"attempted": actual, "point": 20, "arm": "signal"}
    report = {"point": 20, "arm": "signal", "elapsed_seconds": 1,
              "peak_allocated_bytes": 1, "peak_reserved_bytes": 1}
    assert batch.check_caps(20, row, report)["predict"] == 48
    actual["predict"] = 49
    with pytest.raises(Exception):
        batch.check_caps(20, row, report)
