import fcntl
import os

import numpy as np
import pytest

from experiments.spectral_reachability import p1_queue_goal_blind_cuda_20261101_codex as c


def test_build_and_source_identity():
    p = c.build()
    assert p["seed"] == 301000001
    assert p["independent_confirmation"] is False
    assert p["contract"]["encoder_cap"] == p["contract"]["projector_cap"] == 4
    assert p["contract"]["predict_cap"] == 0


def test_prepare_closure_and_tamper(tmp_path):
    path = tmp_path / "prepare.json"
    result = c.prepare(path)
    assert c.verify_prepare(path, result["sha256"])["seed"] == 301000001
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        c.verify_prepare(path, result["sha256"])


def test_task_uses_all_four_arrays_and_raw_history():
    p = c.build()
    packet = c.b.load_blind(p["slots"][0]["blind_npz"], c.BLIND_SHA)
    values = [np.full(192, i, np.float32) for i in range(4)]
    task = c.export_task(p, "1" * 64, packet, values)
    assert task["task"]["goal"] == [3.] * 192
    assert task["task"]["root_context"]["latent_history"] == [v.tolist() for v in values[:3]]
    assert task["task"]["root_raw_action_history"] == [[0., 0.]] * 10
    assert task["new_independent_confirmation"] is False


def test_lock_exclusion_and_exception_release(tmp_path):
    path = tmp_path / "gpu.lock"
    path.touch()
    with pytest.raises(RuntimeError):
        with c.locking.gpu_lock(path):
            fd = os.open(path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            raise RuntimeError("fixture")
    with c.locking.gpu_lock(path):
        assert path.exists()
