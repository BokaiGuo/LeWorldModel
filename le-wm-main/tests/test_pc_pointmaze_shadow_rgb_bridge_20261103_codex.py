"""Transport and lifecycle tests using an isolated deterministic worker."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from experiments import pc_pointmaze_shadow_rgb_bridge_20261103_codex as bridge

FAKE = Path(__file__).parent / "fixtures/pc_shadow_bridge_fake_worker_20261103_codex.py"


def install_fake(monkeypatch, mode="ok", pid_path="-"):
    def launch(_):
        parent, child = socket.socketpair()
        proc = subprocess.Popen([sys.executable, str(FAKE), str(child.fileno()), mode, str(pid_path)],
                                pass_fds=(child.fileno(),), start_new_session=True)
        child.close()
        return proc, parent
    monkeypatch.setattr(bridge, "_scene", lambda: b"isolated test scene")
    monkeypatch.setattr(bridge, "_launch_worker", launch)


def test_reset_and_packet_clock(monkeypatch):
    install_fake(monkeypatch)
    with bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=2) as env:
        assert env.initial_rgb.shape == (224, 224, 3) and env.step_count == 0
        assert env.step([[0, 0], [1, -1]])[0, 0, 0] == 2
        assert env.step_count == 2
        assert env.reset()[0, 0, 0] == 0 and env.step_count == 0
        assert env.step([[0, 0]])[0, 0, 0] == 1
    env.close()


@pytest.mark.parametrize("bad", [[], [[2, 0]], [[float("nan"), 0]], [[0, 0]] * 6, [0, 0]])
def test_invalid_action_does_not_advance_or_poison(monkeypatch, bad):
    install_fake(monkeypatch)
    with bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=2) as env:
        with pytest.raises(ValueError):
            env.step(bad)
        assert env.step_count == 0
        env.step([[0, 0]])
        assert env.step_count == 1


@pytest.mark.parametrize("mode", ["hang", "partial", "exit", "clock", "sequence", "hash", "extra"])
def test_fault_poisoned_and_never_reused(monkeypatch, mode):
    install_fake(monkeypatch, mode)
    env = bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=0.5)
    with pytest.raises((RuntimeError, TimeoutError)):
        env.step([[0, 0]])
    assert env.step_count == 0
    with pytest.raises(RuntimeError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step([[0, 0]])
    env.close()


def test_failure_stops_worker_process_group(monkeypatch, tmp_path):
    path = tmp_path / "child.pid"
    install_fake(monkeypatch, "hang", path)
    env = bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=0.5)
    pid = int(path.read_text())
    with pytest.raises(TimeoutError):
        env.step([[0, 0]])
    for _ in range(100):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("worker descendant survived failure")
    env.close()


def test_failed_reset_poisoned_without_clock_reset(monkeypatch):
    install_fake(monkeypatch, "reset_clock")
    env = bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=2)
    env.step([[0, 0]])
    with pytest.raises(RuntimeError):
        env.reset()
    assert env.step_count == 1
    with pytest.raises(RuntimeError):
        env.step([[0, 0]])
    env.close()


def test_worker_death_between_operations(monkeypatch):
    install_fake(monkeypatch)
    env = bridge.ShadowRGBBridge([1, 2, 3, 4], timeout=2)
    env._process.kill()
    env._process.wait()
    with pytest.raises(RuntimeError):
        env.step([[0, 0]])
    with pytest.raises(RuntimeError):
        env.reset()
    env.close()


def test_bad_reset_configuration_rejected():
    with pytest.raises(ValueError):
        bridge.ShadowRGBBridge([0, 1, float("inf"), 2])
    with pytest.raises(ValueError):
        bridge.ShadowRGBBridge([0, 1, 2, 3], timeout=float("nan"))
