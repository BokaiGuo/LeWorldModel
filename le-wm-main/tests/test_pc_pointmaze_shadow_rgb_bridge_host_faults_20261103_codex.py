"""Opt-in host integration: kill or hang the real shadow process."""
import os
from pathlib import Path
import signal
import time

import pytest

from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge

pytestmark = pytest.mark.skipif(os.environ.get("POINTMAZE_RUN_GPU") != "1",
                                reason="requires explicit NVIDIA host opt-in")


@pytest.mark.parametrize("sig,expected", [(signal.SIGKILL, RuntimeError), (signal.SIGSTOP, TimeoutError)])
def test_real_shadow_failure_poison_and_cleanup(sig, expected):
    env = ShadowRGBBridge([1.0, 1.0, 0.0, 0.0], timeout=30)
    try:
        path = Path(f"/proc/{env._process.pid}/task/{env._process.pid}/children")
        children = [int(x) for x in path.read_text().split()]
        assert len(children) == 1
        shadow_pid = children[0]
        os.kill(shadow_pid, sig)
        env.timeout = 0.5
        with pytest.raises(expected):
            env.step([[0.0, 0.0]])
        assert env.step_count == 0
        with pytest.raises(RuntimeError):
            env.reset()
        for _ in range(100):
            stat = Path(f"/proc/{shadow_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("real shadow process remains alive")
    finally:
        env.close()
