from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p2_b0_isolated_consume_20261102_exclusive as m


def test_isolation_command():
    cmd = m.command(Path("/tmp/historical.py"))
    assert cmd[cmd.index("--ro-bind") + 1:cmd.index("--ro-bind") + 3] == ["/", "/"]
    assert ["--ro-bind", "/tmp/historical.py", str(m.recovery.TARGET)] == cmd[cmd.index("--proc") + 2:cmd.index("--proc") + 5]
    assert ["--dev", "/dev"] == cmd[cmd.index("--dev"):cmd.index("--dev") + 2]
    assert "CUDA_VISIBLE_DEVICES=" in cmd and "PYTHONDONTWRITEBYTECODE=1" in cmd
    assert "--bind" not in cmd


def test_historical_wrong_sha_fails(monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b"wrong"))
    with pytest.raises(ValueError, match="historical source"):
        m.historical_bytes()


def test_git_history_actual_bytes():
    assert m.digest(m.historical_bytes()) == m.recovery.HISTORICAL_SHA
