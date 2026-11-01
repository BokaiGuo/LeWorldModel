"""Synthetic systemd/cgroup proof tests; no service or GPU execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p3_stopped_cgroup_reaped_20261030_v1 as m


@pytest.fixture
def sample(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(m.os.path, "lexists", lambda p: False)
    service = dict.fromkeys(m.PROPERTIES, "")
    service.update(Id=m.UNIT, LoadState="loaded", ActiveState="failed", SubState="failed",
                   MainPID="0", ControlGroup="", Result="exit-code", InvocationID=m.INVOCATION,
                   ExecMainPID="123", ExecMainCode="1", ExecMainStatus="1",
                   ExecMainStartTimestampMonotonic="10", ExecMainExitTimestampMonotonic="20",
                   Slice="app.slice", KillMode="control-group",
                   ExecStart="python -m experiments.spectral_reachability.p3_dual_branch_host_driver_20261030_codex_v1 "
                   "--plan p3_dual_branch_host_driver_20261030_codex_PREP01.json "
                   "--plan-sha256 " + m.PLAN_SHA + " ")
    parent = dict(Id="app.slice", LoadState="loaded", ActiveState="active",
                  ControlGroup="/user.slice/user-1000.slice/user@1000.service/app.slice")
    mount = "12 1 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup rw"
    return service, parent, "0::/\n", mount


def verify(sample):
    s, p, pid, mount = sample
    return m.verify_snapshot(s, p, Path("/sys/fs/cgroup"), pid, mount, "cgroup:[42]", "cgroup:[42]")


def test_proof_reports_bound_reaped_cgroup(sample):
    proof = verify(sample)
    assert proof["target_absent"] and proof["invocation"] == m.INVOCATION
    assert "escaped" in proof["limitations"]


@pytest.mark.parametrize("bad", ["active", "pid", "invocation", "group", "kill", "exit",
                                 "cmd", "slice", "namespace", "mount", "time"])
def test_ambiguous_proof_rejected(sample, bad):
    s, p, pid, mount = deepcopy(sample)
    if bad == "active": s["ActiveState"] = "active"
    elif bad == "pid": s["MainPID"] = "22"
    elif bad == "invocation": s["InvocationID"] = "0" * 32
    elif bad == "group": s["ControlGroup"] = "/somewhere"
    elif bad == "kill": s["KillMode"] = "process"
    elif bad == "exit": s["ExecMainStatus"] = "0"
    elif bad == "cmd": s["ExecStart"] = "wrong command"
    elif bad == "slice": p["ControlGroup"] = "/unexpected"
    elif bad == "namespace": pid = "0::/sandbox\n"
    elif bad == "mount": mount = "12 1 0:30 /else /sys/fs/cgroup rw - cgroup2 cgroup rw"
    else: s["ExecMainExitTimestampMonotonic"] = "10"
    with pytest.raises(ValueError):
        verify((s, p, pid, mount))


def test_existing_cgroup_rejected(sample, monkeypatch):
    monkeypatch.setattr(m.os.path, "lexists", lambda p: True)
    with pytest.raises(ValueError, match="still exists"):
        verify(sample)


def test_namespace_mismatch_rejected(sample):
    s, p, pid, mount = sample
    with pytest.raises(ValueError, match="namespace"):
        m.verify_snapshot(s, p, Path("/sys/fs/cgroup"), pid, mount, "cgroup:[1]", "cgroup:[2]")
