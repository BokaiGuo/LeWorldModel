import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p5_rgb_forward_probe_supervision_20261030_codex_v1 as m


def report():
    return {"created_utc": "2026-10-30T00:00:00+00:00", "disk": {"min_free_mib": 100},
            "source_check": {"ok": True}, "driver_versions": {"ok": True},
            "device_nodes": {"ok": True}, "python_environment": {"ok": True}}


def events():
    gpu = "GPU-12345678-1234-1234-1234-123456789abc"
    created = {"event": "created", "producer_sha256": "code", "probe_sha256": m.PROBE_SHA,
               "contract_sha256": m.PREP_SHA, "baseline_sha256": m.wiring.BASE_SHA,
               "hostgate_sha256": m.HOSTGATE_SHA, "supervisor_source_sha256": m.SUP_SHA,
               "preflight_name": "report", "preflight_sha256": "sha", "max_age_seconds": 300,
               "verified_utc": "2026-10-30T00:00:01+00:00", "preflight_age_seconds": 1.,
               "fresh_static": report(), "watchdog_max_seconds": 100,
               "watchdog": {"manager": "user", "uid": 1000, "unit": "probe.service",
                            "control_group": "/user.slice/user-1000.slice/user@1000.service/app.slice/probe.service",
                            "runtime_max_seconds": 100.,
                            "timeout_stop_seconds": 10., "kill_mode": "control-group", "send_sigkill": True},
               "gpu_uuid": gpu, "run_suffix": "X", "min_free_mib": 100, "min_total_mib": 200,
               "hard_seconds": 50, "runner_seconds": 40, "retry": False}
    snap = {"uuid": gpu, "total_mib": 200, "free_mib": 100, "compute_mode": "Default", "mig_mode": "Disabled"}
    return [created, {"event": "admitted", "snapshot": snap},
            {"event": "launch_attempt", "command": m.command("X", 40),
             "cuda_visible_devices": gpu, "cuda_device_order": "PCI_BUS_ID"},
            {"event": "launched", "pid": 42}, {"event": "finished", "status": "complete_self_report_only",
             "retry": False, "writer_may_be_alive": False, "review": {"journal_sha256": "journal"},
             "execution": {"status": "exited", "returncode": 0, "pid": 42, "elapsed_seconds": 5.}}]


@pytest.fixture
def static_stub(monkeypatch):
    monkeypatch.setattr(m.hostgate, "static_check", lambda data: 100)


def test_supervisor_valid(static_stub):
    assert m.validate_supervisor(events(), "X", "code", report()) == {"journal_sha256": "journal"}


@pytest.mark.parametrize("mutation", ["memory", "total", "uuid", "mig", "compute", "timeout", "returncode",
                                      "pid", "retry", "command", "mapping", "watchdog", "uid", "manager", "age", "contract", "order"])
def test_supervisor_rejects_bad_evidence(static_stub, mutation):
    e = copy.deepcopy(events())
    if mutation == "memory":
        e[1]["snapshot"]["free_mib"] = 99
    elif mutation == "total":
        e[1]["snapshot"]["total_mib"] = 199
    elif mutation == "uuid":
        e[1]["snapshot"]["uuid"] = "wrong"
    elif mutation == "mig":
        e[1]["snapshot"]["mig_mode"] = "Enabled"
    elif mutation == "compute":
        e[1]["snapshot"]["compute_mode"] = "Prohibited"
    elif mutation == "timeout":
        e[-1]["execution"]["status"] = "timeout"
    elif mutation == "returncode":
        e[-1]["execution"]["returncode"] = 1
    elif mutation == "pid":
        e[-1]["execution"]["pid"] = 43
    elif mutation == "retry":
        e[-1]["retry"] = True
    elif mutation == "command":
        e[2]["command"][-1] = "41"
    elif mutation == "mapping":
        e[2]["cuda_visible_devices"] = "other"
    elif mutation == "watchdog":
        e[0]["watchdog"]["kill_mode"] = "process"
    elif mutation == "uid":
        e[0]["watchdog"]["uid"] = 1001
    elif mutation == "manager":
        e[0]["watchdog"]["manager"] = "system"
    elif mutation == "age":
        e[0]["verified_utc"] = "2026-10-30T01:00:00+00:00"
    elif mutation == "contract":
        e[0]["contract_sha256"] = "bad"
    else:
        e[1], e[2] = e[2], e[1]
    with pytest.raises(ValueError):
        m.validate_supervisor(e, "X", "code", report())


def test_actual_process_group_timeout_cpu_only(tmp_path):
    with (tmp_path / "out").open("wb") as out, (tmp_path / "err").open("wb") as err:
        result = m.sup.supervise([sys.executable, "-c", "import time; time.sleep(30)"], {},
                                 str(tmp_path), out, err, 0.2)
    assert result["status"] == "timeout" and result["kill"] == "sigkill_sent"
    assert result["returncode"] is not None and result["elapsed_seconds"] < 5


def test_memory_denial_does_not_launch_and_preserves_log(tmp_path, monkeypatch):
    sdir, rdir = tmp_path / "supervisor", tmp_path / "run"
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "paths", lambda suffix: (sdir, rdir))
    monkeypatch.setattr(m, "preflight", lambda: ({}, {}))
    monkeypatch.setattr(m, "static_binding", lambda args: (report(), report(), 1., events()[0]["watchdog"]))
    gpu = events()[0]["gpu_uuid"]
    monkeypatch.setattr(m.sup, "query_gpu", lambda uuid: {"uuid": uuid, "total_mib": 200, "free_mib": 1})
    monkeypatch.setattr(m.sup, "supervise", lambda *a, **k: pytest.fail("child launched"))
    args = SimpleNamespace(authorization=m.AUTH, gpu_uuid=gpu, run_suffix="X", min_free_mib=100, min_total_mib=200,
                           hard_seconds=50, runner_seconds=40, max_age_seconds=300, watchdog_max_seconds=100,
                           preflight_name="report", preflight_sha="sha", watchdog_unit="probe.service")
    assert m.launch(args) == 2
    log = [json.loads(line) for line in (sdir / "supervisor.jsonl").read_bytes().splitlines()]
    assert [x["event"] for x in log] == ["created", "finished"]
    assert log[-1]["status"] == "partial_or_not_started" and log[-1]["retry"] is False
    assert not rdir.exists()


@pytest.mark.parametrize("failure", ["expired_after_preflight", "watchdog_changed", "timeout"])
def test_launch_rechecks_gate_and_never_retries(tmp_path, monkeypatch, failure):
    sdir, rdir = tmp_path / "supervisor", tmp_path / "run"
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "paths", lambda suffix: (sdir, rdir))
    monkeypatch.setattr(m, "preflight", lambda: ({}, {}))
    watch = events()[0]["watchdog"]
    monkeypatch.setattr(m, "static_binding", lambda args: (report(), report(), 1., watch))
    monkeypatch.setattr(m.hostgate, "load_report", lambda *a: report())
    calls = {"age": 0, "watch": 0, "launch": 0}
    def age(*args):
        calls["age"] += 1
        if failure == "expired_after_preflight" and calls["age"] == 2:
            raise ValueError("expired")
    def watchdog(*args):
        calls["watch"] += 1
        if failure == "watchdog_changed" and calls["watch"] == 2:
            return dict(watch, unit="impostor.service")
        return watch
    def supervise(*args):
        calls["launch"] += 1
        args[-1](42)
        return {"status": "timeout", "returncode": -9, "pid": 42, "elapsed_seconds": 1.}
    monkeypatch.setattr(m.hostgate, "age_ok", age)
    monkeypatch.setattr(m, "user_watchdog", watchdog)
    monkeypatch.setattr(m.sup, "query_gpu", lambda uuid: events()[1]["snapshot"])
    monkeypatch.setattr(m.sup, "supervise", supervise)
    args = SimpleNamespace(authorization=m.AUTH, gpu_uuid=events()[0]["gpu_uuid"], run_suffix="X",
                           min_free_mib=100, min_total_mib=200, hard_seconds=50, runner_seconds=40,
                           max_age_seconds=300, watchdog_max_seconds=100, preflight_name="report",
                           preflight_sha="sha", watchdog_unit="probe.service")
    assert m.launch(args) == 2
    log = [json.loads(line) for line in (sdir / "supervisor.jsonl").read_bytes().splitlines()]
    assert log[-1]["retry"] is False
    assert calls["launch"] == (1 if failure == "timeout" else 0)
    if failure == "timeout":
        assert log[-1]["status"] == "partial" and log[-1]["execution"]["status"] == "timeout"
    else:
        assert log[-1]["status"] == "partial_or_not_started"
        assert all(event["event"] != "launch_attempt" for event in log)


def test_checked_review_rejects_external_sha_before_consuming(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "events.jsonl").write_bytes(b"tampered\n")
    monkeypatch.setattr(m.probe, "review", lambda *args: pytest.fail("should not review"))
    with pytest.raises(ValueError, match="external journal"):
        m.checked_review(directory, {}, {}, "0" * 64)


def show_response(group):
    fields = {"ControlGroup": group, "ActiveState": "active", "SubState": "running", "KillMode": "control-group",
              "SendSIGKILL": "yes", "RuntimeMaxUSec": "100s", "TimeoutStopUSec": "10s"}
    return SimpleNamespace(stdout="".join(f"{k}={v}\n" for k, v in fields.items()))


def test_user_watchdog_binds_user_bus_uid_and_cgroup(tmp_path, monkeypatch):
    group = "/user.slice/user-1000.slice/user@1000.service/app.slice/probe.service"
    cg = tmp_path / "cgroup"
    cg.write_text("0::" + group + "\n")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/impostor")
    def systemctl(command, **kwargs):
        assert command[:4] == ["/usr/bin/systemctl", "--user", "show", "probe.service"]
        assert kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
        assert kwargs["env"]["XDG_RUNTIME_DIR"] == "/run/user/1000"
        return show_response(group)
    watch = m.user_watchdog("probe.service", 100, run=systemctl, cgroup_path=str(cg), uid=1000)
    assert watch["manager"] == "user" and watch["uid"] == 1000 and watch["control_group"] == group


@pytest.mark.parametrize("variant", ["other_unit", "other_uid", "other_cgroup", "scope", "inactive", "duplicate"])
def test_user_watchdog_rejects_identity_impostors(tmp_path, variant):
    group = "/user.slice/user-1000.slice/user@1000.service/app.slice/probe.service"
    proc_group, reported_group = group, group
    if variant == "other_unit":
        reported_group = group.replace("probe.service", "other.service")
    elif variant == "other_uid":
        reported_group = group.replace("1000", "1001")
        proc_group = reported_group
    elif variant == "other_cgroup":
        proc_group = group.replace("probe.service", "other.service")
    elif variant == "scope":
        reported_group = "/user.slice/user-1000.slice/session.scope"
        proc_group = reported_group
    cg = tmp_path / "cgroup"
    cg.write_text("0::" + proc_group + "\n")
    def systemctl(*args, **kwargs):
        answer = show_response(reported_group)
        if variant == "inactive":
            answer.stdout = answer.stdout.replace("ActiveState=active", "ActiveState=inactive")
        elif variant == "duplicate":
            answer.stdout += "ControlGroup=anything\n"
        return answer
    with pytest.raises(ValueError):
        m.user_watchdog("probe.service", 100, run=systemctl, cgroup_path=str(cg), uid=1000)
