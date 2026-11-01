import copy
import datetime
import subprocess
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p5_rgb_launch_preflight_gate_20261030_codex_v1 as gate

GPU = "GPU-12345678-1234-1234-1234-123456789abc"
NOW = datetime.datetime(2026, 10, 30, 0, 0, 30, tzinfo=datetime.timezone.utc)


@pytest.fixture
def good_report():
    path = gate.abi.bridge.ROOT / "results/p5_rgb_host_preflight_20261030_codex_v1_MAIN01.json"
    report = gate.abi.bridge.read_json(path.read_bytes())
    report["created_utc"] = "2026-10-30T00:00:00+00:00"
    report["device_nodes"] = {key: {"exists": True, "character_device": True,
                                    "read_write_access_check": True} for key in gate.DEVICES}
    report["blockers"] = list(gate.PENDING)
    return report


def fake_watchdog(unit, max_seconds):
    return {"unit": unit, "control_group": "/test.service", "runtime_max_seconds": 120,
            "timeout_stop_seconds": 10, "kill_mode": "control-group", "send_sigkill": True}


def run_args():
    return SimpleNamespace(authorization=gate.AUTH, gpu_uuid=GPU, run_suffix="TEST",
                           preflight_name=gate.host.NAME + "_TEST.json", preflight_sha="",
                           min_free_mib=100, min_total_mib=2000, hard_seconds=60, runner_seconds=30,
                           max_age_seconds=60, watchdog_unit="test.service", watchdog_max_seconds=120)


def save_report(tmp_path, monkeypatch, report):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(gate.abi.bridge, "ROOT", tmp_path)
    args = run_args()
    data = gate.abi.bridge.canonical(report)
    (tmp_path / "results" / args.preflight_name).write_bytes(data)
    args.preflight_sha = gate.abi.bridge.digest(data)
    return args


def test_successful_binding_dispatches_only_fake_launcher(tmp_path, monkeypatch, good_report):
    args = save_report(tmp_path, monkeypatch, good_report)
    calls = []
    result = gate.launch(args, launch_fn=lambda x: calls.append(x) or 17,
                         inspect=lambda floor: copy.deepcopy(good_report), clock=lambda: NOW,
                         watchdog_fn=fake_watchdog)
    assert result == 17 and len(calls) == 1
    assert calls[0].gpu_uuid == GPU and calls[0].authorization == gate.wiring.AUTH
    receipt = gate.abi.bridge.read_json(
        (tmp_path / "results" / (gate.NAME + "_binding_TEST.json")).read_bytes())
    assert receipt["preflight_sha256"] == args.preflight_sha
    assert receipt["status"] == "static_verified_runtime_and_authorization_pending"
    assert receipt["external_watchdog_static_check"]["unit"] == "test.service"
    assert receipt["P5_admission"] is False
    with pytest.raises(FileExistsError):
        gate.launch(args, launch_fn=lambda x: pytest.fail("duplicate dispatch"),
                    inspect=lambda floor: copy.deepcopy(good_report), clock=lambda: NOW,
                    watchdog_fn=fake_watchdog)


@pytest.mark.parametrize("damage", ["hash", "stale", "future", "node", "source", "disk",
                                      "false_blockers", "changed_fresh", "watchdog"])
def test_rejections_do_not_dispatch(tmp_path, monkeypatch, good_report, damage):
    fresh = copy.deepcopy(good_report)
    if damage == "stale":
        good_report["created_utc"] = "2026-10-29T23:59:00+00:00"
    elif damage == "future":
        good_report["created_utc"] = "2026-10-30T00:01:00+00:00"
    elif damage == "node":
        good_report["device_nodes"]["/dev/nvidia0"]["exists"] = False
    elif damage == "source":
        good_report["source_check"]["ok"] = False
    elif damage == "disk":
        good_report["disk"]["free_mib"] = 0
    elif damage == "false_blockers":
        good_report["blockers"] = []
    elif damage == "changed_fresh":
        fresh["driver_versions"]["value"]["loaded"] = "0.0"
    args = save_report(tmp_path, monkeypatch, good_report)
    if damage == "hash":
        args.preflight_sha = "0" * 64
    def denied(*args):
        raise ValueError("watchdog denied")
    with pytest.raises(ValueError):
        gate.launch(args, launch_fn=lambda x: pytest.fail("must not dispatch"),
                    inspect=lambda floor: fresh, clock=lambda: NOW,
                    watchdog_fn=denied if damage == "watchdog" else fake_watchdog)


def test_watchdog_requires_matching_bounded_service(tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/system.slice/p5.service\n")
    values = {"ControlGroup": "/system.slice/p5.service", "ActiveState": "active",
              "SubState": "running", "KillMode": "control-group", "SendSIGKILL": "yes",
              "RuntimeMaxUSec": "2min", "TimeoutStopUSec": "10s"}
    def systemctl(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "".join(f"{k}={v}\n" for k, v in values.items()), "")
    result = gate.watchdog("p5.service", 120, run=systemctl, cgroup_path=str(cgroup))
    assert result["runtime_max_seconds"] == 120
    values["KillMode"] = "process"
    with pytest.raises(ValueError, match="cgroup"):
        gate.watchdog("p5.service", 120, run=systemctl, cgroup_path=str(cgroup))
    values["KillMode"] = "control-group"
    values["RuntimeMaxUSec"] = "infinity"
    with pytest.raises(ValueError, match="duration"):
        gate.watchdog("p5.service", 120, run=systemctl, cgroup_path=str(cgroup))


def test_actual_saved_report_blocks():
    name = gate.host.NAME + "_MAIN01.json"
    report = gate.load_report(name, "6e0066e92675933c7791fd9669de585c749756fc1dbbc719ca39594c00b49089")
    with pytest.raises(ValueError, match="static blockers"):
        gate.static_check(report)
