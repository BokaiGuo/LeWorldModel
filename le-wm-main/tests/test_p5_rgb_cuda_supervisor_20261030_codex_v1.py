import os
import subprocess
import sys
import time

import pytest

from experiments.spectral_reachability import p5_rgb_cuda_supervisor_20261030_codex_v1 as sup

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def query_mock(total="24000", free="12000", mig="Disabled", mode="Default", uuid=GPU):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        value = mig if "mig.mode.current" in command[3] else f"{uuid}, {total}, {free}, {mode}"
        return subprocess.CompletedProcess(command, 0, value + "\n", "")
    return run, calls


def test_gpu_admission_uses_uuid_and_rejects_insufficient():
    query, calls = query_mock()
    snapshot = sup.query_gpu(GPU, query)
    sup.admit(snapshot, 10000, 20000)
    assert all(call[0][2] == GPU and call[1]["timeout"] == 10 for call in calls)
    with pytest.raises(ValueError, match="admission denied"):
        sup.admit(snapshot, 13000, 20000)


@pytest.mark.parametrize("args", [
    {"mig": "Enabled"}, {"mode": "Prohibited"}, {"uuid": "GPU-other"},
    {"free": "[N/A]"}, {"total": "100", "free": "200"}])
def test_gpu_denials(args):
    query, _ = query_mock(**args)
    with pytest.raises(ValueError):
        snapshot = sup.query_gpu(GPU, query)
        sup.admit(snapshot, 10, 100)


def test_bad_uuid_and_query_timeout():
    with pytest.raises(ValueError, match="UUID"):
        sup.query_gpu("0")
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("mock", 10)
    with pytest.raises(subprocess.TimeoutExpired):
        sup.query_gpu(GPU, timed_out)


def test_timeout_kills_process_group_and_never_retries(tmp_path):
    marker = tmp_path / "descendant_marker"
    child_script = ("import subprocess,sys,time; "
                    "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
                    "time.sleep(30)")
    descendant = "import time,pathlib,sys; time.sleep(0.8); pathlib.Path(sys.argv[1]).touch()"
    starts = []
    with (tmp_path / "out").open("wb") as out, (tmp_path / "err").open("wb") as err:
        result = sup.supervise([sys.executable, "-c", child_script, descendant, str(marker)],
                               dict(os.environ), tmp_path, out, err, 0.3, starts.append)
    assert result["status"] == "timeout"
    assert result["returncode"] == -9
    assert result["kill"] == "sigkill_sent"
    assert len(starts) == 1
    time.sleep(0.8)
    assert not marker.exists()


def test_normal_exit_and_spawn_error(tmp_path):
    with (tmp_path / "out").open("wb") as out, (tmp_path / "err").open("wb") as err:
        result = sup.supervise([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ),
                               tmp_path, out, err, 3)
        assert result["status"] == "exited" and result["returncode"] == 7
        with pytest.raises(FileNotFoundError):
            sup.supervise([str(tmp_path / "missing")], dict(os.environ), tmp_path, out, err, 1)


def test_invalid_and_missing_journal_preserved(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert sup.journal_summary(run_dir)["status"] == "no_journal"
    path = run_dir / "events.jsonl"
    path.write_bytes(b'{"incomplete":')
    report = sup.journal_summary(run_dir)
    assert report["status"] == "journal_not_verified"
    assert path.read_bytes() == b'{"incomplete":'


def test_source_and_input_preflight_no_torch():
    sup.checked_gate()
    assert "torch" not in sys.modules


def test_main_denial_never_launches(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sup.abi.bridge, "ROOT", tmp_path)
    monkeypatch.setattr(sup, "checked_gate", lambda: None)
    monkeypatch.setattr(sup, "query_gpu", lambda uuid: {
        "uuid": uuid, "total_mib": 24000, "free_mib": 1,
        "compute_mode": "Default", "mig_mode": "Disabled"})
    def forbidden(*args, **kwargs):
        pytest.fail("denied device must not launch")
    monkeypatch.setattr(sup, "supervise", forbidden)
    monkeypatch.setattr(sys, "argv", ["supervisor", "--authorization", sup.AUTH,
                        "--gpu-uuid", GPU, "--min-free-mib", "100",
                        "--min-total-mib", "20000", "--hard-seconds", "5",
                        "--runner-seconds", "2", "--run-suffix", "TEST"])
    assert sup.main() == 2
    log = (tmp_path / "results" / (sup.NAME + "_TEST") / "supervisor.jsonl").read_text()
    assert '"launch_attempt"' not in log
    assert '"partial_or_not_started"' in log
