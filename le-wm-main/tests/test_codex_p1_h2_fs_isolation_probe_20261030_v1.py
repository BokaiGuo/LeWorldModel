import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_p1_h2_fs_isolation_probe_20261030_v1.py"
spec = importlib.util.spec_from_file_location("h2_fs_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_real_namespace_denials(tmp_path):
    report = tmp_path / "audit.json"
    assert probe.run_probe(report)
    data = json.loads(report.read_text())
    assert data["status"] == "passed"
    assert all(data["checks"].values())
    assert not data["production_authorized"]
    assert not data["cuda_initialized"]
    assert not data["network_isolated"]


def test_device_rejected_without_open(tmp_path):
    report = tmp_path / "audit.json"
    assert not probe.run_probe(report, ["/dev/null"])
    assert "explicit" in json.loads(report.read_text())["error"]


def test_report_never_overwritten(tmp_path):
    report = tmp_path / "audit.json"
    report.write_text("preserve")
    with pytest.raises(FileExistsError):
        probe.run_probe(report)
    assert report.read_text() == "preserve"


def test_cli_failure_nonzero(tmp_path):
    report = tmp_path / "audit.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--report", str(report),
                             "--device", "/dev/null"], check=False)
    assert result.returncode != 0
    assert json.loads(report.read_text())["status"] == "failed"
