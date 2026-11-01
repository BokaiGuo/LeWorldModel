import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p5_rgb_host_preflight_20261030_codex_v1 as pre


def test_versions_match_and_mismatch(tmp_path):
    kernel = tmp_path / "version"
    kernel.write_text("595.99.02\n")
    library = tmp_path / "libnvidia-ml.so.1"
    library.symlink_to("libnvidia-ml.so.595.99.02")
    def result(*args, **kwargs):
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(args[0], 0, "595.99.02\n", "")
    assert pre.driver_versions(kernel, library, result)["all_equal"]
    kernel.write_text("595.84\n")
    assert not pre.driver_versions(kernel, library, result)["all_equal"]


def test_missing_or_invalid_metadata_fails_closed(tmp_path):
    kernel = tmp_path / "version"
    kernel.write_text("unknown")
    library = tmp_path / "library"
    library.symlink_to("libnvidia-ml.so.595.99.02")
    with pytest.raises(ValueError):
        pre.driver_versions(kernel, library)
    assert not pre.attempt(lambda: (_ for _ in ()).throw(TimeoutError("timeout")))["ok"]
    nodes = pre.device_nodes([str(tmp_path / "missing")])
    assert not next(iter(nodes.values()))["exists"]


def test_even_good_static_checks_do_not_authorize():
    sources = {"ok": True}
    driver = {"ok": True, "value": {"all_equal": True}}
    nodes = {str(i): {"exists": True, "character_device": True,
                      "read_write_access_check": True} for i in range(3)}
    blockers = pre.assess(sources, driver, nodes, {"ok": True, "free_mib": 9000}, 1000, {"ok": True})
    assert blockers == ["gpu_identity_memory_mig_and_runtime_unverified",
                        "external_watchdog_and_run_authorization_unverified"]


def test_static_failures_are_independent():
    nodes = {"x": {"exists": False, "character_device": False, "read_write_access_check": False}}
    blockers = pre.assess({"ok": False}, {"ok": True, "value": {"all_equal": False}}, nodes,
                          {"ok": True, "free_mib": 99}, 100, {"ok": False})
    assert len(blockers) == 7
    assert "disk_floor_unmet_or_unverified" in blockers
    assert "driver_versions_unverified_or_mismatched" in blockers


def test_real_source_chain_without_torch_or_gpu():
    result = pre.source_check()
    assert result["history_rgb_branches_verified"] == 164
    assert result["physical_receipt_and_future_labels_reopened"] is False
    assert "torch" not in sys.modules


def test_output_is_exclusive_and_denied(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(pre.abi.bridge, "ROOT", tmp_path)
    monkeypatch.setattr(pre, "inspect", lambda minimum: {"status": "blocked", "blockers": ["mock"]})
    args = ["preflight", "--min-disk-free-mib", "100",
            "--output-name", pre.NAME + "_TEST.json"]
    monkeypatch.setattr(sys, "argv", args)
    assert pre.main() == 2
    with pytest.raises(FileExistsError):
        pre.main()


def test_device_permission_error_recorded(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(os, "stat", denied)
    assert pre.device_nodes(["/dev/nvidiactl"])["/dev/nvidiactl"]["error_type"] == "PermissionError"
