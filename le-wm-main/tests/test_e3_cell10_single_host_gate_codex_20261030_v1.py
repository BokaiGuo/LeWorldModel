import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import e3_cell10_durable_claim_codex_20261030_v1 as marker
from experiments.spectral_reachability import e3_cell10_single_host_gate_codex_20261030_v1 as gate


def test_competing_process_routes_share_one_marker(tmp_path):
    assert Path("/tmp") in tmp_path.resolve().parents
    module = "experiments.spectral_reachability.e3_cell10_single_host_gate_codex_20261030_v1"
    def attempt(route):
        return subprocess.run([sys.executable, "-B", "-m", module, "fixture",
                               "--directory", str(tmp_path), "--route", route,
                               "--closure-sha256", "a" * 64],
                              capture_output=True, text=True, check=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gate.ROUTES)) as pool:
        outputs = list(pool.map(attempt, gate.ROUTES))
    assert [x.returncode for x in outputs].count(0) == 1
    assert all(x.returncode == 0 or "File exists" in x.stderr for x in outputs)
    assert (tmp_path / marker.MARKER).is_file()


def test_child_failure_retains_claim_and_blocks_another_route(tmp_path):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "failure")
    previous = marker.DIRECTORY
    with pytest.raises(RuntimeError, match="marker retained"):
        gate.fixture_dispatch(tmp_path, "chunked", "b" * 64, child=failed)
    assert marker.DIRECTORY == previous
    with pytest.raises(FileExistsError):
        gate.fixture_dispatch(tmp_path, "identity", "c" * 64)


def test_production_always_refuses_without_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(marker, "DIRECTORY", tmp_path)
    with pytest.raises(RuntimeError, match="legacy entrypoints"):
        gate.production_dispatch()
    assert not os.path.lexists(tmp_path / marker.MARKER)


def test_invalid_route_and_path_fail_before_claim(tmp_path):
    with pytest.raises(ValueError, match="route"):
        gate.fixture_dispatch(tmp_path, "unlisted", "a" * 64)
    with pytest.raises(ValueError, match="private"):
        gate.fixture_dispatch("/tmp", "chunked", "a" * 64)
    assert not os.path.lexists(tmp_path / marker.MARKER)
