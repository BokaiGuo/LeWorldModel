"""Read-only readiness tests: fake files, mocked disk, no experiment imports."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import d2p_experiment_readiness as readiness


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    policy = tmp_path/"policy"
    policy.mkdir()
    (policy/"weights.pt").write_bytes(b"not-a-real-model")
    (policy/"config.json").write_text('{"model": "fixture"}')
    adapter = tmp_path/"adapter.json"
    adapter.write_text('{"kind": "fixture"}')
    source = tmp_path/"source.py"
    source.write_text("raise RuntimeError('must never execute')\n")
    monkeypatch.setattr(readiness.shutil, "disk_usage", lambda p: SimpleNamespace(free=100))
    return dict(policy=policy, adapter=adapter, output=tmp_path/"new_output",
                sources=[source], minbytes=100)


def test_ready_read_only_and_weights_never_opened(inputs, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "weights.pt", "weights must be stat-only"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = readiness.check_readiness(**inputs)
    assert result["ready_for_host_cuda_request"] is True
    assert result["issues"] == []
    assert "GPU" in result["scope"] and "authorization" in result["scope"]
    assert not inputs["output"].exists()


@pytest.mark.parametrize("free", [0, 99])
def test_insufficient_disk(inputs, monkeypatch, free):
    monkeypatch.setattr(readiness.shutil, "disk_usage", lambda p: SimpleNamespace(free=free))
    assert not readiness.check_readiness(**inputs)["ready_for_host_cuda_request"]


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "100"])
def test_explicit_valid_integer_minbytes_required(inputs, value):
    inputs["minbytes"] = value
    assert readiness.check_readiness(**inputs)["issues"]


@pytest.mark.parametrize("target", ["weights.pt", "config.json", "adapter", "source"])
@pytest.mark.parametrize("mode", ["missing", "empty", "directory", "symlink"])
def test_regular_nonempty_files_required(inputs, tmp_path, target, mode):
    path = (inputs["policy"]/target if target.endswith((".pt", ".json")) else
            inputs["adapter"] if target == "adapter" else inputs["sources"][0])
    if mode == "missing":
        path.unlink()
    elif mode == "empty":
        path.write_bytes(b"")
    elif mode == "directory":
        path.unlink()
        path.mkdir()
    else:
        actual = tmp_path/"actual"
        actual.write_text('{"x": 1}')
        path.unlink()
        path.symlink_to(actual)
    assert readiness.check_readiness(**inputs)["issues"]


@pytest.mark.parametrize("content", ["{", "[]", "null", "{}", '{"x": NaN}',
                                      '{"x": 1, "x": 2}', " "])
@pytest.mark.parametrize("target", ["config", "adapter"])
def test_json_contract(inputs, content, target):
    path = inputs["policy"]/"config.json" if target == "config" else inputs["adapter"]
    path.write_text(content)
    assert readiness.check_readiness(**inputs)["issues"]


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_output_must_be_new(inputs, kind, tmp_path):
    output = inputs["output"]
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("existing")
    else:
        output.symlink_to(tmp_path/"absent")
    assert readiness.check_readiness(**inputs)["issues"]


def test_no_output_parent_creation(inputs):
    inputs["output"] = inputs["output"]/"nested"
    assert readiness.check_readiness(**inputs)["issues"]
    assert not inputs["output"].parent.exists()


@pytest.mark.parametrize("sources", [None, [], "source.py", [None]])
def test_explicit_sources_required(inputs, sources):
    inputs["sources"] = sources
    assert readiness.check_readiness(**inputs)["issues"]


def test_unreadable_source(inputs, monkeypatch):
    original = Path.open
    def denied(path, *args, **kwargs):
        if path == inputs["sources"][0]:
            raise PermissionError("fixture")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    assert readiness.check_readiness(**inputs)["issues"]


def test_disk_error_returns_issue(inputs, monkeypatch):
    def denied(path):
        raise OSError("fixture disk failure")
    monkeypatch.setattr(readiness.shutil, "disk_usage", denied)
    assert readiness.check_readiness(**inputs)["issues"]


def test_fifo_source_rejected_without_open(inputs, tmp_path):
    fifo = tmp_path/"pipe"
    os.mkfifo(fifo)
    inputs["sources"] = [fifo]
    assert readiness.check_readiness(**inputs)["issues"]


def test_readability_metadata_failure(inputs, monkeypatch):
    monkeypatch.setattr(readiness.os, "access", lambda *args: False)
    assert readiness.check_readiness(**inputs)["issues"]


def test_readiness_does_not_change_any_fixture_bytes(inputs):
    paths = [inputs["policy"]/"weights.pt", inputs["policy"]/"config.json",
             inputs["adapter"], *inputs["sources"]]
    before = {path: path.read_bytes() for path in paths}
    assert readiness.check_readiness(**inputs)["ready_for_host_cuda_request"]
    assert before == {path: path.read_bytes() for path in paths}


def test_zero_disk_minimum_is_explicit_and_allowed(inputs, monkeypatch):
    inputs["minbytes"] = 0
    monkeypatch.setattr(readiness.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    assert readiness.check_readiness(**inputs)["ready_for_host_cuda_request"]


def test_cli_only_prints_readiness(inputs, capsys):
    status = readiness.main(["--policy", str(inputs["policy"]), "--adapter", str(inputs["adapter"]),
        "--output", str(inputs["output"]), "--source", str(inputs["sources"][0]), "--minbytes", "100"])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["ready_for_host_cuda_request"] is True
    assert not inputs["output"].exists()
