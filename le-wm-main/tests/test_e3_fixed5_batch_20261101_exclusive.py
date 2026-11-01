import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_fixed5_batch_20261101_exclusive as b


def test_fixed_scope_and_exclusive_write(tmp_path):
    assert b.CELLS == (5, 6, 7, 14, 15)
    target = tmp_path / "x"
    b.write_new(target, {"x": 1})
    with pytest.raises(FileExistsError):
        b.write_new(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 1}


def test_attempt_gap_rejected(tmp_path):
    (tmp_path / "cell05_attempt002").mkdir()
    with pytest.raises(ValueError):
        b.attempts(tmp_path, 5)


def test_interrupted_attempt_is_visible(tmp_path):
    (tmp_path / "cell05_attempt001").mkdir()
    assert b.latest(tmp_path, 5)["status"] == "interrupted_or_unrecorded"


def test_failed_precheck_is_recorded_and_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(b.startup, "precheck", lambda *a: (_ for _ in ()).throw(ValueError("broken")))
    row = dict(cell=5, source="s", source_sha256="a", closure="c", closure_sha256="b",
               authority="a", authority_sha256="c")
    with pytest.raises(ValueError, match="broken"):
        b.execute_one(tmp_path, row, "seal", 1)
    result = json.loads((tmp_path / "cell05_attempt001/result.json").read_text())
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert not (tmp_path / "cell05_attempt001/child_intent.json").exists()


def test_child_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(b.startup, "precheck", lambda *a: None)
    monkeypatch.setattr(b, "load_seal", lambda *a: None)
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=9))
    row = dict(cell=5, source="s", source_sha256="a", closure="c", closure_sha256="b",
               authority="a", authority_sha256="c")
    with pytest.raises(RuntimeError):
        b.execute_one(tmp_path, row, "seal", 1)
    attempt = tmp_path / "cell05_attempt001"
    assert (attempt / "child_intent.json").exists()
    assert json.loads((attempt / "result.json").read_text())["returncode"] == 9


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", Path.cwd())
    monkeypatch.setattr(b, "batch_path", lambda x: tmp_path)
    monkeypatch.setattr(b, "gpu_lock", lambda: contextlib.nullcontext(1))
    monkeypatch.setattr(b, "load_seal", lambda *a: {"cells": [{"cell": c} for c in b.CELLS]})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def test_resume_stops_on_unknown_attempt(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    (tmp_path / "cell05_attempt001").mkdir()
    monkeypatch.setattr(b, "execute_one", lambda *a: pytest.fail("must not run"))
    assert b.run(str(tmp_path), "x", b.CONFIRM)["status"] == "paused_unresolved_attempt"


def test_early_stop_is_not_completed_and_pauses(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    calls = []
    def execute(p, row, seal, fd):
        calls.append(row["cell"])
        return {"status": "native_stopped_prefix"}
    monkeypatch.setattr(b, "execute_one", execute)
    assert b.run(str(tmp_path), "x", b.CONFIRM)["status"] == "paused_native_stop"
    assert calls == [5]


def test_retry_requires_existing_failed_attempt_before_any_run(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(b, "execute_one", lambda *a: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="unresolved"):
        b.run(str(tmp_path), "x", b.CONFIRM, retry_cell=6)


def test_cuda_hidden_refuses_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError):
        b.run(str(tmp_path), "x", b.CONFIRM)


def test_seal_then_detect_input_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "RESULTS", tmp_path)
    monkeypatch.setattr(b, "SHAS", tuple(b.sha(Path(__file__)) for _ in b.CELLS))
    monkeypatch.setattr(b.fixed, "verify", lambda *a: None)
    for cell in b.CELLS:
        source = tmp_path / f"e3_uncovered_fixed_inputs_20261101_exclusive_cell{cell:02d}_RUN01.json"
        source.write_bytes(Path(__file__).read_bytes())
    def collect(source, digest, output):
        return {"sha256": b.write_new(output, {"source": source})}
    def prepare(source, digest, closure, closure_sha, output):
        return {"sha256": b.write_new(output, {"closure": closure})}
    monkeypatch.setattr(b.startup, "collect", collect)
    monkeypatch.setattr(b.startup, "prepare", prepare)
    batch = tmp_path / "e3_fixed5_batch_20261101_exclusive_TEST"
    receipt = b.seal(str(batch))
    assert len(b.load_seal(batch, receipt["sha256"])["cells"]) == 5
    (batch / "cell05_authority.json").write_text("{}")
    with pytest.raises(ValueError, match="drift"):
        b.load_seal(batch, receipt["sha256"])
