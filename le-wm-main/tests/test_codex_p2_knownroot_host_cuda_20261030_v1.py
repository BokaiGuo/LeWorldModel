"""Orchestration tests with no CUDA, model execution or physical environment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as m


@pytest.fixture
def setup(monkeypatch):
    cases = [dict(root_sha256=r, root_index=i) for i, r in enumerate(m.frontier.ROOTS)]
    protocol = {"context": {"cases": cases, "compilation": "synthetic"}}
    monkeypatch.setattr(m.frontier, "read", lambda p: {"rows": [
        {"root_sha256": r, "compiled": {}} for r in m.frontier.ROOTS]})
    monkeypatch.setattr(m.bridge, "check_sources", lambda s: None)
    monkeypatch.setattr(m, "inventory", lambda p: {"source": "hash"})
    class Backend:
        host = {"fixture": True}
        model, device = object(), SimpleNamespace(type="cuda")
        def start(self, protocol): pass
        def check_frozen(self): pass
        def model_hash(self): return "unchanged"
    return protocol, Backend()


def test_four_roots_and_bound_consumption(tmp_path, monkeypatch, setup):
    protocol, backend = setup
    calls, consumed = [], []
    def native(**kw):
        calls.append(kw["ready"]["root_sha256"])
        assert kw["caps"] == m.CAPS
        kw["output"].mkdir()
        (kw["output"] / "report.json").write_text("{}")
    monkeypatch.setattr(m.bank, "execute_native", native)
    monkeypatch.setattr(m.saved, "consume", lambda **kw: consumed.append(kw))
    manifest = {"source_bindings": {"source": "hash"}}
    out = tmp_path / "run"
    result = m._run(manifest=manifest, protocol=protocol, prepared={}, output=out, backend=backend)
    assert calls == list(m.frontier.ROOTS) and len(consumed) == 4
    assert result["status"] == "exploratory_complete" and not result["P2_admission"]
    assert len(json.loads((out / "run_report.json").read_text())["roots"]) == 4
    with pytest.raises(ValueError, match="new absolute"):
        m._run(manifest=manifest, protocol=protocol, prepared={}, output=out, backend=backend)


@pytest.mark.parametrize("failure", ["native", "consumer", "model"])
def test_failure_preserved_no_next_root(tmp_path, monkeypatch, setup, failure):
    protocol, backend = setup
    calls = []
    def native(**kw):
        calls.append(kw)
        assert kw["caps"] == m.CAPS
        if failure == "native":
            raise RuntimeError("synthetic native failure")
        kw["output"].mkdir()
        (kw["output"] / "report.json").write_text("{}")
    monkeypatch.setattr(m.bank, "execute_native", native)
    def consume(**kw):
        if failure == "consumer":
            raise ValueError("synthetic consumer failure")
    monkeypatch.setattr(m.saved, "consume", consume)
    if failure == "model":
        hashes = iter(["before", "after"])
        backend.model_hash = lambda: next(hashes)
    out = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        m._run(manifest={"source_bindings": {"source": "hash"}}, protocol=protocol,
               prepared={}, output=out, backend=backend)
    report = json.loads((out / "run_report.json").read_text())
    assert report["status"] == "partial_or_failed" and len(calls) == 1
    assert report["roots"] == [] and report["error"]


def test_ack_required_before_execution(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["x", "--execute-cuda", "--prepared", "x",
                                    "--prepared-sha256", "x", "--output", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="acknowledgement"):
        m.main()
    assert not (tmp_path / "run").exists()
