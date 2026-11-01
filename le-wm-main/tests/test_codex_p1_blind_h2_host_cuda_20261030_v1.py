"""Blind bundle + synthetic callbacks only; no GPU, simulator, or outcome reads."""
import builtins
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_blind_h2_host_cuda_20261030_v1 as runner

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/codex_p1_blind_bundle_export_20261030_v1_MAIN01.json"
SHA = "b5748d81de35df501cb6b7d2d325642080994701a9fa7a634e50ac1c22462856"


@pytest.fixture(scope="module")
def bundle():
    return runner.h2.blind.read_verified(BUNDLE, SHA)[0]


def fake_plan(index, latent):
    return dict(row_index=index, physical_authorized=False), [{}, {}]


def test_drive_binds_own_rgb_and_never_opens_files(bundle, monkeypatch):
    observed = []
    rows = [dict(index=i, status="not_started") for i in range(16)]
    events = []
    def encode(index, pixels):
        observed.append((index, runner.h2.blind.sha(pixels.tobytes())))
        return np.full(192, index, np.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError("drive must not open any file")
    monkeypatch.setattr(builtins, "open", forbidden)
    runner.drive(bundle, encode, fake_plan, events.append, rows)
    assert observed == [(i, row["cp5_rgb"]["pixel_sha256"]) for i, row in enumerate(bundle["rows"])]
    assert all(row["status"] == "planned" for row in rows)
    assert all(row["cp5_latent_sha256"] == runner.h2.blind.sha(
        np.full(192, i, "<f4").tobytes()) for i, row in enumerate(rows))


def test_failed_encoding_stops_without_replacement(bundle):
    rows = [dict(index=i, status="not_started") for i in range(16)]
    attempted = []
    def encode(index, pixels):
        attempted.append(index)
        if index == 3: raise RuntimeError("synthetic encode fault")
        return np.zeros(192, np.float32)
    with pytest.raises(RuntimeError, match="synthetic encode fault"):
        runner.drive(bundle, encode, fake_plan, lambda e: None, rows)
    assert attempted == [0, 1, 2, 3]
    assert [r["status"] for r in rows[:5]] == ["planned"]*3 + ["attempted", "not_started"]


def test_real_h2_compiler_through_drive_with_analytic_backend(bundle):
    class Backend:
        def __init__(self, ledger):
            self.ledger = ledger
            self.j = np.zeros((192, 10))
            self.j[0, 0] = 1
        def step(self, context, actions):
            for kind in ("action_encoder", "predict"):
                e = self.ledger.begin(kind); self.ledger.end(e)
            c = deepcopy(context)
            z = np.asarray(c["latent_history"][-1]) + self.j @ np.asarray(actions).reshape(10)
            c["latent_history"] = c["latent_history"][1:] + [z.tolist()]
            token = runner.base.d2p.normalized_history(actions)[0].astype(float).tolist()
            c["action_history"] = c["action_history"][1:] + [token]
            return c
        def nominal(self, c):
            return np.asarray(self.step(c, np.zeros((5, 2), np.float32))["latent_history"][-1])
        def jacobian(self, c, operation):
            operation("jacobian", lambda: self.step(c, np.zeros((5, 2), np.float32)))
            return self.j.copy()
        def replay(self, c, actions): return self.step(c, actions)

    rows = [dict(index=i, status="not_started") for i in range(16)]
    def plan(index, latent):
        result = runner.h2.compile_row(bundle, index, latent, lambda packet, ledger: Backend(ledger), lambda e: None)
        return result, [{"synthetic": True}, {"synthetic": True}]
    runner.drive(bundle, lambda i, pixels: np.zeros(192, np.float32), plan, lambda e: None, rows)
    assert all(r["status"] == "planned" and len(r["h2_result"]["raw10"]) == 10 for r in rows)
    assert all(r["h2_result"]["packets"][1]["context"] ==
               r["h2_result"]["packets"][0]["winner"]["context"] for r in rows)
    assert runner.bytes_json(rows).startswith(b"[")


def test_encoder_ledger_order_cuda_and_failure():
    torch = SimpleNamespace(is_grad_enabled=lambda: False)
    ledger = runner.EncodeLedger(torch, lambda: None, lambda e: None)
    ledger.image = 0
    cuda = lambda shape: SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=shape)
    with pytest.raises(ValueError, match="order/cap"):
        ledger.dispatch("projector", cuda((1, 192)), lambda x: x, None)
    with pytest.raises(ValueError, match="CUDA shape"):
        ledger.dispatch("encoder", SimpleNamespace(device=SimpleNamespace(type="cpu"), shape=(1,3,224,224)),
                        lambda x: x, None)
    ledger.dispatch("encoder", cuda((1,3,224,224)), lambda x: x, 1)
    with pytest.raises(RuntimeError):
        ledger.dispatch("projector", cuda((1,192)), lambda x: (_ for _ in ()).throw(RuntimeError()), 1)
    assert ledger.counts() == {"encoder": {"attempted": 1, "completed": 1},
                               "projector": {"attempted": 1, "completed": 0}}
    with pytest.raises(ValueError):
        ledger.dispatch("projector", cuda((1,192)), lambda x: x, 1)


def test_guarded_backend_scopes_every_call(monkeypatch):
    calls = []
    class AD:
        def __init__(self, *args): self.undo = []
        def __enter__(self): calls.append("ad_enter"); return self
        def __exit__(self, *args): calls.append("ad_exit")
    @contextmanager
    def guard(model, ledger):
        calls.append("module_enter")
        try: yield
        finally: calls.append("module_exit")
    class Native:
        def __init__(self, model, device): pass
        def nominal(self, context): calls.append("native"); return 123
    monkeypatch.setattr(runner.obs, "OneAD", AD)
    monkeypatch.setattr(runner.base, "module_guard", guard)
    monkeypatch.setattr(runner.base.d2p, "TorchBackend", Native)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: None))
    backend = runner.GuardedBackend(0, SimpleNamespace(), object(), object(), torch, lambda e: None)
    assert backend.nominal({}) == 123
    assert calls == ["ad_enter", "module_enter", "native", "module_exit", "ad_exit"]


def test_disabled_cuda_writes_partial_and_exclusive_marker(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inputs", lambda *args: (bundle, bundle["model_files"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    target = tmp_path / (runner.NAME+"_failure")
    with pytest.raises(ValueError, match="partial attempt retained"):
        runner.run("unused", SHA, "/unused", target, "ISOLATED_HOST_CUDA_COORDINATED")
    report = json.loads((target/"report.json").read_text())
    assert report["status"] == "partial_or_failed" and len(report["rows"]) == 16
    assert report["prior_outcome_read_contamination"] is True
    assert report["preregistration_guarantee"] is False and report["physical_authorized"] is False
    assert all(r["status"] == "not_started" for r in report["rows"])
    with pytest.raises(ValueError, match="exclusive"):
        runner.run("unused", SHA, "/unused", target, "ISOLATED_HOST_CUDA_COORDINATED")


def test_unacknowledged_fails_before_creating_attempt(tmp_path):
    target = tmp_path/(runner.NAME+"_never")
    with pytest.raises(ValueError, match="coordination"):
        runner.run("unused", SHA, "/unused", target, "NO")
    assert not target.exists()


def test_wrong_local_model_hash_rejected_before_cuda(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.h2.blind, "read_verified", lambda *args: (bundle, {}))
    (tmp_path/"weights.pt").write_bytes(b"not the frozen weights")
    (tmp_path/"config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="model file hashes"):
        runner.inputs("unused", SHA, tmp_path)
