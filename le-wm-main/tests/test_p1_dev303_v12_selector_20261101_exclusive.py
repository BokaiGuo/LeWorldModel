"""CPU-only selector normalization tests."""
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_dev303_v12_selector_20261101_exclusive as m


def test_protocol_is_development_only_and_fixed():
    value = m.protocol()
    assert value["schema"] == m.NAME + ".protocol.v1"
    assert value["P1_admission"] is False
    assert value["selector_execution_authorized"] is False
    assert value["budget"]["predict_samples"] == 54417
    assert len(value["prefix_bank"]["candidates"]) == 17


def test_plan_and_task_hashes_are_fixed():
    assert m.PREP_SHA == "f21eb1abe6517bf272f58e3a96af2291f829a4ebf6379add6c02d0fa09e2f1e0"
    assert m.TASK_SHA == "3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12"


def test_abstention_precedes_kernel_dispatch(monkeypatch, tmp_path):
    value = m.protocol()
    assert value["candidate_status"] == "abstain_no_feasible_candidate"
    prepared = tmp_path / "prep.json"
    prepared.write_text(json.dumps(value))
    monkeypatch.setattr(m.kernel, "run", lambda *_: (_ for _ in ()).throw(AssertionError("dispatched")))
    import pytest
    with pytest.raises(ValueError, match="abstain before CUDA"):
        m.run(prepared, m.kernel.sha(prepared), tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_common_random_numbers_and_restoration(monkeypatch, tmp_path):
    value = m.protocol()
    value["candidate_status"] = "ready"
    value["feasible_candidate_count"] = 1
    monkeypatch.setattr(m, "protocol", lambda: value)
    prepared = tmp_path / "prep.json"
    prepared.write_text(json.dumps(value))
    original = m.kernel.np.random.default_rng

    def fake_run(*_):
        first = m.kernel.np.random.default_rng(1).normal(size=20)
        second = m.kernel.np.random.default_rng(2).normal(size=20)
        assert np.array_equal(first, second)
        return "sealed"

    monkeypatch.setattr(m.kernel, "run", fake_run)
    assert m.run(prepared, m.kernel.sha(prepared), tmp_path / "output") == "sealed"
    assert m.kernel.np.random.default_rng is original
