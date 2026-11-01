from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p2_bank_bind_before_compare_20261030_v1 as m
from test_p2_composition_predict_frontier_20261027_v1 import base


def inputs(tmp_path):
    ready = {"root_sha256": m.frontier.ROOTS[0]}
    events, calls = [], []
    def prov(bank, arm):
        return {"candidate_bank_sha256": m.frontier.digest(
            {"bank": bank, "arm": arm, "contract": m.frontier.contract()})}
    def run(arm, bank, provenance):
        assert (tmp_path / "bank.json").is_file()
        assert (tmp_path / "bank_binding.json").is_file()
        assert events[-1]["kind"] == "arm_started"
        calls.append(arm)
        return {"arm": arm, "complete": True}
    return dict(directory=tmp_path, ready=ready, compile_bank=base,
                make_provenance=prov, run_arm=run, verify_arm=lambda *a: None,
                emit=events.append), events, calls


def test_binding_is_durable_before_both_comparisons(tmp_path):
    args, events, calls = inputs(tmp_path)
    result = m.bind_then_compare(**args)
    assert calls == list(m.ARMS)
    assert [e["kind"] for e in events] == ["bank_bound_before_compare", "arm_started",
                                           "arm_verified", "arm_started", "arm_verified"]
    assert result["bank_file_sha256"] == m.frontier.file_sha(tmp_path / "bank.json")


def test_bank_tamper_stops_second_arm(tmp_path):
    args, events, calls = inputs(tmp_path)
    run = args["run_arm"]
    def tamper(*a):
        result = run(*a)
        (tmp_path / "bank.json").write_text("{}")
        return result
    args["run_arm"] = tamper
    with pytest.raises(ValueError, match="drift"):
        m.bind_then_compare(**args)
    assert len(calls) == 1


def test_bad_provenance_prevents_both_arms(tmp_path):
    args, _, calls = inputs(tmp_path)
    args["make_provenance"] = lambda bank, arm: {"candidate_bank_sha256": "0"*64}
    with pytest.raises(ValueError, match="provenance"):
        m.bind_then_compare(**args)
    assert not calls


def test_bind_journal_failure_prevents_both_arms(tmp_path):
    args, _, calls = inputs(tmp_path)
    def fail(event):
        raise OSError("journal")
    args["emit"] = fail
    with pytest.raises(OSError):
        m.bind_then_compare(**args)
    assert not calls


def test_native_cpu_rejected_before_output(tmp_path):
    class Device:
        type = "cpu"
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="CUDA"):
        m.execute_native(model=None, device=Device(), ready={}, compiled={}, protocol={},
                         prepared={}, caps={}, output=output)
    assert not output.exists()
