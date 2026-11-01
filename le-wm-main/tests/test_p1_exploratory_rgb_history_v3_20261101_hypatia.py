import math

from experiments.spectral_reachability import p1_exploratory_rgb_history_v3_20261101_hypatia as mod


def test_contract_disallows_planner_and_keeps_gate():
    prepared = mod.prepare()
    assert prepared["with_target"] and prepared["native_gate_unchanged"]
    assert "forbidden" in prepared["scope"]
    assert prepared["slots"] == [0, 1] and prepared["freshness"] == "unknown"


def test_native_and_wrapped_angles_are_distinct():
    result = mod.angles([0, 0, 0, 0, -1.5], [0, 0, 0, 0, 5.5], (True, 7))
    assert result["native_eval_success"] is True
    assert result["native_angle_expression"] < 0
    assert result["strict_wrapped_angle"] > math.pi / 9
    assert result["wrapped_diagnostic_success"] is False


def test_journal_chain_bytes(tmp_path):
    journal = mod.Journal(tmp_path / "journal.jsonl")
    journal.emit("one")
    journal.emit("two")
    journal.stream.close()
    assert len((tmp_path / "journal.jsonl").read_bytes().splitlines()) == 2
