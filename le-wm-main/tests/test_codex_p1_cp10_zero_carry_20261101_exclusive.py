import pytest
import torch
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_cp10_zero_carry_20261101_exclusive as probe
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer


def test_zero_only_first_packet_and_preserve_input():
    original = torch.full((3, 5, 2), 0.25)
    result = probe.intervene({"selected_raw": original, "selected_index": 4})
    assert torch.all(result["selected_raw"][0] == 0)
    assert torch.all(result["selected_raw"][1:] == 0.25)
    assert torch.all(original == 0.25)
    assert result["selected_index"] == 4


def test_invalid_plan_fails():
    with pytest.raises(ValueError):
        probe.intervene({"selected_raw": torch.zeros((2, 5, 2))})
    with pytest.raises(ValueError):
        probe.intervene({"selected_raw": torch.full((3, 5, 2), float("nan"))})


def test_recorded_zero_branch_and_ledger():
    directory = Path(__file__).resolve().parents[1] / "results/codex_p1_cp10_zero_carry_20261101_exclusive_RUN_CODEXZERO01"
    if not directory.exists():
        pytest.skip("optional recorded host run absent")
    report = json.loads((directory / "report.json").read_bytes())
    row = report["rows"][0]
    assert report["host"]["device"] == "cuda:0"
    assert row["initial_exact"] and row["status"] == "completed20"
    assert row["executed_raw"][:5] == [[0.0, 0.0]] * 5
    checkpoints = {c["step"]: c for c in row["checkpoints"]}
    nodes = report["nodes"]["s0_structured"]
    for node in nodes:
        boundary = node["boundary"]
        seal = json.loads((directory / f"s0_structured_b{boundary:02}_seal.json").read_bytes())
        receipt = node["receipt"]
        assert seal["receipt"] == receipt
        assert seal["identity"]["observation"]["pixel_sha256"] == checkpoints[boundary]["pixel_sha256"]
        account = receipt["account"]
        ledger = receipt["ledger"]
        assert consumer.consume(ledger["journal"], ledger["sha256"], account["contract_sha256"],
                                "structured", "node", account["run_id"]) == account
    assert nodes[1]["receipt"]["account"]["charged_upper_bounds"]["predict_calls"] == 93
    assert nodes[1]["receipt"]["account"]["charged_upper_bounds"]["replay_calls"] == 17
