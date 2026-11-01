import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_four_arm_cost_rollup_20261101 as roll


REPORT = Path(__file__).resolve().parents[1] / "results/p1_four_arm_extra32_root1_20261031_RUN01/report.json"
SHA = "3e53dfd8a73298df7e08717836d6b30c0baf602239c5d6f661d7910a44b55df2"


@pytest.mark.skipif(not REPORT.exists(), reason="saved local execution unavailable")
def test_real_saved_ledgers():
    out = roll.rollup(REPORT, SHA)
    assert out["arms"]["native"]["planning_charged_upper_bounds"]["predict_calls"] == 684
    assert out["arms"]["structured"]["planning_charged_upper_bounds"]["backward_calls"] == 832
    assert out["arms"]["native"]["physical_visual_meter"]["encoder"]["attempted"] == 4
    assert not out["fairness_proven"] and not out["full_cost_certified"]


def test_reject_wrong_hash():
    with pytest.raises(ValueError, match="digest"):
        roll.rollup(REPORT, "0" * 64)


@pytest.mark.parametrize("mutation", ["missing_node", "wrong_hook", "wrong_account", "bad_meter", "partial"])
def test_reject_tampered_report(tmp_path, monkeypatch, mutation):
    report = json.loads(REPORT.read_bytes())
    report = copy.deepcopy(report)
    row = report["rows"][0]
    nodes = report["nodes"][row["episode_id"]]
    receipt = nodes[0]["receipt"]
    # Keep journals local to the synthetic report while exercising all rollup checks.
    for entries in report["nodes"].values():
        for node in entries:
            node["receipt"]["ledger"]["journal"] = str(tmp_path / "events.jsonl")
    accounts = {r["receipt"]["contract_sha256"]: r["receipt"]["account"]
                for entries in report["nodes"].values() for r in entries}
    monkeypatch.setattr(roll.consumer, "consume", lambda path, sha, contract, arm, level, run: accounts[contract])
    if mutation == "missing_node":
        nodes.pop()
    elif mutation == "wrong_hook":
        receipt["module_hook_audit"]["module_forward_pre_hooks"]["predict_calls"] += 1
    elif mutation == "wrong_account":
        receipt["account"] = dict(receipt["account"], status="failed")
    elif mutation == "bad_meter":
        row["physical_visual_counts"]["encoder"]["completed"] = -1
    else:
        report["status"] = "partial_or_failed"
    payload = json.dumps(report).encode()
    path = tmp_path / "report.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        roll.rollup(path, hashlib.sha256(payload).hexdigest())
