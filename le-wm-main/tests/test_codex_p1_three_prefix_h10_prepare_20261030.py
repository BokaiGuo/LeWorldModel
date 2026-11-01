import copy
import hashlib

import pytest

from experiments.spectral_reachability import codex_p1_three_prefix_h10_prepare_20261030 as mod


@pytest.fixture(scope="module")
def actual():
    return mod.build()


def test_full_matched_roster_and_bytes(actual):
    assert (actual["fixed_pairs"], actual["fixed_anchors"], actual["fixed_branches"]) == (8, 24, 1008)
    assert len(actual["schedule"]) == 1008
    assert [x["index"] for x in actual["schedule"]] == list(range(1008))
    for i, bank in enumerate(actual["probe_banks"]):
        assert len(bank) == 42
        assert bank[0]["raw10_float32_hex"] == bytes(80).hex()
        assert [a["role"] for a in actual["anchors"][3*i:3*i+3]] == ["selected", "zero", "greedy"]
        assert all(hashlib.sha256(bytes.fromhex(b["raw10_float32_hex"])).hexdigest() ==
                   b["raw10_sha256"] for b in bank)
        assert len([s for s in actual["schedule"] if s["logical_pair"] == i]) == 126
    assert actual["P1_admission"] is actual["freshness"] is actual["equal_budget_certified"] is False


def test_bank_rejects_wrong_tail_and_nonfinite():
    with pytest.raises(ValueError):
        mod.bank(b"")
    with pytest.raises(ValueError):
        mod.bank(bytes.fromhex("0000c07f") + bytes(76))


def test_greedy_missing_or_incumbent_mismatch_fails():
    null_p, _ = mod.null.verify(mod.NULL_PREP, mod.NULL_SHA)
    old_p = mod.historical.build()
    analysis = mod.native.read(mod.ANALYSIS, cap=16*1024**2)
    bad = copy.deepcopy(analysis)
    bad["rows"][0]["selected_is_greedy"] = False
    with pytest.raises(ValueError, match="greedy"):
        mod.assemble(null_p, old_p, bad)
    old_p["anchors"][0]["prefix5_float32_hex"] = bytes(40).hex()
    with pytest.raises(ValueError, match="prefix identity"):
        mod.assemble(null_p, old_p, analysis)
    old_p["anchors"][0]["task_id"] = "different"
    with pytest.raises(ValueError, match="same root"):
        mod.assemble(null_p, old_p, analysis)


def test_mismatched_task_and_census(actual):
    assert actual["caps_if_complete_no_early_stop"]["public_step"] == 1008*25
    assert actual["caps_if_complete_no_early_stop"]["physics_step"] == 2*2016 + 10*25200
    assert actual["caps_if_complete_no_early_stop"]["explicit_render"] == 1008*27*3
    assert actual["historical_proposal_and_shared_cost"] == "unknown_not_zero"
    assert actual["execution_implemented"] is actual["saved_consumer_implemented"] is False


def test_verify_rejects_drift_and_exclusive_output(tmp_path, monkeypatch, actual):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    path = results / (mod.NAME + "_fixture.json")
    monkeypatch.setattr(mod, "build", lambda: actual)
    digest = mod.prepare(path)
    assert mod.verify(path, digest) == actual
    with pytest.raises(FileExistsError):
        mod.prepare(path)
    changed = copy.deepcopy(actual)
    changed["fixed_branches"] = 1
    monkeypatch.setattr(mod, "build", lambda: changed)
    with pytest.raises(ValueError, match="drift"):
        mod.verify(path, digest)
