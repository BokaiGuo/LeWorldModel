import copy
import sys

import pytest

from experiments.spectral_reachability import codex_e1_ancestor_integrated_ledger_20261030_v1 as m


@pytest.fixture(scope="module")
def inputs():
    return m.load_verified()


def test_real_join_keeps_partial_unknown(inputs):
    base, observed = inputs
    before = copy.deepcopy(base)
    result = m.join(base, observed)
    assert base == before
    assert result["status"] == "blocked_incomplete"
    assert not result["equal_budget_certified"]
    assert not result["ranking_authorized"]
    assert not result["scientific_admission"]
    for slot in result["slots"]:
        inherited = slot["ledger_input"]["phases"]["inherited"]
        assert inherited["scope"] == "unknown"
        assert inherited["receipt_sha256"] is None
        assert slot["ledger_preflight"]["structural_status"] == "incomplete"
        assert "inherited:not_attributed_per_arm" in slot["ledger_preflight"]["blockers"]
        if slot["slot"] in (2, 5):
            assert slot["partial_inherited_source"]["shared_costs"] == "unknown"
            assert slot["partial_inherited_source"]["cross_stage_overlap"] == "unknown"
            for axis in m.MAP.values():
                value = inherited["measurements"][axis]
                assert value["attempted"] == value["completed"] == 144
                assert value["event_set_sha256"] == observed["rows"][slot["root"]]["event_axes"][
                    next(k for k, v in m.MAP.items() if v == axis)]["event_set_sha256"]
                assert f"inherited:{axis}:unbound_measurement" in slot["ledger_preflight"]["blockers"]
            assert sum(x is not None for x in inherited["measurements"].values()) == 4
        else:
            assert "partial_inherited_source" not in slot
            assert all(x is None for x in inherited["measurements"].values())


@pytest.mark.parametrize("mutation", ["root", "task", "target", "axis", "slot", "prior", "journal"])
def test_invalid_join_rejected(inputs, mutation):
    base, observed = copy.deepcopy(inputs)
    if mutation == "root":
        observed["rows"][1]["root"] = 0
    elif mutation == "task":
        observed["rows"][0]["task_sha256"] = "0" * 64
    elif mutation == "target":
        observed["rows"][0]["target_sha256"] = "0" * 64
    elif mutation == "axis":
        observed["rows"][0]["event_axes"].pop("predict_API")
    elif mutation == "slot":
        base["slots"][2]["slot"] = 1
    elif mutation == "prior":
        base["slots"][2]["ledger_input"]["phases"]["inherited"]["scope"] = "per_arm"
    else:
        observed["event_sha256"] = "0" * 64
    with pytest.raises((ValueError, KeyError)):
        m.join(base, observed)


def test_saved_result_hash_required(tmp_path, monkeypatch):
    source = tmp_path / "saved.json"
    source.write_text("{}")
    monkeypatch.setattr(m, "SAVED", source)
    with pytest.raises(ValueError, match="digest"):
        m.load_verified()


def test_recomputed_audit_must_match_saved(inputs, monkeypatch):
    altered = copy.deepcopy(inputs[1])
    altered["rows"][0]["event_axes"]["predict_API"]["attempted"] += 1
    monkeypatch.setattr(m.ancestor, "analyze", lambda *args: altered)
    with pytest.raises(ValueError, match="differs"):
        m.load_verified()


def test_exclusive_output(tmp_path, monkeypatch, inputs):
    output = tmp_path / "out.json"
    monkeypatch.setattr(m, "OUTPUT", output)
    monkeypatch.setattr(m, "load_verified", lambda: inputs)
    monkeypatch.setattr(sys, "argv", ["audit"])
    assert m.main() == 0
    first = output.read_bytes()
    assert m.main() == 2
    assert output.read_bytes() == first
