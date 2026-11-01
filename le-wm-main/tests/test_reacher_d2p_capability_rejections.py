"""Pure numeric saved-record fixtures, not CPU neural inference."""
import copy
import json

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed, predicates
from experiments.spectral_reachability import audit_reacher_d2p_capability_rejections as a


def fixture():
    bank = [{"index": i, "prefix_cost": 1. if i == 0 else 2.,
        "variants": {"cem_then_gn4": {"terminal_cost": 1.}},
        "common_response_gain": {"bounded_quadratic_gain": 1.,
            "bounded_quadratic_gain_upper_bound": 1.,
            "objective_certificate_passed": True, "physical_labels_used": False}}
        for i in range(17)]
    return bank, {"greedy_index": 0, "native_index": 0, "qualified_indices": []}


def test_each_original_predicate_is_explicit_and_threshold_is_unchanged():
    b, d = fixture()
    b[3]["variants"]["cem_then_gn4"]["terminal_cost"] = .5
    rows = predicates(b, d)
    assert rows[3]["rejected_by"] == ["model_common_gain_certifiably_better_than_controls"]
    b[3]["common_response_gain"]["bounded_quadratic_gain"] = 1.1
    b[3]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] = 1.1
    d.update(native_index=3, qualified_indices=[3])
    assert predicates(b, d)[3]["qualified"]


@pytest.mark.parametrize("bad", ["missing", "order", "greedy", "native", "qualified", "nan", "labels", "certificate", "bounds"])
def test_rejects_changed_bank_or_historical_selector(bad):
    b, d = copy.deepcopy(fixture())
    if bad == "missing": b.pop()
    elif bad == "order": b[1]["index"] = 2
    elif bad == "greedy": d["greedy_index"] = 1
    elif bad == "native": d["native_index"] = 1
    elif bad == "qualified": d["qualified_indices"] = [1]
    elif bad == "nan": b[1]["prefix_cost"] = float("nan")
    elif bad == "labels": b[1]["common_response_gain"]["physical_labels_used"] = True
    elif bad == "certificate": b[1]["common_response_gain"]["objective_certificate_passed"] = False
    else: b[1]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] = .5
    with pytest.raises(ValueError): predicates(b, d)


def test_pairing_uses_root_identity_not_collection_position():
    rows = [{"seed": 3}, {"seed": 1}]
    assert indexed(rows) == {1: {"seed": 1}, 3: {"seed": 3}}
    with pytest.raises(ValueError): indexed(rows + [{"seed": 3}])
    with pytest.raises(ValueError): indexed([{"seed": True}])


def test_gain_equality_does_not_become_certified_strict_improvement():
    b, d = fixture()
    b[1]["variants"]["cem_then_gn4"]["terminal_cost"] = 0.
    assert not predicates(b, d)[1]["qualified"]


def audit_fixture(tmp_path, monkeypatch):
    reference = tmp_path / "source.json"
    predictions = reference.with_suffix(".predictions.json")
    capability = tmp_path / "capability.json"
    reference.write_text("{}")
    plans, measured = [], []
    for seed in (8, 2):
        pt, mt = {}, {}
        for task in a.TASKS:
            b, d = fixture()
            pt[task] = {"bank": b, "decisions": {"cem_then_gn4": d}}
            checks = {"stage": True, "joint": True, "rgb": True, "capability": True}
            mt[task] = {"roles": {"zero": 0, "greedy": 0},
                "cells": {str(i): {"probes": [{"fits": {space: {"common": {
                    "gain_fraction": .2 if i == 0 else .3}} for space in ("physical_rgb_fd", "model_fd")}}
                    for _ in range(2)]} for i in range(17)},
                "mechanism_diagnostics": {f"program_{i}": {"checks": checks,
                    "all_diagnostic_checks": i == 3} for i in range(17)}}
        plans.append({"seed": seed, "tasks": pt})
        measured.append({"seed": seed, "tasks": mt})
    predictions.write_text(json.dumps({"rows": plans}))
    capability.write_text(json.dumps({"sealed_protocol": {"input_bindings": {
        str(p.resolve()): a.digest(p) for p in (reference, predictions)}}}))
    result = {"all_programs": True, "rows": list(reversed(measured))}
    monkeypatch.setattr(a, "verify", lambda p: result)
    return capability, reference, result


def test_audit_joins_reordered_roots_and_retains_all_failures(tmp_path, monkeypatch):
    c, r, _ = audit_fixture(tmp_path, monkeypatch)
    out = a.audit(c, r)
    assert [row["seed"] for row in out["rows"]] == [2, 8]
    assert out["P1_admission"] is False
    assert out["physical_labels_used_for_deployment"] is False
    for task in a.TASKS:
        stats = out["aggregate"][task]
        assert stats["complete_program_cells"] == 34
        assert stats["physical_diagnostic_candidates"] == 2
        assert stats["physical_candidates_rejected_by_model"] == 2
        assert len(out["rows"][0]["tasks"][task]["programs"]) == 17


def test_metric_disagreement_is_not_mislabeled_as_confirmed_model_or_code_failure(tmp_path, monkeypatch):
    c, r, measured = audit_fixture(tmp_path, monkeypatch)
    for row in measured["rows"]:
        for task in a.TASKS:
            for probe in row["tasks"][task]["cells"]["3"]["probes"]:
                probe["fits"]["physical_rgb_fd"]["common"]["gain_fraction"] = .1
    out = a.audit(c, r)
    for row in out["rows"]:
        p = row["tasks"]["full_drive"]["programs"][3]
        assert "physical_joint_vs_rgb_common_gain" in p["diagnostic_disagreements"]
        assert "actual_history_model_vs_physical_rgb_common_gain" not in p["diagnostic_disagreements"]
        assert p["code_defect_confirmed"] is False


@pytest.mark.parametrize("bad", ["partial", "root_set", "duplicate", "binding"])
def test_audit_rejects_incomplete_or_unbound_evidence(tmp_path, monkeypatch, bad):
    c, r, result = audit_fixture(tmp_path, monkeypatch)
    if bad == "partial": result["all_programs"] = False
    elif bad == "root_set": result["rows"].pop()
    elif bad == "duplicate": result["rows"].append(result["rows"][0])
    else: r.write_text('{"changed":true}')
    with pytest.raises(ValueError): a.audit(c, r)
