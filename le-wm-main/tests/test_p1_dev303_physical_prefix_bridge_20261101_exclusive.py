import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_physical_prefix_bridge_20261101_exclusive as subject,
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    actions = []
    candidates = []
    for index, label in enumerate(subject.LABELS):
        value = np.zeros((5, 2), dtype="<f4")
        if index:
            value[0, 0] = index / 20
        raw = value.tobytes()
        candidates.append({
            "index": index, "label": label, "actions": value.tolist(),
            "actions_sha256": sha(raw), "contact_feasible_prior": index in (3, 5),
        })
        actions.append(value)
    source = {
        "seed": subject.SEED, "queue_sha256": subject.QUEUE_SHA,
        "source_receipt_sha256": subject.RECEIPT_SHA,
        "reset_receipt_sha256": "reset",
        "current_rgb_pixel_sha256": "rgb",
        "goal_rgb_pixel_sha256": "goal",
    }
    prep = {
        "schema": "p1_dev303_v12_adapter_20261101_exclusive.prepare.v1",
        "source_receipt_sha256": subject.RECEIPT_SHA,
        "pixel_sha256": ["rgb", "goal"], "candidate_status": "ready",
        "P1_admission": False, "prefix_bank": {"candidates": candidates},
    }
    scores = [{"robust_score": float(20 - i), "cp5_q": float(i)}
              for i in range(17)]
    report = {
        "schema": subject.REPORT_SCHEMA, "status": "model_only_sealed",
        "prepare_sha256": sha(json.dumps(prep, sort_keys=True).encode()),
        "source": source, "selection_input": "rgb_only",
        "simulator_state_used": False, "physical_outcomes_used": False,
        "P1_admission": False, "errors": [], "scores": scores,
        "role_indices": {"selected": 5, "zero": 0,
                         "task_greedy_direct": 4, "random": 7},
    }
    prep_raw = json.dumps(prep, sort_keys=True).encode()
    report["prepare_sha256"] = sha(prep_raw)
    report_raw = json.dumps(report, sort_keys=True).encode()
    return source, prep_raw, report_raw


def test_contract_is_rgb_only_and_allows_repeated_roles():
    source, prep_raw, report_raw = fixture()
    result = subject.build_contract(source, prep_raw, sha(prep_raw), report_raw, sha(report_raw))
    assert result["physical_execution_authorized"] is False
    assert result["P1_admission"] is False
    assert result["roles"][0]["candidate_index"] == 5
    assert result["roles"][2]["candidate_index"] == 4


def test_report_cannot_change_selection_or_open_outcomes():
    source, prep_raw, report_raw = fixture()
    report = json.loads(report_raw)
    report["physical_outcomes_used"] = True
    changed = json.dumps(report, sort_keys=True).encode()
    with pytest.raises(ValueError, match="selector seal"):
        subject.build_contract(source, prep_raw, sha(prep_raw), changed, sha(changed))


def test_actions_are_hash_and_range_bound():
    source, prep_raw, report_raw = fixture()
    prep = json.loads(prep_raw)
    prep["prefix_bank"]["candidates"][3]["actions"][0][0] = 2.0
    changed = json.dumps(prep, sort_keys=True).encode()
    report = json.loads(report_raw)
    report["prepare_sha256"] = sha(changed)
    report_raw2 = json.dumps(report, sort_keys=True).encode()
    with pytest.raises(ValueError, match="action shape/range"):
        subject.build_contract(source, changed, sha(changed), report_raw2, sha(report_raw2))
