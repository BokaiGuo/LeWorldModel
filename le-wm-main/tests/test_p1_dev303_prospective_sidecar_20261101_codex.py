import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_dev303_prospective_sidecar_20261101_codex as m


def write(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    queue1 = {
        "schema": "p1_future_root_queue_20261101_codex.frozen.v1",
        "spec": {
            "slots": [
                {"slot_id": "dev_reused", "role": "development",
                 "reset_seeds": [301000001]},
                {"slot_id": "not_a_confirmation", "role": "confirmation",
                 "reset_seeds": [301000003]},
            ]
        },
        "P1_admission": False,
        "execution_authorized": False,
    }
    queue2 = {
        "schema": "p1_future_root_queue_20261101_codex.frozen.v1",
        "spec": {
            "slots": [
                {"slot_id": "dev_holdout02", "role": "development",
                 "reset_seeds": [301000002]},
                {"slot_id": "reserved_not_confirmation", "role": "confirmation",
                 "reset_seeds": [301000004]},
            ]
        },
        "P1_admission": False,
        "execution_authorized": False,
    }
    ledger1 = [
        {"kind": "reset_attempt", "slot_id": "dev_reused", "attempt": 0,
         "seed": 301000001, "payload": {}},
        {"kind": "reset_return", "slot_id": "dev_reused", "attempt": 0,
         "seed": 301000001, "payload": {"state_sha256": "1" * 64}},
    ]
    ledger2 = [
        {"kind": "reset_attempt", "slot_id": "dev_holdout02", "attempt": 0,
         "seed": 301000002, "payload": {}},
        {"kind": "reset_return", "slot_id": "dev_holdout02", "attempt": 0,
         "seed": 301000002, "payload": {"state_sha256": "2" * 64}},
    ]
    receipt = {
        "schema": "p1_queue_goal_history_20261101_codex.source.v1",
        "history_sha256": "a" * 64,
        "training_membership": "unknown",
        "historical_coverage": "unknown",
        "independent_confirmation": False,
        "P1_admission": False,
    }
    refs = {}
    for name, value in (
        ("dev301_queue", queue1), ("dev302_queue", queue2),
        ("dev301_receipt", receipt), ("dev302_receipt", receipt),
    ):
        path = tmp_path / f"{name}.json"
        refs[name] = {"path": path, "sha256": write(path, value)}
    for name, value in (("dev301_ledger", ledger1), ("dev302_ledger", ledger2)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in value) + "\n",
                        encoding="utf-8")
        refs[name] = {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in m.SELECTOR_SOURCE_NAMES:
        path = tmp_path / Path(name).name
        path.write_text(name, encoding="utf-8")
        refs[name] = {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    refs["forbidden_paths"] = [tmp_path / "p1_dev303_outcome.json"]
    return refs


def test_protocol_freezes_all_requested_axes(tmp_path):
    report = m.build(fixture(tmp_path))
    assert report["status"] == "dev303_prospective_protocol_frozen_waiting_for_source"
    assert report["candidate_library"]["labels"] == [
        "zero", "spectral", "gn", "rgb_nearest_contact",
        "rgb_goal_support_direct", "rgb_goal_support_top_route",
        "rgb_goal_support_bottom_route", "random_control",
    ]
    assert report["contact_geometry"]["input_permission"] == "RGB-only"
    assert report["selector"]["version"] == "v3"
    assert report["feedback"]["count"] == 1
    assert report["budget"]["root_selector"]["tail_samples"] == 64
    assert report["metrics"]["primary"] == "native_success_by_fixed_step_15"
    assert report["failure_and_censoring"]["root_replacement_after_outcome"] == "prohibited"
    assert report["outcome_opened"] is False
    assert report["execution_authorized"] is False
    assert report["P1_admission"] is False


def test_root_scope_is_bounded_nonparticipation_not_blindness(tmp_path):
    report = m.build(fixture(tmp_path))
    root = report["root_source"]
    assert root["seed"] == 301000003
    assert root["previously_declared_in_dev301_queue"] is True
    assert root["attempted_in_dev301_or_dev302_ledgers"] is False
    assert root["development_history_nonparticipation"] == "verified_within_bounded_ledgers"
    assert root["developer_blindness"] == "not_established"
    assert root["training_membership"] == "unknown"
    assert root["source_materialized"] is False


@pytest.mark.parametrize("fault", ["attempted", "queue_role", "sha", "outcome", "admission"])
def test_root_and_outcome_checks_fail_closed(tmp_path, fault):
    refs = fixture(tmp_path)
    if fault == "attempted":
        path = refs["dev301_ledger"]["path"]
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "reset_attempt", "slot_id": "not_a_confirmation",
                                     "attempt": 0, "seed": 301000003, "payload": {}}) + "\n")
        refs["dev301_ledger"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "queue_role":
        path = refs["dev301_queue"]["path"]
        value = json.loads(path.read_text())
        value["spec"]["slots"][1]["role"] = "development"
        refs["dev301_queue"]["sha256"] = write(path, value)
    elif fault == "sha":
        refs["dev302_queue"]["sha256"] = "0" * 64
    elif fault == "outcome":
        refs["forbidden_paths"][0].write_text("do not read", encoding="utf-8")
    else:
        path = refs["dev302_receipt"]["path"]
        value = json.loads(path.read_text())
        value["P1_admission"] = True
        refs["dev302_receipt"]["sha256"] = write(path, value)
    with pytest.raises(ValueError):
        m.build(refs)


def test_exclusive_write(tmp_path):
    output = tmp_path / "p1_dev303_prospective_sidecar_20261101_codex_PREP_MAIN01.json"
    m.write_new(output, {"frozen": True})
    with pytest.raises(FileExistsError):
        m.write_new(output, {"frozen": True})
