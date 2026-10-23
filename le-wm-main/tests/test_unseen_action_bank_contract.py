from __future__ import annotations

import pytest

from experiments.decision_benchmark.audit_unseen_action_bank_contract import (
    audit_unseen_action_bank_contract,
)


def _record(root: str, split: str, candidate: int, sequence: list[list[float]]) -> dict:
    return {
        "instance_id": root,
        "split": split,
        "candidate_id": candidate,
        "action_sequence": sequence,
    }


def _banks() -> tuple[list[dict], list[dict]]:
    old_a = [[0.0, 0.0]] * 8
    old_b = [[0.45, 0.0]] * 8
    new_a = [[0.18, 0.28]] * 2 + [[0.32, -0.16]] * 2 + [[0.12, 0.21]] * 2 + [[0.08, -0.11]] * 2
    new_b = [[-0.27, 0.14]] * 2 + [[0.15, 0.25]] * 2 + [[-0.09, 0.31]] * 2 + [[0.20, -0.18]] * 2
    train = [
        _record("development_train", "train", 0, old_a),
        _record("development_train", "train", 1, old_b),
        _record("development_cal", "calibration", 0, old_a),
        _record("development_cal", "calibration", 1, old_b),
    ]
    unseen = []
    for root, split in (("development_train", "train"), ("development_cal", "calibration"), ("held", "outcome")):
        unseen.extend((_record(root, split, 1000, new_a), _record(root, split, 1001, new_b)))
    return train, unseen


def test_unseen_contract_accepts_disjoint_ids_sequences_templates_and_outcome_roots() -> None:
    train, unseen = _banks()
    report = audit_unseen_action_bank_contract(train, unseen)

    assert report["status"] == "passed"
    assert report["candidate_id_overlap"] == []
    assert report["action_hash_overlap"] == []
    assert report["template_fingerprint_overlap"] == []
    assert report["unseen_root_counts"] == {"train": 1, "calibration": 1, "outcome": 1}


def test_unseen_contract_rejects_candidate_id_relabeling_even_when_action_changes() -> None:
    train, unseen = _banks()
    unseen[0]["candidate_id"] = 0
    unseen[2]["candidate_id"] = 0
    unseen[4]["candidate_id"] = 0
    with pytest.raises(ValueError, match="candidate IDs"):
        audit_unseen_action_bank_contract(train, unseen)


def test_unseen_contract_rejects_literal_action_reuse_even_under_new_id() -> None:
    train, unseen = _banks()
    reused = train[0]["action_sequence"]
    for index in (0, 2, 4):
        unseen[index]["action_sequence"] = reused
    with pytest.raises(ValueError, match="action-sequence hashes"):
        audit_unseen_action_bank_contract(train, unseen)
