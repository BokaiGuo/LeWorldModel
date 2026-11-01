"""Structural checks for the fixed all-candidate crossover."""
import copy
import json

import pytest

from experiments.spectral_reachability import p1_selected_candidates_crossover_20261031 as subject


def test_roster_is_complete_and_action_identity_is_preserved():
    fixed = subject.build()
    branches = list(subject.fixed_branches(fixed))
    assert len(branches) == 64
    assert len({(a["index"], p["index"]) for a, _, p in branches}) == 64
    assert {a["role"] for a, _, _ in branches} == {"zero", "greedy"}
    for anchor, item, probe in branches:
        assert item["prefix5_float32_hex"] == anchor["prefix5_float32_hex"]
        assert item["raw10_sha256"] == probe["raw10_sha256"]
        assert item["raw10_float32_hex"] == probe["raw10_float32_hex"]
        assert probe == fixed["banks"][anchor["logical_pair"]][probe["index"]]


def test_validation_rejects_changed_frozen_preparation(tmp_path):
    fixed = subject.build()
    changed = copy.deepcopy(fixed)
    changed["banks"][0][0]["raw10_sha256"] = "0" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(Exception):
        subject.verify(path, subject.sha(path))
