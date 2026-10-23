from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v3_panel as builder
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v3_calibration as runner


def test_v3_seeds_produce_new_disjoint_action_identities() -> None:
    held = builder.v3_action_sequences(seed=202610222, count=16)
    train = builder.v3_action_sequences(seed=202610223, count=16, train=True)
    held_hashes = {builder.sequence_hash(value) for value in held}
    train_hashes = {builder.sequence_hash(value) for value in train}
    held_templates = {builder.temporal_template_fingerprint(value) for value in held}
    train_templates = {builder.temporal_template_fingerprint(value) for value in train}
    assert len(held_hashes) == len(train_hashes) == 16
    assert len(held_templates) == len(train_templates) == 16
    assert not held_hashes & train_hashes
    assert not held_templates & train_templates


def test_v3_root_split_is_large_and_excludes_prior_outcomes() -> None:
    forbidden = set(range(40))
    split = builder._split_roots(seed=202610221, forbidden=forbidden)
    assert len(split) == 144
    assert not set(split) & forbidden
    assert {name: sum(value == name for value in split.values()) for name in builder.SPLIT_COUNTS} == builder.SPLIT_COUNTS
    assert sum(value == "calibration" for value in split.values()) >= 40


def test_renderer_calibration_never_uses_forbidden_outcome_roots(tmp_path: Path) -> None:
    obses = tmp_path / "obses"
    obses.mkdir()
    for episode in range(1800, 1900):
        (obses / f"episode_{episode:03d}.pth").write_bytes(b"placeholder")
    selected = builder._renderer_calibration_roots(tmp_path, set(range(1800, 1820)))
    assert len(selected) == 40
    assert not set(selected) & set(range(1800, 1820))


def test_v3_compatibility_view_is_metadata_only(tmp_path: Path) -> None:
    source = tmp_path / "native.json"
    payload = {
        "metadata": {"candidate_family": builder.V3_ACTION_OOD_FAMILY},
        "records": [{"instance_id": "fresh", "candidate_id": 0, "action_sequence": np.zeros((20, 2)).tolist()}],
    }
    source.write_text(json.dumps(payload))
    converted = json.loads(runner._compatibility_view(source, tmp_path).read_text())
    assert converted["records"] == payload["records"]
    assert converted["metadata"]["candidate_family"] == runner.ACTION_OOD_FAMILY
    assert converted["metadata"]["underlying_candidate_family"] == builder.V3_ACTION_OOD_FAMILY
