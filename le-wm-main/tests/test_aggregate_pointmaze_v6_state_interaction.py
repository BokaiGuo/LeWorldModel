from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark.aggregate_pointmaze_v6_state_interaction import aggregate
from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash


def _panel(parent: Path, name: str, root: int, action_offset: float, *, design: str = "state_interaction_pairs", schema: str = "pointmaze_temporal_ood_v6_matched_panel.v1") -> Path:
    directory = parent / name
    directory.mkdir()
    actions = [np.full((20, 2), action_offset + candidate, dtype=np.float64) for candidate in range(2)]
    hashes = [sequence_hash(action) for action in actions]
    rows = []
    for split, episode in (("train", root + 10), ("calibration", root + 20), ("outcome", root)):
        for candidate, action in enumerate(actions):
            rows.append({"instance_id": f"root-{episode}", "start_episode_id": str(episode), "split": split, "candidate_id": candidate, "action_sequence": action.tolist(), "candidate_action_hash": hashes[candidate]})
    context = directory / "temporal_ood_v6_context.json"
    context.write_text(json.dumps({"metadata": {"schema": schema}, "records": rows}))
    source = directory / "temporal_ood_v6_source_dataset.json"
    source.write_text(json.dumps({"records": [dict(row) for row in rows if row["split"] == "train"]}))
    build = {"candidate_design": design, "candidate_generalization_regime": "shared_candidate_in_support", "context": str(context), "source_dataset": str(source), "interaction_design_roots": [root + 30], "interaction_pair_design": {"screening_role": "design-root terminal-effect interaction only; no goal or task-cost access", "design_root_count": 64, "proposal_count": 96, "held_pair_hashes": hashes}}
    (directory / "build_v6_matched_report.json").write_text(json.dumps(build))
    outcome_root = f"root-{root}"
    report = {"schema": "pointmaze_temporal_ood_v6_interaction_score.v3", "outcome_root_ids": [outcome_root], "outcome_labels_generated_after_frozen_selection": True, "outcome_labels_preexisted": False, "outcome_labels_sha256": name, "active_policy": "action_only_fallback", "recurrent": {"roots": {outcome_root: {"regret": .1}}}, "action_only": {"roots": {outcome_root: {"regret": .3}}}, "observability_suite": {"candidate_oracle": {}, "privileged_state_effect": {}, "history_cyclic_shuffle": {}, "paired_gain_visual_over_action_only": {}, "paired_gain_visual_over_cyclic_shuffle": {}}}
    path = directory / "temporal_ood_v6_score_report.json"
    path.write_text(json.dumps(report))
    return path


def test_aggregate_requires_two_disjoint_sealed_interaction_panels(tmp_path: Path) -> None:
    first = _panel(tmp_path, "first", 1, .1)
    second = _panel(tmp_path, "second", 2, .4, schema="pointmaze_temporal_ood_v6_matched_panel.v2")
    result = aggregate([first, second], samples=100)
    assert result["outcome_root_count"] == 2
    assert result["held_action_hash_count"] == 4
    assert result["pooled_root_level_paired_recurrent_over_action_only"]["mean_gain"] == pytest.approx(.2)


def test_aggregate_rejects_random_or_reused_mechanism_evidence(tmp_path: Path) -> None:
    first = _panel(tmp_path, "first", 1, .1)
    random = _panel(tmp_path, "random", 2, .4, design="random_matched")
    with pytest.raises(ValueError, match="non-interaction"):
        aggregate([first, random], samples=10)
    overlap = _panel(tmp_path, "overlap", 1, .8)
    with pytest.raises(ValueError, match="outcome roots overlap"):
        aggregate([first, overlap], samples=10)
