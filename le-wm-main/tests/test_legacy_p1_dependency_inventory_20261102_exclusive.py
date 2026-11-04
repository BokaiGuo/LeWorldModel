import copy
import json

import pytest

from experiments.empirical_spectral_audit import legacy_p1_dependency_graph_20261102_exclusive as graph
from experiments.empirical_spectral_audit import legacy_p1_dependency_inventory_20261102_exclusive as audit


def record():
    return {"checkpoint": "/run/a/stage_005.pt",
            "training_lineage": graph.lineage("/run/a/stage_005.pt", "PointMaze", "A", 1),
            "environment": "PointMaze", "outcome_export": {"dataset_sha256": "d", "row_episode_ids": ["1"]},
            "planning": {"planning_seed": 7, "episode_ids": ["0"]},
            "record_path": "r", "stratum": 0}


def test_seed_alone_does_not_identify_training_lineage():
    a = record()
    b = copy.deepcopy(a)
    b["checkpoint"] = "/run/b/stage_010.pt"
    b["training_lineage"] = graph.lineage(b["checkpoint"], "PointMaze", "B", 1)
    assert "same_declared_training_lineage" not in graph.relation(a, b)


def test_episode_join_requires_namespace_and_overlap():
    a = record()
    b = copy.deepcopy(a)
    reason = "same_declared_dataset_and_overlapping_outcome_episode_ids"
    assert reason in graph.relation(a, b)
    b["outcome_export"]["dataset_sha256"] = None
    assert reason not in graph.relation(a, b)
    b["outcome_export"] = {"dataset_sha256": "d", "row_episode_ids": ["2"]}
    assert reason not in graph.relation(a, b)


def test_seed_and_indices_are_only_declared_planning_relation():
    a = record()
    b = copy.deepcopy(a)
    assert "same_declared_planning_seed_and_episode_indices" in graph.relation(a, b)
    b["environment"] = "PushT"
    assert "same_declared_planning_seed_and_episode_indices" not in graph.relation(a, b)


def test_components_are_connectivity_not_independence():
    assert graph.component_table([{}, {}, {}, {}], [{"left": 0, "right": 1}, {"left": 1, "right": 2}]) == [[0, 1, 2], [3]]


def test_all_records_including_ineligible_are_joined():
    a = record()
    b = copy.deepcopy(a)
    b.update(stratum=1, eligible=False, record_path="other")
    result = audit.inventory([{"records": [a, b]}])
    assert result["stratum_groups"][0]["pair_counts"]["same_checkpoint_path"] == 1


def test_conditional_root_reconstruction_and_fail_closed():
    external = [{"episode_id": 1, "window_start": 0}, {"episode_id": 1, "window_start": 1},
                {"episode_id": 2, "window_start": 2}, {"episode_id": 3, "window_start": 1}]
    assert audit.conditional_roots(external, [{"episode_id": 2, "window_start": 2}], [], 4) == ["1:1", "3:1"]
    with pytest.raises(ValueError):
        audit.conditional_roots(external, [], [], 3)
    with pytest.raises(ValueError):
        audit.conditional_roots(external, [], [], 20)


def test_real_fixed_outputs_and_conditional_roots():
    source, digest = audit.read(audit.GRAPH)
    assert digest == audit.GRAPH_SHA
    assert [len(p["records"]) for p in source["panels"]] == [100, 140]
    result = json.loads((audit.OUT / "report.json").read_text())
    counts = result["all_record_inventory"]["pair_reason_counts"]
    assert counts["cross_panel:same_checkpoint_path"] == 50
    assert counts["cross_panel:same_outcome_export_bytes"] == 50
    roots = result["pointmaze_conditional_reconstruction"]["conditional_records"]
    assert len(roots) == 50
    assert all(len(r["root_keys"]) == 100 for r in roots)
    assert len({tuple(r["root_keys"]) for r in roots}) == 1
    assert "Conditional" in result["pointmaze_conditional_reconstruction"]["qualification"]
    assert result["gate_changed"] is False
