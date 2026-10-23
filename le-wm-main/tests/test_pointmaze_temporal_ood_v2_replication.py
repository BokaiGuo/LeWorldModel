from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v2_bank as builder
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v2_replication as replication


def _write_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    records, arrays = [], {}
    states = np.zeros((100, 40, 4), dtype=np.float64)
    actions = np.zeros((100, 40, 2), dtype=np.float64)
    for index in range(100):
        name = f"root-{index:03d}"
        state = [float(index) / 100.0, 0.0, 0.0, 0.0]
        states[index, 20] = state
        actions[index, 5:20] = (index + 1) / 1000.0
        for candidate, control in enumerate((-0.2, 0.2)):
            action = np.repeat(np.asarray([[control, 0.0]], dtype=np.float64), 20, axis=0)
            arrays[f"terminal/{name}/{candidate}"] = np.asarray(state) + np.asarray([control, 0.0, 0.0, 0.0])
            records.append({
                "instance_id": name, "split": ("train", "calibration", "outcome")[index % 3],
                "candidate_id": candidate, "start_episode_id": str(index), "source_frame": 20,
                "source_state_raw": state, "goal_raw": [1.0, 0.0],
                "source_latent_model": [float(index), 0.0],
                "source_history_model": [float(index - 3), 0.0, float(index - 2), 0.0, float(index - 1), 0.0, float(index), 0.0],
                "action_sequence": action.tolist(), "realized_terminal_state_ref": f"arrays.npz::terminal/{name}/{candidate}",
            })
    source = root / "source.json"
    source.write_text(json.dumps({"metadata": {"history_windows": 4}, "records": records}))
    arrays_path = root / "arrays.npz"
    np.savez(arrays_path, **arrays)
    torch.save(torch.as_tensor(states), root / "states.pth")
    torch.save(torch.as_tensor(actions), root / "actions.pth")
    # Precisely twenty roots are v1 outcome roots and cannot enter any v2 arm.
    v1_records = []
    for index in range(100):
        for candidate in range(2):
            v1_records.append({
                "instance_id": f"root-{index:03d}", "split": "outcome" if index < 20 else "train",
                "candidate_action_hash": f"v1-action-{candidate}", "candidate_template_fingerprint": f"v1-template-{candidate}",
            })
    v1 = root / "v1_context.json"
    v1.write_text(json.dumps({"metadata": {"candidate_family": builder.V1_ACTION_OOD_FAMILY}, "records": v1_records}))
    return source, arrays_path, root, v1


def test_v2_bank_is_seeded_and_excludes_all_v1_outcome_roots(tmp_path: Path, monkeypatch) -> None:
    source, _arrays, root, v1 = _write_source(tmp_path)
    monkeypatch.setattr(builder, "_simulate_sequence", lambda state, action: np.asarray(state) + np.asarray([float(action[:, 0].mean()), float(action[:, 1].mean()), 0.0, 0.0]))
    report = builder.build(source, root, v1, tmp_path / "bank", candidate_count=8)
    payload = json.loads(Path(report["context"]).read_text())
    roots = {str(row["instance_id"]) for row in payload["records"]}
    assert all(f"root-{index:03d}" not in roots for index in range(20))
    assert report["root_counts"] == {"train": 32, "calibration": 24, "outcome": 24}
    assert report["v1_outcome_root_overlap"] == []
    assert report["source_action_hash_overlap"] == []
    assert report["v1_action_hash_overlap"] == []
    assert report["source_template_overlap"] == []
    assert report["v1_template_overlap"] == []


def test_v2_action_seed_changes_bank_identity() -> None:
    first = builder.v2_action_sequences(seed=17, count=8)
    second = builder.v2_action_sequences(seed=18, count=8)
    assert not np.array_equal(first, second)
    assert len({builder.sequence_hash(value) for value in first}) == 8
    assert len({builder.temporal_template_fingerprint(value) for value in first}) == 8


def test_replication_policy_requires_both_heldout_and_calibration_gates() -> None:
    good = {"predeclared_success_rule": {"satisfied": True}, "calibration_only_history_policy": {"decision": "history_permitted"}}
    bad_gate = {"predeclared_success_rule": {"satisfied": True}, "calibration_only_history_policy": {"decision": "history_rejected_action_only_fallback"}}
    assert replication._criterion(good)["replicated_policy_criterion"] is True
    assert replication._criterion(bad_gate)["replicated_policy_criterion"] is False


def test_evaluator_compatibility_view_changes_only_family_metadata(tmp_path: Path) -> None:
    source = tmp_path / "v2.json"
    payload = {
        "metadata": {"candidate_family": builder.V2_ACTION_OOD_FAMILY, "root_split_seed": 1},
        "records": [{"instance_id": "root", "candidate_id": 5000, "action_sequence": [[0.1, 0.0]] * 20}],
    }
    source.write_text(json.dumps(payload))
    view = json.loads(replication._evaluator_compatible_context(source, tmp_path).read_text())
    assert view["records"] == payload["records"]
    assert view["metadata"]["candidate_family"] == builder.V1_ACTION_OOD_FAMILY
    assert view["metadata"]["underlying_candidate_family"] == builder.V2_ACTION_OOD_FAMILY
