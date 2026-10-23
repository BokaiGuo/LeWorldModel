from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v5_recurrent as v5


def test_eight_frame_history_is_strictly_pre_candidate() -> None:
    actions = np.zeros((50, 2), dtype=np.float64)
    actions[5:40] = 1.0
    actions[40:] = 9.0
    frames, history = v5.causal_history8(40, actions)
    assert frames == [5, 10, 15, 20, 25, 30, 35, 40]
    assert history.shape == (35, 2)
    assert float(history.max()) == 1.0


def test_recurrent_history_can_modulate_candidate_effect() -> None:
    torch.manual_seed(3)
    model = v5.RecurrentModulatedEffect(latent_dim=8, candidate_dim=40, hidden=16).eval()
    left = torch.randn(2, 8, 10)
    right = torch.randn(2, 8, 10)
    base, candidate = torch.zeros(2, 40), torch.randn(2, 40)
    with torch.no_grad():
        # A learned nonzero FiLM map must make the same candidate's effect
        # depend on history; mixed differences would be identically zero for
        # the former additive b(history) + g(candidate) readout.
        model.modulation.weight.fill_(0.1)
        model.modulation.bias.zero_()
        delta_left = model(left, candidate) - model(left, base)
        delta_right = model(right, candidate) - model(right, base)
    assert not torch.allclose(delta_left, delta_right, atol=1e-6)


def test_recurrent_modulation_starts_at_additive_baseline() -> None:
    model = v5.RecurrentModulatedEffect(latent_dim=8, candidate_dim=40, hidden=16)
    assert torch.count_nonzero(model.modulation.weight) == 0
    assert torch.count_nonzero(model.modulation.bias) == 0


def test_zero_initialized_modulation_receives_an_optimizer_gradient() -> None:
    torch.manual_seed(9)
    model = v5.RecurrentModulatedEffect(latent_dim=8, candidate_dim=40, hidden=16).train()
    history = torch.randn(3, 8, 10)
    candidate = torch.randn(3, 40)
    loss = model(history, candidate).square().mean()
    loss.backward()
    assert model.modulation.weight.grad is not None
    assert float(model.modulation.weight.grad.norm()) > 0.0


def test_recurrent_tensor_contract_has_eight_steps_and_seven_action_blocks() -> None:
    row = {
        "source_history_model": np.zeros((8, 8), dtype=np.float32).reshape(-1).tolist(),
        "history_actions_raw": np.zeros((35, 2), dtype=np.float32).tolist(),
        "action_sequence": np.zeros((20, 2), dtype=np.float32).tolist(),
    }
    history, candidate = v5._recurrent_tensors([row])
    assert history.shape == (1, 8, 10)
    assert candidate.shape == (1, 40)


def test_v5_seeded_bank_is_disjoint_from_its_training_bank() -> None:
    held = v5.v3_action_sequences(seed=v5.ACTION_SEED, train=False)
    train = v5.v3_action_sequences(seed=v5.TRAIN_ACTION_SEED, train=True)
    assert not {v5.sequence_hash(item) for item in held} & {v5.sequence_hash(item) for item in train}
    assert not {v5.temporal_template_fingerprint(item) for item in held} & {v5.temporal_template_fingerprint(item) for item in train}


def test_v5_refuses_confirmation_run_without_explicit_debug_acknowledgement(tmp_path: Path) -> None:
    """The historically unsealed V5 protocol cannot accidentally create new evidence."""
    with pytest.raises(RuntimeError, match="debug-only"):
        v5.run(
            tmp_path / "source.json", tmp_path / "arrays.npz", tmp_path / "context.json",
            tmp_path / "calibration.json", tmp_path / "outcome.json", tmp_path / "output",
        )


def _write_minimal_v5_protocol(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny, CPU-only artifact set with the same binding contracts."""
    held_action = np.ones((20, 2), dtype=np.float64)
    train_action = np.zeros((20, 2), dtype=np.float64)
    held_hash, train_hash = v5.sequence_hash(held_action), v5.sequence_hash(train_action)
    held_template = v5.temporal_template_fingerprint(held_action)
    train_template = v5.temporal_template_fingerprint(train_action)
    held_bank, train_bank = v5._json_hash([held_hash]), v5._json_hash([train_hash])
    source = tmp_path / "temporal_ood_v5_source_dataset.json"
    arrays = tmp_path / "temporal_ood_v5_source_arrays.npz"
    context = tmp_path / "temporal_ood_v5_context.json"
    calibration = tmp_path / "temporal_ood_v5_calibration_outcomes.json"
    outcome = tmp_path / "temporal_ood_v5_outcomes.json"
    manifest = tmp_path / "temporal_ood_v5_protocol_manifest.json"

    source_row = {
        "instance_id": "pointmaze_v5_episode_11",
        "split": "train",
        "start_episode_id": "11",
        "source_frame": v5.SOURCE_FRAME,
        "history_frame_indices": list(range(5, 41, 5)),
        "history_action_interval": [5, 40],
        "history_actions_raw": np.zeros((35, 2), dtype=np.float64).tolist(),
        "candidate_id": 0,
        "candidate_generator": v5.V5_TRAIN_FAMILY,
        "candidate_action_hash": train_hash,
        "candidate_template_fingerprint": train_template,
        "candidate_bank_hash": train_bank,
        "action_sequence": train_action.tolist(),
        "realized_terminal_state_ref": "arrays.npz::terminal/v5/11/0",
    }
    source.write_text(json.dumps({"metadata": {"schema": "pointmaze_temporal_ood_v5_source.v1", "history_frames": v5.HISTORY_FRAMES}, "records": [source_row]}, sort_keys=True))
    np.savez(arrays, **{"terminal/v5/11/0": np.asarray([0.0, 0.0])})

    context_rows = []
    labels = {"calibration": {}, "outcome": {}}
    for split, root in (("train", 11), ("calibration", 22), ("outcome", 33)):
        row = {
            "instance_id": f"pointmaze_v5_episode_{root}",
            "split": split,
            "start_episode_id": str(root),
            "source_frame": v5.SOURCE_FRAME,
            "history_frame_indices": list(range(5, 41, 5)),
            "history_action_interval": [5, 40],
            "history_actions_raw": np.zeros((35, 2), dtype=np.float64).tolist(),
            "candidate_id": 0,
            "candidate_generator": v5.V5_ACTION_FAMILY,
            "candidate_action_hash": held_hash,
            "candidate_template_fingerprint": held_template,
            "candidate_bank_hash": held_bank,
            "action_sequence": held_action.tolist(),
        }
        context_rows.append(row)
        if split not in labels:
            continue
        labels[split][f"{row['instance_id']}/0"] = {
            "real_task_cost": 1.0,
            "realized_terminal_state": [0.0, 0.0],
            "split": split,
            "start_episode_id": str(root),
            "candidate_action_hash": held_hash,
            "candidate_template_fingerprint": held_template,
            "candidate_bank_hash": held_bank,
        }
    context.write_text(json.dumps({"metadata": {"schema": v5.PANEL_SCHEMA, "history_frames": v5.HISTORY_FRAMES, "encoder_checkpoint_sha256": "encoder-sha", "pca_checkpoint_sha256": "pca-sha"}, "records": context_rows}, sort_keys=True))
    calibration.write_text(json.dumps(labels["calibration"], sort_keys=True))
    outcome.write_text(json.dumps(labels["outcome"], sort_keys=True))
    v5.write_protocol_manifest(source, arrays, context, calibration, outcome, manifest)
    return {"source": source, "arrays": arrays, "context": context, "calibration": calibration, "outcome": outcome, "manifest": manifest}


def test_protocol_manifest_accepts_bound_source_context_arrays_and_labels(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    integrity = v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])
    assert integrity["validated"] is True
    assert integrity["root_counts"] == {"train": 1, "calibration": 1, "outcome": 1}


def test_protocol_manifest_rejects_label_candidate_binding_tamper_even_with_updated_file_hash(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    labels = json.loads(paths["calibration"].read_text())
    labels["pointmaze_v5_episode_22/0"]["candidate_template_fingerprint"] = "tampered-template"
    paths["calibration"].write_text(json.dumps(labels, sort_keys=True))
    manifest = json.loads(paths["manifest"].read_text())
    manifest["files"]["calibration_outcomes"]["sha256"] = v5._sha256(paths["calibration"])
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="label.*template|template.*label"):
        v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])


def test_protocol_manifest_rejects_array_key_drift_even_with_updated_file_hash(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    np.savez(paths["arrays"], **{"terminal/v5/11/0": np.asarray([0.0, 0.0]), "unexpected": np.asarray([1.0])})
    manifest = json.loads(paths["manifest"].read_text())
    manifest["files"]["source_arrays"]["sha256"] = v5._sha256(paths["arrays"])
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="array key"):
        v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])


def test_protocol_manifest_rejects_runner_source_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["runner_source_sha256"] = "0" * 64
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="runner source"):
        v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])


def test_protocol_manifest_rejects_root_split_overlap_even_with_updated_context_hash(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    context = json.loads(paths["context"].read_text())
    next(row for row in context["records"] if row["split"] == "calibration")["start_episode_id"] = "11"
    paths["context"].write_text(json.dumps(context, sort_keys=True))
    manifest = json.loads(paths["manifest"].read_text())
    manifest["files"]["context"]["sha256"] = v5._sha256(paths["context"])
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="roots overlap"):
        v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])


def test_protocol_manifest_rejects_checkpoint_provenance_change_even_with_updated_context_hash(tmp_path: Path) -> None:
    paths = _write_minimal_v5_protocol(tmp_path)
    context = json.loads(paths["context"].read_text())
    context["metadata"]["encoder_checkpoint_sha256"] = "other-encoder-sha"
    paths["context"].write_text(json.dumps(context, sort_keys=True))
    manifest = json.loads(paths["manifest"].read_text())
    manifest["files"]["context"]["sha256"] = v5._sha256(paths["context"])
    paths["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="checkpoint_info"):
        v5.validate_protocol_integrity(paths["source"], paths["arrays"], paths["context"], paths["calibration"], paths["outcome"], paths["manifest"])


def test_candidate_bank_contract_rejects_a_root_with_a_missing_candidate() -> None:
    action_a = np.zeros((20, 2), dtype=np.float64)
    action_b = np.ones((20, 2), dtype=np.float64) * 0.25
    rows = []
    hashes = [v5.sequence_hash(action_a), v5.sequence_hash(action_b)]
    templates = [v5.temporal_template_fingerprint(action_a), v5.temporal_template_fingerprint(action_b)]
    bank_hash = v5._json_hash(hashes)
    for candidate_id, action in enumerate((action_a, action_b)):
        if candidate_id == 1:
            continue
        rows.append({
            "instance_id": "root-a", "candidate_id": candidate_id,
            "candidate_generator": v5.V5_ACTION_FAMILY,
            "candidate_action_hash": hashes[candidate_id],
            "candidate_template_fingerprint": templates[candidate_id],
            "candidate_bank_hash": bank_hash, "action_sequence": action.tolist(),
        })
    rows.extend({
        "instance_id": "root-b", "candidate_id": candidate_id,
        "candidate_generator": v5.V5_ACTION_FAMILY,
        "candidate_action_hash": hashes[candidate_id],
        "candidate_template_fingerprint": templates[candidate_id],
        "candidate_bank_hash": bank_hash, "action_sequence": action.tolist(),
    } for candidate_id, action in enumerate((action_a, action_b)))
    with pytest.raises(ValueError, match="candidate bank is incomplete"):
        v5._candidate_bank_contract(rows, family=v5.V5_ACTION_FAMILY, description="context")


def test_calibration_selection_and_authorization_roots_are_disjoint() -> None:
    records = [{"split": "calibration", "start_episode_id": str(root), "candidate_id": 0} for root in range(8)]
    selection, authorization = v5._calibration_roles(records)
    selection_roots = {row["start_episode_id"] for row in selection}
    authorization_roots = {row["start_episode_id"] for row in authorization}
    assert selection_roots.isdisjoint(authorization_roots)
    assert selection_roots | authorization_roots == {str(root) for root in range(8)}


def test_recurrent_model_summary_reports_class_and_exact_parameter_count() -> None:
    model = v5.RecurrentModulatedEffect(latent_dim=8, candidate_dim=40, hidden=16)
    summary = v5._model_summary(model)
    assert summary["class_name"] == "RecurrentModulatedEffect"
    assert summary["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert summary["trainable_parameter_count"] == summary["parameter_count"]
