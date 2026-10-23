from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6


def test_pairwise_counterfactual_contrast_excludes_near_ties() -> None:
    records = [
        {"instance_id": "root", "candidate_id": index}
        for index in range(4)
    ]
    report = v6.pairwise_counterfactual_contrast_metrics(
        records, np.asarray([0.1, 0.5, 0.3, 0.31]), np.asarray([0.0, 0.2, 0.7, 0.701]), margin=0.003,
    )
    assert report["pair_count"] == 2
    assert report["valid_pair_count"] == 1
    assert report["sign_accuracy"] == 1.0


def test_pairwise_counterfactual_contrast_is_optional_for_non_pairwise_bank() -> None:
    records = [{"instance_id": "legacy-root", "candidate_id": index} for index in range(3)]
    report = v6.pairwise_counterfactual_contrast_metrics(
        records, np.asarray([0.1, 0.2, 0.3]), np.asarray([0.0, 0.2, 0.4]), margin=0.003,
    )
    assert report["status"] == "not_applicable"
    assert report["reason"] == "candidate_ids_are_not_contiguous_even_pairs"
    assert report["sign_accuracy"] is None


def test_pairwise_authorization_uses_root_cluster_not_pair_count() -> None:
    records = []
    # Two roots with two pairs each: the first root favours the method on both
    # pairs, while the second favours the baseline on both.  The gain is zero
    # at the root level, regardless of there being four pair observations.
    for root in ("r0", "r1"):
        records.extend({"instance_id": root, "candidate_id": candidate} for candidate in range(4))
    actual = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    method = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    baseline = np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    gain = v6._paired_pairwise_sign_gain(records, method, baseline, actual)
    assert gain["root_count"] == 2
    assert gain["mean_gain"] == 0.0


def _row(root: str, split: str, candidate_id: int, *, signal: float) -> dict[str, object]:
    start = np.stack([np.asarray([0., .2]), np.asarray([signal, signal + .2]), np.asarray([signal, signal + .4])])
    history = np.vstack([start, np.repeat(start[-1:], 5, axis=0)])
    action = np.repeat(np.asarray([[.15 + .15 * candidate_id, -.12 + .1 * candidate_id]]), 20, axis=0)
    visual_state = np.c_[history[:, :2], .5 * history[:, :2]]
    return {"instance_id": root, "start_episode_id": root, "split": split, "candidate_id": candidate_id, "source_frame": 40, "history_frame_indices": [5, 10, 15, 20, 25, 30, 35, 40], "history_action_interval": [5, 40], "source_history_model": history.reshape(-1).tolist(), "source_history_visual_state": visual_state.tolist(), "history_actions_raw": np.repeat(np.asarray([[.02, .01]]), 35, axis=0).tolist(), "action_sequence": action.tolist(), "source_state_raw": [0., 0., 0., 0.], "goal_raw": [.5, -.2, 0., 0.], "realized_terminal_state_ref": f"arrays.npz::terminal/{root}/{candidate_id}"}


def _effect(row: dict[str, object]) -> np.ndarray:
    history = np.asarray(row["source_history_model"], dtype=np.float64).reshape(8, -1)
    action = np.asarray(row["action_sequence"], dtype=np.float64).mean(axis=0)
    return .25 * action + .35 * float(history[-1].mean() - history[0].mean()) * np.asarray([action[1], -action[0]])


def _bind_candidate_metadata(records: list[dict[str, object]]) -> None:
    """Give synthetic fixture rows the same identity contract as the builder."""
    groups: dict[str, list[dict[str, object]]] = {}
    for row in records:
        groups.setdefault(str(row["instance_id"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["candidate_id"]))
        hashes = [v6.sequence_hash(np.asarray(row["action_sequence"], dtype=np.float64)) for row in rows]
        bank = v6.hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row, action_hash in zip(rows, hashes):
            row["candidate_action_hash"] = action_hash
            row["candidate_bank_hash"] = bank


def _write_builder_style_panel(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_rows, context_rows, arrays = [], [], {}
    for index in range(8):
        root = f"train-{index}"
        for candidate in range(3):
            row = _row(root, "train", candidate, signal=(-1. if index % 2 else 1.)); source_rows.append(row)
            arrays[f"terminal/{root}/{candidate}"] = np.r_[_effect(row), [0., 0.]]
            # The production builder records every root in the context artifact,
            # including train roots (with a separate held candidate bank).  The
            # runner now binds those root identities to the source artifact to
            # reject accidental source/context splices.  Keep this fixture on
            # the same contract without embedding the source terminal reference.
            context_rows.append({key: value for key, value in row.items() if key != "realized_terminal_state_ref"})
    for split in ("calibration", "outcome"):
        for index in range(6):
            for candidate in range(3):
                context_rows.append(_row(f"{split}-{index}", split, candidate, signal=(-1. if index % 2 else 1.)))
    _bind_candidate_metadata(source_rows)
    _bind_candidate_metadata(context_rows)
    source, source_arrays, context, calibration = tmp_path / "source.json", tmp_path / "source_arrays.npz", tmp_path / "context.json", tmp_path / "calibration.json"
    source.write_text(json.dumps({"metadata": {"schema": v6.SOURCE_SCHEMA, "history_frames": 8}, "records": source_rows}))
    np.savez(source_arrays, **arrays)
    context.write_text(json.dumps({"metadata": {"schema": v6.CONTEXT_SCHEMA, "history_frames": 8, "history_actions_are_strictly_pre_candidate": True, "outcome_labels_materialized_during_build": False, "candidate_generalization_regime": "shared_candidate_in_support", "calibration_label_identity_bound": True, "goal_qpos": [.5, -.2], "legacy_goal_reproduction": False}, "records": context_rows}))
    labels = {}
    for row in context_rows:
        if row["split"] == "calibration":
            terminal = np.r_[_effect(row), [0., 0.]]
            labels[v6._key(row)] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - np.asarray(row["goal_raw"])[:2])), "realized_terminal_state": terminal.tolist(), "instance_id": row["instance_id"], "candidate_id": row["candidate_id"], "candidate_action_hash": row["candidate_action_hash"], "candidate_bank_hash": row["candidate_bank_hash"]}
    calibration.write_text(json.dumps(labels))
    manifest = tmp_path / "temporal_ood_v6_manifest.json"
    artifact = {
        "source_dataset": {"path": str(source.resolve()), "sha256": v6._sha256(source)},
        "source_arrays": {"path": str(source_arrays.resolve()), "sha256": v6._sha256(source_arrays)},
        "context": {"path": str(context.resolve()), "sha256": v6._sha256(context)},
        "calibration_outcomes": {"path": str(calibration.resolve()), "sha256": v6._sha256(calibration)},
        "outcome_labels": {"path": str((tmp_path / "temporal_ood_v6_outcome_labels.json").resolve()), "materialized": False, "sha256": None},
    }
    manifest.write_text(json.dumps({"schema": v6.MANIFEST_SCHEMA, "formal_protocol": False, "root_counts": {"train": 8, "calibration": 6, "outcome": 6}, "artifacts": artifact}))
    return source, source_arrays, context, calibration


def test_film_history_modulates_pulse_effect_when_interaction_is_nonzero() -> None:
    torch.manual_seed(7)
    model = v6.RecurrentPulseFiLMEffect(history_dim=4, action_dim=2, hidden=8).eval()
    left, right, base, candidate = torch.randn(1, 8, 4), torch.randn(1, 8, 4), torch.zeros(1, 20, 2), torch.randn(1, 20, 2)
    with torch.no_grad():
        model.film.weight.fill_(.1)
        assert not torch.allclose(model(left, candidate) - model(left, base), model(right, candidate) - model(right, base), atol=1e-6)


def test_history_source_family_is_causal_and_does_not_alias_raw_state() -> None:
    row = _row("source", "train", 0, signal=1.0)
    latent, _candidate, _state, _goal = v6._tensors([row])
    visual, _candidate, _state, _goal = v6._tensors([row], history_source="source_history_visual_state")
    hybrid, _candidate, _state, _goal = v6._tensors([row], history_source="hybrid_latent_visual_state")
    # Every family appends only the strictly pre-candidate action blocks.
    assert latent.shape == (1, 8, 4)
    assert visual.shape == (1, 8, 6)
    assert hybrid.shape == (1, 8, 8)
    assert np.allclose(visual[0, :, -2:], latent[0, :, -2:])
    with pytest.raises(ValueError, match="unsupported history source"):
        v6._tensors([row], history_source="source_state_raw")


def test_interaction_diagnostic_detects_a_trained_style_film_cross_difference(tmp_path: Path) -> None:
    source, _arrays, _context, _calibration = _write_builder_style_panel(tmp_path)
    _metadata, rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    model = v6.RecurrentPulseFiLMEffect(history_dim=4, action_dim=2, hidden=8).eval()
    with torch.no_grad():
        model.film.weight.fill_(.1)
    fit = {
        "history_mean": np.zeros(4, dtype=np.float32), "history_scale": np.ones(4, dtype=np.float32),
        "candidate_mean": np.zeros(2, dtype=np.float32), "candidate_scale": np.ones(2, dtype=np.float32),
        "target_mean": np.zeros(2, dtype=np.float32), "target_scale": np.ones(2, dtype=np.float32),
    }
    diagnostic = v6.interaction_diagnostics(model, fit, rows)
    assert diagnostic["film_weight_frobenius"] > 0.0
    assert diagnostic["history_candidate_mixed_difference_l2"] > 1.0e-6
    assert diagnostic["interaction_active_at_1e-6"] is True
    assert diagnostic["full_bank_candidate_count"] == 3
    assert diagnostic["full_bank_history_count"] == 8
    assert isinstance(diagnostic["history_changes_full_bank_canonical_ranking"], bool)


def test_builder_contract_rejects_a_temporally_shifted_history_ledger() -> None:
    source = [_row("train", "train", 0, signal=1.0)]
    context = [
        _row("train", "train", 0, signal=1.0),
        _row("calibration", "calibration", 0, signal=1.0),
        _row("outcome", "outcome", 0, signal=1.0),
    ]
    _bind_candidate_metadata(source)
    _bind_candidate_metadata(context)
    context[-1]["history_action_interval"] = [10, 45]
    with pytest.raises(ValueError, match="history-action interval"):
        v6._validate_builder_contract(
            source,
            context,
            {"candidate_generalization_regime": "shared_candidate_in_support", "calibration_label_identity_bound": True, "goal_qpos": [.5, -.2], "legacy_goal_reproduction": False},
            strict_counts=False,
        )


def test_matched_builder_calibration_is_deterministically_partitioned() -> None:
    rows = [_row(f"cal-{index}", "calibration", candidate, signal=1.) for index in range(8) for candidate in range(2)]
    first, second = v6.deterministic_calibration_split(rows), v6.deterministic_calibration_split(list(reversed(rows)))
    assert first == second
    assert len(first["selection"]) == len(first["authorization"]) == 4
    assert set(first["selection"]).isdisjoint(first["authorization"])


def test_support_regime_contract_rejects_a_shared_panel_with_an_unseen_action() -> None:
    source = [_row("train", "train", 0, signal=1.0)]
    context = [
        _row("train", "train", 0, signal=1.0),
        _row("calibration", "calibration", 0, signal=1.0),
        _row("outcome", "outcome", 0, signal=1.0),
    ]
    for row in context:
        row["action_sequence"] = np.repeat(np.asarray([[.73, -.41]]), 20, axis=0).tolist()
    _bind_candidate_metadata(source)
    _bind_candidate_metadata(context)
    with pytest.raises(ValueError, match="unsupported held actions"):
        v6._validate_builder_contract(
            source,
            context,
            {"candidate_generalization_regime": "shared_candidate_in_support", "calibration_label_identity_bound": True, "goal_qpos": [.5, -.2], "legacy_goal_reproduction": False},
            strict_counts=False,
        )


def test_candidate_identity_rejects_late_row_bank_hash_tamper() -> None:
    """Every candidate row must bind the same executable action bank."""
    records = [
        _row("root", "outcome", candidate, signal=1.0)
        for candidate in range(3)
    ]
    _bind_candidate_metadata(records)
    records[-1]["candidate_bank_hash"] = "tampered-bank"

    with pytest.raises(ValueError, match="candidate bank hash"):
        v6._validate_candidate_identity(records, label="tampered")


def test_v2_contract_requires_declared_support_and_identity_labels() -> None:
    source = [_row("train", "train", 0, signal=1.0)]
    context = [
        _row("train", "train", 0, signal=1.0),
        _row("calibration", "calibration", 0, signal=1.0),
        _row("outcome", "outcome", 0, signal=1.0),
    ]
    _bind_candidate_metadata(source)
    _bind_candidate_metadata(context)
    with pytest.raises(ValueError, match="identity-bound"):
        v6._validate_builder_contract(source, context, {}, strict_counts=False)
    with pytest.raises(ValueError, match="generalization regime"):
        v6._validate_builder_contract(source, context, {"calibration_label_identity_bound": True}, strict_counts=False)


def test_v2_calibration_rejects_swapped_label_candidate_provenance(tmp_path: Path) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    labels = json.loads(calibration.read_text())
    first, second = sorted(labels)[:2]
    labels[first]["candidate_action_hash"] = labels[second]["candidate_action_hash"]
    calibration.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="manifest calibration_outcomes binding"):
        v6.calibrate(source, arrays, context, calibration, tmp_path / "frozen.json", epochs=3, device="cpu")


def test_selection_binds_costs_to_candidate_ids_not_input_row_order() -> None:
    rows = [
        _row("root", "outcome", 2, signal=1.0),
        _row("root", "outcome", 0, signal=1.0),
        _row("root", "outcome", 1, signal=1.0),
    ]
    # The minimum predicted entry is candidate 0, while the realised optimum
    # is candidate 1.  A ranker must not mix this mapping when JSONL rows have
    # been reordered by an external artifact tool.
    ranked = v6._selection(rows, np.asarray([.4, .1, .2]), np.asarray([.8, .5, .0]))
    root = ranked["roots"]["root"]
    assert root["winner_candidate_id"] == 0
    assert root["regret"] == pytest.approx(.5)


def test_cyclic_root_history_rotates_whole_root_histories_without_reordering_candidate_rows() -> None:
    # Deliberately interleave roots and vary candidate order: the control must
    # rotate only the history tensor, not re-sort or otherwise rewrite rows.
    records = [
        _row("bravo", "outcome", 2, signal=2.0),
        _row("alpha", "outcome", 1, signal=1.0),
        _row("charlie", "outcome", 0, signal=3.0),
        _row("bravo", "outcome", 0, signal=2.0),
        _row("alpha", "outcome", 2, signal=1.0),
        _row("charlie", "outcome", 1, signal=3.0),
    ]
    history, candidate, _state, _goal = v6._tensors(records)
    row_identity_before = [
        (row["instance_id"], row["candidate_id"], row["action_sequence"])
        for row in records
    ]

    permuted = v6._cyclic_root_history(records, history)

    # Sorted cyclic donors are alpha <- bravo <- charlie <- alpha.  Every
    # candidate row stays at its original index and receives its donor root's
    # complete shared history.
    donor = {"alpha": "bravo", "bravo": "charlie", "charlie": "alpha"}
    representative = {
        root: history[next(index for index, row in enumerate(records) if row["instance_id"] == root)]
        for root in donor
    }
    for index, row in enumerate(records):
        recipient_root = str(row["instance_id"])
        assert np.array_equal(permuted[index], representative[donor[recipient_root]])
    assert [
        (row["instance_id"], row["candidate_id"], row["action_sequence"])
        for row in records
    ] == row_identity_before
    assert np.array_equal(candidate, v6._tensors(records)[1])


def test_cyclic_root_history_rejects_candidate_rows_with_nonshared_root_history() -> None:
    records = [
        _row("alpha", "outcome", 0, signal=1.0),
        _row("alpha", "outcome", 1, signal=1.0),
        _row("bravo", "outcome", 0, signal=2.0),
        _row("bravo", "outcome", 1, signal=2.0),
    ]
    history, _candidate, _state, _goal = v6._tensors(records)
    history[1, 0, 0] += 1.0

    with pytest.raises(ValueError, match="candidate rows of a root must share one source history"):
        v6._cyclic_root_history(records, history)


def test_history_energy_gate_is_root_level_visual_only_and_preserves_candidate_alignment() -> None:
    records = [
        _row("low", "calibration", candidate, signal=.1) for candidate in range(3)
    ] + [
        _row("high", "calibration", candidate, signal=4.0) for candidate in range(3)
    ] + [
        _row("middle-a", "calibration", candidate, signal=1.0) for candidate in range(3)
    ] + [
        _row("middle-b", "calibration", candidate, signal=2.0) for candidate in range(3)
    ]
    specs = v6._history_gate_specs(records)
    policy = next(spec for spec in specs if spec.get("direction") == "recurrent_when_low" and spec.get("quantile") == .5)
    mask = v6._history_gate_mask(records, policy)
    assert mask.shape == (len(records),)
    for root, rows in v6._group(records).items():
        indices = [index for index, row in enumerate(records) if row["instance_id"] == root]
        assert len(set(mask[indices].tolist())) == 1
    recurrent = np.arange(len(records), dtype=np.float64)
    action = np.full(len(records), -1.0)
    routed = v6._policy_cost(records, recurrent, action, policy)
    assert np.array_equal(routed[mask], recurrent[mask])
    assert np.array_equal(routed[~mask], action[~mask])


def test_frozen_history_energy_gate_is_accepted_by_the_score_contract() -> None:
    records = [
        _row(f"calibration-{index}", "calibration", candidate, signal=float(index))
        for index in range(4)
        for candidate in range(2)
    ]
    partition = v6.deterministic_calibration_split(records)
    policy = {
        "kind": "history_energy_gate",
        "direction": "recurrent_when_high",
        "quantile": 0.5,
        "threshold": 0.25,
    }
    frozen = {
        "schema": v6.FROZEN_SCHEMA,
        "history_gate_enabled": True,
        "root_counts": {"train": 8, "calibration": 4, "outcome": 4},
        "selected_hidden": v6.HIDDEN_WIDTHS[0],
        "selected_contrast_weight": 0.25,
        "selected_temperature": v6.TEMPERATURES[0],
        "selected_policy": policy,
        "authorization": {"permitted": False},
        "calibration_partition": partition,
    }
    assert v6._validate_frozen_selection_contract(
        frozen, records, expected_schema=v6.FROZEN_SCHEMA, history_gate=True,
        expected_root_counts={"train": 8, "calibration": 4, "outcome": 4},
    ) == policy


def test_root_count_contract_supports_a_larger_formal_panel_without_relaxing_validation() -> None:
    counts = v6._validated_root_counts({"train": 128, "calibration": 192, "outcome": 64})
    assert counts == {"train": 128, "calibration": 192, "outcome": 64}
    with pytest.raises(ValueError, match="even"):
        v6._validated_root_counts({"train": 128, "calibration": 191, "outcome": 64})
    with pytest.raises(ValueError, match="exactly"):
        v6._validated_root_counts({"train": 128, "calibration": 192})


def test_rootwise_task_contrast_distinguishes_candidate_order_within_a_root() -> None:
    state = torch.zeros((2, 2))
    goal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    observed_cost = torch.tensor([1.0, 0.0])
    root_indices = torch.tensor([[0, 1]], dtype=torch.long)
    correct = v6.rootwise_task_contrast_loss(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]), state, goal, observed_cost, root_indices,
    )
    inverted = v6.rootwise_task_contrast_loss(
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]), state, goal, observed_cost, root_indices,
    )
    assert float(correct) == pytest.approx(0.0)
    assert float(inverted) > 0.0


def test_score_generates_pointmaze_labels_only_after_frozen_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = v6.calibrate(source, arrays, context, calibration, frozen_path, epochs=3, device="cpu")
    assert frozen["outcome_labels_opened"] is False
    assert Path(frozen["model_checkpoint_path"]).is_file()
    assert frozen["model_checkpoint_sha256"] == v6._sha256(Path(frozen["model_checkpoint_path"]))
    assert not list(tmp_path.glob("*outcome*labels*.json"))
    calls: list[int] = []
    def fake_simulate(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        calls.append(1); return np.r_[.25 * action.mean(axis=0), [0., 0.]]
    monkeypatch.setattr(v6, "_simulate_sequence", fake_simulate)
    def fail_refit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("score must load the calibration-frozen model, never refit")
    monkeypatch.setattr(v6, "_fit", fail_refit)
    report = v6.score(source, arrays, context, frozen_path, tmp_path / "score", device="cpu")
    assert len(calls) == 6 * 3
    assert Path(report["outcome_label_path"]).exists()
    assert report["outcome_labels_sha256"] == v6._sha256(Path(report["outcome_label_path"]))
    assert report["outcome_labels_generated_after_frozen_selection"] is True
    assert report["outcome_labels_preexisted"] is False


def test_geometry_history_family_is_frozen_before_outcome_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "geometry_frozen.json"
    frozen = v6.calibrate(
        source, arrays, context, calibration, frozen_path, epochs=3, device="cpu",
        history_sources=("source_history_visual_state",),
    )
    assert frozen["history_source_candidates"] == ["source_history_visual_state"]
    assert frozen["selected_history_source"] == "source_history_visual_state"
    monkeypatch.setattr(
        v6, "_simulate_sequence", lambda _state, action: np.r_[.25 * action.mean(axis=0), [0., 0.]],
    )
    report = v6.score(source, arrays, context, frozen_path, tmp_path / "geometry_score", device="cpu")
    assert report["selected_history_source"] == "source_history_visual_state"


def test_score_fails_closed_without_frozen_selection(tmp_path: Path) -> None:
    source, arrays, context, _calibration = _write_builder_style_panel(tmp_path)
    with pytest.raises(FileNotFoundError, match="persisted frozen selection"):
        v6.score(source, arrays, context, tmp_path / "missing.json", tmp_path / "score", device="cpu")


def test_score_recovers_a_relocated_hash_bound_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = v6.calibrate(source, arrays, context, calibration, frozen_path, epochs=3, device="cpu")
    frozen["model_checkpoint_path"] = "/relocated/temporal_ood_v6_frozen_selection_model.pt"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(
        v6, "_simulate_sequence", lambda _state, action: np.r_[.25 * action.mean(axis=0), [0., 0.]],
    )

    report = v6.score(source, arrays, context, frozen_path, tmp_path / "score", device="cpu")

    assert report["outcome_labels_generated_after_frozen_selection"] is True


def test_score_rejects_v2_selection_without_a_frozen_model_before_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = v6.calibrate(source, arrays, context, calibration, frozen_path, epochs=3, device="cpu")
    frozen["schema"] = v6.LEGACY_FROZEN_SCHEMA
    frozen.pop("model_checkpoint_path")
    frozen.pop("model_checkpoint_sha256")
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(v6, "_simulate_sequence", lambda *_args: pytest.fail("must not open outcomes"))

    with pytest.raises(ValueError, match="cannot be replayed"):
        v6.score(source, arrays, context, frozen_path, tmp_path / "score", device="cpu")
    assert not (tmp_path / "score").exists()


def test_score_rejects_tampered_contrast_weight_before_opening_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = v6.calibrate(source, arrays, context, calibration, frozen_path, epochs=3, device="cpu")
    selected = float(frozen["selected_contrast_weight"])
    frozen["selected_contrast_weight"] = next(weight for weight in v6.CONTRAST_WEIGHTS if weight != selected)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(v6, "_simulate_sequence", lambda *_args: pytest.fail("must not open outcomes"))

    with pytest.raises(ValueError, match="contrast weight"):
        v6.score(source, arrays, context, frozen_path, tmp_path / "score", device="cpu")
    assert not (tmp_path / "score").exists()


def test_score_rejects_a_frozen_model_from_a_different_runner_source(tmp_path: Path) -> None:
    source, arrays, context, calibration = _write_builder_style_panel(tmp_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = v6.calibrate(source, arrays, context, calibration, frozen_path, epochs=3, device="cpu")
    model_path = Path(frozen["model_checkpoint_path"])
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    payload["runner_source_sha256"] = "0" * 64
    torch.save(payload, model_path)
    frozen["model_checkpoint_sha256"] = v6._sha256(model_path)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="different runner source"):
        v6.score(source, arrays, context, frozen_path, tmp_path / "score", device="cpu")
    assert not (tmp_path / "score").exists()
