from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as v6


def _prior_context(path: Path, roots: list[int], prefix: str) -> Path:
    records = []
    for index, root in enumerate(roots):
        action = np.repeat(np.asarray([[0.3 + index * 0.01, 0.1]], dtype=np.float64), 20, axis=0)
        records.append({
            "start_episode_id": str(root),
            "candidate_action_hash": f"{prefix}-hash-{index}",
            "candidate_template_fingerprint": f"{prefix}-template-{index}",
            "action_sequence": action.tolist(),
        })
    path.write_text(json.dumps({"metadata": {"candidate_family": prefix}, "records": records}))
    return path


def test_v6_prior_identity_audit_includes_metadata_only_interaction_roots(tmp_path: Path) -> None:
    path = _prior_context(tmp_path / "prior.json", [1, 2], "prior")
    payload = json.loads(path.read_text())
    payload["metadata"]["interaction_design_roots"] = [17, 19]
    path.write_text(json.dumps(payload))
    roots, _hashes, _templates, _checksums = v6._all_prior_identities([path])
    assert roots == {1, 2, 17, 19}


def test_v6_prior_identity_audit_rejects_partial_candidate_records_with_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({
        "metadata": {"candidate_family": "partial"},
        "records": [{"start_episode_id": "1", "candidate_action_hash": "hash-only"}],
    }))
    with pytest.raises(ValueError, match="lacks candidate identity fields"):
        v6._all_prior_identities([path])


def test_directional_interaction_pairs_preserve_support_and_legacy_summary_collision() -> None:
    pairs = v6.directional_interaction_pair_proposals(seed=17, count=8)
    assert pairs.shape == (8, 2, v6.TOTAL_STEPS, 2)
    assert float(np.abs(pairs).max()) <= 1.0
    for pair in pairs:
        assert np.allclose(pair[0].mean(axis=0), pair[1].mean(axis=0), rtol=0.0, atol=1e-12)
        assert np.allclose(pair[0].std(axis=0), pair[1].std(axis=0), rtol=0.0, atol=1e-12)
        assert np.allclose(pair[0, 0], pair[1, 0], rtol=0.0, atol=1e-12)
        assert np.allclose(pair[0, -1], pair[1, -1], rtol=0.0, atol=1e-12)
        assert not np.array_equal(pair[0], pair[1])


def test_frozen_preflight_ledger_binds_context_bytes_and_declared_panel_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"frozen-state-pool")
    context = _prior_context(tmp_path / "prior.json", [1, 2], "prior")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    ledger = tmp_path / "preflight.json"
    ledger.write_text(json.dumps({
        "schema": v6.FROZEN_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight",
        "checks": {"all": True},
        "prior_context_sha256": context_sha256,
        "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 128, "calibration": 192, "outcome": 64},
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
            "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
        },
    }))
    provenance = v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 128, "calibration": 192, "outcome": 64},
        candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]),
    )
    assert provenance is not None
    assert provenance["context_count"] == 1
    changed = dict(context_sha256); changed[str(context.resolve())] = "0" * 64
    with pytest.raises(ValueError, match="exactly match"):
        v6.frozen_preflight_provenance(
            ledger, dataset_root=dataset, prior_context_sha256=changed,
            split_counts={"train": 128, "calibration": 192, "outcome": 64},
            candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]),
        )


def test_v7_frozen_preflight_binds_seeds_and_declared_root_range(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"fresh-v7-state-pool")
    context = _prior_context(tmp_path / "prior.json", [1, 2], "prior")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    construction = {
        "root_seed": 202610371, "held_action_seed": 202610372, "train_action_seed": 202610373,
        "root_id_min": 10296, "root_id_max_exclusive": 10808,
    }
    ledger = tmp_path / "v7_preflight.json"
    ledger.write_text(json.dumps({
        "schema": v6.PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 128, "calibration": 192, "outcome": 64},
        "construction": construction,
        "post_execution_aliasing_contract": {
            "representation_key": "source_history_model",
            "radius_quantile": 0.25,
            "contrast_gap_threshold": 0.05,
            "max_reported_pairs": 64,
        },
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
            "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
            "shared_candidate_in_support": True,
            "held_action_hash_count": 16,
            "held_train_action_hash_overlap_count": 16,
            "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "scorer_training_action_hash_count": 16,
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": 0.001,
            "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
            "design_candidate_cost_spread_q10": 0.05,
            "router_feature_contract": "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1",
            "router_feature_dimension": 95,
            "router_train_root_count": 128,
            "minimum_router_train_root_surplus": 16,
            "router_kind": "advantage_ridge",
            "router_action_representation": "full_ordered_20x2",
            "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        },
    }))
    provenance = v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 128, "calibration": 192, "outcome": 64},
        candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
    )
    assert provenance and provenance["construction"] == construction
    assert provenance["post_execution_aliasing_contract"]["max_reported_pairs"] == 64
    with pytest.raises(ValueError, match="construction seeds"):
        v6.frozen_preflight_provenance(
            ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
            split_counts={"train": 128, "calibration": 192, "outcome": 64},
            candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]),
            construction={**construction, "root_seed": construction["root_seed"] + 1},
        )


def test_frozen_preflight_binds_extension_seed_via_manifest_without_changing_runner_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"fresh-extension-state-pool")
    (dataset / "extension_manifest.json").write_text(json.dumps({"seed": 202611100}))
    context = _prior_context(tmp_path / "prior.json", [1, 2], "prior")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    construction = {
        "root_seed": 11, "held_action_seed": 12, "train_action_seed": 13,
        "root_id_min": 100, "root_id_max_exclusive": 200,
    }
    ledger = tmp_path / "extension_preflight.json"
    ledger.write_text(json.dumps({
        "schema": v6.PLANNING_SHAPED_ADVANTAGE_V7_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 128, "calibration": 192, "outcome": 64},
        "construction": {**construction, "extension_seed": 202611100},
        "post_execution_aliasing_contract": {
            "representation_key": "source_history_model", "radius_quantile": 0.25,
            "contrast_gap_threshold": 0.05, "max_reported_pairs": 64,
        },
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
            "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
            "shared_candidate_in_support": True, "held_action_hash_count": 16,
            "held_train_action_hash_overlap_count": 16, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "scorer_training_action_hash_count": 16, "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": 0.001, "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
            "design_candidate_cost_spread_q10": 0.05,
            "router_feature_contract": "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1",
            "router_feature_dimension": 95, "router_train_root_count": 128,
            "minimum_router_train_root_surplus": 16, "router_kind": "advantage_ridge",
            "router_action_representation": "full_ordered_20x2",
            "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        },
    }))
    assert v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 128, "calibration": 192, "outcome": 64},
        candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
    ) is not None
    (dataset / "extension_manifest.json").write_text(json.dumps({"seed": 202611101}))
    with pytest.raises(ValueError, match="extension manifest seed"):
        v6.frozen_preflight_provenance(
            ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
            split_counts={"train": 128, "calibration": 192, "outcome": 64},
            candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
        )


def test_v9_panel_accepts_the_declared_bootstrap_lower_advantage_router(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"bootstrap-router-state-pool")
    context = _prior_context(tmp_path / "prior.json", [1, 2], "prior")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    construction = {
        "root_seed": 21, "held_action_seed": 22, "train_action_seed": 23,
        "root_id_min": 200, "root_id_max_exclusive": 584,
    }
    candidate_bank = {
        "candidate_design": "expanded_tail_robust_goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
        "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
        "shared_candidate_in_support": True, "held_action_hash_count": 16,
        "held_train_action_hash_overlap_count": 16, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "scorer_training_action_hash_count": 16, "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
        "design_best_second_cost_gap_q10": 0.001, "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
        "design_candidate_cost_spread_q10": 0.05,
        "router_feature_contract": "opportunity_compact_v1_predicted_score_action_delta",
        "router_feature_dimension": 15, "router_train_root_count": 160,
        "minimum_router_train_root_surplus": 16, "router_kind": "advantage_bootstrap_lower",
        "router_action_representation": "full_ordered_20x2",
        "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        "router_training_scope": "scorer_disagreement_only",
        "opportunity_gate": "cost_planning_candidate_disagreement",
        "minimum_router_opportunity_root_count": 31, "minimum_calibration_opportunity_roots_per_role": 8,
    }
    ledger = tmp_path / "v9-bootstrap.json"
    ledger.write_text(json.dumps({
        "schema": v6.OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 160, "calibration": 160, "outcome": 64},
        "construction": construction, "post_execution_aliasing_contract": v6.V7_POST_EXECUTION_ALIASING_CONTRACT,
        "candidate_bank": candidate_bank,
    }))
    assert v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 160, "calibration": 160, "outcome": 64},
        candidate_design="expanded_tail_robust_goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]),
        construction=construction,
    ) is not None


def test_v8_shared_future_preflight_requires_the_same_support_and_headroom_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"fresh-v8-state-pool")
    context = _prior_context(tmp_path / "prior-v8.json", [1, 2], "prior-v8")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    construction = {
        "root_seed": 202610381, "held_action_seed": 202610382, "train_action_seed": 202610383,
        "root_id_min": 10808, "root_id_max_exclusive": 11320,
    }
    ledger = tmp_path / "v8_preflight.json"
    ledger.write_text(json.dumps({
        "schema": v6.SHARED_FUTURE_V8_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 128, "calibration": 192, "outcome": 64}, "construction": construction,
        "candidate_bank": {
            "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
            "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
            "shared_candidate_in_support": True, "held_action_hash_count": 16,
            "held_train_action_hash_overlap_count": 16, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "scorer_training_action_hash_count": 16, "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": 0.001, "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
            "design_candidate_cost_spread_q10": 0.05,
        },
    }))
    provenance = v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 128, "calibration": 192, "outcome": 64},
        candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
    )
    assert provenance and provenance["schema"] == v6.SHARED_FUTURE_V8_PREFLIGHT_SCHEMA
    assert provenance["post_execution_aliasing_contract"] is None


def test_v9_opportunity_preflight_binds_its_own_gate_and_cannot_downgrade_to_v7(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"fresh-v9-state-pool")
    context = _prior_context(tmp_path / "prior-v9.json", [1, 2], "prior-v9")
    _roots, _hashes, _templates, context_sha256 = v6._all_prior_identities([context])
    construction = {
        "root_seed": 202610391, "held_action_seed": 202610392, "train_action_seed": 202610393,
        "root_id_min": 11320, "root_id_max_exclusive": 11832,
    }
    candidate_bank = {
        "candidate_design": "goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
        "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
        "shared_candidate_in_support": True, "held_action_hash_count": 16,
        "held_train_action_hash_overlap_count": 16, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "scorer_training_action_hash_count": 16, "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
        "design_best_second_cost_gap_q10": 0.001, "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
        "design_candidate_cost_spread_q10": 0.05,
        "router_feature_contract": "opportunity_compact_v1_predicted_score_action_delta",
        "router_feature_dimension": 15, "router_train_root_count": 160,
        "minimum_router_train_root_surplus": 16, "router_kind": "advantage_ridge",
        "router_action_representation": "full_ordered_20x2",
        "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        "router_training_scope": "scorer_disagreement_only",
        "opportunity_gate": "cost_planning_candidate_disagreement",
        "minimum_router_opportunity_root_count": 31,
        "minimum_calibration_opportunity_roots_per_role": 8,
    }
    ledger = tmp_path / "v9_preflight.json"
    ledger.write_text(json.dumps({
        "schema": v6.OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 160, "calibration": 160, "outcome": 64},
        "construction": construction,
        "post_execution_aliasing_contract": v6.V7_POST_EXECUTION_ALIASING_CONTRACT,
        "candidate_bank": candidate_bank,
    }))
    provenance = v6.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
        split_counts={"train": 160, "calibration": 160, "outcome": 64},
        candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
    )
    assert provenance and provenance["schema"] == v6.OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA
    assert provenance["post_execution_aliasing_contract"] == v6.V7_POST_EXECUTION_ALIASING_CONTRACT
    candidate_bank.pop("opportunity_gate")
    ledger.write_text(json.dumps({
        "schema": v6.OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "checks": {"all": True},
        "prior_context_sha256": context_sha256, "dataset_states_sha256": v6._sha256(states),
        "root_counts": {"train": 160, "calibration": 160, "outcome": 64},
        "construction": construction,
        "post_execution_aliasing_contract": v6.V7_POST_EXECUTION_ALIASING_CONTRACT,
        "candidate_bank": candidate_bank,
    }))
    with pytest.raises(ValueError, match="opportunity-gating"):
        v6.frozen_preflight_provenance(
            ledger, dataset_root=dataset, prior_context_sha256=context_sha256,
            split_counts={"train": 160, "calibration": 160, "outcome": 64},
            candidate_design="goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
        )


def test_v6_matched_banks_are_hash_disjoint_and_share_support() -> None:
    train = v6.v6_matched_action_sequences(seed=v6.TRAIN_ACTION_SEED, count=v6.TRAIN_CANDIDATES)
    held = v6.v6_matched_action_sequences(seed=v6.HELD_ACTION_SEED, count=v6.HELD_CANDIDATES)
    assert train.shape == (64, 20, 2)
    assert held.shape == (16, 20, 2)
    assert not {v6.sequence_hash(value) for value in train} & {v6.sequence_hash(value) for value in held}
    diagnostics = v6.support_distance_diagnostics(train, held)
    assert diagnostics["train_within_envelope"]
    assert diagnostics["held_within_envelope"]
    assert diagnostics["pulse_schedule"] == {"pulse_count": 5, "pulse_width": 4, "total_steps": 20}
    assert diagnostics["amplitude_envelope"] == {"minimum": 0.28, "maximum": 0.78}
    assert diagnostics["held_to_nearest_train_pulse_l2_mean"] > 0.0


def test_new_v6_panels_default_to_free_region_goal_and_reject_legacy_target() -> None:
    assert np.array_equal(v6.DEFAULT_GOAL_QPOS, np.asarray([0.8, 0.8]))
    assert np.array_equal(v6.validate_goal_qpos([0.8, 0.8]), v6.DEFAULT_GOAL_QPOS)
    with pytest.raises(ValueError, match="wall-sensitive legacy target"):
        v6.validate_goal_qpos([1.2, 1.2])
    assert np.array_equal(
        v6.validate_goal_qpos([1.2, 1.2], allow_wall_sensitive_goal=True),
        v6.LEGACY_GOAL,
    )


def test_v6_action_seed_override_creates_a_fresh_matched_bank() -> None:
    first = v6.v6_matched_action_sequences(seed=v6.HELD_ACTION_SEED, count=v6.HELD_CANDIDATES)
    fresh = v6.v6_matched_action_sequences(seed=v6.HELD_ACTION_SEED + 1000, count=v6.HELD_CANDIDATES)
    assert {v6.sequence_hash(item) for item in first}.isdisjoint({v6.sequence_hash(item) for item in fresh})
    assert v6.support_distance_diagnostics(first, fresh)["held_within_envelope"] is True


def test_shared_candidate_regime_removes_candidate_extrapolation_but_keeps_root_holdout() -> None:
    held, train = v6.temporal_memory_candidate_banks(
        held_seed=v6.HELD_ACTION_SEED,
        train_seed=v6.TRAIN_ACTION_SEED,
        regime="shared_candidate_in_support",
    )
    held_hashes = {v6.sequence_hash(value) for value in held}
    train_hashes = {v6.sequence_hash(value) for value in train}
    assert held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert train.shape == (v6.TRAIN_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert held_hashes <= train_hashes
    diagnostics = v6.support_distance_diagnostics(train, held)
    assert diagnostics["candidate_hash_overlap_count"] == v6.HELD_CANDIDATES
    assert diagnostics["held_exact_train_match_fraction"] == 1.0
    assert diagnostics["held_to_nearest_train_pulse_l2_mean"] == pytest.approx(0.0)


def test_opportunity_router_ledger_requires_exact_shared_scorer_bank() -> None:
    assert v6._requires_exact_shared_scorer_bank({
        "schema": v6.OPPORTUNITY_ROUTER_V9_PREFLIGHT_SCHEMA,
    }) is True
    assert v6._requires_exact_shared_scorer_bank({
        "schema": v6.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
    }) is True
    assert v6._requires_exact_shared_scorer_bank({"schema": "unrelated"}) is False


def test_state_interaction_pairs_are_energy_matched_and_can_be_screened_without_goal_costs() -> None:
    proposals = v6.state_interaction_pair_proposals(seed=91, count=12)
    assert proposals.shape == (12, 2, v6.TOTAL_STEPS, 2)
    assert np.allclose(np.square(proposals[:, 0]).sum(axis=(1, 2)), np.square(proposals[:, 1]).sum(axis=(1, 2)))
    assert np.allclose(proposals[:, 0].mean(axis=1), proposals[:, 1].mean(axis=1))
    assert np.allclose(proposals[:, 0].std(axis=1), proposals[:, 1].std(axis=1))
    assert np.allclose(proposals[:, 0, 0], proposals[:, 1, 0])
    assert np.allclose(proposals[:, 0, -1], proposals[:, 1, -1])
    assert all(not np.array_equal(left, right) for left, right in proposals)
    states = np.asarray([[.2, .3, -1.0, .5], [.4, .6, .8, -1.2], [.7, .8, .3, .1]])

    def fake_simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # There is no goal/cost argument: screening sees terminal effects only.
        return np.r_[state[:2] + state[2:] * action[0], state[2:]]

    held, report = v6.screen_state_interaction_pairs(states, proposals, simulator=fake_simulator)
    assert held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert report["screening_role"].startswith("design-root terminal-effect interaction")
    assert len(report["selected_pair_indices"]) == v6.HELD_CANDIDATES // 2
    assert len(report["selected_pair_interaction_scale"]) == v6.HELD_CANDIDATES // 2


def test_state_interaction_design_requires_shared_candidate_support() -> None:
    with pytest.raises(ValueError, match="requires shared_candidate_in_support"):
        v6.build_panel(
            Path("unused"), [], Path("unused"), encoder_checkpoint=Path("unused"),
            pca_checkpoint=Path("unused"), dinov2_source=Path("unused"),
            candidate_design="state_interaction_pairs", candidate_generalization_regime="numeric_ood",
        )


def test_goal_balanced_interaction_pairs_are_selected_only_from_disjoint_design_state_costs() -> None:
    proposals = v6.state_interaction_pair_proposals(seed=117, count=16)
    states = np.asarray([[.1, .2, .7, -.2], [.5, .4, -.4, .6], [.8, .6, .2, .3], [.3, .9, -.5, -.1]])

    def fake_simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Screening has only a state, a proposed action and the declared goal;
        # it receives no train, calibration or outcome row.
        return np.r_[state[:2] + state[2:] * action[0], state[2:]]

    held, report = v6.screen_goal_balanced_interaction_pairs(
        states, proposals, np.asarray([.8, .8]), simulator=fake_simulator,
    )
    assert held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert report["declared_goal_qpos"] == [.8, .8]
    assert report["screening_role"].startswith("disjoint-design-root, declared-goal")
    assert 0.0 <= report["design_winner_entropy_normalized"] <= 1.0
    assert report["design_distinct_winner_count"] >= 1
    headroom = report["design_oracle_headroom"]
    assert headroom["task_cost"] == "terminal_euclidean_distance_to_declared_goal"
    assert headroom["root_count"] == len(states)
    assert headroom["candidate_count"] == v6.HELD_CANDIDATES
    assert headroom["best_second_cost_gap"]["minimum"] >= 0.0
    assert 0.0 <= headroom["best_second_cost_gap"]["ge_0p003_fraction"] <= 1.0
    assert headroom["candidate_cost_spread"]["q10"] >= 0.0


def test_robust_goal_balanced_pairs_require_two_disjoint_design_folds() -> None:
    proposals = v6.state_interaction_pair_proposals(seed=211, count=16)
    states = np.asarray([
        [.10, .20, .70, -.20], [.50, .40, -.40, .60],
        [.80, .60, .20, .30], [.30, .90, -.50, -.10],
        [.25, .35, .55, -.45], [.65, .15, -.35, .50],
    ])

    def fake_simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.r_[state[:2] + state[2:] * action[0], state[2:]]

    held, report = v6.screen_robust_goal_balanced_interaction_pairs(
        states, proposals, np.asarray([.8, .8]), simulator=fake_simulator,
    )
    assert held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert report["screening_role"].startswith("two disjoint design-root folds")
    assert report["design_fold_root_counts"] == [3, 3]
    assert len(report["design_fold_headroom"]) == 2
    assert all(item["candidate_count"] == v6.HELD_CANDIDATES for item in report["design_fold_headroom"])
    assert all(0.0 <= item["winner_entropy_normalized"] <= 1.0 for item in report["design_fold_headroom"])
    assert report["design_all_root_headroom"]["root_count"] == len(states)
    tail_held, tail_train, tail_report = v6.tail_robust_goal_balanced_interaction_candidate_banks(
        states, np.asarray([.8, .8]), proposal_seed=211, supplemental_seed=212, simulator=fake_simulator,
    )
    assert tail_held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert tail_train.shape == (v6.TRAIN_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert tail_report["candidate_design"] == "tail_robust_goal_balanced_interaction_pairs"
    assert "best-second-gap-q10" in tail_report["selection_order"]
    expanded_held, expanded_train, expanded_report = v6.expanded_tail_robust_goal_balanced_interaction_candidate_banks(
        states, np.asarray([.8, .8]), proposal_seed=211, supplemental_seed=212,
        proposal_count=128, simulator=fake_simulator,
    )
    assert expanded_held.shape == (v6.HELD_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert expanded_train.shape == (v6.TRAIN_CANDIDATES, v6.TOTAL_STEPS, 2)
    assert expanded_report["candidate_design"] == "expanded_tail_robust_goal_balanced_interaction_pairs"
    with pytest.raises(ValueError, match="cannot reduce"):
        v6.expanded_tail_robust_goal_balanced_interaction_candidate_banks(
            states, np.asarray([.8, .8]), proposal_seed=211, supplemental_seed=212,
            proposal_count=95, simulator=fake_simulator,
        )
    with pytest.raises(ValueError, match="requires shared_candidate_in_support"):
        v6.build_panel(
            Path("unused"), [], Path("unused"), encoder_checkpoint=Path("unused"),
            pca_checkpoint=Path("unused"), dinov2_source=Path("unused"),
            candidate_design="robust_goal_balanced_interaction_pairs", candidate_generalization_regime="numeric_ood",
        )


def test_v6_root_split_excludes_all_roots_from_each_supplied_context(tmp_path: Path) -> None:
    first = _prior_context(tmp_path / "first.json", [1, 2, 3], "first")
    second = _prior_context(tmp_path / "second.json", [7, 8], "second")
    roots, hashes, templates, checksums = v6._all_prior_identities([first, second])
    assert roots == {1, 2, 3, 7, 8}
    assert len(hashes) == len(templates) == 5
    assert set(checksums) == {str(first.resolve()), str(second.resolve())}
    split = v6._split_roots_v6(seed=v6.ROOT_SEED, forbidden=roots)
    assert not set(split) & roots
    assert {name: sum(value == name for value in split.values()) for name in v6.SPLIT_COUNTS} == v6.SPLIT_COUNTS
    assert max(split) < v6.STATE_ACTION_ROOT_COUNT


def test_v6_root_range_constrains_both_evaluation_and_interaction_design_roots() -> None:
    allowed = range(200, 520)
    split = v6._split_roots_v6(
        seed=13,
        forbidden={201, 202},
        root_count=600,
        root_id_min=min(allowed),
        root_id_max_exclusive=max(allowed) + 1,
    )
    design = v6._interaction_design_roots(
        seed=13,
        forbidden={201, 202},
        evaluation_roots=set(split),
        root_count=600,
        root_id_min=min(allowed),
        root_id_max_exclusive=max(allowed) + 1,
    )
    assert set(split).issubset(allowed)
    assert set(design).issubset(allowed)
    assert not set(split) & set(design)


def test_v6_root_range_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="root range"):
        v6._eligible_roots(root_count=10, root_id_min=8, root_id_max_exclusive=8)


def test_v6_interaction_design_excludes_renderer_chart_roots(tmp_path: Path) -> None:
    forbidden = {1, 2, 3}
    evaluation = set(range(10, 154))
    chart = set(range(1800, 1840))
    design = v6._interaction_design_roots(seed=17, forbidden=forbidden | chart, evaluation_roots=evaluation)
    assert len(design) == v6.INTERACTION_DESIGN_ROOTS
    assert not set(design) & forbidden
    assert not set(design) & evaluation
    assert not set(design) & chart


def test_v6_renderer_chart_excludes_only_v6_roots_and_records_historical_overlap(tmp_path: Path) -> None:
    obses = tmp_path / "obses"
    obses.mkdir()
    for episode in range(1800, 1900):
        (obses / f"episode_{episode:03d}.pth").write_bytes(b"stored-rgb")
    v6_roots = set(range(1800, 1810))
    historical_roots = set(range(1800, 1900))
    chart = v6._renderer_chart_roots_v6(tmp_path, v6_roots)
    provenance = v6._renderer_chart_provenance(chart, v6_roots, historical_roots)
    assert len(chart) == 40
    assert not set(chart) & v6_roots
    assert provenance["historical_context_root_overlap"] == chart
    assert provenance["historical_context_root_overlap_count"] == 40
    assert "simulator-state regression targets" in provenance["role"]
    assert "never V6 candidate supervision, calibration labels, or outcome labels" in provenance["role"]
    assert provenance["uses_simulator_state_targets_for_renderer_calibration"] is True
    assert provenance["v6_outcome_labels_materialized_from_chart"] is False


def test_v6_history_is_eight_frames_and_strictly_pre_candidate() -> None:
    actions = np.zeros((48, 2), dtype=np.float64)
    actions[5:40] = 1.0
    actions[40:] = 9.0
    frames, history = v6.causal_history8(40, actions)
    assert frames == [5, 10, 15, 20, 25, 30, 35, 40]
    assert history.shape == (35, 2)
    assert float(history.max()) == 1.0


def test_v6_artifact_writer_never_materializes_outcome_labels(tmp_path: Path) -> None:
    paths = v6._write_panel_artifacts(
        tmp_path,
        source_records=[{"instance_id": "train-root"}],
        source_arrays={"terminal/v6/train-root/0": np.asarray([1.0, 2.0])},
        context_records=[{"instance_id": "outcome-root", "outcome_label_status": "not_materialized_during_build"}],
        calibration_labels={"calibration-root/0": {"real_task_cost": 0.4}},
        context_metadata={"schema": v6.PANEL_SCHEMA, "candidate_family": v6.V6_ACTION_FAMILY},
        manifest_base={"support_distance_diagnostics": {}},
    )
    assert paths["source_dataset"].exists()
    assert paths["source_arrays"].exists()
    assert paths["context"].exists()
    assert paths["calibration_outcomes"].exists()
    assert not paths["outcome_labels_reserved"].exists()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["artifacts"]["outcome_labels"] == {
        "path": str(paths["outcome_labels_reserved"].resolve()), "materialized": False, "sha256": None,
    }
    paths["outcome_labels_reserved"].write_text("must not be read or overwritten")
    with pytest.raises(RuntimeError, match="outcome-label"):
        v6._write_panel_artifacts(
            tmp_path,
            source_records=[], source_arrays={}, context_records=[], calibration_labels={},
            context_metadata={"schema": v6.PANEL_SCHEMA, "candidate_family": v6.V6_ACTION_FAMILY},
            manifest_base={},
        )


def test_v6_artifact_writer_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="non-empty V6 output"):
        v6._write_panel_artifacts(
            tmp_path,
            source_records=[], source_arrays={}, context_records=[], calibration_labels={},
            context_metadata={"schema": v6.PANEL_SCHEMA, "candidate_family": v6.V6_ACTION_FAMILY},
            manifest_base={},
        )


def test_declared_enlarged_calibration_panel_preserves_split_contract() -> None:
    counts = v6._validated_split_counts({"train": 128, "calibration": 160, "outcome": 64})
    split = v6._split_roots_v6(seed=7, forbidden=set(), root_count=512, split_counts=counts)
    assert {name: sum(value == name for value in split.values()) for name in counts} == counts


def test_declared_calibration_panel_requires_even_authorization_pool() -> None:
    with pytest.raises(ValueError, match="even"):
        v6._validated_split_counts({"train": 64, "calibration": 49, "outcome": 32})
