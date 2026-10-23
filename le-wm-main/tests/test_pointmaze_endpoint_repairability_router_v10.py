import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_endpoint_repairability_router_v10 import (
    EXTENSION_SEED,
    ROOT_RANGE,
    SPLIT_COUNTS,
    _design_quality,
    _extension_contract,
)
from experiments.gdd_wm.run_pointmaze_endpoint_repairability_router_v10 import (
    _apply_threshold,
    _calibrate,
    _calibration_role,
    _summary,
)
from experiments.gdd_wm import run_pointmaze_endpoint_repairability_router_v10 as v10


def _row(*, eligible: bool, endpoint_better: bool, score: float) -> dict:
    cost = 1.0 if endpoint_better else 0.0
    endpoint = 0.0 if endpoint_better else 1.0
    return {
        "routing_eligible": eligible,
        "candidate_disagreement": eligible,
        "router_score_pre_outcome": score,
        "cost_candidate_id": "cost", "endpoint_candidate_id": "endpoint",
        "routed_candidate_id": "cost", "released_endpoint": False,
        "cost_regret": cost, "endpoint_regret": endpoint, "routed_regret": cost,
        "endpoint_beats_cost": endpoint_better,
    }


def test_v10_extension_contract_requires_a_new_independent_root_range(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1", "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0],
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
        "base_root_count": ROOT_RANGE[0], "extended_root_count": ROOT_RANGE[1],
    }), encoding="utf-8")
    assert all(_extension_contract(tmp_path)["checks"].values())
    assert SPLIT_COUNTS == {"train": 192, "calibration": 128, "outcome": 64}


def test_v10_robust_design_quality_reads_aggregate_contract() -> None:
    quality = _design_quality({"design_all_root_headroom": {
        "winner_entropy_normalized": 0.6,
        "best_second_gap_q10": 0.002,
        "best_second_gap_ge_0p003_fraction": 0.7,
        "candidate_cost_spread_q10": 0.1,
    }})
    assert quality["winner_entropy_normalized"] == 0.6
    assert quality["best_second_gap_q10"] == 0.002
    with pytest.raises(ValueError, match="incomplete"):
        _design_quality({"design_all_root_headroom": {"winner_entropy_normalized": 0.6}})


def test_v10_extension_contract_rejects_reusing_v9_roots(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1", "seed": EXTENSION_SEED,
        "new_episode_count": 512, "new_root_id_range": [11320, 11831],
        "base_root_count": 11320, "extended_root_count": 11832,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="violates frozen contract"):
        _extension_contract(tmp_path)


def test_v10_threshold_never_releases_same_choice_roots() -> None:
    rows = {
        "same": _row(eligible=False, endpoint_better=True, score=99.0),
        "different": _row(eligible=True, endpoint_better=True, score=1.0),
    }
    routed = _apply_threshold(rows, threshold=0.0)
    assert routed["same"]["released_endpoint"] is False
    assert routed["same"]["routed_candidate_id"] == "cost"
    assert routed["different"]["released_endpoint"] is True
    assert _summary(routed)["endpoint_release_count"] == 1


def test_v10_calibration_requires_opportunity_support_in_both_roles() -> None:
    rows, count, index = {}, {0: 0, 1: 0}, 0
    while min(count.values()) < 9:
        root = f"root-{index}"
        role = _calibration_role(root)
        if count[role] < 9:
            rows[root] = _row(eligible=True, endpoint_better=True, score=float(index))
            count[role] += 1
        index += 1
    result = _calibrate(rows, seed=17)
    assert result["routing_authorized"] is True
    assert result["deployment_policy"] == "routed"
    sparse = {root: _row(eligible=False, endpoint_better=True, score=1.0) for root in rows}
    with pytest.raises(ValueError, match="opportunity count"):
        _calibrate(sparse, seed=17)


def test_v10_static_ledger_binds_endpoint_arm_and_rejects_its_removal(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = dataset / "states.pth"
    states.write_bytes(b"v10-state-pool")
    context = tmp_path / "prior.json"
    context.write_text(json.dumps({"metadata": {"candidate_family": "prior"}, "records": [{
        "start_episode_id": "1", "candidate_action_hash": "prior-action",
        "candidate_template_fingerprint": "prior-template", "action_sequence": np.zeros((20, 2)).tolist(),
    }]}), encoding="utf-8")
    _roots, _hashes, _templates, context_sha256 = panel._all_prior_identities([context])
    construction = {
        "root_seed": 202610971, "held_action_seed": 202610972, "train_action_seed": 202610973,
        "root_id_min": 16312, "root_id_max_exclusive": 16824,
    }
    bank = {
        "candidate_design": "expanded_tail_robust_goal_balanced_interaction_pairs", "goal_qpos": [0.8, 0.8],
        "held_shape": [16, 20, 2], "train_shape": [64, 20, 2],
        "shared_candidate_in_support": True, "held_action_hash_count": 16,
        "held_train_action_hash_overlap_count": 16, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "scorer_training_action_hash_count": 16, "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
        "design_best_second_cost_gap_q10": 0.001, "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
        "design_candidate_cost_spread_q10": 0.05,
        "router_feature_contract": "opportunity_compact_v1_predicted_score_action_delta",
        "router_feature_dimension": 15, "router_train_root_count": 192,
        "minimum_router_train_root_surplus": 16, "router_kind": "advantage_ridge",
        "router_action_representation": "full_ordered_20x2",
        "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        "router_training_scope": "scorer_disagreement_only",
        "opportunity_gate": "cost_endpoint_candidate_disagreement",
        "minimum_router_opportunity_root_count": 31,
        "minimum_calibration_opportunity_roots_per_role": 8,
        "repair_arm": "endpoint_coupled_counterfactual",
    }
    ledger = tmp_path / "v10-preflight.json"
    split_roots = {
        "train": list(range(16312, 16504)),
        "calibration": list(range(16504, 16632)),
        "outcome": list(range(16632, 16696)),
    }

    def write() -> None:
        ledger.write_text(json.dumps({
            "schema": panel.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
            "status": "passed_static_preflight", "checks": {"all": True},
            "prior_context_sha256": context_sha256, "dataset_states_sha256": panel._sha256(states),
            "root_counts": SPLIT_COUNTS, "split_roots": split_roots, "construction": construction,
            "post_execution_aliasing_contract": panel.V7_POST_EXECUTION_ALIASING_CONTRACT,
            "candidate_bank": bank,
        }), encoding="utf-8")

    write()
    receipt = panel.frozen_preflight_provenance(
        ledger, dataset_root=dataset, prior_context_sha256=context_sha256, split_counts=SPLIT_COUNTS,
        candidate_design="expanded_tail_robust_goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
    )
    assert receipt and receipt["schema"] == panel.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA
    bank.pop("repair_arm")
    write()
    with pytest.raises(ValueError, match="endpoint-coupled repair-arm"):
        panel.frozen_preflight_provenance(
            ledger, dataset_root=dataset, prior_context_sha256=context_sha256, split_counts=SPLIT_COUNTS,
        candidate_design="expanded_tail_robust_goal_balanced_interaction_pairs", goal_qpos=np.asarray([0.8, 0.8]), construction=construction,
        )


def test_v10_runner_binds_repair_routing_before_outcome_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the execution boundary without treating this fixture as CUDA evidence."""

    root_ids: dict[str, int] = {}

    def row(root: str, split: str, candidate: int) -> dict:
        root_id = root_ids.setdefault(root, len(root_ids) + 1)
        action = np.zeros((20, 2), dtype=np.float64)
        action[:, 0] = -0.2 if candidate == 0 else 0.2
        return {
            "instance_id": root, "start_episode_id": str(root_id), "split": split, "candidate_id": f"candidate-{candidate}",
            "candidate_action_hash": panel.sequence_hash(action), "source_history_model": [0.1, 0.2, 0.3, 0.4],
            "source_latent_model": [0.3, 0.4], "source_state_raw": [0.0, 0.0, 0.0, 0.0],
            "goal_raw": [0.8, 0.8], "action_sequence": action.tolist(), "terminal_ref": f"{root}/{candidate}",
        }

    role_counts, calibration_roots, index = {0: 0, 1: 0}, [], 0
    while min(role_counts.values()) < 9:
        root = f"calibration-{index}"
        role = _calibration_role(root)
        if role_counts[role] < 9:
            calibration_roots.append(root)
            role_counts[role] += 1
        index += 1
    # The fixture still satisfies the real 15D + 16-root static capacity
    # condition even though model fitting itself is monkeypatched.
    train_roots, outcome_roots = [f"train-{index}" for index in range(31)], ["outcome-0", "outcome-1"]
    source_rows = [row(root, "train", candidate) for root in train_roots for candidate in range(2)]
    context_rows = [
        *(row(root, "calibration", candidate) for root in calibration_roots for candidate in range(2)),
        *(row(root, "outcome", candidate) for root in outcome_roots for candidate in range(2)),
    ]
    ledger = tmp_path / "preflight.json"
    bank = {
        "shared_candidate_in_support": True, "held_action_hash_count": 2,
        "held_train_action_hash_overlap_count": 2, "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "scorer_training_action_hash_count": 2,
        "router_feature_contract": "opportunity_compact_v1_predicted_score_action_delta",
        "router_feature_dimension": 15, "router_train_root_count": len(train_roots),
        "minimum_router_train_root_surplus": 16,
        "router_kind": "advantage_ridge", "router_action_representation": "full_ordered_20x2",
        "router_training_scope": "scorer_disagreement_only", "opportunity_gate": "cost_endpoint_candidate_disagreement",
        "minimum_router_opportunity_root_count": 31, "minimum_calibration_opportunity_roots_per_role": 8,
        "repair_arm": "endpoint_coupled_counterfactual",
    }
    ledger.write_text(json.dumps({
        "schema": panel.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
        "status": "passed_static_preflight", "candidate_bank": bank,
        "root_counts": {"train": len(train_roots), "calibration": len(calibration_roots), "outcome": len(outcome_roots)},
        "split_roots": {
            "train": [root_ids[root] for root in train_roots],
            "calibration": [root_ids[root] for root in calibration_roots],
            "outcome": [root_ids[root] for root in outcome_roots],
        },
        "post_execution_aliasing_contract": panel.V7_POST_EXECUTION_ALIASING_CONTRACT,
    }), encoding="utf-8")
    ledger_sha256 = panel._sha256(ledger)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": source_rows}), encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"metadata": {
        "candidate_generalization_regime": "shared_candidate_in_support",
        "candidate_design": "goal_balanced_interaction_pairs", "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "frozen_preflight_ledger": {
            "schema": panel.ENDPOINT_REPAIRABILITY_ROUTER_V10_PREFLIGHT_SCHEMA,
            "path": str(ledger), "sha256": ledger_sha256,
        },
    }, "records": context_rows}), encoding="utf-8")
    arrays = tmp_path / "arrays.npz"
    np.savez(arrays, ignored=np.zeros(1))
    labels = tmp_path / "calibration.json"
    labels.write_text("{}", encoding="utf-8")
    order: list[str] = []

    def cost_predictor(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.tile(np.asarray([0.0, 1.0]), len(feature) // 2), np.zeros((len(feature), 4))

    def endpoint_predictor(feature: np.ndarray, _goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.tile(np.asarray([1.0, 0.0]), len(feature) // 2), np.zeros((len(feature), 4))

    def fake_models(*_args, **_kwargs):
        return {"arm": "cost"}, cost_predictor, {"arm": "endpoint"}, endpoint_predictor

    def fake_crossfit(*_args, **_kwargs):
        order.append("crossfit")
        return (lambda values: np.ones(len(values))), {"selection_protocol": "fixture"}, {}, {"root": 1.0}, {"root": True}

    monkeypatch.setattr(v10, "require_cuda_allocation", lambda _device: {"status": "fixture-cuda"})
    monkeypatch.setattr(v10, "_terminal_from_ref", lambda _row, _arrays: np.zeros(4))
    monkeypatch.setattr(v10, "_fit_models", fake_models)
    monkeypatch.setattr(v10, "_crossfit_router", fake_crossfit)
    monkeypatch.setattr(v10, "_calibration_terminal", lambda rows, _labels: np.zeros((len(rows), 4)))
    monkeypatch.setattr(v10, "_fresh_terminal", lambda rows: (order.append("outcome") or np.zeros((len(rows), 4))))
    monkeypatch.setattr(v10, "_root_costs", lambda rows, _terminal: {
        root: np.asarray([1.0, 0.0]) for root in {str(row["instance_id"]) for row in rows}
    })
    report = v10.run(
        source_dataset=source, source_arrays=arrays, context=context, calibration_labels=labels,
        output=tmp_path / "output", device="cuda", ensemble_size=1,
    )
    assert report["calibration"]["routing"]["deployment_policy"] == "routed"
    assert report["outcome"]["authorized_deployment"]["endpoint_release_count"] == len(outcome_roots)
    assert order == ["crossfit", "outcome"]
    assert report["cuda_preflight"]["status"] == "fixture-cuda"
    # Insufficient cross-fit opportunities stop before a final scorer,
    # calibration-label read, or sealed outcome execution.
    def insufficient_crossfit(*_args, **_kwargs):
        raise v10._CrossfitOpportunityError(
            train_root_count=len(train_roots), opportunity_count=30,
            minimum_opportunity_count=31, mean_cost_minus_endpoint_regret=0.0,
        )

    order.clear()
    monkeypatch.setattr(v10, "_crossfit_router", insufficient_crossfit)
    sealed_labels = tmp_path / "unreadable-until-crossfit-passes.json"
    sealed_labels.write_text("this is deliberately not JSON", encoding="utf-8")
    train_rejected = v10.run(
        source_dataset=source, source_arrays=arrays, context=context, calibration_labels=sealed_labels,
        output=tmp_path / "train-rejected-output", device="cuda", ensemble_size=1,
    )
    assert train_rejected["status"] == "rejected_pre_calibration_crossfit_opportunity"
    assert train_rejected["calibration"]["materialized"] is False
    assert train_rejected["outcome"]["materialized"] is False
    assert order == []
    monkeypatch.setattr(v10, "_crossfit_router", fake_crossfit)
    # A valid calibration opportunity failure must preserve the sealed outcome
    # instead of opening it merely to explain an under-powered router.
    def same_endpoint_predictor(feature: np.ndarray, _goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cost_predictor(feature)

    def same_models(*_args, **_kwargs):
        return {"arm": "cost"}, cost_predictor, {"arm": "endpoint"}, same_endpoint_predictor

    order.clear()
    monkeypatch.setattr(v10, "_fit_models", same_models)
    rejected = v10.run(
        source_dataset=source, source_arrays=arrays, context=context, calibration_labels=labels,
        output=tmp_path / "rejected-output", device="cuda", ensemble_size=1,
    )
    assert rejected["status"] == "rejected_pre_outcome_calibration_opportunity"
    assert rejected["outcome"]["materialized"] is False
    assert order == ["crossfit"]
    tampered = json.loads(context.read_text(encoding="utf-8"))
    original_action_value = tampered["records"][0]["action_sequence"][0][0]
    tampered["records"][0]["action_sequence"][0][0] = 0.99
    context.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_action_hash does not match"):
        v10._validate_v10_contract(v10._load(source), v10._load(context))
    tampered["records"][0]["action_sequence"][0][0] = original_action_value
    context.write_text(json.dumps(tampered), encoding="utf-8")
    original_root = tampered["records"][0]["start_episode_id"]
    tampered["records"][0]["start_episode_id"] = "99999"
    context.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="split-root identities differ"):
        v10._validate_v10_contract(v10._load(source), v10._load(context))
    tampered["records"][0]["start_episode_id"] = original_root
    context.write_text(json.dumps(tampered), encoding="utf-8")
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        v10._validate_v10_contract(v10._load(source), v10._load(context))


def test_v10_launcher_is_gpu_only_and_executes_an_endpoint_repair() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_endpoint_repairability_router_v10_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "point_maze_v41_endpoint_repairability_r4_extension_20261022" in text
    assert "point_maze_v42_endpoint_repairability_r5_extension_20261022" in text
    assert "--root-id-min 16312 --root-id-max-exclusive 16824" in text
    assert "expanded_tail_robust_goal_balanced_interaction_pairs" in text
    assert "--train-root-count 192 --calibration-root-count 128 --outcome-root-count 64" in text
    assert "--recurrent-smoke" in text and "refusing CPU fallback" in text
    assert "stopped before outcome materialization" in text
    assert "run_pointmaze_endpoint_repairability_router_v10" in text
    assert text.index("--recurrent-smoke") < text.index('--input-root "${BASE_POOL}"')
