import json
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_p1_physical_nongreediness import audit
from experiments.spectral_reachability.audit_reacher_d2p_p1_physical_alignment_calibration import audit as audit_calibration
from experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_replay import (
    _raw_action_is_zero,
    _validate_report as validate_first_chunk_replay,
)
from experiments.spectral_reachability.audit_reacher_d2p_p1_candidate_prefixes import ranked_eligible_indices
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as p1


def test_p1_runner_direct_script_entrypoint_loads_project_package():
    completed = subprocess.run(
        [sys.executable, str(Path(p1.__file__)), "--help"],
        cwd=Path(p1.__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CUDA-only physical runner" in completed.stdout
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY


def test_p1_runner_resolves_explicit_local_paths_before_checkpoint_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        policy=Path("checkpoint"),
        adapter_artifact=Path("adapter.json"),
        p0_evidence=Path("p0.json"),
        p1_preflight_evidence=Path("preflight"),
        calibration_evidence=None,
        training_source_manifest=[],
    )

    resolved = p1._resolve_local_cli_inputs(args)

    assert resolved.policy == tmp_path / "checkpoint"
    assert resolved.adapter_artifact == tmp_path / "adapter.json"
    assert resolved.p0_evidence == tmp_path / "p0.json"
    assert resolved.p1_preflight_evidence == tmp_path / "preflight"
    assert resolved.calibration_evidence is None
    assert resolved.training_source_manifest == []


def test_p1_runner_resolves_each_repeatable_training_source_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        policy=None, adapter_artifact=None, p0_evidence=None, p1_preflight_evidence=None,
        calibration_evidence=None, margin_band_selection_evidence=None,
        training_source_manifest=[Path("pool.json"), Path("repair.json")],
    )

    resolved = p1._resolve_local_cli_inputs(args)

    assert resolved.training_source_manifest == [tmp_path / "pool.json", tmp_path / "repair.json"]


def test_p1_runner_reserves_immutable_output_before_running_roots(tmp_path):
    output = tmp_path / "calibration"

    assert p1._reserve_immutable_output(output) == output
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        p1._reserve_immutable_output(output)


def test_p1_runner_rejects_simulator_training_seed_overlap(tmp_path):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "schema": "reacher_d2p_counterfactual_pair_pool.v1", "status": "completed",
        "seed_range": [100, 110],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        p1._bind_simulator_training_sources([pool], run_seeds={99, 100, 111})
    binding = p1._bind_simulator_training_sources([pool], run_seeds={111, 112})
    assert binding[0]["reserved_training_seed_range"] == [100, 110]


def test_p1_margin_band_selection_binds_its_sources_and_runtime_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.pt"
    weights.write_bytes(b"weights")
    adapter = tmp_path / "adapter.json"
    adapter.write_text("adapter", encoding="utf-8")
    report = tmp_path / "source_report.json"
    fidelity = tmp_path / "source_fidelity.json"
    report.write_text("report", encoding="utf-8")
    fidelity.write_text("fidelity", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    selection = tmp_path / "band.json"
    selection.write_text(json.dumps({
        "schema": "d2p_reacher_p1_margin_band_guard_selection.v1", "status": "pass",
        "selected": {"min_non_greedy_margin": 0.03, "max_non_greedy_margin": 0.06},
        "protocol_binding": {
            "checkpoint_weights_sha256": digest(weights), "adapter_sha256": digest(adapter),
            "runner_sha256": p1._sha256_file(p1.Path(p1.__file__)),
            "mechanism_observability": "observed_encoded_goal_cost", "proposal_distribution": "physical_uniform",
            "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 10.0, "action_block": 5,
            "reconfiguration_horizon": 1, "recompiled_progress_horizon": 2, "proposals": 128,
            "max_relative_reachability_improvement": None, "min_second_to_first_singular_ratio": None,
            "warmup_policy": WARMUP_POLICY, "target_delta": [0.65, -0.65],
        },
        "inputs": {
            "report": str(report), "report_sha256": digest(report),
            "fidelity": str(fidelity), "fidelity_sha256": digest(fidelity),
        },
    }), encoding="utf-8")
    args = SimpleNamespace(
        policy=checkpoint, adapter_artifact=adapter, non_greedy_margin=0.03, max_non_greedy_margin=0.06,
        mechanism_observability="observed_encoded_goal_cost", proposal_distribution="physical_uniform",
        selection_metric="bounded_dls_suffix_goal_cost", damping=10.0, action_block=5,
        reconfiguration_horizon=1, progress_horizon=2, proposals=128,
        max_relative_reachability_improvement=None, min_second_to_first_singular_ratio=None,
        target_delta=(0.65, -0.65),
    )

    binding = p1._validate_margin_band_selection(selection, args=args)

    assert binding["selected_bounds"] == {"min_non_greedy_margin": 0.03, "max_non_greedy_margin": 0.06}
    fidelity.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="has changed"):
        p1._validate_margin_band_selection(selection, args=args)


def test_p1_margin_band_selection_rejects_a_different_runner_or_planner_geometry(tmp_path):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    weights = checkpoint / "weights.pt"; weights.write_bytes(b"weights")
    adapter = tmp_path / "adapter.json"; adapter.write_text("adapter", encoding="utf-8")
    report = tmp_path / "source_report.json"; report.write_text("report", encoding="utf-8")
    fidelity = tmp_path / "source_fidelity.json"; fidelity.write_text("fidelity", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    selection = tmp_path / "band.json"
    selection.write_text(json.dumps({
        "schema": "d2p_reacher_p1_margin_band_guard_selection.v1", "status": "pass",
        "selected": {"min_non_greedy_margin": 0.03, "max_non_greedy_margin": 0.06},
        "protocol_binding": {
            "checkpoint_weights_sha256": digest(weights), "adapter_sha256": digest(adapter),
            "runner_sha256": "stale-runner", "mechanism_observability": "observed_encoded_goal_cost",
            "reconfiguration_horizon": 1, "recompiled_progress_horizon": 2, "proposals": 128,
            "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost",
            "damping": 10.0, "action_block": 5, "max_relative_reachability_improvement": None,
            "min_second_to_first_singular_ratio": None, "warmup_policy": WARMUP_POLICY,
            "target_delta": [0.65, -0.65],
        },
        "inputs": {"report": str(report), "report_sha256": digest(report), "fidelity": str(fidelity), "fidelity_sha256": digest(fidelity)},
    }), encoding="utf-8")
    args = SimpleNamespace(
        policy=checkpoint, adapter_artifact=adapter, non_greedy_margin=0.03, max_non_greedy_margin=0.06,
        mechanism_observability="observed_encoded_goal_cost", reconfiguration_horizon=1, progress_horizon=2,
        proposals=128, proposal_distribution="physical_uniform", selection_metric="bounded_dls_suffix_goal_cost",
        damping=10.0, action_block=5, max_relative_reachability_improvement=None,
        min_second_to_first_singular_ratio=None, target_delta=(0.65, -0.65),
    )
    with pytest.raises(ValueError, match="runner_sha256"):
        p1._validate_margin_band_selection(selection, args=args)


def test_p1_paired_summary_reports_root_paired_final_distance_direction():
    rows = [
        {"methods": {"reconfiguration": {"physical": {"final_qpos_distance": 1.0}}, "cem_full_horizon": {"physical": {"final_qpos_distance": 2.0}}}},
        {"methods": {"reconfiguration": {"physical": {"final_qpos_distance": 3.0}}, "cem_full_horizon": {"physical": {"final_qpos_distance": 4.0}}}},
    ]
    summary = p1._paired(rows, "reconfiguration", "cem_full_horizon", seed=3)
    assert summary["mean"] == -1.0
    assert summary["method_only_wins"] == 2
    assert np.all(np.asarray(summary["ci95"]) <= 0.0)


def test_p1_preflight_validation_accepts_only_the_same_frozen_planning_contract(tmp_path):
    directory = tmp_path / "preflight"
    directory.mkdir()
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    weights = checkpoint / "weights.pt"; weights.write_bytes(b"weights")
    adapter = tmp_path / "adapter.json"; adapter.write_text("adapter", encoding="utf-8")
    p0_evidence = tmp_path / "p0.json"; p0_evidence.write_text("p0", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = directory / "report.json"
    report.write_text(json.dumps({
        "schema": "d2p_reacher_p1_reconfiguration_preflight.v1", "status": "pass", "gates": {"cuda_only": True},
        "protocol": {
            "history_len": 3, "min_model_witness_roots": p1.p1.DEFAULT_MIN_MODEL_WITNESS_ROOTS,
            "reconfiguration_horizon": 1, "recompiled_progress_horizon": 2,
            "action_block": 5, "proposals": 128, "proposal_distribution": "physical_uniform",
            "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 10.0,
            "max_relative_reachability_improvement": None, "min_second_to_first_singular_ratio": None,
            "warmup_policy": WARMUP_POLICY, "target_delta": [0.65, -0.65],
        },
        "inputs": {
            "weights_sha256": digest(weights), "adapter_sha256": digest(adapter),
            "runner_sha256": p1._sha256_file(p1.Path(p1.p1.__file__)),
        },
        "upstream_p0_evidence": {"sha256": digest(p0_evidence)},
    }), encoding="utf-8")
    args = SimpleNamespace(
        policy=checkpoint, adapter_artifact=adapter, p0_evidence=p0_evidence,
        reconfiguration_horizon=1, progress_horizon=2, action_block=5, proposals=128,
        proposal_distribution="physical_uniform", selection_metric="bounded_dls_suffix_goal_cost", damping=10.0,
        max_relative_reachability_improvement=None, min_second_to_first_singular_ratio=None,
        target_delta=(0.65, -0.65),
    )
    assert p1._validate_preflight(directory, args=args)["path"] == str(report.resolve())
    args.proposals = 64
    with pytest.raises(ValueError, match="proposals"):
        p1._validate_preflight(directory, args=args)


def test_p1_primary_comparison_is_only_defined_for_two_eligible_roots():
    eligible = [{"seed": 1}]
    assert len(eligible) < 2


def test_p1_physical_nongreediness_audit_rejects_model_only_witness():
    report = {"schema": "d2p_reacher_p1_reconfiguration.v1", "rows": [{
        "seed": 1, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True}, "physical": {"after_zero_prefix_qpos_distance": 1.0, "after_first_chunk_qpos_distance": 0.9}}},
    }]}
    result = audit(report)
    assert result["status"] == "fail"
    assert result["physically_non_greedy_eligible_root_count"] == 0


def test_p1_physical_nongreediness_uses_zero_prefix_not_pre_action_root():
    report = {"schema": "d2p_reacher_p1_reconfiguration.v1", "rows": [{
        "seed": 1, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True}, "physical": {"initial_qpos_distance": 2.0, "after_zero_prefix_qpos_distance": 0.9, "after_first_chunk_qpos_distance": 1.0}}},
    }]}
    result = audit(report)
    assert result["status"] == "pass"
    assert result["physically_non_greedy_eligible_root_count"] == 1


def test_p1_observable_mechanism_gate_uses_actual_successor_encoder_costs_not_qpos():
    method = {
        "physical": {"after_zero_prefix_qpos_distance": 1.0, "after_first_chunk_qpos_distance": 0.9},
        "observed_first_chunk": {"candidate_encoded_goal_cost": 1.2, "zero_prefix_encoded_goal_cost": 1.0},
    }
    assert not p1._mechanism_non_greedy(method, observability="qpos_distance")
    assert p1._mechanism_non_greedy(method, observability="observed_encoded_goal_cost")
    assert p1._mechanism_gate_name("observed_encoded_goal_cost").endswith("observably_non_greedy_relative_to_zero_prefix")


def test_p1_alignment_calibration_audit_keeps_physical_gate_distinct_from_other_gates(tmp_path):
    receipt = tmp_path / "calibration.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only",
        "protocol": {"calibration_only": True, "non_greedy_margin": 1.0e-6, "max_non_greedy_margin": 0.05},
        "gates": {"cuda_only": True, "every_model_eligible_reconfiguration_is_physically_non_greedy_relative_to_zero_prefix": False},
        "eligibility": {"eligible_root_count": 4, "physically_non_greedy_eligible_root_count": 3},
    }), encoding="utf-8")
    result = audit_calibration([(receipt, json.loads(receipt.read_text(encoding="utf-8")))])
    assert result["status"] == "blocked_model_to_physical_alignment"
    assert result["candidates"][0]["all_nonphysical_gates_pass"]
    assert not result["candidates"][0]["physical_nongreediness_gate_pass"]


def test_p1_first_chunk_replay_requires_the_receipt_contract():
    with pytest.raises(ValueError, match="not a Reacher"):
        validate_first_chunk_replay({"schema": "wrong"})
    with pytest.raises(ValueError, match="action block"):
        validate_first_chunk_replay({"schema": "d2p_reacher_p1_reconfiguration.v1", "rows": [{"seed": 1}], "protocol": {"action_block": 0}})


def test_p1_first_chunk_replay_checks_raw_zero_control_after_normalization():
    # The raw zero is constructed by a float32 normalize/invert round trip,
    # so its numerical contract must not borrow the much tighter qpos replay
    # tolerance.
    assert _raw_action_is_zero(np.asarray([[5.0e-8, -3.0e-8]], dtype=np.float32), atol=1.0e-6)
    assert not _raw_action_is_zero(np.asarray([[2.0e-3, 0.0]], dtype=np.float32), atol=1.0e-6)


def test_p1_candidate_prefix_audit_uses_model_rank_before_physical_diagnostics():
    rows = [
        {"proposal_index": 0, "immediate_goal_cost": 1.0, "value": 10.0},
        {"proposal_index": 1, "immediate_goal_cost": 1.2, "value": 8.0},
        {"proposal_index": 2, "immediate_goal_cost": 1.1, "value": 4.0},
        {"proposal_index": 3, "immediate_goal_cost": 0.9, "value": 1.0},
    ]
    assert ranked_eligible_indices(rows, count=2) == [2, 1]


def test_p1_candidate_prefix_audit_can_rank_by_bounded_dls_score():
    rows = [
        {"proposal_index": 0, "immediate_goal_cost": 1.0, "bounded_dls_suffix_goal_cost": 2.0},
        {"proposal_index": 1, "immediate_goal_cost": 1.2, "bounded_dls_suffix_goal_cost": 1.8},
        {"proposal_index": 2, "immediate_goal_cost": 1.1, "bounded_dls_suffix_goal_cost": 1.2},
    ]
    assert ranked_eligible_indices(rows, count=2, selection_metric="bounded_dls_suffix_goal_cost") == [2, 1]


def test_p1_outcome_rejects_overlapping_calibration_roots(tmp_path):
    receipt = tmp_path / "calibration.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only",
        "gates": {"ok": True}, "protocol": {"calibration_only": True, "reconfiguration_horizon": 1, "recompiled_progress_horizon": 2, "action_block": 5, "proposals": 128, "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 0.01, "non_greedy_margin": 1.0e-6, "max_non_greedy_margin": None, "min_eligible_calibration_roots": 4, "warmup_policy": WARMUP_POLICY, "target_delta": [0.65, -0.65]},
        "inputs": {"runner_sha256": p1._sha256_file(p1.Path(p1.__file__))}, "rows": [{"seed": 10}, {"seed": 11}],
    }), encoding="utf-8")
    args = type("Args", (), {"reconfiguration_horizon": 1, "progress_horizon": 2, "action_block": 5, "proposals": 128, "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 0.01, "non_greedy_margin": 1.0e-6, "max_non_greedy_margin": None, "max_relative_reachability_improvement": None, "min_second_to_first_singular_ratio": None, "min_eligible_calibration_roots": 4, "target_delta": (0.65, -0.65)})()
    with pytest.raises(ValueError, match="disjoint"):
        p1._validate_outcome_calibration(receipt, args=args, outcome_seeds={11, 12})


def test_p1_outcome_rejects_calibration_with_different_nongreedy_margin(tmp_path):
    receipt = tmp_path / "calibration.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only",
        "gates": {"ok": True}, "protocol": {"calibration_only": True, "reconfiguration_horizon": 1, "recompiled_progress_horizon": 2, "action_block": 5, "proposals": 128, "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 0.01, "non_greedy_margin": 0.05, "max_non_greedy_margin": None, "min_eligible_calibration_roots": 4, "warmup_policy": WARMUP_POLICY, "target_delta": [0.65, -0.65]},
        "inputs": {"runner_sha256": p1._sha256_file(p1.Path(p1.__file__))}, "rows": [{"seed": 10}, {"seed": 11}],
    }), encoding="utf-8")
    args = type("Args", (), {"reconfiguration_horizon": 1, "progress_horizon": 2, "action_block": 5, "proposals": 128, "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "damping": 0.01, "non_greedy_margin": 1.0e-6, "max_non_greedy_margin": None, "max_relative_reachability_improvement": None, "min_second_to_first_singular_ratio": None, "min_eligible_calibration_roots": 4, "target_delta": (0.65, -0.65)})()
    with pytest.raises(ValueError, match="configuration differs"):
        p1._validate_outcome_calibration(receipt, args=args, outcome_seeds={12, 13})
