import hashlib
import json

import pytest

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.d2p_stage_admission import P0_INTEGRITY_KEYS
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY
from experiments.spectral_reachability.reacher_d2p_packet_support import WARMUP_POLICY as SOURCE_WARMUP_POLICY


def test_source_supported_p1_rejects_uniform_p0_receipt(tmp_path):
    receipt = tmp_path / "p0.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "stage_gate": {"p1_unlocked": True},
        "gates": {key: True for key in P0_INTEGRITY_KEYS},
        "comparisons_outcome_only": {"cem": {"ci95": [-.5, -.1]}},
        "protocol": {"calibration_only": False, "context_support": "uniform", "warmup_policy": WARMUP_POLICY, "qpos_readout_binding": {"enabled": False}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="context-support"):
        p1._validate_upstream_p0(
            receipt, expected_context_support="source_contiguous",
            expected_warmup_policy=SOURCE_WARMUP_POLICY + ".contiguous_history",
            expected_qpos_binding={"enabled": False},
        )


def test_source_supported_p1_accepts_only_matching_source_p0_receipt(tmp_path):
    receipt = tmp_path / "p0.json"
    policy = SOURCE_WARMUP_POLICY + ".contiguous_history"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "stage_gate": {"p1_unlocked": True},
        "gates": {key: True for key in P0_INTEGRITY_KEYS},
        "comparisons_outcome_only": {"cem": {"ci95": [-.5, -.1]}},
        "protocol": {"calibration_only": False, "context_support": "source_contiguous", "warmup_policy": policy, "qpos_readout_binding": {"enabled": False}},
    }), encoding="utf-8")

    binding = p1._validate_upstream_p0(
        receipt, expected_context_support="source_contiguous", expected_warmup_policy=policy, expected_qpos_binding={"enabled": False},
    )

    assert binding["context_support"] == "source_contiguous"
    assert binding["warmup_policy"] == policy
    assert binding["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_p0_outcome_rejects_calibration_from_a_different_context_distribution(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema": "d2p_reacher_damping_selection.v1",
        "status": "pass",
        "selected": {"damping": 1.0},
        "calibration_root_seeds": [10, 11],
        "calibration_contract": {
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "runner": p0._sha256_file(p0.Path(p0.__file__)),
            "relinearization_steps": 4, "d2p_restarts": 1,
            "d2p_initialization": "screened_gaussian_proposals",
            "d2p_proposal_samples": 128, "d2p_proposal_elites": 4,
            "context_mode": "executed_warmup", "context_support": "uniform",
            "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
            "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore",
            "warmup_policy": WARMUP_POLICY,
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="context-support"):
        p0._validate_outcome_calibration(
            (selection,), damping=1.0, relinearization_steps=4, d2p_restarts=1,
            d2p_initialization="screened_gaussian_proposals", d2p_proposal_samples=128,
            d2p_proposal_elites=4, outcome_seeds={20, 21},
            context_support="source_contiguous",
            expected_warmup_policy=SOURCE_WARMUP_POLICY + ".contiguous_history",
            expected_qpos_binding={"enabled": False},
        )
