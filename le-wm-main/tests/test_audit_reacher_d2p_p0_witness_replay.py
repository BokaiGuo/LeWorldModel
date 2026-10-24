import subprocess
import sys
from pathlib import Path

from experiments.spectral_reachability import audit_reacher_d2p_p0_witness_replay as replay_audit
from experiments.spectral_reachability.audit_reacher_d2p_p0_witness_replay import _validate_receipt
import pytest
import torch


def _receipt(**overrides):
    value = {
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "protocol": {
            "outcome_only": True,
            "future_suffix_only_optimized": True,
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "context_mode": "executed_warmup",
            "warmup_policy": "seeded_bounded_excitation_v1",
            "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
            "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore",
        },
        "inputs": {"runner_sha256": "current-runner-hash"},
        "rows": [{"seed": 1}],
    }
    value.update(overrides)
    return value


def test_witness_replay_rejects_non_outcome_or_future_prefix_receipts(monkeypatch):
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as runner

    monkeypatch.setattr(runner, "_sha256_file", lambda _: "current-runner-hash")
    _validate_receipt(_receipt())
    with pytest.raises(ValueError, match="outcome-only"):
        _validate_receipt(_receipt(protocol={"outcome_only": False, "future_suffix_only_optimized": True}))
    with pytest.raises(ValueError, match="runner hash"):
        _validate_receipt(_receipt(inputs={"runner_sha256": "other"}))
    bad_protocol = dict(_receipt()["protocol"])
    bad_protocol["execution_root_reconstruction"] = "reset_and_replay"
    with pytest.raises(ValueError, match="execution snapshot"):
        _validate_receipt(_receipt(protocol=bad_protocol))


def test_normalized_witness_replay_rejects_action_dimension_mismatch():
    from experiments.spectral_reachability.audit_reacher_d2p_p0_witness_replay import _frozen_cost_from_normalized_witness

    with pytest.raises(ValueError, match="incompatible"):
        _frozen_cost_from_normalized_witness(
            torch.nn.Identity(), torch.empty(1), torch.empty(1), torch.empty(1, 3, 10), torch.empty(2, 9),
        )


def test_witness_replay_records_either_model_cost_order_without_spurious_gate():
    rows = [
        {"methods": {"d2p_goal_progress": {"replayed_terminal_cost": 0.1}, "cem": {"replayed_terminal_cost": 0.2}}},
        {"methods": {"d2p_goal_progress": {"replayed_terminal_cost": 0.2}, "cem": {"replayed_terminal_cost": 0.1}}},
        {"methods": {"d2p_goal_progress": {"replayed_terminal_cost": 0.3}, "cem": {"replayed_terminal_cost": 0.3}}},
    ]
    assert replay_audit._model_cost_order_counts(rows) == (1, 1, 1)
    rows[0]["methods"]["d2p_goal_progress"]["replayed_terminal_cost"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        replay_audit._model_cost_order_counts(rows)


def test_witness_replay_direct_script_entrypoint_loads_project_package():
    completed = subprocess.run(
        [sys.executable, str(Path(replay_audit.__file__)), "--help"],
        cwd=Path(replay_audit.__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CUDA replay audit" in completed.stdout
