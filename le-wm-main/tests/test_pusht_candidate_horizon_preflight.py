import pytest
import numpy as np

from experiments import preflight_pusht_candidate_horizon as preflight_module
from pathlib import Path

from experiments.preflight_pusht_candidate_horizon import (
    SCHEMA,
    authorize_calibration,
    reconstruct_frozen_cohort,
    validate_mode_aware_runner_parameters,
    validate_learned_runtime_contract,
)


def _row(index: int, reachable: bool) -> dict[str, object]:
    return {
        "root_id": index,
        "root_sha256": f"root-{index}",
        "root_action_bank_sha256": f"bank-{index}",
        "root_candidate_count": 9,
        "oracle_reachable": reachable,
    }


def _preflight(authorized: bool = True) -> dict[str, object]:
    rows = [_row(0, authorized), _row(1, False)]
    return {
        "schema": SCHEMA,
        "protocol": {"protocol_sha256": "protocol-hash"},
        "calibration_authorization": {"authorized": authorized},
        "cohorts": {"outcome": rows},
    }


def test_authorization_is_calibration_only_and_requires_both_count_and_rate():
    result = authorize_calibration([_row(0, True), _row(1, False), _row(2, False)], minimum_reachable_roots=1, minimum_reachable_rate=0.25)
    assert result["authorized"] is True
    assert result["reachable_roots"] == 1
    rejected = authorize_calibration([_row(0, True), _row(1, False), _row(2, False)], minimum_reachable_roots=2, minimum_reachable_rate=0.10)
    assert rejected["authorized"] is False
    assert rejected["decision"] == "candidate_or_horizon_limited"


def test_authorization_uses_the_same_oracle_arm_as_the_future_learned_selector():
    row = _row(0, True)
    row["oracle_arm_pose_success"] = {
        "one_pulse": False,
        "two_pulse": False,
        "three_pulse_bounded": True,
    }
    rejected = authorize_calibration([row], minimum_reachable_roots=1, minimum_reachable_rate=0.5)
    assert rejected["authorized"] is False
    accepted = authorize_calibration(
        [row], minimum_reachable_roots=1, minimum_reachable_rate=0.5,
        required_oracle_arm="three_pulse_bounded",
    )
    assert accepted["authorized"] is True


def test_runtime_contract_requires_exact_ordered_root_and_bank_manifest():
    preflight = _preflight()
    runtime = {
        "protocol_sha256": "protocol-hash",
        "roots": preflight["cohorts"]["outcome"],
    }
    validate_learned_runtime_contract(preflight, runtime)
    runtime["roots"] = list(reversed(runtime["roots"]))
    with pytest.raises(ValueError, match="roots or ordered candidate banks"):
        validate_learned_runtime_contract(preflight, runtime)


def test_runtime_contract_rejects_candidate_limited_preflight_and_protocol_drift():
    with pytest.raises(ValueError, match="candidate_or_horizon_limited"):
        validate_learned_runtime_contract(_preflight(authorized=False), {"protocol_sha256": "protocol-hash", "roots": []})
    with pytest.raises(ValueError, match="protocol fingerprint"):
        validate_learned_runtime_contract(_preflight(), {"protocol_sha256": "other", "roots": []})


def test_cuda_launcher_has_bounded_preflight_and_no_cpu_fallback():
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_candidate_horizon_preflight_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "GPU_PREFLIGHT_ATTEMPTS" in text
    assert "experiments.gpu_preflight --device cuda --recurrent-smoke" in text
    assert "experiments.preflight_pusht_candidate_horizon" in text
    assert "--device cuda" in text
    assert "CPU fallback" in text


def test_matched_launcher_requires_preflight_authorization_before_learned_run():
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_mode_aware_preflight_matched_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "run_pusht_candidate_horizon_preflight_cuda.sh" in text
    assert "calibration_authorization" in text
    assert "candidate_or_horizon_limited" in text
    assert "--oracle-feasibility-preflight" in text
    assert "--device cuda" in text


def test_reconstruct_frozen_cohort_rejects_seed_or_root_generator_drift(monkeypatch):
    roots = [
        {"state": np.zeros(7), "goal": np.ones(7)},
        {"state": np.ones(7), "goal": np.zeros(7)},
    ]
    monkeypatch.setattr(preflight_module, "collect_nonterminal_boundary_contact_roots", lambda *_args, **_kwargs: roots)
    expected = [preflight_module.root_fingerprint(root) for root in roots]
    artifact = {
        "schema": SCHEMA,
        "protocol": {"cohort_contract": {"outcome": {"root_seed": 9, "root_count": 2, "oversample_factor": 4}}},
        "cohorts": {"outcome": [{"root_sha256": value} for value in expected]},
    }
    rebuilt = reconstruct_frozen_cohort(object(), artifact, cohort="outcome")
    assert len(rebuilt) == 2
    artifact["cohorts"]["outcome"] = list(reversed(artifact["cohorts"]["outcome"]))
    with pytest.raises(ValueError, match="differs"):
        reconstruct_frozen_cohort(object(), artifact, cohort="outcome")


def test_mode_aware_contract_rejects_horizon_or_bank_mismatch():
    artifact = {
        "schema": SCHEMA,
        "protocol": {
            "required_oracle_arm": "two_pulse",
            "pulse_steps": 2,
            "macro_steps": 18,
            "action_bank_config": {
                "inward_magnitude": 0.08,
                "tangent_magnitudes": [0.06, 0.10, 0.16],
                "contact_normal_magnitudes": [0.02, 0.04, 0.08],
            },
        },
    }
    common = {
        "root_protocol": "boundary_contact", "boundary_action_frame": True,
        "inward_magnitude": 0.08, "tangent_magnitudes": (0.06, 0.10, 0.16),
        "contact_normal_magnitudes": (0.02, 0.04, 0.08), "pulse_steps": 2,
        "closure_steps": 18, "lookahead": 2,
    }
    validate_mode_aware_runner_parameters(artifact, **common)
    with pytest.raises(ValueError, match="lookahead"):
        validate_mode_aware_runner_parameters(artifact, **{**common, "lookahead": 1})
    with pytest.raises(ValueError, match="tangent magnitudes"):
        validate_mode_aware_runner_parameters(artifact, **{**common, "tangent_magnitudes": (0.04, 0.10)})
