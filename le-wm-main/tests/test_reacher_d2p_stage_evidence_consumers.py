import json

import pytest

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.d2p_stage_admission import P0_INTEGRITY_KEYS


def evidence():
    root, goal = "a" * 64, "b" * 64
    def method(name):
        return {"execution_identity": {"reset_rgb_sha256": root, "goal_rgb_sha256": goal},
                "receipt": {"predict_accounting_scope": "planner_only_excludes_post_execution_diagnostic_encoder_and_backward",
                            "actual_planning_predict": {"phase": "planning", "method": name,
                                "completed": True, "failed_predict_calls": 0, "horizon_tokens": 3,
                                "predict_calls": 3, "predict_samples": 3, "context_lengths": {"3": 3}}}}
    return {"schema": "d2p_reacher_p0_goal_progress.v1", "status": "completed_without_efficacy_verdict",
        "stage_gate": {"p1_unlocked": True}, "gates": {key: True for key in P0_INTEGRITY_KEYS},
        "comparisons_outcome_only": {"cem": {"ci95": [-.5, -.1]}},
        "rows": [{"split": "outcome", "seed": 1, "root_index": 0,
                  "executed_context": {"root_rgb_sha256": root, "goal_rgb_sha256": goal},
                  "methods": {name: method(name) for name in ("d2p_goal_progress", "cem")}}],
        "protocol": {"calibration_only": False, "context_support": "fixture", "warmup_policy": "fixture", "qpos_readout_binding": None}}


def validate(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return p1._validate_upstream_p0(path, expected_context_support="fixture", expected_warmup_policy="fixture", expected_qpos_binding=None)


def test_strict_historical_positive_remains_admissible(tmp_path):
    assert validate(tmp_path, evidence())["p1_unlocked"] is True


@pytest.mark.parametrize("interval", ([0, 0], [-.2, 0], [-.2, .1], [float("nan"), -1]))
def test_consumer_cannot_trust_legacy_true_flag_over_actual_interval(tmp_path, interval):
    report = evidence()
    report["comparisons_outcome_only"]["cem"]["ci95"] = interval
    with pytest.raises(ValueError):
        validate(tmp_path, report)


@pytest.mark.parametrize("defect", ("empty_gates", "missing_gate", "missing_comparison", "calibration"))
def test_upstream_missing_or_calibration_evidence_fails_closed(tmp_path, defect):
    report = evidence()
    if defect == "empty_gates":
        report["gates"] = {}
    elif defect == "missing_gate":
        del report["gates"][P0_INTEGRITY_KEYS[0]]
    elif defect == "missing_comparison":
        report["comparisons_outcome_only"] = {}
    else:
        report["protocol"]["calibration_only"] = True
    with pytest.raises(ValueError):
        validate(tmp_path, report)
