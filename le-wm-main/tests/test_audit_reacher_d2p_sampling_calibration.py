"""Synthetic receipt fixtures, never experimental performance evidence."""
import json
import hashlib
from copy import deepcopy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_sampling_calibration import EXPECTED, audit


PROTOCOL = {
    "seeds": [20261040, 20261041], "history_len": 3, "future_horizon": 2,
    "action_block": 5, "target_delta": [.65, -.65], "split": "known_calibration_only",
    "context_support": "uniform", "sampling_parameter_box": [-3., 3.],
    "sampling_coordinate": "pre_tanh_parameters_not_raw_action_box",
    "cem_parameter_domain": "P0_unbounded_Gaussian_pre_tanh_parameters",
    "methods": list(EXPECTED), "samples": 32, "mppi_iterations": 4,
    "temperature": 1., "noise_scale": 1., "cem_steps": 4, "cem_elite": 4,
    "P1_admission": False, "efficacy_claim": False, "fair_budget_claim": False,
    "readout": "none; frozen original latent objective",
    "mppi_claim_boundary": "clipped Gaussian heuristic without importance correction; not faithful MPPI; negative results cannot exclude MPPI",
    "budget_policy": "count all initial/proposal/updated-nominal scoring; no candidate-count matching",
    "inputs": {"/synthetic/weights.pt": "a" * 64, "/synthetic/config.json": "b" * 64,
               "/synthetic/adapter.npz": "c" * 64},
}


def fixture():
    rows = []
    for seed in (20261040, 20261041):
        root = hashlib.sha256(str(seed).encode()).hexdigest()
        goal = hashlib.sha256(f"goal{seed}".encode()).hexdigest()
        methods = {}
        for name, (calls, samples, cc, cs) in EXPECTED.items():
            receipt = {"actual_planning_predict": {"phase": "planning", "method": name,
                "horizon_tokens": 2, "completed": True, "failed_predict_calls": 0,
                "predict_calls": calls, "predict_samples": samples, "context_lengths": {"3": calls}},
                "planner_synchronized_wall_seconds": .01}
            if name != "cem":
                receipt.update(cost_calls=cc, cost_evaluations=cs,
                    candidate_bridge={"candidate_cost_calls": cc, "candidate_cost_samples": cs,
                        "completed_candidate_cost_calls": cc, "completed_candidate_cost_samples": cs,
                        "failed_candidate_cost_calls": 0, "normalized_action_lower": [-2.] * 10,
                        "normalized_action_upper": [2.] * 10, "model_input_dtype": "torch.float32",
                        "model_device": "cuda:0", "horizon_tokens": 2,
                        "coordinate": "unconstrained_parameters_tanh_mapped_to_normalized_physical_box",
                        "counter_semantics": "candidate_cost counters are submitted after CUDA conversion; completed counters require valid output",
                        "scope": "candidate scoring only; not predict/encoder/backward or total-compute accounting"})
            else:
                receipt["candidate_bridge"] = None
            methods[name] = {"seed": seed, "method": name, "root_rgb_sha256": root,
                "goal_rgb_sha256": goal, "execution_identity": {
                "reset_rgb_sha256": root, "goal_rgb_sha256": goal}, "receipt": receipt,
                "physical": {"initial_qpos_distance": 1., "final_qpos_distance": .5,
                             "min_qpos_distance": .5, "physical_progress": .5},
                "params": [[[0.] * 10 for _ in range(2)]], "raw_actions": [[0., 0.] for _ in range(10)],
                "normalized_actions": [[0.] * 10 for _ in range(2)]}
        rows.append({"seed": seed, "split": "calibration", "methods": methods,
            "executed_context": {"root_rgb_sha256": root, "goal_rgb_sha256": goal,
                "history_rgb_sha256": [root] * 3, "warmup_actions_sha256": "d" * 64,
                "warmup_policy": "seeded_full_packet_uniform_v2"}})
    return {"schema": "reacher_d2p_sampling_calibration.v1", "status": "calibration_completed",
        "P1_admission": False, "efficacy_claim": False, "fair_budget_claim": False,
        "synthetic": True, "gates": {"fixture_only": True}, "rows": rows,
        "protocol": deepcopy(PROTOCOL)}


def write_fixture(tmp_path, report):
    for row in report["rows"]:
        for name, method in row["methods"].items():
            plan = {k: v for k, v in method.items() if k not in ("physical", "execution_identity")}
            plan["synthetic"] = True
            (tmp_path / f"seed_{row['seed']}.{name}.plan.json").write_text(json.dumps(plan))
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(report))
    return path


def test_expected_accounting_fixture(tmp_path):
    path = write_fixture(tmp_path, fixture())
    result = audit(path)
    assert result["status"] == "accounting_receipts_valid"
    assert len(result["counts"]) == 6 and result["P1_admission"] is False


@pytest.mark.parametrize("name", list(EXPECTED))
@pytest.mark.parametrize("field", ["raw_actions", "normalized_actions"])
def test_in_bounds_but_wrong_action_witness_rejected(tmp_path, name, field):
    report = fixture()
    report["rows"][0]["methods"][name][field][0][0] = .25
    # Both plan and report agree, so only an independent chart check catches it.
    with pytest.raises(ValueError, match="witness does not match"):
        audit(write_fixture(tmp_path, report))


def test_nonzero_time_varying_parameter_mapping_accepts_correct_order(tmp_path):
    import numpy as np
    report = fixture()
    params = np.linspace(-.8, .8, 20).reshape(2, 10)
    for row in report["rows"]:
        for method in row["methods"].values():
            method["params"] = [params.tolist()]
            method["raw_actions"] = np.tanh(params).reshape(10, 2).tolist()
            method["normalized_actions"] = (2*np.tanh(params)).tolist()
    assert audit(write_fixture(tmp_path, report))["status"] == "accounting_receipts_valid"


@pytest.mark.parametrize("field,value", [("predict_samples", 256), ("completed", False),
                                         ("failed_predict_calls", 1)])
def test_malformed_receipt_rejected(tmp_path, field, value):
    report = fixture()
    report["rows"][0]["methods"]["clipped_gaussian_mppi_heuristic"]["receipt"]["actual_planning_predict"][field] = value
    path = write_fixture(tmp_path, report)
    with pytest.raises(ValueError):
        audit(path)


@pytest.mark.parametrize("field", ["params", "raw_actions", "normalized_actions", "receipt",
                                   "seed", "method", "root_rgb_sha256", "goal_rgb_sha256"])
def test_plan_field_binding(tmp_path, field):
    path = write_fixture(tmp_path, fixture())
    plan_path = tmp_path / "seed_20261040.cem.plan.json"
    plan = json.loads(plan_path.read_text())
    plan.pop(field)
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        audit(path)


def test_missing_plan_fails(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture()))
    with pytest.raises(ValueError):
        audit(path)


@pytest.mark.parametrize("field,value", [
    ("params", [[[True] * 10] * 2]), ("params", [[["0"] * 10] * 2]),
    ("params", [[[0.] * 10]]), ("raw_actions", [[0., 0.]] * 9),
    ("raw_actions", [[0.] * 3] * 10), ("normalized_actions", [[0.] * 9] * 2),
    ("root_rgb_sha256", "g" * 64), ("goal_rgb_sha256", "x"),
])
def test_strict_witnesses(tmp_path, field, value):
    report = fixture()
    report["rows"][0]["methods"]["cem"][field] = value
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


@pytest.mark.parametrize("value", [True, -1., float("nan"), float("inf"), "0"])
def test_wall_time_invalid(tmp_path, value):
    report = fixture()
    report["rows"][0]["methods"]["cem"]["receipt"]["planner_synchronized_wall_seconds"] = value
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


@pytest.mark.parametrize("key", list(PROTOCOL))
def test_every_protocol_field_required(tmp_path, key):
    report = fixture()
    del report["protocol"][key]
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


def test_histogram_sum_and_bool_counts(tmp_path):
    report = fixture()
    report["rows"][0]["methods"]["cem"]["receipt"]["actual_planning_predict"]["context_lengths"] = {"3": 7}
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


def test_complex_numeric_validator():
    from experiments.spectral_reachability.audit_reacher_d2p_sampling_calibration import _number
    with pytest.raises(ValueError):
        _number(1 + 0j)


@pytest.mark.parametrize("field,value", [
    ("completed_candidate_cost_calls", 8), ("completed_candidate_cost_samples", 132),
    ("failed_candidate_cost_calls", 1), ("failed_candidate_cost_calls", False),
    ("normalized_action_lower", [True] * 10), ("normalized_action_upper", [-3.] * 10),
    ("model_input_dtype", "torch.complex64"), ("model_device", "cpu"),
])
def test_bridge_completion_and_contract(tmp_path, field, value):
    report = fixture()
    report["rows"][0]["methods"]["clipped_gaussian_mppi_heuristic"]["receipt"]["candidate_bridge"][field] = value
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


@pytest.mark.parametrize("key", list(PROTOCOL))
def test_every_protocol_value_fixed(tmp_path, key):
    report = fixture()
    report["protocol"][key] = "not-the-frozen-value"
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


@pytest.mark.parametrize("field", ["params", "raw_actions", "normalized_actions", "receipt",
                                   "seed", "method", "root_rgb_sha256", "goal_rgb_sha256"])
def test_changed_plan_values_rejected(tmp_path, field):
    path = write_fixture(tmp_path, fixture())
    plan_path = tmp_path / "seed_20261040.cem.plan.json"
    plan = json.loads(plan_path.read_text())
    plan[field] = "tampered"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan/report"):
        audit(path)


@pytest.mark.parametrize("mutation", [
    lambda m: m["receipt"]["actual_planning_predict"].update(failed_predict_calls=False),
    lambda m: m["receipt"]["actual_planning_predict"].update(context_lengths={"3": True}),
    lambda m: m["physical"].update(physical_progress=True),
    lambda m: m["physical"].pop("final_qpos_distance"),
    lambda m: m["execution_identity"].update(goal_rgb_sha256="e" * 64),
    lambda m: m.update(goal_rgb_sha256="e" * 64),
    lambda m: m.update(seed=True),
])
def test_report_bindings_and_strict_numeric_types(tmp_path, mutation):
    report = fixture()
    mutation(report["rows"][0]["methods"]["cem"])
    with pytest.raises(ValueError):
        audit(write_fixture(tmp_path, report))


def test_all_synthetic_plans_bound_with_real_hashes(tmp_path):
    result = audit(write_fixture(tmp_path, fixture()))
    assert result["synthetic"] is True
    assert len(result["plan_bindings"]) == 6
    for binding in result["plan_bindings"]:
        from pathlib import Path
        assert hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    assert result["efficacy_claim"] is False and result["fair_budget_claim"] is False


def test_duplicate_json_keys_fail_closed(tmp_path):
    path = write_fixture(tmp_path, fixture())
    plan = tmp_path / "seed_20261040.cem.plan.json"
    plan.write_text('{"seed":20261040,"seed":20261040}')
    with pytest.raises(ValueError, match="duplicate"):
        audit(path)
