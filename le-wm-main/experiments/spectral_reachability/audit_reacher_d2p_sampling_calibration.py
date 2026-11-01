"""Independent fixed-protocol receipts audit, not efficacy or P1 admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

EXPECTED = {
    "clipped_gaussian_mppi_heuristic": (18, 266, 9, 133),
    "bestofn_random": (4, 66, 2, 33),
    "cem": (8, 256, None, None),
}

FIXED_PROTOCOL = {
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
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float), "finite real number required; bool/complex/coercion forbidden")
    _require(math.isfinite(value), "nonfinite number")
    return value


def _counter(value):
    _require(type(value) is int and value >= 0, "nonnegative integer counter required")
    return value


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
             "real 64hex SHA256 required")
    return value


def _array(value, shape):
    if not shape:
        return _number(value)
    _require(type(value) is list and len(value) == shape[0], f"exact array shape {shape} required")
    return [_array(item, shape[1:]) for item in value]


def _equal(left, right):
    # JSON structural comparison that does not let True == 1 bypass binding.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _load(raw):
    return json.loads(raw, object_pairs_hook=_pairs)


def audit(path):
    """Bind saved plans and report, not proof of execution or pre-outcome timing.

    No neural imports. Plan filenames are derived, never taken from report paths.
    Source-seal verification, when requested separately, proves byte consistency
    only; this receipt audit does not claim runtime authenticity.
    """
    try:
        return _audit(Path(path))
    except (KeyError, TypeError, OSError, OverflowError) as error:
        raise ValueError(f"missing/malformed report or plan: {error}") from error


def _audit(path):
    raw = Path(path).read_bytes()
    report = _load(raw)
    if (report["schema"] != "reacher_d2p_sampling_calibration.v1"
            or report["status"] != "calibration_completed"
            or any(report[k] is not False for k in ("P1_admission", "efficacy_claim", "fair_budget_claim"))
            or not report["gates"] or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete calibration-only report with all integrity gates required")
    rows = report["rows"]
    if [_counter(r["seed"]) for r in rows] != [20261040, 20261041]:
        raise ValueError("fixed two-root calibration protocol required")
    protocol = report["protocol"]
    _require(set(protocol) == set(FIXED_PROTOCOL) | {"inputs"}, "complete fixed protocol required")
    for key, value in FIXED_PROTOCOL.items():
        _require(_equal(protocol[key], value), f"unsupported calibration protocol: {key}")
    inputs = protocol["inputs"]
    _require(type(inputs) is dict and len(inputs) == 3, "three bound input files required")
    for filename, digest in inputs.items():
        _require(isinstance(filename, str) and Path(filename).is_absolute(), "absolute input binding required")
        _sha(digest)
    _require(sum(Path(p).name == "weights.pt" for p in inputs) == 1
             and sum(Path(p).name == "config.json" for p in inputs) == 1, "weights/config bindings required")
    counts = []
    roots = set()
    plan_bindings = []
    for row in rows:
        context = row["executed_context"]
        root = _sha(context["root_rgb_sha256"])
        goal = _sha(context["goal_rgb_sha256"])
        _sha(context["warmup_actions_sha256"])
        _require(type(context["history_rgb_sha256"]) is list and len(context["history_rgb_sha256"]) == 3,
                 "three history hashes required")
        for digest in context["history_rgb_sha256"]:
            _sha(digest)
        _require(context["warmup_policy"] == "seeded_full_packet_uniform_v2", "fixed warmup required")
        if row["split"] != "calibration" or root in roots or set(row["methods"]) != set(EXPECTED):
            raise ValueError("distinct calibration roots and complete method set required")
        roots.add(root)
        reference_bridge = row["methods"]["bestofn_random"]["receipt"]["candidate_bridge"]
        reference_lower = np.asarray(_array(reference_bridge["normalized_action_lower"], (10,)), dtype=float)
        reference_upper = np.asarray(_array(reference_bridge["normalized_action_upper"], (10,)), dtype=float)
        _require(np.all(reference_lower < reference_upper), "invalid shared action bounds")
        for name, (calls, samples, cost_calls, cost_samples) in EXPECTED.items():
            method = row["methods"][name]
            _require(_counter(method["seed"]) == row["seed"] and method["method"] == name,
                     "method seed/name mismatch")
            _require(_sha(method["goal_rgb_sha256"]) == goal, "method goal mismatch")
            if (method["root_rgb_sha256"] != root
                    or method["execution_identity"]["reset_rgb_sha256"] != root
                    or method["execution_identity"]["goal_rgb_sha256"] != context["goal_rgb_sha256"]):
                raise ValueError("execution identity mismatch")
            receipt = method["receipt"]
            _require(_number(receipt["planner_synchronized_wall_seconds"]) >= 0, "negative walltime")
            actual = receipt["actual_planning_predict"]
            for key in ("horizon_tokens", "failed_predict_calls", "predict_calls", "predict_samples"):
                _counter(actual[key])
            if (actual["phase"] != "planning" or actual["method"] != name
                    or actual["horizon_tokens"] != 2 or actual["completed"] is not True
                    or actual["failed_predict_calls"] != 0
                    or actual["predict_calls"] != calls or actual["predict_samples"] != samples):
                raise ValueError("actual prediction counts do not match independent protocol formula")
            histogram = actual["context_lengths"]
            _require(type(histogram) is dict and set(histogram) == {"3"}, "fixed context histogram required")
            _require(sum(_counter(v) for v in histogram.values()) == calls, "context histogram calls mismatch")
            if name != "cem":
                bridge = receipt["candidate_bridge"]
                for key in ("cost_calls", "cost_evaluations"):
                    _counter(receipt[key])
                for key in ("candidate_cost_calls", "candidate_cost_samples", "completed_candidate_cost_calls",
                            "completed_candidate_cost_samples", "failed_candidate_cost_calls", "horizon_tokens"):
                    _counter(bridge[key])
                _require(bridge["completed_candidate_cost_calls"] == bridge["candidate_cost_calls"]
                         and bridge["completed_candidate_cost_samples"] == bridge["candidate_cost_samples"]
                         and bridge["failed_candidate_cost_calls"] == 0, "submitted work not fully completed")
                lower = _array(bridge["normalized_action_lower"], (10,))
                upper = _array(bridge["normalized_action_upper"], (10,))
                _require(all(a < b for a, b in zip(lower, upper)), "invalid normalized bounds")
                _require(np.array_equal(lower, reference_lower) and np.array_equal(upper, reference_upper),
                         "methods use different normalized action bounds")
                _require(bridge["model_input_dtype"] == "torch.float32"
                         and re.fullmatch(r"cuda:[0-9]+", bridge["model_device"]) is not None
                         and bridge["horizon_tokens"] == 2, "bridge dtype/device/horizon mismatch")
                _require(bridge["coordinate"] == "unconstrained_parameters_tanh_mapped_to_normalized_physical_box"
                         and bridge["counter_semantics"] == "candidate_cost counters are submitted after CUDA conversion; completed counters require valid output"
                         and bridge["scope"] == "candidate scoring only; not predict/encoder/backward or total-compute accounting",
                         "bridge coordinate/counting scope mismatch")
                if (receipt["cost_calls"] != cost_calls or receipt["cost_evaluations"] != cost_samples
                        or bridge["candidate_cost_calls"] != cost_calls
                        or bridge["candidate_cost_samples"] != cost_samples):
                    raise ValueError("initial/proposal/updated-nominal scoring count mismatch")
            else:
                _require(receipt["candidate_bridge"] is None, "CEM bridge must be absent explicitly")
            physical = method["physical"]
            _require(set(physical) == {"initial_qpos_distance", "final_qpos_distance", "min_qpos_distance", "physical_progress"},
                     "complete physical labels required")
            for value in physical.values():
                _number(value)
            parameters = np.asarray(_array(method["params"], (1, 2, 10)), dtype=float)[0]
            raw_actions = _array(method["raw_actions"], (10, 2))
            normalized = _array(method["normalized_actions"], (2, 10))
            _require(all(-1-1e-6 <= v <= 1+1e-6 for action in raw_actions for v in action), "raw action bounds")
            # Independent NumPy reconstruction of the released affine/tanh
            # action chart. Float32 torch execution permits rounding, not a
            # different witness; all methods use the same physical [-1,1] box.
            squashed = np.tanh(parameters)
            expected_normalized = ((reference_lower + reference_upper) / 2
                                   + (reference_upper - reference_lower) / 2 * squashed)
            _require(np.allclose(normalized, expected_normalized, rtol=2e-6, atol=2e-6),
                     "normalized witness does not match parameter mapping")
            _require(np.allclose(raw_actions, squashed.reshape(10, 2), rtol=2e-6, atol=2e-6),
                     "raw witness does not match parameter mapping or packet order")
            if name != "cem":
                _require(all(-3 <= v <= 3 for action in method["params"][0] for v in action), "sampling parameter bounds")
                _require(all(a <= v <= b for action in normalized for a, v, b in zip(lower, action, upper)),
                         "normalized action outside bridge bounds")
            plan_path = path.parent / f"seed_{row['seed']}.{name}.plan.json"
            plan_raw = plan_path.read_bytes()
            plan = _load(plan_raw)
            for key in ("seed", "method", "params", "raw_actions", "normalized_actions", "receipt",
                        "root_rgb_sha256", "goal_rgb_sha256"):
                _require(_equal(plan[key], method[key]), f"plan/report field mismatch: {key}")
            plan_bindings.append({"seed": row["seed"], "method": name,
                                  "path": str(plan_path.resolve()), "sha256": hashlib.sha256(plan_raw).hexdigest()})
            counts.append({"seed": row["seed"], "method": name,
                           "predict_calls": calls, "predict_samples": samples})
    return {"schema": "reacher_d2p_sampling_calibration_audit.v1",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "status": "accounting_receipts_valid", "counts": counts,
            "plan_bindings": plan_bindings, "synthetic": report.get("synthetic", False),
            "P1_admission": False, "efficacy_claim": False, "fair_budget_claim": False,
            "scope": "saved receipt consistency; not independent runtime or physical replay proof"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.report), indent=2, sort_keys=True))
