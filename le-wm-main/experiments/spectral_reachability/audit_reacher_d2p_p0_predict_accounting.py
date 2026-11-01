"""Independent pure receipt audit of calibration-only Reacher P0 predict work.

No runner/helper imports, neural execution, outcome reading, budget matching or
efficacy claim. Checks receipt consistency, not authenticity of the CUDA run.
Expected seeds must be supplied independently; no outcome-selected root filter.
Legacy forward_rollout_evaluations has heterogeneous method-specific semantics
and is NOT the unit of actual predict invocations or batch samples. Numerical
equality, including random's zero, never establishes equivalent units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

METHODS = ("d2p_goal_progress", "local_gauss_newton", "action_gd",
           "grasp_style_restart_gd", "cem", "random")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid integer {name}")
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(raw):
    return json.loads(raw, object_pairs_hook=_unique_pairs)


def load_report(path):
    """Reject duplicate object keys before any method/phase can be overwritten."""
    return _decode(Path(path).read_bytes())


def _phase(receipt, method, horizon, history):
    _require(isinstance(receipt, dict), "missing predict receipt")
    _require(receipt["phase"] == "planning" and receipt["method"] == method,
             "mixed phase/method binding")
    _require(receipt["completed"] is True, "incomplete planning phase")
    _require(_integer(receipt["failed_predict_calls"], "failed_predict_calls") == 0,
             "failed predict calls")
    _require(_integer(receipt["horizon_tokens"], "horizon_tokens", 1) == horizon,
             "mixed horizon binding")
    calls = _integer(receipt["predict_calls"], "predict_calls")
    samples = _integer(receipt["predict_samples"], "predict_samples")
    _require(samples >= calls and (calls > 0 or samples == 0), "batch samples inconsistent with calls")
    histogram = receipt["context_lengths"]
    _require(isinstance(histogram, dict), "context histogram must be an object")
    count = 0
    for key, value in histogram.items():
        _require(isinstance(key, str) and key == str(history), "invalid/mixed context length")
        count += _integer(value, "histogram count", 1)
    _require(count == calls, "context histogram does not sum to predict calls")
    return calls, samples


def audit(report, *, expected_seeds):
    """Return a bounded analysis or raise ValueError on any invalid receipt.

    This schema audits only saved planning receipts. Evaluation receipts are
    absent in this P0 schema; exclusion is reported scope, not independently
    proven coverage. Encoder/backward/physical work are never added to predict.
    """
    try:
        return _audit(report, expected_seeds=expected_seeds)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed/missing receipt field: {error}") from error


def _audit(report, *, expected_seeds):
    expected = list(expected_seeds)
    _require(expected and all(type(s) is int and s >= 0 for s in expected)
             and len(set(expected)) == len(expected), "expected seeds must be nonempty and duplicate-free")
    _require(report["schema"] == "d2p_reacher_p0_goal_progress.v1", "wrong source schema")
    protocol = report["protocol"]
    _require(report["status"] == "calibration_only" and protocol["calibration_only"] is True
             and protocol["outcome_only"] is False, "calibration-only audit required")
    _require(report["stage_gate"]["p1_unlocked"] is False, "calibration cannot admit P1")
    gates = report["gates"]
    _require(isinstance(gates, dict) and gates and all(v is True for v in gates.values()),
             "source integrity gates must all be true")
    names = protocol["methods"]
    _require(isinstance(names, list) and len(names) == len(METHODS)
             and set(names) == set(METHODS), "missing/duplicate/extra protocol methods")
    horizon = _integer(protocol["future_horizon"], "future_horizon", 1)
    history = _integer(protocol["history_len"], "history_len", 1)
    source_rows = report["rows"]
    _require(isinstance(source_rows, list) and len(source_rows) == len(expected), "root coverage mismatch")
    seeds, indices, fingerprints = set(), set(), set()
    rows = []
    for row in source_rows:
        seed = _integer(row["seed"], "seed")
        index = _integer(row["root_index"], "root_index")
        fingerprint = row["executed_context"]["root_rgb_sha256"]
        _require(isinstance(fingerprint, str) and bool(fingerprint.strip()), "missing root fingerprint")
        _require(seed not in seeds and index not in indices and fingerprint not in fingerprints,
                 "duplicate root seed/index/fingerprint")
        seeds.add(seed)
        indices.add(index)
        fingerprints.add(fingerprint)
        _require(row["split"] == "calibration", "outcome row in calibration audit")
        methods = row["methods"]
        _require(isinstance(methods, dict) and set(methods) == set(METHODS), "root method coverage mismatch")
        summaries = {}
        for method in METHODS:
            receipt = methods[method]["receipt"]
            calls, samples = _phase(receipt["actual_planning_predict"], method, horizon, history)
            old = _integer(receipt["forward_rollout_evaluations"], "legacy forward_rollout_evaluations")
            backward = _integer(receipt["backward_vjp_rows"], "legacy backward_vjp_rows")
            summaries[method] = {
                "predict_calls": calls, "predict_samples": samples,
                "horizon_tokens": horizon,
                "context_lengths": dict(receipt["actual_planning_predict"]["context_lengths"]),
                "legacy_forward_rollout_evaluations": old,
                "legacy_backward_vjp_rows_separate": backward,
                "numeric_equal_to_old": {"predict_calls": calls == old, "predict_samples": samples == old},
                "old_and_actual_same_unit": False,
            }
        rows.append({"seed": seed, "root_index": index, "root_rgb_sha256": fingerprint, "methods": summaries,
                     "d2p_vs_cem_resource_coordinates": {
                         "predict_samples_equal": summaries["d2p_goal_progress"]["predict_samples"] == summaries["cem"]["predict_samples"],
                         "predict_calls_equal": summaries["d2p_goal_progress"]["predict_calls"] == summaries["cem"]["predict_calls"],
                         "full_compute_match_claim": False,
                     }})
    _require(seeds == set(expected) and indices == set(range(len(expected))), "root coverage/binding mismatch")
    return {
        "schema": "audit_reacher_d2p_p0_predict_accounting.v1",
        "status": "accounting_receipts_valid", "root_count": len(rows),
        "expected_seeds": expected, "source_gate_count": len(gates),
        "rows": rows, "old_and_actual_same_unit": False,
        "evaluation_receipts_available": False,
        "scope": "Saved planning receipt consistency only; evaluator exclusion is declared, not independently instrumented here.",
        "unit_warning": "Legacy forward_rollout_evaluations mixes method-specific counting; numerical equality never means equal units or equal compute.",
        "unmeasured_resources": ["encoder forwards/samples", "actual backward invocations", "physical execution"],
        "efficacy_claim": False, "compute_matched_claim": False, "P1_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    analysis = audit(_decode(raw), expected_seeds=args.expected_seeds)
    analysis["source_report"] = str(args.report.resolve())
    analysis["source_sha256"] = hashlib.sha256(raw).hexdigest()
    # Exclusive creation: never overwrite an existing artifact or the source.
    with args.output.open("x") as stream:
        json.dump(analysis, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": analysis["status"], "root_count": analysis["root_count"],
                      "output": str(args.output), "source_sha256": analysis["source_sha256"]}))


if __name__ == "__main__":
    main()
