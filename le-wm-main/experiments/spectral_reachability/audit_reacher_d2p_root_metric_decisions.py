"""Independent known-panel evaluation of actual certified metric decisions.

Never choose the best physical candidate: join every sealed model decision by
root identity and retain its selected action, including all false positives.
This retrospective diagnostic cannot open P1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import TASKS, digest
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler


def evaluate(score, diagnostic):
    if (score.get("status") != "completed" or not score.get("gates")
            or not all(v is True for v in score["gates"].values())
            or score.get("actual_certified_qps") != 408
            or diagnostic.get("independent_metric_fits") != 1224):
        raise ValueError("complete certified CUDA scores and independent fits required")
    left, right = indexed(score["rows"]), indexed(diagnostic["rows"])
    if len(left) != 8 or set(left) != set(right):
        raise ValueError("all eight duplicate-free root identities must agree")
    rows, errors = [], []
    for seed in sorted(left):
        tasks = {}
        if set(left[seed]["tasks"]) != set(TASKS) or set(right[seed]["tasks"]) != set(TASKS):
            raise ValueError("all original tasks required")
        for task in TASKS:
            plan, measured = left[seed]["tasks"][task], right[seed]["tasks"][task]
            if compiler.select(plan["bank"]) != plan["decision"]:
                raise ValueError("sealed selector must reproduce exactly")
            programs = measured["programs"]
            if [p["index"] for p in programs] != list(range(17)):
                raise ValueError("complete independent program bank required")
            for b, p in zip(plan["bank"], programs):
                gain = float(p["fits"]["model"]["gain"])
                lo = float(b["common_response_gain"]["bounded_quadratic_gain"])
                hi = float(b["common_response_gain"]["bounded_quadratic_gain_upper_bound"])
                if not all(math.isfinite(v) for v in (gain, lo, hi)) or lo > hi:
                    raise ValueError("finite ordered gain intervals required")
                tolerance = 1e-8 * max(1., abs(gain), abs(lo), abs(hi))
                if not lo - tolerance <= gain <= hi + tolerance:
                    raise ValueError("independent point solution outside CUDA gain certificate")
                errors.append(max(lo - gain, gain - hi, 0.))
            decision = plan["decision"]
            if decision["greedy_index"] != measured["fixed_greedy_index"]:
                raise ValueError("fixed control identity changed")
            chosen = programs[decision["native_index"]]
            tasks[task] = {
                "selected_index": chosen["index"], "greedy_index": decision["greedy_index"],
                "model_witness": decision["reconfiguration_witness_found"],
                "original_selected_index": plan["original_decision"]["native_index"],
                "selected_original_physical_conjunction": chosen["original_physical_diagnostic_conjunction"],
                "selected_rgb_gain_positive_both_scales": chosen["physical_rgb_positive_point_margin_both_scales"],
                "selected_rgb_gain_minus_controls": chosen["physical_rgb_gain_minus_fixed_controls"],
                "selected_passes_both_diagnostic_checks": bool(chosen["original_physical_diagnostic_conjunction"]
                    and chosen["physical_rgb_positive_point_margin_both_scales"]),
            }
        rows.append({"seed": seed, "tasks": tasks})
    aggregate = {}
    for task in TASKS:
        all_rows = [r["tasks"][task] for r in rows]
        witnesses = [r for r in all_rows if r["model_witness"]]
        aggregate[task] = {"retained_roots": len(all_rows), "model_witness_roots": len(witnesses),
            "selected_original_physical_conjunction": sum(r["selected_original_physical_conjunction"] for r in witnesses),
            "selected_rgb_gain_positive_both_scales": sum(r["selected_rgb_gain_positive_both_scales"] for r in witnesses),
            "selected_passes_both_diagnostic_checks": sum(r["selected_passes_both_diagnostic_checks"] for r in witnesses),
            "selection_changed_roots": sum(r["selected_index"] != r["original_selected_index"] for r in all_rows)}
    return {"schema": "reacher_d2p_root_metric_decision_audit.v1", "rows": rows,
        "aggregate": aggregate, "independent_certificates_checked": len(errors),
        "max_point_gain_interval_violation": max(errors), "P1_admission": False,
        "physical_labels_used_for_selection": False,
        "scope": "Known-panel retrospective actual-selection check, not fresh calibration or efficacy. "
                 "No root/task/program omitted; no physical-best candidate substituted for selected action."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable decision audit required")
    score, diagnostic = [json.loads(p.read_text()) for p in (args.scores, args.diagnostic)]
    # Both independently generated reports must bind exactly the same model bank.
    score_bindings = score["sealed_protocol"]["input_bindings"]
    diagnostic_bindings = diagnostic["protocol"]["input_bindings"]
    common = set(score_bindings) & set(diagnostic_bindings)
    if not any(p.endswith(".predictions.json") for p in common):
        raise ValueError("same explicit sealed prediction binding required")
    for p in common:
        if score_bindings[p] != diagnostic_bindings[p] or digest(Path(p)) != score_bindings[p]:
            raise ValueError("bound input changed")
    for path, receipt in ((args.scores, score), (args.diagnostic, diagnostic)):
        if digest(path.with_suffix(".protocol.json")) != receipt["protocol_sha256"]:
            raise ValueError("protocol sidecar seal changed")
    result = evaluate(score, diagnostic)
    result["bindings"] = {str(p.resolve()): digest(p) for p in (args.scores, args.diagnostic)}
    result["implementation_sha256"] = digest(Path(__file__))
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"aggregate": result["aggregate"], "certificates": result["independent_certificates_checked"]}, indent=2))


if __name__ == "__main__":
    main()
