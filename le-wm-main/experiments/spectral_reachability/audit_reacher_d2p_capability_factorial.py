"""Evaluator-only response/residual factorial attribution; no neural inference.

Separates the two factors previously grouped as predicted-state-or-residual
disagreement. These oracle substitutions are not a deployable correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as capability
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed, predicates


def factorial_fits(predicted, actual, model_residual, physical_residual):
    predicted=np.asarray(predicted,np.float64)
    actual=np.asarray(actual,np.float64)
    if predicted.ndim!=2 or actual.shape!=predicted.shape:
        raise ValueError("aligned complete predicted/actual-history response matrices required")
    return {f"{state}_response__{anchor}_residual":capability.geometry(j,r)
        for state,j in (("predicted",predicted),("actual_history",actual))
        for anchor,r in (("model",model_residual),("physical",physical_residual))}


def control_margins(fits,greedy):
    if type(greedy) is not int or not 0<=greedy<17 or set(fits)!=set(range(17)):
        raise ValueError("complete all17 program identities and fixed greedy required")
    return {i:{variant:fit["gain"]-max(fits[c][variant]["gain"] for c in (0,greedy))
        for variant,fit in variants.items()} for i,variants in fits.items()}


def audit(report_path,reference_path):
    verified=capability.verify(report_path)
    if not verified["all_programs"]:
        raise ValueError("complete all17/all8 capability audit required")
    report=json.loads(report_path.read_text())
    predictions_path=reference_path.with_suffix(".predictions.json")
    for p in (reference_path,predictions_path):
        if report["sealed_protocol"]["input_bindings"].get(str(p.resolve()))!=capability.digest(p):
            raise ValueError("same bound source/predictions required")
    plans=indexed(json.loads(predictions_path.read_text())["rows"])
    measured=indexed(report["rows"])
    diagnoses=indexed(verified["rows"])
    if set(plans)!=set(measured) or set(plans)!=set(diagnoses):
        raise ValueError("duplicate-free equal root identities required")
    rows=[]
    solves=0
    max_gap=0.
    for seed in sorted(plans):
        tasks={}
        target=capability.finite(measured[seed]["goal_vector"],(192,))
        for task in capability.TASKS:
            plan=plans[seed]["tasks"][task]
            cells=measured[seed]["tasks"][task]["cells"]
            predicates(plan["bank"],plan["decisions"]["cem_then_gn4"])
            model_residual=capability.finite(plan["common_residual"],(192,))
            physical_residual=target-capability.finite(cells["0"]["center"]["observed"],(192,))
            fits={}
            for i in range(17):
                cell=cells[str(i)]
                n=10*len(capability.TASKS[task])
                predicted=capability.finite(cell["predicted_state_response"],(192,n))
                actual=capability.finite(cell["actual_history_response"],(192,n))
                fits[i]=factorial_fits(predicted,actual,model_residual,physical_residual)
                solves+=4
                receipt=plan["bank"][i]["common_response_gain"]
                baseline=fits[i]["predicted_response__model_residual"]
                tolerance=1e-8*max(1.,baseline["baseline_squared_residual"])
                gain=baseline["gain"]
                if (abs(baseline["baseline_squared_residual"]-receipt["baseline_squared_residual"])>tolerance
                        or gain<receipt["bounded_quadratic_gain"]-tolerance
                        or gain>receipt["bounded_quadratic_gain_upper_bound"]+tolerance):
                    raise ValueError("independent factorial baseline does not reproduce original certified QP")
                max_gap=max(max_gap,abs(gain-receipt["bounded_quadratic_gain"]))
            greedy=plan["decisions"]["cem_then_gn4"]["greedy_index"]
            margins=control_margins(fits,greedy)
            tasks[task]={"fixed_greedy_index":greedy,"programs":[{"index":i,"fits":fits[i],
                "gain_minus_control_max":margins[i],
                "original_model_gain_qualified":plan["bank"][i]["common_response_gain"]["bounded_quadratic_gain"]>
                    max(plan["bank"][c]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] for c in (0,greedy)),
                "physical_diagnostic_conjunction":diagnoses[seed]["tasks"][task]["mechanism_diagnostics"][f"program_{i}"]["all_diagnostic_checks"]}
                for i in range(17)]}
        rows.append({"seed":seed,"tasks":tasks})
    if len(rows)!=8 or solves!=1632:
        raise ValueError("all eight roots, three tasks,17 programs and four factorial combinations required")
    return {"schema":"reacher_d2p_capability_factorial.v1","rows":rows,
        "independent_factorial_fits":solves,"max_original_qp_gain_abs_difference":max_gap,
        "bindings":{str(p.resolve()):capability.digest(p) for p in (report_path,reference_path,predictions_path)},
        "implementation_sha256":capability.digest(Path(__file__)),
        "independent_capability_analyzer_sha256":capability.digest(Path(capability.__file__)),
        "P1_admission":False,"physical_labels_used_for_deployment":False,"neural_inference":False,
        "scope":"Known-panel evaluator-only factorial attribution. Physical residual and actual-history successor responses are oracle substitutions, not deployment inputs or efficacy evidence. Point-estimate gain margins do not replace original certified selector bounds."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable attribution output required")
    result=audit(args.capability,args.reference)
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__": main()
