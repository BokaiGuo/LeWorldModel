"""Fixed model-root response preconditioning; saved-array diagnosis only.

Q = .01 * (J0 J0.T + .01 I)^-1. J0 is the original model-predicted
zero-prefix suffix response, never a measured response or a fitted readout.
This is an output-coordinate conditioning candidate, not a calibrated physical
task metric, a new success definition, or a deployable P1 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as independent
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed, predicates

RIDGE=.01
POLICY="model-root-response-metric-v1"


class RootMetric:
    def __init__(self,response):
        j=np.asarray(response,np.float64)
        if j.ndim!=2 or min(j.shape)<1 or not np.isfinite(j).all():
            raise ValueError("finite complete model-root response required")
        self.u,self.s,_=np.linalg.svd(j,full_matrices=False)
        self.correction=np.sqrt(RIDGE/(self.s*self.s+RIDGE))-1
        self.rows=j.shape[0]
        self.fingerprint=hashlib.sha256(np.ascontiguousarray(j).tobytes()).hexdigest()

    def apply(self,value):
        x=np.asarray(value,np.float64)
        if x.ndim not in (1,2) or x.shape[0]!=self.rows or not np.isfinite(x).all():
            raise ValueError("finite aligned vector or response matrix required")
        projected=self.u.T@x
        adjustment=self.correction*projected if x.ndim==1 else self.correction[:,None]*projected
        return x+self.u@adjustment


def margins(values,greedy):
    v=independent.finite(values,(17,))
    if type(greedy) is not int or not 0<=greedy<17:
        raise ValueError("fixed original model greedy required")
    return (v-max(v[0],v[greedy])).tolist()


def audit(report_path,reference_path):
    checked=independent.verify(report_path)
    if not checked["all_programs"]: raise ValueError("complete all17 capability coverage required")
    report=json.loads(report_path.read_text())
    prediction_path=reference_path.with_suffix(".predictions.json")
    for p in (reference_path,prediction_path):
        if report["sealed_protocol"]["input_bindings"].get(str(p.resolve()))!=independent.digest(p):
            raise ValueError("same original bound source/predictions required")
    plans=indexed(json.loads(prediction_path.read_text())["rows"])
    measured=indexed(report["rows"])
    diagnostic=indexed(checked["rows"])
    if set(plans)!=set(measured) or set(plans)!=set(diagnostic) or len(plans)!=8:
        raise ValueError("all eight unique root identities required")
    rows=[]
    solves=0
    for seed in sorted(plans):
        tasks={}
        target=independent.finite(measured[seed]["goal_vector"],(192,))
        for task in independent.TASKS:
            plan=plans[seed]["tasks"][task]
            predicates(plan["bank"],plan["decisions"]["cem_then_gn4"])
            n=10*len(independent.TASKS[task])
            # Sole metric-construction input is the pre-existing model array.
            metric=RootMetric(independent.finite(plan["bank"][0]["response"],(192,n)))
            cells=measured[seed]["tasks"][task]["cells"]
            model_r=metric.apply(independent.finite(plan["common_residual"],(192,)))
            actual_r=metric.apply(target-independent.finite(cells["0"]["center"]["observed"],(192,)))
            fits=[]
            for i in range(17):
                model_j=metric.apply(independent.finite(plan["bank"][i]["response"],(192,n)))
                model_fit={**independent.geometry(model_j,model_r),"physical_labels_used":False}
                physical=[]
                for probe in independent.response_matrices(cells[str(i)],task,measured[seed]["goal_admission"]["limited_joints"]):
                    physical.append(independent.geometry(metric.apply(probe["matrices"]["physical_rgb_fd"]),actual_r))
                fits.append({"model":model_fit,"physical_rgb":physical})
                solves+=3
            greedy=plan["decisions"]["cem_then_gn4"]["greedy_index"]
            model_m=margins([f["model"]["gain"] for f in fits],greedy)
            physical_m=[margins([f["physical_rgb"][k]["gain"] for f in fits],greedy) for k in range(2)]
            programs=[]
            for i in range(17):
                physical_conjunction=diagnostic[seed]["tasks"][task]["mechanism_diagnostics"][f"program_{i}"]["all_diagnostic_checks"]
                programs.append({"index":i,"fits":fits[i],"model_gain_minus_fixed_controls":model_m[i],
                    "physical_rgb_gain_minus_fixed_controls":[m[i] for m in physical_m],
                    "model_positive_point_margin":model_m[i]>1e-10,
                    "physical_rgb_positive_point_margin_both_scales":all(m[i]>1e-10 for m in physical_m),
                    "original_physical_diagnostic_conjunction":physical_conjunction})
            tasks[task]={"fixed_greedy_index":greedy,"model_root_response_sha256":metric.fingerprint,
                "model_root_response_singular_values":metric.s.tolist(),"programs":programs}
        rows.append({"seed":seed,"tasks":tasks})
    if solves!=1224: raise ValueError("complete three fits per all408 programs required")
    aggregate={}
    for task in independent.TASKS:
        candidates=[p for r in rows for p in r["tasks"][task]["programs"] if p["original_physical_diagnostic_conjunction"]]
        aggregate[task]={"original_physical_diagnostic_candidates":len(candidates),
            "model_positive_point_margin":sum(p["model_positive_point_margin"] for p in candidates),
            "physical_rgb_positive_point_margin_both_scales":sum(p["physical_rgb_positive_point_margin_both_scales"] for p in candidates),
            "both_positive_point_margins":sum(p["model_positive_point_margin"] and p["physical_rgb_positive_point_margin_both_scales"] for p in candidates)}
    return {"schema":"reacher_d2p_root_metric_diagnostic.v1","policy":POLICY,"rows":rows,"aggregate":aggregate,
        "independent_metric_fits":solves,"P1_admission":False,"neural_inference":False,
        "physical_labels_used_to_construct_metric":False,"physical_labels_used_for_deployment":False,
        "point_estimates_not_selector_certificates":True,
        "scope":"All known roots/programs retained. Fixed model-only output conditioning, not a physically calibrated metric or independent efficacy. Physical probes only evaluate the candidate; no outcomes choose a root, ridge, program, or deployment action."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol_path=args.output.with_suffix(".protocol.json")
    if args.output.exists() or protocol_path.exists(): raise FileExistsError("new immutable metric diagnostic paths required")
    prediction_path=args.reference.with_suffix(".predictions.json")
    protocol={"policy":POLICY,"ridge":RIDGE,"formula":"Q=.01*(J0@J0.T+.01*I)^-1",
        "metric_source":"original model-predicted zero-prefix suffix Jacobian; no physical labels",
        "input_bindings":{str(p.resolve()):independent.digest(p) for p in (args.capability,args.reference,prediction_path)},
        "implementation_sha256":independent.digest(Path(__file__)),
        "analyzer_sha256":independent.digest(Path(independent.__file__)),"P1_admission":False}
    with protocol_path.open("x") as f: json.dump(protocol,f,indent=2,allow_nan=False)
    result=audit(args.capability,args.reference)
    result.update(protocol=protocol,protocol_sha256=independent.digest(protocol_path))
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"aggregate":result["aggregate"],"fits":result["independent_metric_fits"],"P1_admission":False},indent=2))


if __name__=="__main__": main()
