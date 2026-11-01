"""Host-CUDA certified scoring of sealed model arrays; no physical outcomes.

This does not load a neural model or remeasure efficacy. It proves the exact
new model-only selector, keeping all original stage/terminal tests and bounds.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_root_metric_score as scoring
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, TASKS

OFFICIAL="eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol_path=args.output.with_suffix(".protocol.json")
    archive=args.output.with_suffix(".implementation")
    if any(p.exists() for p in (args.output,protocol_path,archive)):
        raise FileExistsError("new immutable certified score output and sidecars required")
    prior=json.loads(args.protocol.read_text())
    predictions=json.loads(args.predictions.read_text())
    if (predictions.get("protocol_sha256")!=digest(args.protocol)
            or prior.get("weights_sha256")!=OFFICIAL or digest(args.policy/"weights.pt")!=OFFICIAL):
        raise ValueError("sealed original model predictions and official frozen weights required")
    seeds=prior["root_seeds"]
    if (len(seeds)!=8 or any(type(s) is not int for s in seeds) or len(set(seeds))!=8
            or [r["seed"] for r in predictions["rows"]]!=seeds):
        raise ValueError("all eight distinct original ordered model roots required")
    modules={"auditor":Path(__file__),"scoring":Path(scoring.__file__),
        "qp":Path(qp.__file__),"compiler":Path(compiler.__file__)}
    bindings={str(p.resolve()):digest(p) for p in (args.predictions,args.protocol)}
    protocol={"policy":scoring.POLICY,"ridge":scoring.RIDGE,"root_seeds":seeds,
        "weights_sha256":OFFICIAL,"input_bindings":bindings,
        "implementation_sha256":{k:digest(p) for k,p in modules.items()},
        "expected_certified_qps":408,"expected_metric_svds":24,
        "new_neural_predict_samples":0,"new_vjp_rows":0,
        "original_stage_terminal_and_certified_comparison_rules_unchanged":True,
        "physical_outcomes_read":False,"P1_admission":False}
    with protocol_path.open("x") as f: json.dump(protocol,f,indent=2,allow_nan=False)
    archive.mkdir()
    for k,p in modules.items(): shutil.copyfile(p,archive/(k+".py"))
    host=shared.require_cuda_allocation("cuda")
    rows=[]
    count=0
    for row in predictions["rows"]:
        tasks={}
        if set(row["tasks"])!=set(TASKS): raise ValueError("all three original tasks required")
        for task in TASKS:
            plan=row["tasks"][task]
            projected=scoring.model_rows(plan["bank"])
            if any(len(b["response"])!=192 or any(len(v)!=10*len(TASKS[task]) for v in b["response"]) for b in projected):
                raise ValueError("all192 original output coordinates required")
            result=scoring.rescore(projected,plan["common_residual"],device="cuda")
            if result["original_decision"]!=plan["decisions"]["cem_then_gn4"]:
                raise ValueError("original selector must reproduce exactly before metric intervention")
            count+=result["certified_qps"]
            tasks[task]=result
        rows.append({"seed":row["seed"],"tasks":tasks})
        print(json.dumps({"seed":row["seed"],"model_only_decisions":{t:v["decision"] for t,v in tasks.items()}}),flush=True)
    gates={"cuda_arithmetic_only":all(t["metric"]["device"].startswith("cuda") for r in rows for t in r["tasks"].values()),
        "all8_roots":[r["seed"] for r in rows]==seeds,"all24_metric_svds":sum(len(r["tasks"]) for r in rows)==24,
        "all408_certified_qps":count==408,"all_original_selectors_exact":True,
        "inputs_unchanged":all(digest(Path(p))==sha for p,sha in bindings.items()),
        "weights_unchanged":digest(args.policy/"weights.pt")==OFFICIAL,
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items())}
    result={"schema":"reacher_d2p_root_metric_certified_score.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(protocol_path),"cuda_host":host,
        "rows":rows,"actual_certified_qps":count,"gates":gates,"P1_admission":False,
        "scope":"Known sealed model-bank score intervention only; no fresh neural inference, physical outcomes, independent calibration or efficacy. Original stage/terminal costs and all certificate comparison rules retained."}
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    if not all(gates.values()): raise RuntimeError("certified model-score integrity failed")
    print(json.dumps({"gates":gates,"actual_certified_qps":count}),flush=True)


if __name__=="__main__": main()
