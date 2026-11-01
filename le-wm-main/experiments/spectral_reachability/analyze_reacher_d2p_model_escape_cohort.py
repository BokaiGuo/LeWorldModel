"""Independent source, action, cost, plateau and rank checks for model admission."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from experiments.spectral_reachability import reacher_d2p_model_escape_admission as admission
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest,finite
from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation import verify_goal_receipt
from experiments.spectral_reachability.analyze_reacher_d2p_goal_encoder_metric import square_root
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability.reacher_d2p_masked_action_chart import TASKS
from experiments.spectral_reachability.reacher_d2p_packet_support import receipt_source

GATES = {"cuda_only","all32_roots","all3_tasks","source_disjoint","predict_accounting",
    "encoder_accounting","action_vjp_accounting","parameters_unchanged","buffers_unchanged",
    "all_modules_eval","weights_unchanged","config_unchanged","adapter_unchanged",
    "reference_unchanged","implementation_unchanged"}


def check_task(task,plan,weighted_goal):
    costs,actions = {},{}
    columns = [2*i+j for i in range(15) for j in TASKS[task]]
    inactive = [i for i in range(2) if i not in TASKS[task]]
    for name in ("zero","gn4","gn8","gn16","cem"):
        item = plan[name]
        vector,raw = finite(item["weighted_vector"],(192,)),finite(item["raw_actions"],(15,2))
        if np.max(np.abs(raw)) > 1 or np.count_nonzero(raw[:,inactive]):
            raise ValueError("all model actions must obey literal torque mask and bounds")
        if name == "zero" and np.count_nonzero(raw):
            raise ValueError("zero model reference must be literal raw zero")
        cost = float(np.mean((vector-weighted_goal)**2))
        if not np.isclose(cost,item["cost"],rtol=1e-10,atol=1e-12):
            raise ValueError("independent full-coordinate model cost differs")
        costs[name],actions[name] = cost,raw.ravel()[columns]
    ordered = [costs[k] for k in ("zero","gn4","gn8","gn16","cem")]
    if any(b>a+1e-12 for a,b in zip(ordered,ordered[1:])):
        raise ValueError("model optimizer lost a previous incumbent")
    previous = costs["zero"]
    for name,iterations in (("gn4",4),("gn8",4),("gn16",8)):
        receipt = plan["gn_receipts"][name]
        if receipt["iterations"] != iterations or receipt["physical_labels_used"] is not False or len(receipt["trace"]) != iterations:
            raise ValueError("complete fixed4/4/8 GN trace required")
        for i,step in enumerate(receipt["trace"]):
            if (step["iteration"] != i+1 or not np.isclose(previous,step["cost_before"],rtol=1e-10,atol=1e-12)
                    or not np.isfinite(step["cost_after"]) or step["cost_after"] > step["cost_before"]
                    or step["accepted_scale"] not in (0.,1.,.5,.25,.125,.0625,.03125)):
                raise ValueError("GN phase continuity/incumbent trace changed")
            cert = step["qp_certificate"]
            if (cert["active_set_terminated"] is not True or cert["box_feasibility_violation"] != 0
                    or not str(cert["device"]).startswith("cuda") or cert["objective_duality_gap_upper_bound"] < 0
                    or not np.isfinite(cert["objective_duality_gap_upper_bound"])):
                raise ValueError("CUDA finite affine-QP receipt required")
            previous = step["cost_after"]
        if not np.isclose(previous,costs[name],rtol=1e-10,atol=1e-12):
            raise ValueError("GN checkpoint is not its last traced incumbent")
    cem = plan["cem_receipt"]
    if (cem["samples"] != 128 or cem["steps"] != 3 or cem["elite"] != 16
            or cem["flat_gn_incumbent"] is not True or cem["singleton_incumbent_guard"] is not True
            or cem["physical_labels_used"] is not False or plan["physical_future_labels_read"] is not False
            or plan["actual_predict_samples"] != 1554 or plan["logical_action_vjp_rows"] != 3073):
        raise ValueError("fixed complete CEM/compute contract required")
    action = actions["gn16"].astype(np.float32)
    gradient = finite(plan["gn16_gradient"],action.shape).astype(np.float32)
    expected_norm = float(np.max(np.abs(action-np.clip(action-gradient,-1,1))))
    if not np.isclose(expected_norm,plan["gn16_projected_gradient_inf"],rtol=1e-6,atol=1e-7):
        raise ValueError("GN16 projected scalar-gradient receipt mismatch")
    plateau,escape = costs["gn8"]-costs["gn16"],costs["gn16"]-costs["cem"]
    eligible = costs["zero"] > 0 and plateau <= 1e-6 and escape > 1e-6
    return {"gn8_to_gn16_improvement":plateau,"gn16_to_cem_improvement":escape,
        "finite_budget_plateau":plateau <= 1e-6,"sampling_escape":escape > 1e-6,"eligible":bool(eligible),
        "normalized_escape":escape/costs["zero"] if costs["zero"] > 0 else 0.,
        "remaining_gn16_fraction":costs["gn16"]/costs["zero"] if costs["zero"] > 0 else 0.}


def audit(path,adapter):
    report = json.loads(path.read_text())
    pp = path.with_suffix(".protocol.json")
    protocol = report["sealed_protocol"]
    if (report["schema"] != "reacher_d2p_model_escape_cohort.v1" or report["status"] != "completed"
            or set(report["gates"]) != GATES or any(v is not True for v in report["gates"].values())
            or report["P1_admission"] is not False or report["physical_future_branches"] != 0
            or json.loads(pp.read_text()) != protocol or digest(pp) != report["protocol_sha256"]
            or protocol["contract"] != admission.CONTRACT or protocol["physical_future_branches"] != 0
            or protocol["selection_uses_only_model_predictions"] is not True
            or protocol["goal_contract"] != "joint-feasible-v2-soft-source" or protocol["target_delta"] != [.65,-.65]
            or protocol["tasks"] != {k:list(v) for k,v in TASKS.items()}
            or digest(adapter) != protocol["adapter_sha256"]):
        raise ValueError("complete frozen32-root model-only cohort contract required")
    for key,sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(key+".py")) != sha:
            raise ValueError("archived model-admission implementation changed")
    excluded = set()
    for binding in protocol["exclusion_bindings"]:
        ref = Path(binding["path"])
        if digest(ref) != binding["sha256"]: raise ValueError("source exclusion reference changed")
        old = json.loads(ref.read_text())
        excluded.update(r["source_episode"] for r in old["rows"])
        excluded.update(old["sealed_protocol"]["excluded_source_episodes"])
    if excluded != set(protocol["excluded_source_episodes"]) or len(excluded) < 248:
        raise ValueError("all prior248 source episodes must remain excluded")
    dataset,source_sha = receipt_source(adapter)
    if source_sha != protocol["source_sha256_receipt"]:
        raise ValueError("source receipt changed; never rehash huge source in memory")
    with h5py.File(dataset,"r",swmr=True) as handle:
        lengths,offsets = np.asarray(handle["ep_len"],np.int64),np.asarray(handle["ep_offset"],np.int64)
    source_pool = protocol["source_pool"]
    # Independently reconstruct the fixed metadata scan rather than call the
    # writer's admission function or trust its stored rejection reasons.
    accepted,rejected,seen = [],[],set()
    eligible_episodes = np.flatnonzero(lengths >= 20)
    for seed in range(source_pool["first_seed"],source_pool["first_seed"]+128):
        rng = np.random.default_rng(seed ^ 0x5A7E)
        episode = int(eligible_episodes[int(rng.integers(len(eligible_episodes)))])
        start = int(offsets[episode])+int(rng.integers(0,int(lengths[episode])-20+1))
        row = {"seed":seed,"source_episode":episode,"source_start_row":start}
        if episode in excluded or episode in seen:
            rejected.append({**row,"reason":"previous_source_episode" if episode in excluded else "pool_source_duplicate"})
        else:
            accepted.append(row)
            seen.add(episode)
            if len(accepted) == 32: break
    if (source_pool["roots"] != accepted or source_pool["metadata_rejections"] != rejected
            or source_pool["scan_limit"] != 128 or source_pool["source_values_or_outcomes_used"] is not False):
        raise ValueError("fixed metadata-only source scan changed")
    rows = indexed(report["rows"])
    if len(rows) != 32 or set(rows) != {r["seed"] for r in accepted}:
        raise ValueError("complete32-root duplicate-free source join required")
    ranked,samples,vjp,encoder,input_vjp = [],0,0,0,0
    for source_row in accepted:
        seed = source_row["seed"]
        row = rows[seed]
        if (json.loads((path.with_suffix(".roots")/f"root_{seed}.json").read_text()) != row
                or any(row[k] != source_row[k] for k in source_row) or set(row["tasks"]) != set(TASKS)):
            raise ValueError("all initial-root sidecars and model tasks must agree")
        transform = verify_goal_receipt(row["goal_metric"])
        target = transform @ finite(row["goal_vector"],(192,))
        evidence = {task:check_task(task,row["tasks"][task],target) for task in TASKS}
        active = [evidence[t]["normalized_escape"] if evidence[t]["eligible"] else 0. for t in ("shoulder_only","wrist_only")]
        ranked.append({"seed":seed,"source_episode":source_row["source_episode"],"eligible":any(v>0 for v in active),
            "rank_max":max(active),"rank_min":min(active),"tasks":evidence})
        samples += sum(t["actual_predict_samples"] for t in row["tasks"].values())
        vjp += sum(t["logical_action_vjp_rows"] for t in row["tasks"].values())
        encoder += row["goal_metric"]["encoder_forwards"]
        input_vjp += row["goal_metric"]["input_vjp_rows"]
    ranked.sort(key=lambda r:(-r["rank_max"],-r["rank_min"],r["seed"]))
    model_eligible = [r for r in ranked if r["eligible"]]
    selected = [r["seed"] for r in model_eligible[:8]] if len(model_eligible) >= 8 else []
    stored = report["selection"]
    if (stored["eligible_count"] != len(model_eligible) or stored["selected_seeds"] != selected
            or stored["status"] != ("model_cohort_admitted" if selected else "insufficient_model_eligible_roots")
            or stored["task_retention"] != list(TASKS) or stored["P1_admission"] is not False
            or stored["physical_future_labels_read"] is not False):
        raise ValueError("independent model-only plateau/escape ranking differs")
    for actual,saved in zip(ranked,stored["ranked_pool"]):
        if any(actual[k] != saved[k] for k in ("seed","source_episode","eligible")):
            raise ValueError("independent model-only root order differs")
        for task in TASKS:
            for key,value in actual["tasks"][task].items():
                other = saved["tasks"][task][key]
                if (type(value) is bool and value is not other) or (type(value) is not bool and not np.isclose(value,other,rtol=1e-9,atol=1e-12)):
                    raise ValueError("independent plateau/escape task predicate differs")
    if (samples != 149184 or report["actual_predict_samples"] != samples or vjp != 295008
            or encoder != 416 or input_vjp != 6144):
        raise ValueError("whole-pool neural work accounting differs")
    return {"schema":"reacher_d2p_model_escape_cohort_analysis.v1","eligible_count":len(model_eligible),
        "selected_seeds":selected,"ranked_pool":ranked,"predict_samples":samples,"logical_action_vjp_rows":vjp,
        "input_vjp_rows":input_vjp,"encoder_forwards":encoder,"physical_future_branches":0,
        "aggregate":{task:{"model_plateau_roots":sum(r["tasks"][task]["finite_budget_plateau"] for r in ranked),
            "model_sampling_escape_roots":sum(r["tasks"][task]["sampling_escape"] for r in ranked),
            "eligible_roots":sum(r["tasks"][task]["eligible"] for r in ranked)} for task in TASKS},
        "bindings":{str(p.resolve()):digest(p) for p in (path,pp,adapter)},"P1_admission":False,
        "scope":"Independent full32-root model-only source/rank/cost check. Not physical efficacy, mathematical stationarity or P1 admission. Does not infer optimality of nonlinear search from affine QP receipts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    archive = args.output.with_suffix(".implementation")
    if args.output.exists() or archive.exists(): raise FileExistsError("new immutable cohort analysis required")
    result = audit(args.report,args.adapter_artifact)
    modules = {"analyzer":Path(__file__),"admission_contract":Path(admission.__file__),
        "array_geometry":Path(finite.__code__.co_filename),"goal_metric_receipt":Path(verify_goal_receipt.__code__.co_filename),
        "goal_metric_square_root":Path(square_root.__code__.co_filename),"source_receipt":Path(receipt_source.__code__.co_filename),
        "root_identity":Path(indexed.__code__.co_filename)}
    archive.mkdir()
    result["implementation_sha256"] = {k:digest(p) for k,p in modules.items()}
    for key,path in modules.items():
        shutil.copyfile(path,archive/(key+".py"))
        if digest(archive/(key+".py")) != result["implementation_sha256"][key]: raise RuntimeError("analysis archive changed")
    with args.output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("eligible_count","selected_seeds","aggregate")},indent=2))


if __name__ == "__main__": main()
