"""Saved-array audit of the full fixed-Q compiler and all physical branches.

Reconstruct spectrum/action witnesses, objective certificates, and selectors;
then join all original and new physical branches by explicit root identity.
No physical observation is used to choose a program or change a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, finite, geometry, TASKS, delta
from experiments.spectral_reachability.analyze_reacher_d2p_root_metric_confirmation import selector
from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation import verify_goal_receipt, contrast
from experiments.spectral_reachability.audit_reacher_d2p_root_metric import RootMetric
from experiments.spectral_reachability.audit_reacher_d2p_root_metric_score import OFFICIAL
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission

VARIANT = "goal_metric_cem_then_gn4"
REQUIRED_GATES = {"cuda_only","all8_roots","source_disjoint","all912_physical_branches",
    "all1536_input_vjps","all104_encoder_forwards","all_full_split_exact","parameters_unchanged",
    "buffers_unchanged","all_modules_eval","predict_sample_accounting","predictions_unchanged",
    "weights_unchanged","config_unchanged","adapter_unchanged","reference_unchanged",
    "implementation_unchanged","fixed_render_receipts"}


def difference32(left,right):
    """Reproduce declared float32 neural residual arithmetic, not a new fit."""
    return (np.asarray(left,np.float32)-np.asarray(right,np.float32)).astype(np.float64)


def verify_fit(receipt, fitted):
    values = finite([receipt["baseline_squared_residual"],receipt["bounded_quadratic_objective"],
        receipt["bounded_quadratic_gain"],receipt["bounded_quadratic_gain_upper_bound"]],(4,))
    baseline,objective,lo,hi = values
    certificate = receipt["qp_certificate"]
    gap = float(certificate["objective_duality_gap_upper_bound"])
    tolerance = 1e-8*max(1.,baseline)
    if (receipt["objective_certificate_passed"] is not True or receipt["physical_labels_used"] is not False
            or certificate["active_set_terminated"] is not True or certificate["box_feasibility_violation"] != 0
            or not str(certificate["device"]).startswith("cuda") or not np.isfinite(gap) or gap < 0 or gap > tolerance
            or lo > hi or not lo-tolerance <= fitted["gain"] <= hi+tolerance
            or not objective-gap-tolerance <= fitted["regularized_objective"] <= objective+tolerance
            or abs(baseline-fitted["baseline_squared_residual"]) > tolerance
            or abs((baseline-objective)-lo) > tolerance):
        raise ValueError("independent model QP/own objective certificate mismatch")


def verify_native(bank, decision):
    objectives = [b["own_response_gain"]["bounded_quadratic_objective"] for b in bank]
    gaps = [b["own_response_gain"]["qp_certificate"]["objective_duality_gap_upper_bound"] for b in bank]
    upper = [b["prefix_cost"]+v/192. for b,v in zip(bank,objectives)]
    lower = [b["prefix_cost"]+max(0.,v-gap)/192. for b,v,gap in zip(bank,objectives,gaps)]
    native = min(range(17),key=lambda i:(upper[i],i))
    greedy = min(range(17),key=lambda i:(bank[i]["prefix_cost"],i))
    witness = (native != 0 and bank[native]["prefix_cost"] > bank[0]["prefix_cost"]+1e-6
        and bank[native]["terminal_cost"] < bank[greedy]["terminal_cost"]-1e-6
        and objectives[native] < min(max(0.,objectives[i]-gaps[i]) for i in (0,greedy))
        and upper[native] < min(lower[i] for i in (0,greedy)))
    if (decision["native_index"] != native or decision["greedy_index"] != greedy
            or decision["reconfiguration_witness_found"] != bool(witness)
            or decision["physical_labels_used"] is not False or decision["P1_admission"] is not False
            or not np.allclose(decision["total_score_upper"],upper,rtol=1e-12,atol=1e-12)
            or not np.allclose(decision["total_score_lower"],lower,rtol=1e-12,atol=1e-12)):
        raise ValueError("independent original stage-plus-own-reach selector mismatch")


def verify_spectrum(plan,transform,task):
    width = 5*len(TASKS[task])
    response = finite(plan["root_response"],(192,width))
    weighted = transform @ response
    singular = np.linalg.svd(weighted,compute_uv=False)
    if not np.allclose(singular,plan["root_weighted_singular_values"],rtol=1e-7,atol=1e-10):
        raise ValueError("independent full goal-weighted root spectrum mismatch")
    columns = [2*i+j for i in range(5) for j in TASKS[task]]
    prefixes = [finite(b["raw_actions"],(15,2))[:5].ravel()[columns] for b in plan["bank"]]
    if np.count_nonzero(prefixes[0]):
        raise ValueError("literal zero prefix changed")
    gram = weighted.T @ weighted
    for band,amplitude in enumerate((.25,.75)):
        for direction in range(4):
            i = 1+band*8+direction*2
            plus,minus = prefixes[i:i+2]
            if not np.array_equal(plus,-minus) or not np.isclose(np.max(np.abs(plus)),amplitude,rtol=0,atol=1e-7):
                raise ValueError("complete fixed +/- amplitude action witnesses required")
            unit = plus/np.linalg.norm(plus)
            error = np.linalg.norm(gram@unit-singular[direction]**2*unit)
            if error > 1e-4*max(singular[0]**2,1e-30):
                raise ValueError("prefix is not its declared goal-weighted spectral direction")


def model_mechanism_diagnostics(bank,decision):
    """Separate absent model candidates from stage/reach tradeoff rejection.

    This evaluator never substitutes another program for the sealed selection.
    All quantities come from predictions, without physical outcome fields.
    """
    verify_native(bank,decision)
    greedy,selected = decision["greedy_index"],decision["native_index"]
    objectives = [b["own_response_gain"]["bounded_quadratic_objective"] for b in bank]
    gaps = [b["own_response_gain"]["qp_certificate"]["objective_duality_gap_upper_bound"] for b in bank]
    rows = []
    for i,b in enumerate(bank):
        checks = {"non_greedy_vs_zero":i != 0 and b["prefix_cost"] > bank[0]["prefix_cost"]+1e-6,
            "terminal_better_than_greedy":b["terminal_cost"] < bank[greedy]["terminal_cost"]-1e-6,
            "own_reach_better_than_controls":objectives[i] < min(max(0.,objectives[c]-gaps[c]) for c in (0,greedy)),
            "total_score_better_than_controls":decision["total_score_upper"][i] < min(decision["total_score_lower"][c] for c in (0,greedy))}
        rows.append({"index":i,"checks":checks,"all_model_checks":all(checks.values()),
            "stage_minus_greedy":b["prefix_cost"]-bank[greedy]["prefix_cost"],
            "own_reach_per_coordinate_minus_greedy":(objectives[i]-objectives[greedy])/192.,
            "score_upper_minus_greedy":decision["total_score_upper"][i]-decision["total_score_upper"][greedy]})
    return {"selected_is_greedy":selected == greedy,"programs":rows,
        "condition_counts":{key:sum(r["checks"][key] for r in rows) for key in rows[0]["checks"]},
        "all_model_checks_count":sum(r["all_model_checks"] for r in rows),
        "physical_labels_used":False,"selection_changed":False}


def branch_metrics(plan, observation, target, admission, task, transform, *, weighted):
    raw = finite(plan["raw_actions"],(15,2))
    inactive = [i for i in range(2) if i not in TASKS[task]]
    if (np.max(np.abs(raw)) > 1 or np.count_nonzero(raw[:,inactive])
            or not np.array_equal(raw,finite(observation["raw_actions"],(15,2)))):
        raise ValueError("literal same-action mask/bounds witness required")
    packets = observation["packets"]
    if [p["packet"] for p in packets] != [1,2,3]:
        raise ValueError("all three physical packets required")
    for i,p in enumerate(packets):
        sha = hashlib.sha256(np.ascontiguousarray(raw[i*5:(i+1)*5],dtype=np.float32).tobytes()).hexdigest()
        if sha != p["action_sha256"] or len(p["root_rgb_sha256"]) != 64:
            raise ValueError("physical action/RGB identity mismatch")
        finite(p["observed_vector"],(192,))
        finite(p["qpos"],(2,))
    predicted = finite(plan["terminal_vector"],(192,))
    predicted_cost = (float(np.mean((transform @ difference32(predicted,target))**2)) if weighted
        else float(np.mean((predicted-target)**2)))
    if not np.isclose(predicted_cost,plan["terminal_cost"],rtol=2e-6,atol=1e-7):
        raise ValueError("same full-coordinate predicted objective mismatch")
    prefix = finite(packets[0]["observed_vector"],(192,))-target
    terminal = finite(packets[-1]["observed_vector"],(192,))-target
    return {"observed_prefix_cost":float(np.mean(prefix**2)),"observed_terminal_cost":float(np.mean(terminal**2)),
        "goal_Q_observed_prefix_cost":float(np.mean((transform@prefix)**2)),
        "goal_Q_observed_terminal_cost":float(np.mean((transform@terminal)**2)),
        "prefix_joint_distance":float(np.linalg.norm(delta(admission["feasible_target_qpos"],packets[0]["qpos"],admission["limited_joints"]))),
        "final_joint_distance":float(np.linalg.norm(delta(admission["feasible_target_qpos"],packets[-1]["qpos"],admission["limited_joints"]))) }


def analyze(path):
    report = json.loads(path.read_text())
    pp,pred_path = path.with_suffix(".protocol.json"),path.with_suffix(".predictions.json")
    protocol,predictions = json.loads(pp.read_text()),json.loads(pred_path.read_text())
    if (report["schema"] != "reacher_d2p_full_goal_metric_compiler.v1" or report["status"] != "completed"
            or not report["gates"] or not all(v is True for v in report["gates"].values())
            or not REQUIRED_GATES.issubset(report["gates"])
            or protocol != report["sealed_protocol"] or digest(pp) != report["protocol_sha256"]
            or digest(pred_path) != report["predictions_sha256"] or predictions["protocol_sha256"] != digest(pp)
            or protocol["weights_sha256"] != OFFICIAL or protocol["ridge"] != .01
            or protocol["policy"] != "full-goal-input-cometric-compiler-v1"
            or protocol["metric_policy"] != "goal-encoder-input-cometric-v1"
            or protocol["expected_predict_samples_per_task"] != {"original":16830,"goal_metric":16734}
            or protocol["expected_encoder_forwards"] != 104 or protocol["expected_input_vjp_rows"] != 1536
            or protocol["target_delta"] != [.65,-.65] or protocol["physical_branches"] != 912
            or protocol["goal_contract"] != "joint-feasible-v2-soft-source"
            or protocol["tasks"] != {k:list(v) for k,v in TASKS.items()}
            or protocol["all_forecasts_sealed_before_future_execution"] is not True
            or protocol["same_metric_in_spectrum_stage_reach_and_search"] is not True
            or protocol["compute_matched"] is not False or protocol["physical_recompilation"] is not False
            or report["P1_admission"] is not False):
        raise ValueError("complete unchanged full-compiler diagnostic contract required")
    for name,sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha:
            raise ValueError("archived implementation changed")
    excluded = set()
    for binding in protocol["exclusion_bindings"]:
        source_path = Path(binding["path"])
        if digest(source_path) != binding["sha256"]: raise ValueError("exclusion reference changed")
        prior = json.loads(source_path.read_text())
        excluded.update(r["source_episode"] for r in prior["rows"])
        excluded.update(prior["sealed_protocol"].get("excluded_source_episodes",[]))
    if excluded != set(protocol["excluded_source_episodes"]) or len(excluded) < 240:
        raise ValueError("all240 previous sources must remain excluded")
    plans,actual,sources = [indexed(x) for x in (predictions["rows"],report["rows"],protocol["source_roots"])]
    if len(plans) != 8 or set(plans) != set(actual) or set(plans) != set(sources) or set(plans) != set(protocol["root_seeds"]):
        raise ValueError("complete duplicate-free eight-root join required")
    if len({r["source_episode"] for r in sources.values()}) != 8 or {r["source_episode"] for r in sources.values()}&excluded:
        raise ValueError("fresh source disjointness changed")
    rows,fits,branches,samples = [],0,0,0
    for seed in sorted(plans):
        plan,observed = plans[seed],actual[seed]
        if json.loads((path.with_suffix(".roots")/f"root_{seed}.json").read_text()) != observed:
            raise ValueError("physical root sidecar changed")
        for key in ("source_episode","source_start_row"):
            if plan[key] != sources[seed][key] or observed[key] != sources[seed][key]:
                raise ValueError("joined source root identity mismatch")
        target = finite(plan["goal_vector"],(192,))
        transform = verify_goal_receipt(plan["goal_metric"])
        validate_goal_admission(protocol["goal_contract"],plan["goal_admission"])
        if set(plan["tasks"]) != set(TASKS) or set(observed["tasks"]) != set(TASKS):
            raise ValueError("all tasks required")
        tasks = {}
        for task in TASKS:
            family = plan["tasks"][task]
            if set(family) != {"original","goal_metric"} or set(observed["tasks"][task]) != set(family):
                raise ValueError("both complete compiler families required")
            original,new = family["original"],family["goal_metric"]
            width = 10*len(TASKS[task])
            for name,method in family.items():
                if [b["index"] for b in method["bank"]] != list(range(17)) or not all(b["full_split_exact"] is True for b in method["bank"]):
                    raise ValueError("all17 split/full-equal model programs required")
                if method["actual_predict_samples"] != protocol["expected_predict_samples_per_task"][name]:
                    raise ValueError("fixed model sample budget mismatch")
                samples += method["actual_predict_samples"]
            selector(original["bank"],original["scores"]["original_decision"])
            selector(original["scores"]["bank"],original["scores"]["decision"])
            selector(new["bank"],new["historical_common_rule_decision"])
            verify_native(new["bank"],new["decision"])
            verify_spectrum(new,transform,task)
            root_metric = RootMetric(finite(original["bank"][0]["response"],(192,width)))
            residual = finite(original["common_residual"],(192,))
            for old,scored,b in zip(original["bank"],original["scores"]["bank"],new["bank"]):
                response = finite(old["response"],(192,width))
                verify_fit(old["common_response_gain"],geometry(response,residual))
                verify_fit(scored["common_response_gain"],geometry(root_metric.apply(response),root_metric.apply(residual)))
                if any(old[k] != scored[k] for k in ("index","prefix_cost","terminal_cost")):
                    raise ValueError("old root metric changed original terminal/stage costs")
                response = transform @ finite(b["response"],(192,width))
                verify_fit(b["common_response_gain"],geometry(response,transform @ finite(new["common_residual"],(192,))))
                verify_fit(b["own_response_gain"],geometry(response,transform @ difference32(target,b["own_zero_suffix_vector"])))
                stage = float(np.mean((transform @ difference32(b["prefix_vector"],target))**2))
                if not np.isclose(stage,b["prefix_cost"],rtol=1e-8,atol=1e-10):
                    raise ValueError("consistent goal-weighted stage mismatch")
                fits += 4
            measured = {}
            for name,method in family.items():
                expected = {"bank_"+str(i) for i in range(17)}|{"flat_gn","flat_cem_gn_incumbent"}
                obs = observed["tasks"][task][name]
                if set(obs) != expected: raise ValueError("all19 physical branches required for each family")
                lookup = {"bank_"+str(b["index"]):b for b in method["bank"]}|method["controls"]
                measured[name] = {key:branch_metrics(branch,obs[key],target,plan["goal_admission"],task,transform,
                    weighted=name=="goal_metric") for key,branch in lookup.items()}
                branches += len(obs)
            d,od,ed = new["decision"],original["scores"]["decision"],original["scores"]["original_decision"]
            controls = {"original_metric":measured["original"]["bank_"+str(od["native_index"])],
                "original_euclidean":measured["original"]["bank_"+str(ed["native_index"])],
                "original_greedy":measured["original"]["bank_"+str(od["greedy_index"])],
                "original_zero":measured["original"]["bank_0"],
                "own_greedy":measured["goal_metric"]["bank_"+str(d["greedy_index"])],
                "own_zero":measured["goal_metric"]["bank_0"],
                **{name+"_"+key:measured[name][key] for name in family for key in ("flat_gn","flat_cem_gn_incumbent")}}
            selected = measured["goal_metric"]["bank_"+str(d["native_index"])]
            checks = {"model_witness":d["reconfiguration_witness_found"],
                "physical_joint_stage_worse_than_own_zero":selected["prefix_joint_distance"]>controls["own_zero"]["prefix_joint_distance"]+1e-6,
                "physical_joint_terminal_better_than_own_greedy":selected["final_joint_distance"]<controls["own_greedy"]["final_joint_distance"]-1e-6,
                "raw_rgb_stage_worse_than_own_zero":selected["observed_prefix_cost"]>controls["own_zero"]["observed_prefix_cost"]+1e-6,
                "raw_rgb_terminal_better_than_own_greedy":selected["observed_terminal_cost"]<controls["own_greedy"]["observed_terminal_cost"]-1e-6}
            tasks[task] = {"decision":d,"selected":selected,"checks":checks,"all_stage_terminal_checks":all(checks.values()),
                "model_mechanism_diagnostics":model_mechanism_diagnostics(new["bank"],d),
                "controls":controls,"all_bank_metrics":measured,
                "selected_minus_controls":{name:{q:selected[q]-control[q] for q in
                    ("observed_terminal_cost","goal_Q_observed_terminal_cost","final_joint_distance")} for name,control in controls.items()}}
        rows.append({"seed":seed,"source_episode":plan["source_episode"],"tasks":tasks})
    stats = report["render_stats"]
    if (fits != 1632 or branches != 912 or samples != report["actual_predict_samples"]
            or stats["requests"] <= 0 or stats["raw_render_passes"] != 2*stats["requests"]
            or stats["exact_state_checks"] != 2*stats["requests"] or stats["dither_state_changes"] != 0):
        raise ValueError("full model/physical/render accounting mismatch")
    aggregate = {task:{"roots":8,"model_witnesses":sum(r["tasks"][task]["checks"]["model_witness"] for r in rows),
        "all_stage_terminal_checks":sum(r["tasks"][task]["all_stage_terminal_checks"] for r in rows),
        "paired_contrasts":{name:{q:contrast([r["tasks"][task]["selected_minus_controls"][name][q] for r in rows])
            for q in ("observed_terminal_cost","goal_Q_observed_terminal_cost","final_joint_distance")}
            for name in rows[0]["tasks"][task]["controls"]}} for task in TASKS}
    return {"schema":"reacher_d2p_full_goal_metric_compiler_analysis.v1","independent_qp_fits":fits,
        "physical_branches":branches,"analyzer_sha256":digest(Path(__file__)),
        "bindings":{str(p.resolve()):digest(p) for p in (path,pp,pred_path)},"rows":rows,"aggregate":aggregate,
        "P1_admission":False,"scope":"Full consistent-Q action compilation, fixed15-action execution. No real capability-gain proof, matched strong baselines or feedback replanning admission."}


def archive_analysis_sources(output):
    """Bind every directly used project helper, not just the CLI source."""
    sources = {"analyzer":Path(__file__),
        "physical_geometry":Path(geometry.__code__.co_filename),
        "baseline_selector":Path(selector.__code__.co_filename),
        "goal_receipt_and_contrasts":Path(verify_goal_receipt.__code__.co_filename),
        "root_metric":Path(sys.modules[RootMetric.__module__].__file__),
        "checkpoint_contract":Path(sys.modules["experiments.spectral_reachability.audit_reacher_d2p_root_metric_score"].__file__),
        "root_identity":Path(indexed.__code__.co_filename),
        "goal_admission":Path(validate_goal_admission.__code__.co_filename)}
    archive = output.with_suffix(".implementation")
    archive.mkdir()
    hashes = {key:digest(path) for key,path in sources.items()}
    for key,path in sources.items():
        destination = archive/(key+".py")
        shutil.copyfile(path,destination)
        if digest(destination) != hashes[key] or digest(path) != hashes[key]:
            raise RuntimeError("analysis helper changed while archiving")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if any(p.exists() for p in (args.output,args.output.with_suffix(".analyzer.py"),args.output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable full compiler analysis required")
    result = analyze(args.report)
    result["implementation_sha256"] = archive_analysis_sources(args.output)
    shutil.copyfile(__file__,args.output.with_suffix(".analyzer.py"))
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({t:{"witnesses":v["model_witnesses"],"stage_terminal":v["all_stage_terminal_checks"],
        "joint_vs_original_greedy":v["paired_contrasts"]["original_greedy"]["final_joint_distance"]} for t,v in result["aggregate"].items()},indent=2))


if __name__ == "__main__": main()
