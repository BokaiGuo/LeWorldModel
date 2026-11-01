"""All-root fresh paired decision consequences; saved arrays only, never P1."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability.audit_reacher_d2p_root_metric import RootMetric
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, finite, geometry, TASKS
from experiments.spectral_reachability.analyze_reacher_d2p_masked_program_pilot import branch_metrics, independent_decision
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission

REQUIRED_GATES = {"cuda_only","all8_roots","source_disjoint","all456_physical_branches",
    "all408_certified_metric_qps","all_full_split_exact","parameters_unchanged","buffers_unchanged",
    "all_modules_eval","predict_sample_accounting","all_task_compute_receipts","predictions_unchanged",
    "weights_unchanged","adapter_unchanged","reference_unchanged","implementation_unchanged","fixed_render_receipts"}


def selector(bank, sealed):
    greedy, native, witnesses = independent_decision(bank)
    if (greedy != sealed["greedy_index"] or native != sealed["native_index"]
            or witnesses != sealed["qualified_indices"]
            or bool(witnesses) != sealed["reconfiguration_witness_found"]
            or sealed["physical_labels_used"] is not False):
        raise ValueError("independent original selector disagrees with sealed decision")


def summarize(report, predictions):
    if (report.get("schema") != "reacher_d2p_root_metric_confirmation.v1" or report.get("status") != "completed"
            or not REQUIRED_GATES.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete integrity-passing host CUDA report required")
    protocol = report["sealed_protocol"]
    if (protocol["policy"] != "model-root-response-metric-v1" or protocol["ridge"] != .01
            or protocol["tasks"] != {k:list(v) for k,v in TASKS.items()}
            or protocol["goal_contract"] != "joint-feasible-v2-soft-source"
            or protocol["stage_terminal_thresholds_unchanged"] is not True
            or protocol["all_forecasts_sealed_before_future_execution"] is not True
            or len(set(protocol["excluded_source_episodes"])) < 224):
        raise ValueError("predeclared fixed intervention contract required")
    model, physical = indexed(predictions["rows"]), indexed(report["rows"])
    if len(model) != 8 or set(model) != set(physical) or set(model) != set(protocol["root_seeds"]):
        raise ValueError("complete unique8 root sets required")
    roots = indexed(protocol["source_roots"])
    if set(roots) != set(model): raise ValueError("planned source root identities disagree")
    episodes, rows, fits = set(), [], 0
    for seed in sorted(model):
        forecast, actual = model[seed], physical[seed]
        for key in ("source_episode", "source_start_row"):
            if forecast[key] != actual[key] or actual[key] != roots[seed][key]:
                raise ValueError("source root identity changed")
        episode = forecast["source_episode"]
        if episode in episodes or episode in protocol["excluded_source_episodes"]:
            raise ValueError("source episode overlap")
        episodes.add(episode)
        admission = forecast["goal_admission"]
        validate_goal_admission(protocol["goal_contract"], admission)
        target = finite(forecast["goal_vector"], (192,))
        if set(forecast["tasks"]) != set(TASKS) or set(actual["tasks"]) != set(TASKS):
            raise ValueError("all tasks required")
        tasks = {}
        for task in TASKS:
            plan = forecast["tasks"][task]
            bank, scores = plan["bank"], plan["scores"]
            if (plan["actual_predict_samples"] != 16830 or scores["certified_qps"] != 17
                    or not scores["metric"]["device"].startswith("cuda")
                    or scores["metric"]["physical_labels_used"] is not False):
                raise ValueError("complete CUDA model-only compute receipts required")
            if ([b["index"] for b in bank] != list(range(17))
                    or [b["index"] for b in scores["bank"]] != list(range(17))
                    or any(b["full_split_exact"] is not True for b in bank)):
                raise ValueError("all17 original programs required")
            selector(bank, scores["original_decision"])
            selector(scores["bank"], scores["decision"])
            width = 10*len(TASKS[task])
            metric = RootMetric(finite(bank[0]["response"], (192,width)))
            common = metric.apply(finite(plan["common_residual"], (192,)))
            for b,s in zip(bank,scores["bank"]):
                if (b["index"] != s["index"] or b["prefix_cost"] != s["prefix_cost"]
                        or b["terminal_cost"] != s["terminal_cost"]):
                    raise ValueError("intervention changed stage or terminal criterion")
                gain = geometry(metric.apply(finite(b["response"], (192,width))), common)["gain"]
                receipt = s["common_response_gain"]
                lo,hi = receipt["bounded_quadratic_gain"],receipt["bounded_quadratic_gain_upper_bound"]
                tolerance = 1e-8*max(1.,abs(gain),abs(lo),abs(hi))
                if (receipt["objective_certificate_passed"] is not True or receipt["physical_labels_used"] is not False
                        or not np.isfinite([lo,hi,gain]).all() or lo>hi or not lo-tolerance<=gain<=hi+tolerance):
                    raise ValueError("independent model-only weighted QP certificate disagrees")
                fits += 1
            observed = actual["tasks"][task]
            expected_names = {"bank_"+str(i) for i in range(17)} | set(plan["controls"])
            if set(observed) != expected_names or len(expected_names) != 19:
                raise ValueError("all19 physical action witnesses required")
            cells = {"bank_"+str(b["index"]):branch_metrics(b, observed["bank_"+str(b["index"])], target, admission, task) for b in bank}
            cells.update({name:branch_metrics(b,observed[name],target,admission,task) for name,b in plan["controls"].items()})
            identities = {"metric":"bank_"+str(scores["decision"]["native_index"]),
                "euclidean":"bank_"+str(scores["original_decision"]["native_index"]),
                "greedy":"bank_"+str(scores["decision"]["greedy_index"]),"zero":"bank_0",
                **{name:name for name in plan["controls"]}}
            methods = {name:cells[key] for name,key in identities.items()}
            checks = {}
            for name,key in (("metric","decision"),("euclidean","original_decision")):
                decision = scores[key]
                selected,zero,greedy = methods[name],methods["zero"],methods["greedy"]
                checks[name] = {"model_witness":decision["reconfiguration_witness_found"],
                    "observed_stage_worse_than_zero":selected["observed_prefix_cost"]>zero["observed_prefix_cost"]+1e-6,
                    "observed_terminal_rgb_better_than_greedy":selected["observed_terminal_cost"]<greedy["observed_terminal_cost"]-1e-6,
                    "observed_terminal_joint_better_than_greedy":selected["final_joint_distance"]<greedy["final_joint_distance"]-1e-6}
                checks[name]["selected_physical_conjunction"] = all(checks[name].values())
            tasks[task] = {"branch_identities":identities,"methods":methods,"checks":checks,"all_bank_metrics":cells}
        rows.append({"seed":seed,"source_episode":episode,"tasks":tasks})
    aggregate = {}
    for task in TASKS:
        values = [r["tasks"][task] for r in rows]
        aggregate[task] = {
            "retained_roots":len(values),
            "model_witness_roots":{name:sum(v["checks"][name]["model_witness"] for v in values) for name in ("metric","euclidean")},
            "selected_physical_conjunction_roots":{name:sum(v["checks"][name]["selected_physical_conjunction"] for v in values) for name in ("metric","euclidean")},
            "mean_final_joint_distance":{name:float(np.mean([v["methods"][name]["final_joint_distance"] for v in values])) for name in values[0]["methods"]},
            "metric_minus_control":{control:{quantity:paired_difference([v["methods"]["metric"][quantity] for v in values],
                [v["methods"][control][quantity] for v in values]) for quantity in ("observed_terminal_cost","final_joint_distance")}
                for control in ("euclidean","greedy","zero","flat_gn","flat_cem_gn_incumbent")}}
        # The shared bootstrap helper has a historical joint/known-panel label.
        # Preserve its arithmetic, but explicitly label the actual quantity/cohort.
        for contrasts in aggregate[task]["metric_minus_control"].values():
            for quantity,contrast in contrasts.items():
                contrast["metric"] = (("full192 RGB latent mean squared cost" if quantity=="observed_terminal_cost"
                    else "joint-topology-aware distance") +
                    " difference; negative favors metric; fresh source-disjoint paired descriptive comparison, not confirmatory")
    if fits != 408: raise ValueError("all408 independent model gain fits required")
    if report["actual_predict_samples"] != 16830*24: raise ValueError("total compute receipt changed")
    return {"schema":"reacher_d2p_root_metric_confirmation_analysis.v1","rows":rows,"aggregate":aggregate,
        "independent_model_gain_fits":fits,"P1_admission":False,"physical_labels_used_for_selection":False,
        "scope":"Fresh8 root-paired descriptive score intervention, not calibrated P1 or compute-matched efficacy. "
                "True RGB common gain is not probed here; physical conjunction alone cannot establish it."}


def verify_files(report_path):
    """Read-only verifier reusable before any downstream physical experiment."""
    report = json.loads(report_path.read_text())
    pp = report_path.with_suffix(".protocol.json")
    predictions_path = report_path.with_suffix(".predictions.json")
    predictions = json.loads(predictions_path.read_text())
    if (digest(pp) != report["protocol_sha256"] or json.loads(pp.read_text()) != report["sealed_protocol"]
            or predictions["protocol_sha256"] != report["protocol_sha256"] or digest(predictions_path) != report["predictions_sha256"]):
        raise ValueError("protocol/model forecast seal changed")
    for key,sha in report["sealed_protocol"]["implementation_sha256"].items():
        if digest(report_path.with_suffix(".implementation")/(key+".py")) != sha:
            raise ValueError("archived implementation changed")
    for row in report["rows"]:
        if json.loads((report_path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text()) != row:
            raise ValueError("physical root sidecar changed")
    excluded = set()
    for binding in report["sealed_protocol"]["exclusion_bindings"]:
        path = Path(binding["path"])
        if digest(path) != binding["sha256"]: raise ValueError("exclusion reference changed")
        prior = json.loads(path.read_text())
        excluded.update(r["source_episode"] for r in prior["rows"])
        excluded.update(prior["sealed_protocol"].get("excluded_source_episodes", []))
    if excluded != set(report["sealed_protocol"]["excluded_source_episodes"]):
        raise ValueError("source exclusions not reconstructed from references")
    stats = report["render_stats"]
    if (stats["policy"] != "fixed-warm-render-v1" or type(stats["requests"]) is not int or stats["requests"] < 1
            or stats["raw_render_passes"] != 2*stats["requests"] or stats["exact_state_checks"] != 2*stats["requests"]
            or stats["dither_state_changes"] != 0):
        raise ValueError("complete fixed warm render receipts required")
    result = summarize(report,predictions)
    result["bindings"] = {str(p.resolve()):digest(p) for p in (report_path,pp,predictions_path)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable fresh analysis required")
    archive = args.output.with_suffix(".implementation")
    if archive.exists(): raise FileExistsError("new immutable analyzer source archive required")
    result = verify_files(args.report)
    result["implementation_sha256"] = digest(Path(__file__))
    modules = {"analyzer":Path(__file__), "metric":Path(RootMetric.__init__.__code__.co_filename),
        "geometry":Path(geometry.__code__.co_filename), "physical_metrics":Path(branch_metrics.__code__.co_filename),
        "paired_stats":Path(paired_difference.__code__.co_filename), "goal_contract":Path(validate_goal_admission.__code__.co_filename),
        "identity_join":Path(indexed.__code__.co_filename)}
    archive.mkdir()
    result["analysis_implementation_sha256"] = {k:digest(p) for k,p in modules.items()}
    for k,p in modules.items():
        shutil.copyfile(p,archive/(k+".py"))
        if digest(archive/(k+".py")) != result["analysis_implementation_sha256"][k]:
            raise RuntimeError("analyzer archive changed during copy")
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"aggregate":result["aggregate"],"independent_model_gain_fits":result["independent_model_gain_fits"]},indent=2))


if __name__ == "__main__":
    main()
