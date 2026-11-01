"""Independent full-response SciPy and same-prefix physical audit.

Only numerical/postprocessing work; no neural CPU fallback. Common-residual
diagnostic and proposal-native own-residual objective stay separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

from experiments.spectral_reachability.analyze_reacher_d2p_masked_task_preflight import finite, TASKS, columns
from experiments.spectral_reachability.analyze_reacher_d2p_masked_program_pilot import branch_metrics, independent_decision
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission
from experiments.spectral_reachability.reacher_d2p_proposal_native_selection import CONTRACT as PROPOSAL_CONTRACT

VARIANTS = ("old_gn4", "cem_incumbent", "cem_then_gn4")
REQUIRED_GATES = {"cuda_only","all8_roots","all_tasks_and1224_branches","all408_original_physical_replays_exact",
    "all_model_incumbents_preserved","all_full_split_exact","all_full_responses_saved","predict_sample_accounting",
    "all_task_compute_receipts","parameters_unchanged","buffers_unchanged","all_modules_eval","predictions_unchanged",
    "reference_unchanged","weights_file_unchanged","implementation_unchanged"}


def verify_gain(receipt,response,residual):
    n = response.shape[1]
    matrix = np.concatenate((response,.1*np.eye(n)))
    target = np.concatenate((residual,np.zeros(n)))
    fit = lsq_linear(matrix,target,bounds=(-1,1),tol=1e-12,max_iter=1000,lsq_solver="exact")
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("independent full-response bounded solve failed")
    baseline = float(residual@residual)
    objective = float(np.sum((matrix@fit.x-target)**2))
    gain = baseline-objective
    if (receipt.get("objective_certificate_passed") is not True or receipt.get("physical_labels_used") is not False
            or not np.isclose(gain,receipt["bounded_quadratic_gain"],rtol=1e-8,atol=1e-8)
            or not np.isclose(objective,receipt["bounded_quadratic_objective"],rtol=1e-8,atol=1e-8)
            or not np.isclose(baseline,receipt["baseline_squared_residual"],rtol=1e-10,atol=1e-10)
            or not np.allclose(np.linalg.svd(response,compute_uv=False),receipt["response_singular_values"],rtol=1e-8,atol=1e-8)):
        raise ValueError("stored CUDA geometry differs from independent full-response solve")
    certificate = receipt["qp_certificate"]
    gap = certificate["objective_duality_gap_upper_bound"]
    if (certificate["active_set_terminated"] is not True or certificate["box_feasibility_violation"] != 0
            or not certificate["device"].startswith("cuda")
            or not np.isfinite(gap) or gap<0 or gap > 1e-8*max(1.,baseline)
            or ("bounded_quadratic_gain_upper_bound" in receipt and not np.isclose(
                receipt["bounded_quadratic_gain_upper_bound"],receipt["bounded_quadratic_gain"]+gap,rtol=1e-10,atol=1e-10))):
        raise ValueError("strict CUDA QP certificate required")
    return {"bounded_gain":gain,"bounded_objective":objective,"baseline_squared_residual":baseline,
        "remaining_fraction":float(np.sum((residual-response@fit.x)**2)/max(baseline,1e-12)),
        "scope":"Frozen-model local linearization only, not physical control-gain evidence."}


def roots(rows,seeds):
    indexed = {r["seed"]:r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!=set(seeds):
        raise ValueError("duplicate-free root identity set equality required")
    return indexed


def independent_proposal_decision(bank,variant):
    """Separate scalar implementation; full own-state QPs checked beforehand."""
    values = [b["own_response_gain"]["bounded_quadratic_objective"] for b in bank]
    gaps = [b["own_response_gain"]["qp_certificate"]["objective_duality_gap_upper_bound"] for b in bank]
    upper = [b["prefix_cost"]+values[i]/192 for i,b in enumerate(bank)]
    lower = [b["prefix_cost"]+max(0.,values[i]-gaps[i])/192 for i,b in enumerate(bank)]
    native = min(range(17),key=lambda i:(upper[i],i))
    greedy = min(range(17),key=lambda i:(bank[i]["prefix_cost"],i))
    witness = (native != 0 and bank[native]["prefix_cost"]-bank[0]["prefix_cost"]>1e-6
        and bank[greedy]["variants"][variant]["terminal_cost"]-bank[native]["variants"][variant]["terminal_cost"]>1e-6
        and all(values[native]<max(0.,values[i]-gaps[i]) and upper[native]<lower[i] for i in (0,greedy)))
    return {"native_index":native,"greedy_index":greedy,"total_score_upper":upper,"total_score_lower":lower,
        "reconfiguration_witness_found":bool(witness),"physical_labels_used":False,"P1_admission":False,
        "contract":"reacher_d2p_proposal_native_selection.v1"}


def summarize(report,predictions,reference,reference_predictions):
    if (report.get("schema")!="reacher_d2p_masked_suffix_search.v1" or report.get("status")!="completed"
            or not REQUIRED_GATES.issubset(report.get("gates",{})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("strict completed same-prefix host CUDA receipt required")
    if (reference.get("schema")!="reacher_d2p_masked_program_pilot.v1" or reference.get("status")!="completed"
            or not reference.get("gates") or not all(v is True for v in reference["gates"].values())):
        raise ValueError("complete original baseline report required")
    protocol = report["sealed_protocol"]
    proposal_enabled = "proposal_native_contract" in protocol
    goal_contract = protocol.get("goal_contract","joint-feasible-v1")
    if goal_contract != reference["sealed_protocol"].get("goal_contract","joint-feasible-v1"):
        raise ValueError("reference goal contract must remain unchanged")
    if proposal_enabled and protocol["proposal_native_contract"] != PROPOSAL_CONTRACT:
        raise ValueError("explicit sealed proposal-native formula required")
    if (protocol["tasks"]!={k:list(v) for k,v in TASKS.items()} or protocol["variants"]!=list(VARIANTS)
            or protocol["prefix_packets"]!=1 or protocol["future_packets"]!=2 or protocol["bank_count"]!=17
            or protocol["cem"]!={"samples":128,"steps":3,"elite":16,"seed_law":"root seed + 100000 + prefix index","initial":"same fixed old GN4 suffix"}
            or protocol["gn_after_cem_iterations"]!=4 or protocol["damping"]!=.01
            or any(protocol[k] is not True for k in ("prefixes_goals_thresholds_unchanged","all_predictions_sealed_before_any_future_execution","no_post_training"))
            or protocol["compute_matched"] is not False):
        raise ValueError("unchanged same-prefix intervention protocol required")
    seeds = protocol["root_seeds"]
    if len(seeds)!=8 or len(set(seeds))!=8 or seeds!=reference["sealed_protocol"]["root_seeds"]:
        raise ValueError("the same eight complete original roots required")
    actual,forecast,old_actual,old_forecast = [roots(r,seeds) for r in
        (report["rows"],predictions["rows"],reference["rows"],reference_predictions["rows"])]
    if protocol.get("source_disjoint_confirmation") is True:
        excluded = protocol["excluded_source_episodes"]
        episodes = {r["source_episode"] for r in forecast.values()}
        if (excluded != reference["sealed_protocol"]["excluded_source_episodes"] or len(set(excluded))<208
                or len(episodes)!=8 or episodes & set(excluded)):
            raise ValueError("complete fresh episode-disjoint confirmation required")
    output,solves = [],0
    for seed in sorted(seeds):
        p,m,old,old_m = forecast[seed],actual[seed],old_forecast[seed],old_actual[seed]
        if any(p[k]!=old[k] or m[k]!=old[k] or old_m[k]!=old[k] for k in ("source_episode","source_start_row")):
            raise ValueError("same source metadata required")
        if any(p[k]!=old[k] for k in ("goal_vector","root_rgb_sha256","goal_rgb_sha256","goal_admission","initial_snapshot")):
            raise ValueError("same root, goal, full original history/state required")
        target = finite(p["goal_vector"],(192,))
        admission = p["goal_admission"]
        validate_goal_admission(goal_contract,admission)
        if set(p["tasks"])!=set(TASKS) or set(m["tasks"])!=set(TASKS):
            raise ValueError("all fixed tasks required")
        tasks = {}
        for task in TASKS:
            plan,observations = p["tasks"][task],m["tasks"][task]
            old_plan = old["tasks"][task]
            bank = plan["bank"]
            if len(bank)!=17 or [b["index"] for b in bank]!=list(range(17)) or set(observations)!={"bank_"+str(i) for i in range(17)}:
                raise ValueError("all original prefix indices retained")
            if plan["actual_predict_samples"]!=14555:
                raise ValueError("actual per-task neural accounting changed")
            common = finite(plan["common_residual"],(192,))
            if plan["common_residual"]!=old_plan["common_model_residual"]:
                raise ValueError("fixed common residual changed")
            details = []
            for b in bank:
                i = b["index"]
                old_b = old_plan["bank"][i]
                if b["prefix_vector"]!=old_b["prefix_vector"] or b["prefix_cost"]!=old_b["prefix_cost"]:
                    raise ValueError("original prefix model prediction/cost changed")
                response = finite(b["response"],(192,len(columns(task,2))))
                own = finite(b["own_residual"],(192,))
                zero = finite(b["zero_suffix_vector"],(192,))
                expected_own = (np.asarray(p["goal_vector"],np.float32)-np.asarray(b["zero_suffix_vector"],np.float32)).astype(np.float64)
                if not np.array_equal(own,expected_own):
                    raise ValueError("state-specific residual is not goal minus this state's zero suffix")
                geometry = {"common":verify_gain(b["common_response_gain"],response,common),"own":verify_gain(b["own_response_gain"],response,own)}
                solves += 2
                if set(b["variants"])!=set(VARIANTS) or set(observations["bank_"+str(i)])!=set(VARIANTS):
                    raise ValueError("every suffix variant must be executed")
                original = b["variants"]["old_gn4"]
                if any(original[k]!=old_b[k] for k in ("raw_actions","terminal_vector","terminal_cost","full_split_exact")):
                    raise ValueError("original complete model witness changed")
                if observations["bank_"+str(i)]["old_gn4"]!=old_m["tasks"][task]["bank_"+str(i)]:
                    raise ValueError("full old physical branch does not reproduce exactly")
                costs,metrics = [],{}
                for name in VARIANTS:
                    variant = b["variants"][name]
                    if variant["raw_actions"][:5]!=old_b["raw_actions"][:5] or variant["full_split_exact"] is not True:
                        raise ValueError("prefix action changed or full/split path differs")
                    metrics[name] = branch_metrics(variant,observations["bank_"+str(i)][name],target,admission,task)
                    costs.append(variant["terminal_cost"])
                if costs!=sorted(costs,reverse=True):
                    raise ValueError("model incumbent worsened")
                cem,gn = b["cem_receipt"],b["gn_receipt"]
                if (cem["seed"]!=seed+100000+i or (cem["samples"],cem["steps"],cem["elite"])!=(128,3,16)
                        or cem["model_terminal_sample_evaluations"]!=386 or cem["physical_labels_used"] is not False
                        or gn["physical_labels_used"] is not False or len(gn["trace"])!=4
                        or any(s["cost_after"]>s["cost_before"] for s in gn["trace"])):
                    raise ValueError("fixed model-only suffix-search receipts required")
                details.append({"index":i,"geometry":geometry,"prefix_cost":b["prefix_cost"],"methods":metrics,
                    "own_vs_common_residual_l2":float(np.linalg.norm(own-common))})
            decisions,selected = {},{}
            for name in VARIANTS:
                rows = [{"index":b["index"],"prefix_cost":b["prefix_cost"],"terminal_cost":b["variants"][name]["terminal_cost"],
                    "common_response_gain":b["common_response_gain"]} for b in bank]
                greedy,native,qualified = independent_decision(rows)
                d = plan["decisions"][name]
                if (d["greedy_index"]!=greedy or d["native_index"]!=native or d["qualified_indices"]!=qualified
                        or d["reconfiguration_witness_found"] is not bool(qualified) or d["physical_labels_used"] is not False):
                    raise ValueError("independent unchanged selector decision mismatch")
                decisions[name] = d
                selected[name] = {"native":details[native]["methods"][name],"greedy":details[greedy]["methods"][name]}
            if decisions["old_gn4"]!=old_plan["decision"]:
                raise ValueError("original decision not exactly reproduced")
            # Counterfactual objective semantics only, never used to select a deployed branch.
            best_own = min(range(17),key=lambda i:(details[i]["geometry"]["own"]["bounded_objective"],i))
            best_common = max(range(17),key=lambda i:(details[i]["geometry"]["common"]["bounded_gain"],-i))
            tasks[task] = {"bank":details,"decisions":decisions,"selected":selected,
                "own_linear_objective_best_index":best_own,"common_linear_gain_best_index":best_common,
                "diagnostic_objectives_disagree":best_own!=best_common}
            if proposal_enabled:
                if set(plan.get("proposal_native_decisions",{})) != set(VARIANTS):
                    raise ValueError("all proposal-native decisions must be presealed")
                pd,ps = {},{}
                for name in VARIANTS:
                    d = independent_proposal_decision(bank,name)
                    if plan["proposal_native_decisions"][name] != d:
                        raise ValueError("proposal-native sealed selection differs from independent formula")
                    pd[name] = d
                    ps[name] = {"native":details[d["native_index"]]["methods"][name],
                        "greedy":details[d["greedy_index"]]["methods"][name]}
                tasks[task]["proposal_native"] = {"decisions":pd,"selected":ps}
            elif "proposal_native_decisions" in plan:
                raise ValueError("unsealed proposal-native selection must not be reported")
        output.append({"seed":seed,"tasks":tasks})
    if solves!=816 or report["physical_branches"]!=1224 or report["actual_model_predict_samples"]!=349320:
        raise ValueError("all 816 geometry solves,1224 physical branches and349320 CUDA samples required")
    aggregate = {}
    for task in TASKS:
        cells = [r["tasks"][task] for r in output]
        all_branch = [b for c in cells for b in c["bank"]]
        comparisons = {}
        for name in VARIANTS[1:]:
            contrast = paired_difference([np.mean([b["methods"][name]["final_joint_distance"] for b in c["bank"]]) for c in cells],
                [np.mean([b["methods"]["old_gn4"]["final_joint_distance"] for b in c["bank"]]) for c in cells])
            panel = "fresh source-disjoint paired diagnostic" if protocol.get("source_disjoint_confirmation") is True else "known-root diagnostic"
            contrast["metric"] = "Per-root mean over all17 fixed prefixes: joint-topology-aware distance difference; negative favors intervention; descriptive "+panel
            comparisons[name] = contrast
        aggregate[task] = {"model_witness_roots":{v:sum(bool(c["decisions"][v]["qualified_indices"]) for c in cells) for v in VARIANTS},
            "common_witness_actual_non_greedy_roots":{v:sum(bool(c["decisions"][v]["qualified_indices"]) and
                c["selected"][v]["native"]["observed_prefix_cost"]>c["bank"][0]["methods"][v]["observed_prefix_cost"]+1e-6 for c in cells) for v in VARIANTS},
            "common_witness_actual_non_greedy_and_terminal_rgb_better_roots":{v:sum(bool(c["decisions"][v]["qualified_indices"]) and
                c["selected"][v]["native"]["observed_prefix_cost"]>c["bank"][0]["methods"][v]["observed_prefix_cost"]+1e-6 and
                c["selected"][v]["native"]["observed_terminal_cost"]<c["selected"][v]["greedy"]["observed_terminal_cost"]-1e-6 for c in cells) for v in VARIANTS},
            "mean_all_prefix_final_joint_distance":{v:float(np.mean([b["methods"][v]["final_joint_distance"] for b in all_branch])) for v in VARIANTS},
            "mean_selected_final_joint_distance":{v:float(np.mean([c["selected"][v]["native"]["final_joint_distance"] for c in cells])) for v in VARIANTS},
            "same_prefix_model_improvement_count":{v:sum(b["methods"][v]["predicted_terminal_cost"]<b["methods"]["old_gn4"]["predicted_terminal_cost"]-1e-7 for b in all_branch) for v in VARIANTS[1:]},
            "same_prefix_physical_distance_improvement_count":{v:sum(b["methods"][v]["final_joint_distance"]<b["methods"]["old_gn4"]["final_joint_distance"]-1e-8 for b in all_branch) for v in VARIANTS[1:]},
            "descriptive_paired_root_comparisons":comparisons,
            "own_and_common_objective_best_index_disagree_roots":sum(c["diagnostic_objectives_disagree"] for c in cells)}
        if proposal_enabled:
            aggregate[task]["proposal_native"] = {
                "model_witness_roots":{v:sum(c["proposal_native"]["decisions"][v]["reconfiguration_witness_found"] for c in cells) for v in VARIANTS},
                "mean_selected_final_joint_distance":{v:float(np.mean([c["proposal_native"]["selected"][v]["native"]["final_joint_distance"] for c in cells])) for v in VARIANTS},
                "decisions_changed_vs_common":{v:sum(c["proposal_native"]["decisions"][v]["native_index"]!=c["decisions"][v]["native_index"] for c in cells) for v in VARIANTS},
                "actual_non_greedy_roots":{v:sum(c["proposal_native"]["selected"][v]["native"]["observed_prefix_cost"]>c["bank"][0]["methods"][v]["observed_prefix_cost"]+1e-6 for c in cells) for v in VARIANTS},
                "witness_actual_non_greedy_and_terminal_rgb_better_roots":{v:sum(c["proposal_native"]["decisions"][v]["reconfiguration_witness_found"] and
                    c["proposal_native"]["selected"][v]["native"]["observed_prefix_cost"]>c["bank"][0]["methods"][v]["observed_prefix_cost"]+1e-6 and
                    c["proposal_native"]["selected"][v]["native"]["observed_terminal_cost"]<c["proposal_native"]["selected"][v]["greedy"]["observed_terminal_cost"]-1e-6 for c in cells) for v in VARIANTS},
                "paired_vs_same_suffix_greedy":{v:paired_difference([c["proposal_native"]["selected"][v]["native"]["final_joint_distance"] for c in cells],
                    [c["proposal_native"]["selected"][v]["greedy"]["final_joint_distance"] for c in cells]) for v in VARIANTS}}
            for contrast in aggregate[task]["proposal_native"]["paired_vs_same_suffix_greedy"].values():
                contrast["metric"] = "Joint-topology-aware distance difference; negative favors proposal-native; descriptive paired diagnostic, not formal efficacy"
    return {"root_seeds":sorted(seeds),"rows":output,"aggregate":aggregate,"independent_scipy_geometry_solves":solves,
        "proposal_native_presealed":proposal_enabled,"source_disjoint_confirmation":protocol.get("source_disjoint_confirmation") is True,
        "physical_branches":1224,"actual_model_predict_samples":349320,"P1_admission":False,"scope":protocol["scope"]}


def verify_files(path):
    report = json.loads(path.read_text())
    protocol = report["sealed_protocol"]
    pp = path.with_suffix(".predictions.json")
    predictions = json.loads(pp.read_text())
    if (digest(path.with_suffix(".protocol.json"))!=report["protocol_sha256"] or json.loads(path.with_suffix(".protocol.json").read_text())!=protocol
            or digest(pp)!=report["predictions_sha256"] or predictions["protocol_sha256"]!=report["protocol_sha256"]):
        raise ValueError("immutable prediction/protocol seals changed")
    ref_path = Path(protocol["reference_path"])
    reference = json.loads(ref_path.read_text())
    old_prediction_path = ref_path.with_suffix(".predictions.json")
    old_predictions = json.loads(old_prediction_path.read_text())
    if digest(ref_path)!=protocol["reference_sha256"] or digest(old_prediction_path)!=protocol["reference_prediction_sha256"]:
        raise ValueError("original model/outcome references changed")
    if (digest(ref_path.with_suffix(".protocol.json"))!=reference["protocol_sha256"]
            or digest(old_prediction_path)!=reference["predictions_sha256"] or old_predictions["protocol_sha256"]!=reference["protocol_sha256"]):
        raise ValueError("original baseline seals changed")
    for k,sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(k+".py"))!=sha:
            raise ValueError("immutable implementation copy changed")
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text())!=row:
            raise ValueError("physical root sidecar changed")
    return {"report_sha256":digest(path),**summarize(report,predictions,reference,old_predictions)}


def plot(summary,output):
    # All roots, tasks and prefixes; figure-generation line-template style.
    plt.rcParams.update({"font.size":10,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
    colors = ("#333333","#0072B2","#D55E00")
    labels = ("Original GN4","CEM + original incumbent","CEM then GN4")
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    maximum = max(np.mean([b["methods"][v]["final_joint_distance"] for b in r["tasks"][t]["bank"]]) for r in summary["rows"] for t in TASKS for v in VARIANTS)
    for ax,task in zip(axes,TASKS):
        for v,label,color in zip(VARIANTS,labels,colors):
            y = [np.mean([b["methods"][v]["final_joint_distance"] for b in r["tasks"][task]["bank"]]) for r in summary["rows"]]
            ax.plot(range(1,9),y,"o-",label=label,color=color)
        ax.set(xlabel="Root (all eight)",ylabel="Mean actual final joint distance (rad)\nAll 17 identical prefixes retained",xticks=range(1,9),ylim=(0,maximum*1.3))
        ax.text(.03,.97,task.replace("_"," "),transform=ax.transAxes,va="top",fontsize=9)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc="upper right")
    for suffix in ("png","pdf"): fig.savefig(output/("Figure_1."+suffix),dpi=300)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,task in zip(axes,TASKS):
        for i,color in zip(range(8),plt.cm.viridis(np.linspace(.1,.9,8))):
            cell = summary["rows"][i]["tasks"][task]
            greedy = cell["decisions"]["old_gn4"]["greedy_index"]
            control = cell["bank"][greedy]
            x = [b["geometry"]["common"]["bounded_gain"]-control["geometry"]["common"]["bounded_gain"] for b in cell["bank"]]
            y = [control["geometry"]["own"]["bounded_objective"]-b["geometry"]["own"]["bounded_objective"] for b in cell["bank"]]
            ax.scatter(x,y,s=12,alpha=.65,color=color)
        ax.axhline(0,color="gray",linewidth=.7)
        ax.axvline(0,color="gray",linewidth=.7)
        bottom,top = ax.get_ylim()
        ax.set_ylim(bottom,top+.12*(top-bottom))
        ax.set(xlabel="Common-residual gain advantage vs greedy",ylabel="Own-residual linear-cost advantage vs greedy\nModel diagnostic only, not physical benefit")
        ax.text(.03,.97,task.replace("_"," ")+"; all 136 programs",transform=ax.transAxes,va="top",fontsize=9)
        ax.grid(alpha=.2)
    for suffix in ("png","pdf"): fig.savefig(output/("Figure_2."+suffix),dpi=300)
    plt.close(fig)
    if summary["proposal_native_presealed"]:
        fig,axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
        maximum = max(r["tasks"][t]["proposal_native"]["selected"]["cem_then_gn4"][k]["final_joint_distance"]
            for r in summary["rows"] for t in TASKS for k in ("native","greedy"))
        maximum = max(maximum,max(r["tasks"][t]["selected"]["cem_then_gn4"]["native"]["final_joint_distance"] for r in summary["rows"] for t in TASKS))
        for ax,t in zip(axes,TASKS):
            for source,key,label,color in (("proposal_native","native","Proposal stage + own V",colors[2]),
                    ("common","native","Common-gain diagnostic",colors[1]),("proposal_native","greedy","Same-bank greedy",colors[0])):
                y = [(r["tasks"][t]["proposal_native"]["selected"] if source=="proposal_native" else r["tasks"][t]["selected"])["cem_then_gn4"][key]["final_joint_distance"] for r in summary["rows"]]
                ax.plot(range(1,9),y,"o-",label=label,color=color)
            ax.set(xlabel="Root (all eight)",ylabel="Actual final joint distance (rad)\nSame CEM then GN4 suffix bank",xticks=range(1,9),ylim=(0,maximum*1.3))
            ax.text(.03,.97,t.replace("_"," "),transform=ax.transAxes,va="top",fontsize=9)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8,loc="upper right")
        for suffix in ("png","pdf"): fig.savefig(output/("Figure_3."+suffix),dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError("new immutable analysis directory required")
    summary = verify_files(args.report)
    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir/"summary.json",summary)
    plot(summary,args.output_dir)
    (args.output_dir/"notes.txt").write_text(summary["scope"]+"\nAll8 roots,3 masks,17 fixed prefixes,3 suffix solvers retained. Original408 physical branches exactly replayed; all1224 branches retained. All neural work host CUDA on original frozen checkpoint. Full192D matrices and common/own residuals independently checked with816 SciPy bounded solves. Model incumbent descent is not physical monotonicity; compare root-level means, not408 independent programs. Common-residual diagnostic remains separate from any explicitly presealed proposal-native stage + own-reach cost. Neither selector opens P1.\n")
    (args.output_dir/"figures.tex").write_text("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n\\caption{Same-prefix physical outcomes under three suffix-search contracts, averaging all17 fixed prefixes per root. The intervention preserves the model incumbent, not a physical guarantee.}\n\\end{figure}\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n\\caption{Common-residual gain and state-specific residual objective differ even for the same full response matrix. All programs and roots are retained; these are model-local diagnostics, not deployed selectors or physical benefits.}\n\\end{figure}\n")
    if summary["proposal_native_presealed"]:
        with (args.output_dir/"figures.tex").open("a") as handle:
            handle.write("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_3.pdf}\n\\caption{Presealed proposal-native and common-residual selections from the same CEM then GN4 program bank, with the same-bank greedy control. Every root is retained. This source-disjoint diagnostic is not compute-matched efficacy or P1 admission.}\n\\end{figure}\n")
    write_new(args.output_dir/"manifest.json",{"report_sha256":digest(args.report),"analysis_sha256":digest(Path(__file__)),
        "artifact_sha256":{p.name:digest(p) for p in args.output_dir.iterdir() if p.is_file()}})
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2))


if __name__ == "__main__": main()
