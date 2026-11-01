"""Full-coordinate same-action outcome audit; no neural inference or selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_masked_task_preflight import finite, joint_residual, TASKS
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission

CONTROLS = ("flat_gn", "flat_cem_gn_incumbent")
REQUIRED_GATES = {"cuda_only", "all8_roots", "source_disjoint", "all456_physical_branches", "all_full_split_exact",
    "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "predict_sample_accounting", "all_task_compute_receipts",
    "predictions_unchanged", "reference_unchanged", "weights_file_unchanged", "implementation_unchanged"}


def independent_decision(bank):
    if len(bank) != 17 or [r["index"] for r in bank] != list(range(17)):
        raise ValueError("all 17 ordered prefixes required")
    greedy = min(range(17), key=lambda i:(bank[i]["prefix_cost"],i))
    bound = max(bank[i]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] for i in (0,greedy))
    witnesses = [i for i in range(1,17) if bank[i]["prefix_cost"]-bank[0]["prefix_cost"] > 1e-6
        and bank[greedy]["terminal_cost"]-bank[i]["terminal_cost"] > 1e-6
        and bank[i]["common_response_gain"]["bounded_quadratic_gain"] > bound]
    return greedy, min(witnesses,key=lambda i:(bank[i]["terminal_cost"],i)) if witnesses else greedy, witnesses


def branch_metrics(plan, observation, target, admission, task):
    raw = finite(plan["raw_actions"],(15,2))
    inactive = [j for j in range(2) if j not in TASKS[task]]
    if np.max(np.abs(raw)) > 1 or np.count_nonzero(raw[:,inactive]) or not np.array_equal(raw,finite(observation["raw_actions"],(15,2))):
        raise ValueError("same-action literal-mask execution witness required")
    if len(observation["packets"]) != 3 or [p["packet"] for p in observation["packets"]] != [1,2,3]:
        raise ValueError("all three physical packet observations required")
    for j,packet in enumerate(observation["packets"]):
        sha = hashlib.sha256(np.ascontiguousarray(raw[j*5:(j+1)*5],dtype=np.float32).tobytes()).hexdigest()
        if packet["action_sha256"] != sha or len(packet["root_rgb_sha256"]) != 64:
            raise ValueError("physical packet action hash/RGB identity changed")
        finite(packet["observed_vector"],(192,))
        finite(packet["qpos"],(2,))
    prefix, terminal = observation["packets"][0],observation["packets"][-1]
    predicted = finite(plan["terminal_vector"],(192,))
    terminal_cost = float(np.mean((predicted-target)**2))
    if not np.isclose(terminal_cost,plan["terminal_cost"],rtol=2e-6,atol=1e-7):
        raise ValueError("full-coordinate predicted terminal cost changed")
    return {"predicted_terminal_cost":terminal_cost,
        "observed_prefix_cost":float(np.mean((finite(prefix["observed_vector"],(192,))-target)**2)),
        "observed_terminal_cost":float(np.mean((finite(terminal["observed_vector"],(192,))-target)**2)),
        "final_joint_distance":float(np.linalg.norm(joint_residual(admission["feasible_target_qpos"],terminal["qpos"],admission["limited_joints"]))),
        "terminal_vector_l2_error":float(np.linalg.norm(predicted-finite(terminal["observed_vector"],(192,))))}


def summarize(report,predictions):
    if (report.get("schema") != "reacher_d2p_masked_program_pilot.v1" or report.get("status") != "completed"
            or not REQUIRED_GATES.issubset(report.get("gates",{})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("strict completed CUDA program pilot required")
    protocol = report["sealed_protocol"]
    if (protocol["tasks"] != {k:list(v) for k,v in TASKS.items()} or protocol["prefix_packets"] != 1 or protocol["future_packets"] != 2
            or protocol["spectral_bank"] != {"directions":4,"amplitudes":[.25,.75],"signs":[-1,1],"literal_zero_index":0,"count":17}
            or protocol["suffix_gn_iterations"] != 4 or protocol["flat_gn_iterations"] != 4 or protocol["damping"] != .01
            or protocol["line_search"] != [1.,.5,.25,.125,.0625,.03125] or protocol["non_greedy_margin"] != 1e-6
            or protocol["terminal_margin"] != 1e-6 or protocol["goal_contract"] not in ("joint-feasible-v1","joint-feasible-v2-soft-source")
            or protocol["target_delta"] != [.65,-.65] or protocol["no_post_training"] is not True
            or protocol["all_predictions_sealed_before_any_future_execution"] is not True or protocol["compute_matched"] is not False):
        raise ValueError("unchanged masked-program protocol required")
    seeds = sorted(protocol["root_seeds"])
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("all eight unique roots required")
    def roots(rows):
        indexed = {r["seed"]:r for r in rows}
        if len(indexed) != len(rows) or sorted(indexed) != seeds:
            raise ValueError("duplicate-free root-identity join required")
        return indexed
    observed,forecast,source = roots(report["rows"]),roots(predictions["rows"]),roots(protocol["source_roots"])
    episodes = [r["source_episode"] for r in source.values()]
    excluded = set(protocol["excluded_source_episodes"])
    if len(set(episodes)) != 8 or len(excluded) < 200 or set(episodes)&excluded:
        raise ValueError("all fresh sources disjoint from at least200 previous episodes required")
    output = []
    for seed in seeds:
        p,m = forecast[seed],observed[seed]
        if any(p[k] != source[seed][k] or m[k] != source[seed][k] for k in ("source_episode","source_start_row")):
            raise ValueError("source identities changed")
        target = finite(p["goal_vector"],(192,))
        admission = p["goal_admission"]
        validate_goal_admission(protocol["goal_contract"],admission)
        if set(p["tasks"]) != set(TASKS) or set(m["tasks"]) != set(TASKS):
            raise ValueError("every task retained")
        tasks = {}
        for task in TASKS:
            plan,measurement = p["tasks"][task],m["tasks"][task]
            bank = plan["bank"]
            if (set(plan["controls"]) != set(CONTROLS) or plan["trained_modules"] is not False or plan["compute_matched"] is not False
                    or plan["actual_predict_samples"] != 2482 or set(measurement) != {"bank_"+str(i) for i in range(17)}|set(CONTROLS)):
                raise ValueError("complete method/compute contracts required")
            common = finite(plan["common_model_residual"],(192,))
            baseline = float(common@common)
            for branch in bank:
                if branch["full_split_exact"] is not True:
                    raise ValueError("split/full prediction equality required")
                prefix = finite(branch["prefix_vector"],(192,))
                if not np.isclose(np.mean((prefix-target)**2),branch["prefix_cost"],rtol=2e-6,atol=1e-7):
                    raise ValueError("complete prefix cost changed")
                gain = branch["common_response_gain"]
                if (gain["objective_certificate_passed"] is not True or gain["physical_labels_used"] is not False
                        or gain["qp_certificate"]["active_set_terminated"] is not True or gain["qp_certificate"]["box_feasibility_violation"] != 0
                        or not gain["qp_certificate"]["device"].startswith("cuda")
                        or not np.isclose(gain["baseline_squared_residual"],baseline,rtol=1e-10,atol=1e-10)
                        or not np.isclose(gain["bounded_quadratic_gain"],baseline-gain["bounded_quadratic_objective"],rtol=1e-10,atol=1e-10)
                        or not np.isclose(gain["bounded_quadratic_gain_upper_bound"]-gain["bounded_quadratic_gain"],gain["qp_certificate"]["objective_duality_gap_upper_bound"],rtol=1e-4,atol=1e-10)):
                    raise ValueError("model-only CUDA response certificate algebra changed")
                trace = branch["suffix_receipt"]["trace"]
                if len(trace) != 4 or any(v["cost_after"] > v["cost_before"] for v in trace):
                    raise ValueError("all four nonlinear descent steps required")
            if np.count_nonzero(np.asarray(bank[0]["raw_actions"])[:5]):
                raise ValueError("zero-prefix reference is not literal raw zero")
            greedy,native,witnesses = independent_decision(bank)
            decision = plan["decision"]
            if (decision["native_index"] != native or decision["greedy_index"] != greedy or decision["qualified_indices"] != witnesses
                    or decision["reconfiguration_witness_found"] is not bool(witnesses) or decision["physical_labels_used"] is not False):
                raise ValueError("independent model-only selector mismatch")
            values = {"bank_"+str(b["index"]):branch_metrics(b,measurement["bank_"+str(b["index"])],target,admission,task) for b in bank}
            values.update({k:branch_metrics(b,measurement[k],target,admission,task) for k,b in plan["controls"].items()})
            reference = values["bank_0"]["observed_prefix_cost"]
            margins = [{"index":i,"predicted":bank[i]["prefix_cost"]-bank[0]["prefix_cost"],
                "observed":values["bank_"+str(i)]["observed_prefix_cost"]-reference} for i in range(1,17)]
            gain_threshold = max(bank[i]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] for i in (0,greedy))
            prefix_set = {i for i in range(1,17) if bank[i]["prefix_cost"] > bank[0]["prefix_cost"]+1e-6}
            terminal_set = {i for i in range(1,17) if bank[i]["terminal_cost"] < bank[greedy]["terminal_cost"]-1e-6}
            gain_set = {i for i in range(1,17) if bank[i]["common_response_gain"]["bounded_quadratic_gain"] > gain_threshold}
            tasks[task] = {"native_index":native,"greedy_index":greedy,"qualified_indices":witnesses,"prefix_margins":margins,
                "candidate_predicate_counts":{"non_greedy":len(prefix_set),"terminal_better_than_greedy":len(terminal_set),
                    "certified_common_response_gain":len(gain_set),"non_greedy_and_terminal":len(prefix_set&terminal_set),
                    "non_greedy_and_gain":len(prefix_set&gain_set),"all_three":len(prefix_set&terminal_set&gain_set)},
                "native_observed_non_greedy":bool(values["bank_"+str(native)]["observed_prefix_cost"]>reference+1e-6),
                "methods":{"native":values["bank_"+str(native)],"greedy":values["bank_"+str(greedy)],**{k:values[k] for k in CONTROLS}},
                "all_branch_metrics":values}
        output.append({"seed":seed,"tasks":tasks})
    if report["physical_branches"] != 456 or report["actual_model_predict_samples"] != 59568:
        raise ValueError("full actual branch/predict inventory required")
    aggregate = {}
    for task in TASKS:
        cells = [r["tasks"][task] for r in output]
        margins = [m for c in cells for m in c["prefix_margins"]]
        aggregate[task] = {"model_witness_roots":sum(bool(c["qualified_indices"]) for c in cells),
            "all128_candidate_predicate_counts":{k:sum(c["candidate_predicate_counts"][k] for c in cells) for k in cells[0]["candidate_predicate_counts"]},
            "witness_roots_observed_non_greedy":sum(bool(c["qualified_indices"]) and c["native_observed_non_greedy"] for c in cells),
            "all128_prefix_sign_agreements":sum(np.sign(v["predicted"])==np.sign(v["observed"]) for v in margins),
            "mean_final_joint_distance":{name:float(np.mean([c["methods"][name]["final_joint_distance"] for c in cells])) for name in ("native","greedy",*CONTROLS)},
            "native_vs_control_joint_distance":{name:paired_difference([c["methods"]["native"]["final_joint_distance"] for c in cells],
                [c["methods"][name]["final_joint_distance"] for c in cells]) for name in ("greedy",*CONTROLS)}}
        aggregate[task]["all128_prefix_sign_agreements"] = int(aggregate[task]["all128_prefix_sign_agreements"])
        for contrast in aggregate[task]["native_vs_control_joint_distance"].values():
            contrast["metric"] = "joint-topology-aware distance difference; negative favors native; descriptive exploratory root-paired comparison"
    return {"root_seeds":seeds,"rows":output,"aggregate":aggregate,"physical_branches":456,"actual_model_predict_samples":59568,
        "P1_admission":False,"scope":protocol["scope"],"gain_verification":"Stored CUDA certificate algebra checked; full response matrices are not retained, so these are not independent SciPy gain solves."}


def verify_files(path):
    report = json.loads(path.read_text())
    protocol_path,prediction_path = path.with_suffix(".protocol.json"),path.with_suffix(".predictions.json")
    predictions = json.loads(prediction_path.read_text())
    if (digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]
            or digest(prediction_path) != report["predictions_sha256"] or predictions["protocol_sha256"] != report["protocol_sha256"]):
        raise ValueError("protocol/prediction seal changed")
    for key,sha in report["sealed_protocol"]["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(key+".py")) != sha:
            raise ValueError("bound implementation copy changed")
    bound_excluded = set()
    for binding in report["sealed_protocol"]["exclusion_bindings"]:
        if digest(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("source exclusion reference changed")
        prior = json.loads(Path(binding["path"]).read_text())
        bound_excluded.update(r["source_episode"] for r in prior["rows"])
        bound_excluded.update(prior["sealed_protocol"].get("excluded_source_episodes",[]))
    if bound_excluded != set(report["sealed_protocol"]["excluded_source_episodes"]):
        raise ValueError("excluded sources differ from bound complete references")
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text()) != row:
            raise ValueError("physical root sidecar mismatch")
    return {"report_sha256":digest(path),**summarize(report,predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new immutable analysis directory required")
    result = verify_files(args.report)
    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir/"summary.json",result)
    plt.rcParams.update({"font.size":10,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
    colors = ("#333333","#0072B2","#D55E00")
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    max_distance = max(r["tasks"][task]["methods"][name]["final_joint_distance"] for r in result["rows"] for task in TASKS
        for name in ("native","flat_gn","flat_cem_gn_incumbent"))
    for ax,task,color in zip(axes,TASKS,colors):
        margins = [m for r in result["rows"] for m in r["tasks"][task]["prefix_margins"]]
        x,y = np.array([m["predicted"] for m in margins]),np.array([m["observed"] for m in margins])
        ax.scatter(x,y,s=13,alpha=.65,color=color)
        ax.axhline(0,color="gray",linewidth=.7)
        ax.axvline(0,color="gray",linewidth=.7)
        lower,upper = min(x.min(),y.min()),max(x.max(),y.max())
        ax.plot([lower,upper],[lower,upper],"--",color="gray",linewidth=.7)
        ax.set(xlabel="Predicted RGB prefix-cost difference",ylabel="Observed RGB prefix-cost difference")
        ax.text(.03,.97,task.replace("_"," ")+"; all 128 prefixes",transform=ax.transAxes,va="top",fontsize=9)
        ax.grid(alpha=.2)
    for suffix in ("png","pdf"): fig.savefig(args.output_dir/("Figure_1."+suffix),dpi=300)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,task in zip(axes,TASKS):
        for name,label,color in zip(("native","flat_gn","flat_cem_gn_incumbent"),("Native (greedy if no witness)","Flat GN","Flat CEM + GN incumbent"),colors):
            ax.plot(range(1,9),[r["tasks"][task]["methods"][name]["final_joint_distance"] for r in result["rows"]],"o-",label=label,color=color)
        ax.set(xlabel="Root (all eight)",ylabel="Actual final joint distance (rad)",xticks=range(1,9))
        ax.text(.03,.97,task.replace("_"," "),transform=ax.transAxes,va="top",fontsize=9)
        ax.set_ylim(0,1.3*max_distance)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc="upper right")
    for suffix in ("png","pdf"): fig.savefig(args.output_dir/("Figure_2."+suffix),dpi=300)
    plt.close(fig)
    (args.output_dir/"notes.txt").write_text("All eight fresh roots, all three exact actuator masks, all 17 spectral prefixes and both flat controls retained. Same-action complete 192D observed RGB, literal-zero reference, original frozen checkpoint and host CUDA. Reconfiguration selection checked independently without outcomes. Predictions presealed before all 456 physical branches. Exploratory unequal-compute code/mechanism audit, not P1 admission or theoretical rejection. Gain certificate algebra checked, not independent SciPy verification of unsaved response matrices.\n")
    (args.output_dir/"figures.tex").write_text("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n\\caption{Same-action predicted and actual prefix-goal cost differences from the literal-zero reference, all roots and nonzero spectral prefixes retained.}\n\\end{figure}\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n\\caption{Actual three-packet terminal joint distances for all roots and exact actuator tasks. Methods are not compute-matched; this pilot does not admit the reconfiguration claim.}\n\\end{figure}\n")
    write_new(args.output_dir/"manifest.json",{"report_sha256":digest(args.report),"analysis_sha256":digest(Path(__file__)),
        "artifact_sha256":{p.name:digest(p) for p in args.output_dir.iterdir() if p.is_file()}})
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))


if __name__ == "__main__": main()
