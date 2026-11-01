"""Independent full-coordinate response/box-fit audit, no neural inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from experiments.spectral_reachability import reacher_d2p_capability_reference as source

TASKS={"full_drive":(0,1),"shoulder_only":(0,),"wrist_only":(1,)}
SCALES=(.1,.25)
GATES={"cuda_only","all8_roots","all_cells","all_physical_futures","predict_accounting",
    "parameters_unchanged","buffers_unchanged","all_modules_eval","references_unchanged","weights_unchanged","implementation_unchanged"}


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(value,shape):
    a=np.asarray(value,np.float64)
    if a.shape!=shape or not np.isfinite(a).all(): raise ValueError("finite full-coordinate arrays required")
    return a


def delta(left,right,limited):
    d=finite(left,(2,))-finite(right,(2,))
    if len(limited)!=2 or any(type(b) is not bool for b in limited): raise ValueError("explicit joint topology required")
    for i,bounded in enumerate(limited):
        if not bounded: d[i]=(d[i]+np.pi)%(2*np.pi)-np.pi
    return d


def geometry(j,residual):
    j=np.asarray(j,np.float64)
    if j.ndim!=2 or not np.isfinite(j).all(): raise ValueError("finite full response matrix required")
    r=finite(residual,(j.shape[0],))
    a=np.concatenate((j,.1*np.eye(j.shape[1])))
    b=np.concatenate((r,np.zeros(j.shape[1])))
    fit=lsq_linear(a,b,bounds=(-1,1),tol=1e-12,max_iter=1000,lsq_solver="exact")
    if not fit.success or not np.isfinite(fit.x).all(): raise ValueError("independent bounded physical diagnostic fit failed")
    base=float(r@r)
    objective=float(np.sum((a@fit.x-b)**2))
    if objective>base+1e-8*max(1.,base): raise ValueError("numeric box fit lost feasible zero")
    return {"baseline_squared_residual":base,"regularized_objective":objective,"gain":base-objective,
        "gain_fraction":float((base-objective)/base) if base>1e-20 else 0.,
        "singular_values":np.linalg.svd(j,compute_uv=False).tolist(),"physical_labels_used":True,"deployed":False}


def cosine(a,b):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.sum(a*b)/denominator) if denominator>1e-20 else None


def response_matrices(cell,task,limited):
    n=10*len(TASKS[task])
    actual=finite(cell["actual_history_response"],(192,n))
    predicted=finite(cell["predicted_state_response"],(192,n))
    if cell["center_repeat_exact"] is not True or [p["scale"] for p in cell["probes"]]!=list(SCALES):
        raise ValueError("both complete fixed-scale probes and exact repeated center required")
    result=[]
    for probe in cell["probes"]:
        endpoints=probe["endpoints"]
        if len(endpoints)!=n or [e["coordinate"] for e in endpoints]!=list(range(n)):
            raise ValueError("all ordered active future coordinates required")
        matrices={"model_fd":[],"physical_rgb_fd":[],"physical_qpos_fd":[]}
        step=float(np.float32(probe["scale"]))
        for i,e in enumerate(endpoints):
            for sign,multiplier in (("plus",1),("minus",-1)):
                endpoint=e[sign]
                raw=finite(endpoint["raw_actions"],(10,2))
                expected=np.zeros((10,2),np.float64)
                expected[i//len(TASKS[task]),TASKS[task][i%len(TASKS[task])]]=multiplier*step
                if not np.array_equal(raw,expected): raise ValueError("physical probe does not match exact reduced raw coordinate/mask/sign")
                finite(endpoint["model"],(192,))
                finite(endpoint["observed"],(192,))
                finite(endpoint["qpos"],(2,))
            matrices["model_fd"].append((np.asarray(e["plus"]["model"])-np.asarray(e["minus"]["model"]))/ (2*step))
            matrices["physical_rgb_fd"].append((np.asarray(e["plus"]["observed"])-np.asarray(e["minus"]["observed"]))/ (2*step))
            matrices["physical_qpos_fd"].append(delta(e["plus"]["qpos"],e["minus"]["qpos"],limited)/(2*step))
        matrices={k:np.stack(v,axis=1) for k,v in matrices.items()}
        result.append({"scale":probe["scale"],"matrices":matrices,
            "ad_vs_model_fd_relative_error":float(np.linalg.norm(actual-matrices["model_fd"])/max(np.linalg.norm(actual),1e-20)),
            "actual_model_vs_physical_rgb_cosine":cosine(actual,matrices["physical_rgb_fd"]),
            "predicted_model_vs_physical_rgb_cosine":cosine(predicted,matrices["physical_rgb_fd"]),
            "physical_rgb_vs_model_fd_relative_error":float(np.linalg.norm(matrices["physical_rgb_fd"]-matrices["model_fd"])/max(np.linalg.norm(matrices["physical_rgb_fd"]),1e-20))})
    return result


def paired(values):
    a=finite(values,(8,))
    rng=np.random.default_rng(20261026)
    means=a[rng.integers(0,8,size=(20000,8))].mean(axis=1)
    return {"mean":float(a.mean()),"ci95_descriptive":np.quantile(means,[.025,.975]).tolist(),
        "improved":int(np.sum(a>1e-10)),"worsened":int(np.sum(a < -1e-10)),"tied":int(np.sum(np.abs(a)<=1e-10)),
        "scope":"Known-eight-root descriptive paired capability difference; positive is larger bounded linear gain, not task success or formal inference."}


def mechanism_checks(stage,zero_stage,terminal_distance,greedy_distance,terminal_rgb,greedy_rgb,capability_differences):
    values=finite([stage,zero_stage,terminal_distance,greedy_distance,terminal_rgb,greedy_rgb],(6,))
    differences=finite(capability_differences,(2,2))
    if np.any(values<0): raise ValueError("nonnegative complete physical metrics required")
    checks={"physical_first_stage_worse_than_zero":stage>zero_stage+1e-6,
        "physical_terminal_joint_distance_better_than_greedy":terminal_distance<greedy_distance-1e-6,
        "physical_terminal_rgb_better_than_greedy":terminal_rgb<greedy_rgb-1e-6,
        "common_physical_joint_gain_better_than_zero_and_greedy_at_both_scales":bool(np.all(differences>1e-10))}
    return {"checks":checks,"all_diagnostic_checks":all(checks.values()),"P1_admission":False,
        "scope":"Known-panel conjunction of original C2 ingredients, not a new calibrated P1 admission rule."}


def verify_render_receipts(receipts,seeds):
    """Reconstruct accounting independently of the runtime's receipt predicate."""
    fail="complete ordered fixed two-pass render/state receipts required"
    if (not isinstance(seeds,list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds))!=len(seeds) or not isinstance(receipts,list) or len(receipts)!=len(seeds)):
        raise ValueError(fail)
    totals={"requests":0,"raw_render_passes":0,"exact_state_checks":0,
        "warmup_retained_differences":0,"max_warmup_retained_pixel_error":0,"dither_state_changes":0}
    for seed,receipt in zip(seeds,receipts):
        if not isinstance(receipt,dict) or type(receipt.get("seed")) is not int or receipt["seed"]!=seed:
            raise ValueError(fail)
        stats=receipt.get("stats")
        if (not isinstance(stats,dict) or stats.get("policy")!="fixed-warm-render-v1"
                or any(type(stats.get(k)) is not int or stats[k]<0 for k in totals)):
            raise ValueError(fail)
        if (stats["requests"]==0 or stats["raw_render_passes"]!=2*stats["requests"]
                or stats["exact_state_checks"]!=2*stats["requests"] or stats["dither_state_changes"]!=0
                or stats["warmup_retained_differences"]>stats["requests"]
                or stats["max_warmup_retained_pixel_error"]>255
                or (stats["warmup_retained_differences"]==0)!=(stats["max_warmup_retained_pixel_error"]==0)):
            raise ValueError(fail)
        for k in totals:
            totals[k]=max(totals[k],stats[k]) if k=="max_warmup_retained_pixel_error" else totals[k]+stats[k]
    return {"policy":"fixed-warm-render-v1","roots":len(seeds),**totals}


def verify(path):
    report=json.loads(path.read_text())
    protocol=report["sealed_protocol"]
    expected_gates=GATES|{"partial_alignment_exact"} if protocol.get("all_programs") is True else GATES
    policy=protocol.get("render_policy","native")
    render_accounting=None
    if policy=="fixed-warm-render-v1":
        expected_gates=expected_gates|{"render_policy_verified"}
        render_accounting=verify_render_receipts(report.get("render_receipts"),protocol["seeds"])
    elif policy!="native": raise ValueError("unknown observation rendering policy")
    if (report.get("schema")!="reacher_d2p_prefix_capability.v1" or report.get("status")!="completed"
            or set(report.get("gates",{}))!=expected_gates or any(v is not True for v in report["gates"].values())
            or report.get("P1_admission") is not False or protocol.get("physical_labels_used_for_selection") is not False
            or digest(path.with_suffix(".protocol.json"))!=report["protocol_sha256"]
            or json.loads(path.with_suffix(".protocol.json").read_text())!=protocol):
        raise ValueError("complete immutable explicit host-CUDA capability audit required")
    for name,sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py"))!=sha: raise ValueError("archived source changed")
    for p,sha in protocol["input_bindings"].items():
        if digest(Path(p))!=sha: raise ValueError("original input changed")
    references=[]
    for p in protocol["input_bindings"]:
        if not p.endswith(".predictions.json"):
            candidate=json.loads(Path(p).read_text())
            if candidate.get("schema") in (source.SUFFIX, source.METRIC, source.FULL):
                references.append((Path(p),candidate))
    if len(references)!=1: raise ValueError("exact original suffix physical report required")
    original_path,original=references[0]
    metric_source = original["schema"] == "reacher_d2p_root_metric_confirmation.v1"
    full_source = original["schema"] == source.FULL
    simple_source = metric_source or full_source
    candidate_roles = (("native_candidate","common_candidate") if full_source else
        ("euclidean_candidate", "metric_candidate") if metric_source else ("common_candidate", "feasible_selector"))
    if full_source:
        if (protocol.get("source_contract") != source.FULL
                or protocol.get("reference_zero_suffix_prediction_saved") is not True
                or policy != "fixed-warm-render-v1" or protocol.get("all_programs") is True):
            raise ValueError("explicit full-Q compiler capability source contract required")
        from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_compiler import analyze as verify_full_source
        verify_full_source(original_path)
    elif metric_source:
        if (protocol.get("source_contract") != original["schema"]
                or protocol.get("reference_zero_suffix_prediction_saved") is not False
                or policy != "fixed-warm-render-v1" or protocol.get("all_programs") is True):
            raise ValueError("explicit fixed metric confirmation capability contract required")
        from experiments.spectral_reachability.analyze_reacher_d2p_root_metric_confirmation import verify_files as verify_metric_source
        verify_metric_source(original_path)
    elif protocol.get("source_contract", original["schema"]) != original["schema"]:
        raise ValueError("capability source contract changed")
    old_actual={r["seed"]:r for r in original["rows"]}
    old_plans={r["seed"]:r for r in json.loads(original_path.with_suffix(".predictions.json").read_text())["rows"]}
    def old_packets(seed, task, index):
        return source.packets(source.physical_branch(old_actual[seed],task,index,original["schema"]),original["schema"])
    seeds=protocol["seeds"]
    if len(seeds)!=8 or len(set(seeds))!=8 or [r["seed"] for r in report["rows"]]!=seeds:
        raise ValueError("all eight distinct root identities required")
    rows,cells,solves,predict,futures=[],0,0,0,0
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text())!=row: raise ValueError("per-root receipt changed")
        tasks={}
        if any(row[k]!=old_plans[row["seed"]][k] for k in ("source_episode","source_start_row","goal_vector","goal_admission")):
            raise ValueError("original source/root/goal identity changed")
        target=finite(row["goal_vector"],(192,))
        limited=row["goal_admission"]["limited_joints"]
        target_qpos=finite(row["goal_admission"]["feasible_target_qpos"],(2,))
        if set(row["tasks"])!=set(TASKS): raise ValueError("all three actuator tasks required")
        for task,measurements in row["tasks"].items():
            roles=measurements["roles"]
            if full_source:
                if roles != source.full_roles(source.task_plan(old_plans[row["seed"]],task,source.FULL)):
                    raise ValueError("full compiler roles must equal original model decisions")
            elif metric_source:
                plan = old_plans[row["seed"]]["tasks"][task]
                expected_roles = {"euclidean_candidate":plan["scores"]["original_decision"]["native_index"],
                    "metric_candidate":plan["scores"]["decision"]["native_index"],
                    "greedy":min(range(17), key=lambda i:(plan["bank"][i]["prefix_cost"],i)), "zero":0}
                if roles != expected_roles:
                    raise ValueError("metric/Euclidean roles must equal all original sealed model decisions")
            n=10*len(TASKS[task])
            if roles!=protocol["roles"][str(row["seed"])][task] or set(measurements["cells"])!={str(i) for i in roles.values()}:
                raise ValueError("all presealed roles/prefix identities required")
            if protocol.get("all_programs") is True and (set(measurements["cells"])!={str(i) for i in range(17)}
                    or any(roles.get(f"program_{i}")!=i for i in range(17))):
                raise ValueError("complete all17-program capability coverage required")
            zero=measurements["cells"]["0"]["center"]
            common_rgb=target-finite(zero["observed"],(192,))
            common_qpos=delta(target_qpos,zero["qpos"],limited)
            summaries={}
            for index,cell in measurements["cells"].items():
                old_bank=source.task_plan(old_plans[row["seed"]],task,original["schema"])["bank"][int(index)]
                old_packet=old_packets(row["seed"], task, int(index))[0]
                old_raw = source.raw_actions(old_bank,original["schema"])
                if (cell["prefix_packet"]!=old_packet or cell["prefix_raw_actions"]!=old_raw[:5]
                        or cell["predicted_state_response"]!=old_bank["response"]
                        or (not simple_source and cell["predicted_zero"]!=old_bank["zero_suffix_vector"])
                        or (full_source and cell["predicted_zero"]!=old_bank["own_zero_suffix_vector"])):
                    raise ValueError("exact original prefix/full model response replay changed")
                if cell["predict_samples"]!=9+8*n: raise ValueError("complete per-cell accounting required")
                if np.count_nonzero(finite(cell["center"]["raw_actions"],(10,2)))!=0: raise ValueError("response anchor must be literal raw zero")
                own_rgb=target-finite(cell["center"]["observed"],(192,))
                own_qpos=delta(target_qpos,cell["center"]["qpos"],limited)
                probes=[]
                for probe in response_matrices(cell,task,limited):
                    matrices=probe.pop("matrices")
                    fits={}
                    for name,j in matrices.items():
                        common,own=(common_qpos,own_qpos) if name=="physical_qpos_fd" else (common_rgb,own_rgb)
                        fits[name]={"common":geometry(j,common),"own":geometry(j,own)}
                        solves+=2
                    probes.append({**probe,"fits":fits})
                summaries[index]={"probes":probes,"prefix_packet":cell["prefix_packet"],
                    "actual_vs_predicted_response_relative_error":float(np.linalg.norm(np.asarray(cell["actual_history_response"])-np.asarray(cell["predicted_state_response"]))/max(np.linalg.norm(cell["predicted_state_response"]),1e-20))}
                cells+=1
                predict+=9+8*n
                futures+=2+4*n
            diagnoses={}
            zero_prefix=measurements["cells"]["0"]["prefix_packet"]["observed_vector"]
            zero_stage=float(np.mean((np.asarray(zero_prefix)-target)**2))
            def terminal_metrics(index):
                packet=old_packets(row["seed"], task, index)[-1]
                return (float(np.linalg.norm(delta(packet["qpos"],target_qpos,limited))),
                    float(np.mean((np.asarray(packet["observed_vector"])-target)**2)))
            gd,grgb=terminal_metrics(roles["greedy"])
            diagnostic_roles=candidate_roles+(tuple(f"program_{i}" for i in range(17)) if protocol.get("all_programs") is True else ())
            for role in diagnostic_roles:
                index=roles[role]
                stage=float(np.mean((np.asarray(measurements["cells"][str(index)]["prefix_packet"]["observed_vector"])-target)**2))
                td,trgb=terminal_metrics(index)
                gain=lambda index,k:summaries[str(index)]["probes"][k]["fits"]["physical_qpos_fd"]["common"]["gain_fraction"]
                differences=[[gain(index,k)-gain(roles[control],k) for control in ("zero","greedy")] for k in range(2)]
                diagnoses[role]=mechanism_checks(stage,zero_stage,td,gd,trgb,grgb,differences)
            tasks[task]={"roles":roles,"cells":summaries,"mechanism_diagnostics":diagnoses}
        rows.append({"seed":row["seed"],"tasks":tasks})
    if (cells!=protocol["cells"] or predict!=protocol["expected_predict_samples"] or predict!=report["actual_predict_samples"]
            or futures!=protocol["expected_physical_futures"] or futures!=report["physical_futures"]):
        raise ValueError("complete whole-panel accounting required")
    if render_accounting is not None:
        for receipt,row in zip(report["render_receipts"],report["rows"]):
            # Each observed physical future needs at least one actual terminal
            # RGB request, independent of extra history/restore/goal renders.
            minimum=sum(len(t["cells"])*(2+40*len(TASKS[task])) for task,t in row["tasks"].items())
            if receipt["stats"]["requests"]<minimum:
                raise ValueError("render request receipts cannot undercount observed physical futures")
    aligned=None
    if protocol.get("all_programs") is True:
        binding=protocol["alignment_reference"]
        alignment_path=Path(binding["path"])
        if digest(alignment_path)!=binding["sha256"]: raise ValueError("original partial audit changed")
        verify(alignment_path)
        alignment=json.loads(alignment_path.read_text())
        indexed={r["seed"]:r for r in report["rows"]}
        aligned=0
        for old in alignment["rows"]:
            for task,t in old["tasks"].items():
                for index,cell in t["cells"].items():
                    if indexed[old["seed"]]["tasks"][task]["cells"][index]!=cell:
                        raise ValueError("complete original partial physical/model cell changed")
                    aligned+=1
        if aligned!=binding["expected_exact_cells"]: raise ValueError("complete partial alignment required")
    aggregate={}
    for task in TASKS:
        scales={}
        for k,scale in enumerate(SCALES):
            def value(row,role,space):
                t=row["tasks"][task]
                return t["cells"][str(t["roles"][role])]["probes"][k]["fits"][space]["common"]["gain_fraction"]
            scales[str(scale)]={space:{f"{a}_minus_{b}":paired([value(r,a,space)-value(r,b,space) for r in rows])
                for a,b in ((a,b) for a in candidate_roles for b in ("greedy","zero"))}
                for space in ("physical_rgb_fd","physical_qpos_fd")}
        aggregate[task]=scales
    mechanism_candidates={task:[{"seed":r["seed"],"programs":[i for i in range(17)
        if r["tasks"][task]["mechanism_diagnostics"][f"program_{i}"]["all_diagnostic_checks"]]} for r in rows]
        for task in TASKS} if protocol.get("all_programs") is True else None
    return {"rows":rows,"aggregate":aggregate,"cells":cells,"independent_bounded_fits":solves,"partial_alignment_cells":aligned,
        "all_programs":protocol.get("all_programs",False),"complete_bank_mechanism_candidates":mechanism_candidates,
        "physical_futures":futures,"predict_samples":predict,"render_accounting":render_accounting,"P1_admission":False,
        "scope":protocol["scope"],"report_sha256":digest(path)}


def plot(result,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"savefig.dpi":300,
        "axes.spines.top":False,"axes.spines.right":False})
    role_styles=(("common_candidate","Common candidate","#0072B2","o","-"),
        ("feasible_selector","Feasible selector","#E69F00","s","--"),
        ("greedy","Greedy","#CC79A7","x",":"),("zero","Zero prefix","#56B4E9","^","-."))
    if "metric_candidate" in result["rows"][0]["tasks"][next(iter(TASKS))]["roles"]:
        role_styles=(("euclidean_candidate","Euclidean selector","#0072B2","o","-"),
            ("metric_candidate","Fixed metric selector","#E69F00","s","--"),*role_styles[2:])
    elif "native_candidate" in result["rows"][0]["tasks"][next(iter(TASKS))]["roles"]:
        role_styles=(("native_candidate","Full-Q stage + own reach","#0072B2","o","-"),
            ("common_candidate","Historical common rule","#E69F00","s","--"),*role_styles[2:])
    seeds=[r["seed"] for r in result["rows"]]
    for k,scale in enumerate(SCALES):
        fig,axes=plt.subplots(3,2,figsize=(11,8),layout="constrained")
        for i,task in enumerate(TASKS):
            for j,(space,label) in enumerate((("physical_rgb_fd","RGB"),("physical_qpos_fd","Joint"))):
                ax=axes[i,j]
                if result["all_programs"]:
                    for index in range(17):
                        values=[row["tasks"][task]["cells"][str(index)]["probes"][k]["fits"][space]["common"]["gain_fraction"] for row in result["rows"]]
                        ax.plot(range(8),values,color=".65",lw=.5,alpha=.4,marker=".",markersize=2)
                for role,title,color,marker,style in role_styles:
                    values=[]
                    for row in result["rows"]:
                        t=row["tasks"][task]
                        values.append(t["cells"][str(t["roles"][role])]["probes"][k]["fits"][space]["common"]["gain_fraction"])
                    ax.plot(range(8),values,label=title,color=color,marker=marker,linestyle=style,lw=1,markersize=4)
                ax.set(xticks=range(8),xticklabels=[str(s)[-6:] for s in seeds],ylim=(-.03,1.03),
                    ylabel=task.replace("_"," ")+"\n"+label+" bounded gain fraction",xlabel="All eight known roots (seed suffix)")
                ax.grid(axis="y",alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=4,frameon=False)
        for ext in ("png","pdf"): fig.savefig(output/f"Figure_{k+1}.{ext}",bbox_inches="tight")
        plt.close(fig)
    (output/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Known-root prefix capability audit at raw perturbation scale 0.1. All roots and four labeled roles are retained; coincident roles overplot. In the all-program extension, gray curves additionally retain all seventeen original prefixes. The common residual within each root is fixed to the actual zero-prefix zero-suffix outcome. Full measured response matrices are fitted in the unchanged raw action box with absolute-action regularization 0.01. RGB and joint-coordinate bounded linear gains are distinct diagnostics, not global reachability, reconfiguration success, or independent efficacy.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{The same full-root audit at fixed perturbation scale 0.25. Scales were fixed before probing and are not selected by outcomes. All probes are evaluator-only; no physical state, measured future, or fitted response is deployed as planner input.}
\end{figure}
""")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.report)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plot(result,args.output_dir)
    for name,value in (("summary.json",result),("notes.txt",{"scope":result["scope"],"neural_cpu_inference":False,"independent_bounded_fits":result["independent_bounded_fits"]})):
        with (args.output_dir/name).open("x") as f: json.dump(value,f,indent=2,allow_nan=False)
    manifest={"report_sha256":digest(args.report),"script_sha256":digest(Path(__file__)),
        "artifacts":{p.name:digest(p) for p in args.output_dir.iterdir()}}
    with (args.output_dir/"manifest.json").open("x") as f: json.dump(manifest,f,indent=2)
    print(json.dumps({"cells":result["cells"],"fits":result["independent_bounded_fits"],"aggregate":result["aggregate"]},indent=2))


if __name__=="__main__": main()
