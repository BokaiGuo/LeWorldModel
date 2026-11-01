"""Audit every fixed candidate and display solver-vs-task metric consequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from experiments.spectral_reachability.analyze_pusht_d2p_physical_calibration import digest, vector

MODES = ("random8_one_shot", "full192_one_shot", "full192_gn4")
REQUIRED_GATES = {"cuda_only", "all64_root_candidates", "all192_real_branches", "parameters_unchanged",
    "buffers_unchanged", "all_modules_eval", "prerun_seal_unchanged", "extra_sources_unchanged", "weights_file_unchanged"}


def costs(row, mode):
    goal = vector(row["goal_vector"])
    branch = row["branches"][mode]
    predicted = vector(branch["suffix_receipt"]["predicted_terminal_vector"])
    actual = vector(branch["final_vector"])
    pcost, ocost = float(np.mean((predicted-goal)**2)), float(np.mean((actual-goal)**2))
    if abs(pcost-branch["suffix_receipt"]["predicted_goal_cost"]) > 1e-6:
        raise ValueError("suffix prediction metric differs from full coordinates")
    return np.array([pcost, ocost, branch["final_pose"]["pose_cost"]])


def summarize(report):
    if (report.get("schema") != "pusht_full_metric_bank.v1" or report.get("status") != "completed"
            or set(report.get("gates", {})) != REQUIRED_GATES
            or any(v is not True for v in report["gates"].values())
            or report.get("actual_predict_samples") != 5056
            or report["protocol"]["outcome_opened"] is not False):
        raise ValueError("complete unchanged all-bank CUDA diagnostic required")
    rows = sorted(report["rows"], key=lambda r: (r["root_id"], r["proposal_index"]))
    if [(r["root_id"], r["proposal_index"]) for r in rows] != [(i,j) for i in range(4) for j in range(16)]:
        raise ValueError("all64 unique fixed root-candidate pairs required")
    identities = [rows[i*16]["root_sha256"] for i in range(4)]
    if len(set(identities)) != 4 or identities != report["protocol"]["roots"]:
        raise ValueError("root identities differ from sealed protocol")
    for row in rows:
        if set(row["branches"]) != set(MODES) or row["root_sha256"] != identities[row["root_id"]]:
            raise ValueError("full solver/root coverage required")
        base = row["branches"][MODES[0]]
        vector(row["goal_vector"])
        for mode in MODES:
            branch = row["branches"][mode]
            if branch["successor_rgb_sha256"] != base["successor_rgb_sha256"] or branch["successor_state"] != base["successor_state"]:
                raise ValueError("same-prefix physical replay drifted")
            np.testing.assert_array_equal(branch["observed_prefix_vector"], base["observed_prefix_vector"])
            costs(row, mode)
            imagined = row["predicted"]["suffixes"][mode]
            cost = float(np.mean((vector(imagined["predicted_terminal_vector"])-vector(row["goal_vector"]))**2))
            if abs(cost-imagined["predicted_goal_cost"]) > 1e-6:
                raise ValueError("pre-execution imagined suffix metric changed")
            if mode == MODES[-1]:
                receipt = branch["suffix_receipt"]["nonlinear"]
                if (receipt["final_objective"] > min(receipt["initial_objective"], receipt["zero_objective"])+1e-6
                        or len(receipt["iterations"]) != 4
                        or any(r["after"] > r["before"]+1e-6 for r in receipt["iterations"])):
                    raise ValueError("bounded solver failed actual-objective incumbent contract")
    matrix = {mode: np.stack([costs(row, mode) for row in rows]) for mode in MODES}
    summary = {"schema": "pusht_full_metric_bank_analysis.v1", "root_count": 4,
        "all_root_candidate_pairs": 64, "real_branches": 192, "predict_samples": report["actual_predict_samples"],
        "mean_metrics": {mode: dict(zip(("predicted_rgb_mse", "actual_rgb_mse", "actual_pose"), values.mean(axis=0).tolist())) for mode, values in matrix.items()},
        "p1_admitted": False, "outcome_opened": False,
        "scope": "Known calibration bank; solver budgets unequal. Lower RGB objective is not physical-pose success."}
    comparisons = {}
    for mode in MODES[1:]:
        delta = matrix[mode]-matrix[MODES[0]]
        comparisons[mode] = {name: {"mean_delta": float(delta[:, i].mean()),
            "improved": int(np.sum(delta[:,i] < -1e-8)), "worsened": int(np.sum(delta[:,i] > 1e-8)),
            "tied": int(np.sum(np.abs(delta[:,i]) <= 1e-8))}
            for i,name in enumerate(("predicted_rgb_mse", "actual_rgb_mse", "actual_pose"))}
    summary["same_prefix_vs_random8"] = comparisons
    rgb, pose, predicted_rgb = [], [], []
    selectors = []
    for root in range(4):
        group = rows[root*16:(root+1)*16]
        goal = vector(group[0]["goal_vector"])
        immediate = [float(np.mean((vector(r["predicted"]["predicted_prefix_vector"])-goal)**2)) for r in group]
        observed = [float(np.mean((vector(r["branches"][MODES[0]]["observed_prefix_vector"])-goal)**2)) for r in group]
        p = [r["branches"][MODES[0]]["prefix_pose"]["pose_cost"] for r in group]
        rgb.append((np.asarray(observed)-observed[0]).tolist())
        pose.append((np.asarray(p)-p[0]).tolist())
        predicted_rgb.append((np.asarray(immediate)-immediate[0]).tolist())
        choices = {}
        for mode in MODES:
            scores = [r["predicted"]["suffixes"][mode]["linear"]["value"] for r in group]
            eligible = [j for j in range(1,16) if immediate[j] > immediate[0] and scores[j] < scores[0]]
            choices[mode] = min(eligible, key=lambda j: (scores[j],j)) if eligible else None
        selectors.append({"root_id": root, "model_linear_only_selection": choices})
    summary["prefix_actual_rgb_margins"] = rgb
    summary["prefix_actual_pose_margins"] = pose
    summary["prefix_predicted_rgb_margins"] = predicted_rgb
    summary["model_score_only_diagnostic_selections"] = selectors
    summary["nonzero_prefix_rgb_sign_agreement"] = int(np.sum(np.sign(np.asarray(rgb)[:,1:]) == np.sign(np.asarray(predicted_rgb)[:,1:])))
    return summary, matrix


def verify_stages(path, report):
    if json.loads((path.parent/"protocol.json").read_text()) != report["protocol"]:
        raise ValueError("sealed protocol changed")
    upstream = Path(report["upstream_path"])
    if digest(upstream) != report["upstream_sha256"]:
        raise ValueError("upstream receipt changed")
    seal = json.loads((path.parent/"prerun_seal.json").read_text())
    for item in seal["inputs"]:
        if digest(Path(item["snapshot"])) != item["sha256"]:
            raise ValueError("pre-run implementation/upstream snapshot changed")
    for name, sha in report["protocol"]["extra_source_sha256"].items():
        if digest(path.parent/"prerun_inputs"/(name+".py")) != sha:
            raise ValueError("pre-run auditor/solver snapshot changed")
    for index in range(4):
        plan_path = path.parent/"stages"/f"root_{index}.plan.json"
        plan = json.loads(plan_path.read_text())
        if plan["future_physical_labels_used"] is not False or len(plan["predictions"]) != 16:
            raise ValueError("root plan causal seal invalid")
    for row in report["rows"]:
        plan_path = path.parent/"stages"/f"root_{row['root_id']}.plan.json"
        plan = json.loads(plan_path.read_text())
        if plan["predictions"][row["proposal_index"]] != row["predicted"]:
            raise ValueError("pre-execution bank prediction differs")
        for mode, branch in row["branches"].items():
            stage_path = Path(branch["stage_path"])
            stage = json.loads(stage_path.read_text())
            if (digest(stage_path) != branch["stage_sha256"] or stage["root_plan_sha256"] != digest(plan_path)
                    or stage["future_suffix_observations_used"] is not False
                    or stage["forecast"] != branch["suffix_receipt"]
                    or stage["observed_successor_vector"] != branch["observed_prefix_vector"]
                    or stage["suffix_action"] != branch["suffix_action"] or stage["mode"] != mode):
                raise ValueError("causal suffix receipt changed")


def plot(summary, matrix, output):
    # Styling follows figure-generation's line/heatmap template; no selected
    # subset, outlier clipping, or internal ledger identifiers in figure text.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3,1,figsize=(11,8),layout="constrained")
    for i,(ax,label) in enumerate(zip(axes,("Predicted RGB MSE change", "Actual RGB MSE change", "Actual pose-cost change"))):
        for mode,color,legend in ((MODES[1],"#E69F00","Full objective, one-shot"),
                                  (MODES[2],"#0072B2","Full objective, 4-step refinement")):
            ax.plot(np.arange(64),matrix[mode][:,i]-matrix[MODES[0]][:,i],"o-",color=color,
                    label=legend,markersize=3,linewidth=.8)
        ax.axhline(0,color="black",linewidth=.7)
        for x in (15.5,31.5,47.5): ax.axvline(x,color="gray",linewidth=.6,alpha=.6)
        ax.set(ylabel=label,xlabel="All 64 fixed root-candidate pairs; lower is better")
    axes[0].legend(frameon=False)
    for ext in ("png","pdf"): fig.savefig(output/("Figure_1."+ext),bbox_inches="tight")
    plt.close(fig)
    fig,axes = plt.subplots(2,1,figsize=(11,5.6),layout="constrained")
    for ax,key,label in zip(axes,("prefix_actual_rgb_margins","prefix_actual_pose_margins"),
                            ("Actual RGB MSE margin", "Actual block-pose margin")):
        values = np.asarray(summary[key])
        scale = max(float(np.max(np.abs(values))),1e-8)
        im = ax.imshow(values,cmap="PuOr",vmin=-scale,vmax=scale,aspect="auto")
        ax.set(xticks=range(16),yticks=range(4),xlabel="All 16 fixed proposals (0 = raw zero)",ylabel="Known calibration root")
        for i,row in enumerate(summary["model_score_only_diagnostic_selections"]):
            j = row["model_linear_only_selection"][MODES[0]]
            if j is not None:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor="black",linewidth=2))
                ax.text(j,i,f"{values[i,j]:+.3f}" if key == "prefix_actual_rgb_margins" else f"{values[i,j]:+.2f}",
                        ha="center",va="center",fontsize=9,color="black",
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":1})
        ax.text(0,1.03,"Positive = non-greedy vs zero; boxes = original model selections",transform=ax.transAxes,fontsize=9)
        fig.colorbar(im,ax=ax,label=label)
    for ext in ("png","pdf"): fig.savefig(output/("Figure_2."+ext),bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    summary,matrix = summarize(report)
    verify_stages(args.report,report)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plot(summary,matrix,args.output_dir)
    summary["report_sha256"] = digest(args.report)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (args.output_dir/"notes.txt").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (args.output_dir/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All 64 fixed root-candidate pairs: exact full-coordinate suffix objectives and four-step bounded refinement versus the legacy random-eight-coordinate one-shot solver. Predictions and real RGB errors can improve while physical block-pose cost worsens. This is a known-calibration, unequal-compute diagnostic, not a policy or generalization result.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Actual prefix margins against the same-root zero control for all candidates. Black boxes indicate the original model-only selections; no root is removed. The RGB and block-pose panels have separate labeled scales: they are different task metrics, not interchangeable measures of prediction fidelity.}
\end{figure}
""")
    (args.output_dir/"manifest.json").write_text(json.dumps({"report_sha256":digest(args.report),
        "analyzer_sha256":digest(Path(__file__)),"artifacts":{p.name:digest(p) for p in args.output_dir.iterdir()}},indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if "margins" not in k},indent=2))


if __name__ == "__main__": main()
