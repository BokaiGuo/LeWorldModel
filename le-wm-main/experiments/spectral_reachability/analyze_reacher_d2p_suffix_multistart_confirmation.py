"""Verify fresh suffix seals, root identities and physical metrics; plot all."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.reacher_d2p_suffix_multistart import NAMES, choose_candidate


FAMILIES = ("dispatch", "greedy", "zero", "cem")


def paired(first, second):
    result = paired_difference(first, second)
    result["metric"] = "wrapped joint distance difference; negative favors first; prospective pilot, descriptive only"
    return result


def summarize(report, prefix_seal, suffix_seal):
    required = {"cuda_only", "weights_unchanged", "all_declared_roots", "source_episodes_disjoint", "all_execution_gates",
        "all24_branches_per_root", "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_suffix_multistart_confirmation.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict integrity-passing multistart evidence required")
    protocol = report["sealed_protocol"]
    if protocol["physical_labels_used_for_planning"] is not False or protocol["all_suffix_choices_sealed_before_terminal_execution"] is not True:
        raise ValueError("sealed model-only protocol required")
    rows = sorted(report["rows"], key=lambda row: row["seed"])
    seeds = sorted(protocol["root_seeds"])
    prefix = {row["seed"]: row for row in prefix_seal["rows"]}
    suffix = {(row["seed"], row["family"]): row for row in suffix_seal["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds
            or len(prefix) != len(prefix_seal["rows"]) or sorted(prefix) != seeds
            or len(suffix) != len(suffix_seal["rows"]) or set(suffix) != {(seed, family) for seed in seeds for family in FAMILIES}):
        raise ValueError("complete duplicate-free root/family identity joins required")
    episodes = [row["source_episode"] for row in rows]
    if len(set(episodes)) != 8 or set(episodes) & set(protocol["excluded_source_episodes"]):
        raise ValueError("fresh source-episode disjointness required")
    expected = {family + "_" + name for family in FAMILIES for name in (*NAMES, "model_best")}
    distances, rgbs, root_rows = {name: [] for name in expected}, {name: [] for name in expected}, []
    for row in rows:
        seed = row["seed"]
        if any(row[key] != prefix[seed][key] for key in ("source_episode", "source_start_row")):
            raise ValueError("prefix source identity changed")
        if (set(row["methods"]) != expected or set(row["branches"]) != set(FAMILIES)
                or not row.get("gates") or not all(v is True for v in row["gates"].values())):
            raise ValueError("all 24 physical branches and strict gates required")
        selections, model_costs = {}, {}
        for family in FAMILIES:
            branch = row["branches"][family]
            if branch != suffix[seed, family] or any(branch[key] != row[key] for key in ("source_episode", "source_start_row")):
                raise ValueError("sealed suffix/source changed")
            receipt = branch["receipt"]
            best = choose_candidate(receipt["model_costs"])
            if (receipt["physical_labels_used"] is not False or receipt["selected"] != best
                    or row["methods"][family + "_model_best"] != row["methods"][family + "_" + best]):
                raise ValueError("model-only incumbent selection changed")
            selections[family], model_costs[family] = best, receipt["model_costs"]
            for name in (*NAMES, "model_best"):
                key = family + "_" + name
                method = row["methods"][key]
                actions = np.asarray(method["action_witness_raw"])
                if (actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6
                        or not np.array_equal(actions[:20], branch["prefix_actions_raw"])
                        or not np.array_equal(actions[20:], branch["suffix_actions_raw"][best if name == "model_best" else name])):
                    raise ValueError("bounded sealed same-prefix action witness required")
                distance, rgb = (method["physical"][k] for k in ("final_qpos_distance", "final_rgb_cost"))
                if not np.isfinite([distance, rgb]).all() or min(distance, rgb) < 0:
                    raise ValueError("finite nonnegative physical outcomes required")
                distances[key].append(distance)
                rgbs[key].append(rgb)
        decision = prefix[seed]["receipt"]["dispatch"]
        root_rows.append({"seed": seed, "selections": selections, "model_costs": model_costs,
            "model_witness": decision["reconfiguration_witness_found"],
            "actual_rgb_non_greedy": row["branches"]["dispatch"]["prefix_rgb_cost"] >= row["branches"]["zero"]["prefix_rgb_cost"] + 1e-6,
            "distances": {name: values[-1] for name, values in distances.items()}})
    means = {name: float(np.mean(values)) for name, values in distances.items()}
    if means != report["aggregate"]["mean_distance"] or report["aggregate"]["physical_branches"] != 192:
        raise ValueError("aggregate or physical branch count drift")
    repairs = {family: paired(distances[family + "_model_best"], distances[family + "_incumbent"]) for family in FAMILIES}
    comparisons = {family: paired(distances["dispatch_model_best"], distances[family + "_model_best"]) for family in FAMILIES[1:]}
    eligible = [r for r in root_rows if r["model_witness"]]
    return {"root_seeds": seeds, "physical_branches": 192, "excluded_prior_source_episodes": len(protocol["excluded_source_episodes"]),
        "mean_distance": means, "mean_rgb_cost": {name: float(np.mean(values)) for name, values in rgbs.items()},
        "within_prefix_repairs": repairs, "dispatch_comparisons": comparisons,
        "model_witnesses": len(eligible), "witness_actual_rgb_non_greedy": sum(r["actual_rgb_non_greedy"] for r in eligible),
        "rows": root_rows, "actual_model_predict_samples": report["actual_model_predict_samples"],
        "scope": "Prospective numerical repair pilot, unequal compute. Paired bootstrap descriptive only, no multiplicity adjustment or P1 admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    prefix_path, suffix_path, protocol_path = (args.report.with_suffix(s) for s in (".selection.json", ".suffix.selection.json", ".protocol.json"))
    prefix, suffix = json.loads(prefix_path.read_text()), json.loads(suffix_path.read_text())
    if (digest(prefix_path) != report["selection_sha256"] or digest(suffix_path) != report["suffix_selection_sha256"]
            or suffix["prefix_selection_sha256"] != digest(prefix_path) or json.loads(protocol_path.read_text()) != report["sealed_protocol"]
            or prefix["protocol_sha256"] != digest(protocol_path) or suffix["protocol_sha256"] != digest(protocol_path)):
        raise ValueError("protocol/prefix/suffix seals changed")
    summary = summarize(report, prefix, suffix)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, family, label in zip(axes, FAMILIES, ("Dispatch", "Greedy", "Neutral", "Full CEM")):
        for row in summary["rows"]:
            a, b = (row["distances"][family + "_" + name] for name in ("incumbent", "model_best"))
            ax.plot([0, 1], [a, b], "o-", color="#0072B2" if b <= a else "#E69F00", markersize=3, alpha=.8)
        ax.set(xticks=[0, 1], xticklabels=["Local", "Multistart"], xlabel=label)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final wrapped joint distance")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    x = np.arange(1, 9)
    for i, (family, label, color, marker) in enumerate(zip(FAMILIES, ("Dispatch", "Greedy", "Neutral", "Full CEM"),
            ("#0072B2", "#E69F00", "#009E73", "#CC79A7"), ("o", "s", "^", "D"))):
        values = [r["distances"][family + "_model_best"] for r in summary["rows"]]
        ax.scatter(x + (i - 1.5) * .12, values, label=label, color=color, marker=marker, s=35)
    ax.set(xticks=x, xlabel="Prospective source-disjoint root ordinal (all eight)", ylabel="Final wrapped joint distance")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.16), frameon=False)
    ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight prospective source-episode-disjoint roots. Each segment compares the same prefix with its local cold/warm suffix incumbent and model-only multistart selection. Orange denotes physical worsening despite model-cost selection. Four prefix families receive the repair; planning compute differs.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Every prospective root after numerical suffix repair, including unchanged word dispatch, same-bank greedy, neutral, and full CEM prefixes. Model-only suffix choices were sealed across all roots before terminal measurements. This is a repair pilot, not P1 admission or a resource-matched mechanism comparison.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({key: summary[key] for key in ("within_prefix_repairs", "dispatch_comparisons", "model_witnesses", "witness_actual_rgb_non_greedy", "scope")}, indent=2))


if __name__ == "__main__":
    main()
