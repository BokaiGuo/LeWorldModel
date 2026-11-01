"""All-word/all-root seal verification and numerical-contract audit figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.audit_reacher_d2p_state_native_selector import FAMILIES
from experiments.spectral_reachability.reacher_d2p_state_suffix import choose_state_candidate
from experiments.spectral_reachability.reacher_d2p_refined_dispatch import select_refined_bank
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission


def summarize(report, prefix_seal, suffix_seal):
    fresh = report.get("schema") == "reacher_d2p_state_native_confirmation.v1"
    required = {"cuda_only", "weights_unchanged", "all_declared_roots", "all17_words_every_root", "all_core_equivalence_gates",
        "all_physical_gates", "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"}
    required |= {"source_episodes_disjoint", "alignment_unchanged", "exclusions_unchanged"} if fresh else {"reference_unchanged"}
    if (report.get("schema") not in ("reacher_d2p_state_native_selector_audit.v1", "reacher_d2p_state_native_confirmation.v1")
            or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict integrity-passing native-selector evidence required")
    protocol = report["sealed_protocol"]
    # Old immutable evidence predates explicit versioning; never relabel it
    # as a physically admitted task, even if some roots happen to be legal.
    goal_contract = protocol.get("goal_contract", "legacy-periodic-replay")
    if goal_contract == "joint-feasible-v1" and any(report["gates"].get(k) is not True for k in ("explicit_goal_contract", "all_goal_admissions")):
        raise ValueError("feasible goal contract requires explicit admission gates")
    if protocol["physical_labels_used_for_planning"] is not False or protocol["all_suffix_choices_sealed_before_terminal_execution"] is not True:
        raise ValueError("sealed model-only choices required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    prefix = {r["seed"]: r for r in prefix_seal["rows"]}
    suffix = {(r["seed"], r["family"]): r for r in suffix_seal["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds
            or len(prefix) != len(prefix_seal["rows"]) or sorted(prefix) != seeds
            or len(suffix) != len(suffix_seal["rows"]) or set(suffix) != {(s, f) for s in seeds for f in FAMILIES}):
        raise ValueError("all unique root/family identities required")
    if fresh and (len({r["source_episode"] for r in rows}) != 8
            or {r["source_episode"] for r in rows} & set(protocol["excluded_source_episodes"])):
        raise ValueError("fresh roots must use unique, excluded-episode-disjoint sources")
    expected = {family + "_" + variant for family in FAMILIES for variant in ("cold", "carry", "native")} | {"old_dispatch_historical"}
    distances, rgb_costs = {name: [] for name in expected}, {name: [] for name in expected}
    roots, old_score_matrix, native_score_matrix = [], [], []
    for row in rows:
        seed, choice = row["seed"], prefix[row["seed"]]
        validate_goal_admission(goal_contract, choice.get("goal_admission"))
        if goal_contract == "joint-feasible-v1" and choice["goal_admission"]["requested_delta"] != protocol["target_delta"]:
            raise ValueError("goal delta differs from sealed task contract")
        if any(row[k] != choice[k] for k in ("source_episode", "source_start_row")):
            raise ValueError("source identity mismatch")
        if (set(row["methods"]) != expected or set(row["branches"]) != set(FAMILIES)
                or not row.get("gates") or not all(v is True for v in row["gates"].values())
                or not choice.get("gates") or not all(v is True for v in choice["gates"].values())):
            raise ValueError("all16 branches and strict per-root gates required")
        words, score_rows = choice["word_records"], choice["score_rows"]
        if (len(words) != 17 or len(score_rows) != 17 or [w["index"] for w in words] != list(range(17))
                or [s["proposal_index"] for s in score_rows] != list(range(17))):
            raise ValueError("all17 unique ordered word scores required")
        if choice["native_decision"] != select_refined_bank(score_rows, non_greedy_margin=protocol["non_greedy_margin"]):
            raise ValueError("native prefix selection changed")
        old_scores, new_scores = [], []
        for word, score in zip(words, score_rows):
            receipt = word["native_receipt"]
            chosen = choose_state_candidate(receipt["model_costs"])
            commands = np.asarray(word["planned_commands"])
            if (receipt["physical_labels_used"] is not False or receipt["selected"] != chosen
                    or commands.shape != (1, 2, 10) or not np.isfinite(commands).all() or np.abs(commands).max() > 1
                    or not np.isfinite([score["immediate_goal_cost"], word["original_fine8_cost"]]).all()
                    or min(score["immediate_goal_cost"], word["original_fine8_cost"]) < 0
                    or score["refined_goal_cost"] != receipt["model_costs"][chosen]
                    or score["refined_goal_cost"] > word["original_fine8_cost"]
                    or not np.isfinite(word["partition_max_abs_error"])
                    or not 0 <= word["partition_max_abs_error"] <= protocol["partition_atol"]):
                raise ValueError("imagined word numerical/witness contract failed")
            old_scores.append(word["original_fine8_cost"])
            new_scores.append(score["refined_goal_cost"])
        old_score_matrix.append(old_scores)
        native_score_matrix.append(new_scores)
        old, new = choice["historical_decision"], choice["native_decision"]
        old_rows = [{**score, "refined_goal_cost": cost} for score, cost in zip(score_rows, old_scores)]
        if old != select_refined_bank(old_rows, non_greedy_margin=protocol["non_greedy_margin"]):
            raise ValueError("historical prefix selection changed")
        indices = {"old_dispatch": old["selected_program_index"], "native_dispatch": new["selected_program_index"],
            "greedy": new["same_bank_greedy_index"], "zero": 0}
        for family in FAMILIES:
            branch = row["branches"][family]
            if (branch != suffix[seed, family] or any(branch[k] != row[k] for k in ("source_episode", "source_start_row"))
                    or branch["prefix_index"] != indices.get(family)):
                raise ValueError("sealed suffix/source/prefix-index mismatch")
            if not np.isfinite(branch["prefix_rgb_cost"]) or branch["prefix_rgb_cost"] < 0:
                raise ValueError("finite nonnegative observed prefix RGB cost required")
            receipt = branch["receipt"]
            if receipt["physical_labels_used"] is not False or receipt["selected"] != choose_state_candidate(receipt["model_costs"]):
                raise ValueError("observed suffix selection changed")
            if family in indices and branch["planned_commands"] != words[indices[family]]["planned_commands"]:
                raise ValueError("selected imagined command witness was not carried exactly")
            variants = ("cold", "carry", "native", "historical") if family == "old_dispatch" else ("cold", "carry", "native")
            for variant in variants:
                name = family + "_" + variant
                method = row["methods"][name]
                actions = np.asarray(method["action_witness_raw"])
                if (actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6
                        or not np.array_equal(actions[:20], branch["prefix_actions_raw"])
                        or not np.array_equal(actions[20:], branch["suffix_actions_raw"][variant])):
                    raise ValueError("bounded sealed complete action witness required")
                distance, rgb = (method["physical"][k] for k in ("final_qpos_distance", "final_rgb_cost"))
                if not np.isfinite([distance, rgb]).all() or min(distance, rgb) < 0:
                    raise ValueError("finite nonnegative physical outcomes required")
                distances[name].append(distance)
                rgb_costs[name].append(rgb)
        if not new["reconfiguration_witness_found"] and row["methods"]["native_dispatch_native"] != row["methods"]["greedy_native"]:
            raise ValueError("fallback must execute identical greedy actions/outcomes")
        old_immediate = score_rows[old["selected_program_index"]]["immediate_goal_cost"] - score_rows[0]["immediate_goal_cost"]
        new_immediate = score_rows[new["selected_program_index"]]["immediate_goal_cost"] - score_rows[0]["immediate_goal_cost"]
        old_actual = row["branches"]["old_dispatch"]["prefix_rgb_cost"] - row["branches"]["zero"]["prefix_rgb_cost"]
        new_actual = row["branches"]["native_dispatch"]["prefix_rgb_cost"] - row["branches"]["zero"]["prefix_rgb_cost"]
        global_best = min(range(17), key=lambda i: (score_rows[i]["refined_goal_cost"], i))
        roots.append({"seed": seed, "old_prefix": old["selected_program_index"], "native_prefix": new["selected_program_index"],
            "old_witness": old["reconfiguration_witness_found"], "native_witness": new["reconfiguration_witness_found"],
            "old_model_immediate_margin": old_immediate, "native_model_immediate_margin": new_immediate,
            "old_actual_rgb_immediate_margin": old_actual, "native_actual_rgb_immediate_margin": new_actual,
            "actual_rgb_non_greedy": new_actual >= protocol["non_greedy_margin"],
            "global_model_best_word_index": global_best,
            "global_model_best_is_non_greedy": global_best != 0 and score_rows[global_best]["immediate_goal_cost"]
                >= score_rows[0]["immediate_goal_cost"] + protocol["non_greedy_margin"],
            "model_cost_above_global_min": score_rows[new["selected_program_index"]]["refined_goal_cost"] - score_rows[global_best]["refined_goal_cost"],
            "distances": {name: values[-1] for name, values in distances.items()}})
    means = {name: float(np.mean(values)) for name, values in distances.items()}
    if means != report["aggregate"]["mean_distance"] or report["aggregate"]["physical_branches"] != 128:
        raise ValueError("aggregate or physical branch count drift")
    comparisons = {name: paired_difference(distances["native_dispatch_native"], distances[name]) for name in
        ("old_dispatch_historical", "old_dispatch_native", "greedy_native", "zero_native", "cem_native")}
    rgb_comparisons = {name: paired_difference(rgb_costs["native_dispatch_native"], rgb_costs[name]) for name in
        ("old_dispatch_historical", "old_dispatch_native", "greedy_native", "zero_native", "cem_native")}
    for comparison in rgb_comparisons.values():
        comparison["metric"] = "terminal observed RGB latent cost difference, negative favors first; descriptive pilot"
    within = {family: paired_difference(distances[family + "_native"], distances[family + "_carry"]) for family in FAMILIES}
    eligible = [r for r in roots if r["native_witness"]]
    return {"root_seeds": seeds, "goal_contract": goal_contract,
        "goal_reflected_roots": [s for s in seeds if prefix[s].get("goal_admission") and prefix[s]["goal_admission"]["flipped_joint_indices"]],
        "imagined_words": 136, "physical_branches": 128,
        "mean_distance": means, "mean_rgb_cost": {name: float(np.mean(values)) for name, values in rgb_costs.items()},
        "comparisons": comparisons, "rgb_comparisons": rgb_comparisons, "native_vs_carry": within,
        "old_score_matrix": old_score_matrix, "native_score_matrix": native_score_matrix,
        "prefix_choices_changed": sum(r["old_prefix"] != r["native_prefix"] for r in roots),
        "historical_witnesses": sum(r["old_witness"] for r in roots), "native_witnesses": len(eligible),
        "native_actual_rgb_non_greedy_witnesses": sum(r["actual_rgb_non_greedy"] for r in eligible),
        "rows": roots, "actual_model_predict_samples": report["actual_model_predict_samples"],
        "scope": ("Source-episode-disjoint numerical-contract confirmation pilot" if fresh else "Known-root numerical contract intervention")
            + ", unequal compute; root bootstrap descriptive, no P1 admission or theory verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    prefix_path, suffix_path, protocol_path = (args.report.with_suffix(s) for s in (".selection.json", ".suffix.selection.json", ".protocol.json"))
    prefix, suffix = json.loads(prefix_path.read_text()), json.loads(suffix_path.read_text())
    if (digest(prefix_path) != report["selection_sha256"] or digest(suffix_path) != report["suffix_selection_sha256"]
            or json.loads(protocol_path.read_text()) != report["sealed_protocol"] or suffix["prefix_selection_sha256"] != digest(prefix_path)
            or prefix["protocol_sha256"] != digest(protocol_path) or suffix["protocol_sha256"] != digest(protocol_path)):
        raise ValueError("protocol/prefix/suffix seals changed")
    summary = summarize(report, prefix, suffix)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    # Heatmap template: retain every word and root. No outcome-based filtering.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True, layout="constrained")
    old, new = np.array(summary["old_score_matrix"]), np.array(summary["native_score_matrix"])
    data = (np.log10(np.maximum(old, 1e-12)), np.log10(np.maximum(new, 1e-12)))
    lo, hi = min(m.min() for m in data), max(m.max() for m in data)
    for ax, matrix, label in zip(axes, data, ("Original fine8 imagined cost (log10)", "Native imagined cost (log10)")):
        im = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=lo, vmax=hi)
        ax.set(xticks=range(17), yticks=range(8), yticklabels=range(1, 9), xlabel="Action word index (all seventeen)")
        ax.tick_params(axis="x", labelsize=8)
        for i, root in enumerate(summary["rows"]):
            ax.add_patch(Rectangle((root["old_prefix"] - .47, i - .43), .94, .86, fill=False, edgecolor="white", linewidth=1.4, linestyle="--"))
            ax.add_patch(Rectangle((root["native_prefix"] - .38, i - .35), .76, .7, fill=False, edgecolor="#CC79A7", linewidth=1.6))
        fig.colorbar(im, ax=ax, location="top", label=label)
    ordinal_label = "Source root ordinal (all eight)"
    axes[0].set_ylabel(ordinal_label)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    styles = (("old_dispatch_historical", "Historical", "#888888", "o"),
        ("old_dispatch_native", "Old prefix / native tail", "#56B4E9", "s"),
        ("native_dispatch_native", "Native prefix / native tail", "#0072B2", "D"),
        ("greedy_native", "Greedy / native tail", "#E69F00", "^"),
        ("zero_native", "Neutral / native tail", "#009E73", "v"),
        ("cem_native", "Full CEM / native tail", "#CC79A7", "P"))
    x = np.arange(1, 9)
    for offset, (name, label, color, marker) in enumerate(styles):
        ax.scatter(x + (offset - 2.5) * .10, [r["distances"][name] for r in summary["rows"]], label=label, color=color, marker=marker, s=32)
    ax.set(xticks=x, xlabel=ordinal_label, ylabel="Final wrapped joint distance")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.22), fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Every action word at every declared root. Original fine8 and native two-chart/CEM imagined endpoint costs use one color scale. White dashed outlines mark original prefix choices; magenta outlines mark native choices. All numerical decisions are sealed before terminal measurements.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{All-root physical outcomes separate suffix-runtime repair from prefix-score intervention, with matched greedy, neutral, and full-CEM prefix controls. All decisions precede terminal outcomes. Planning compute differs; this is a numerical-contract experiment, not confirmatory mechanism efficacy.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("mean_distance", "comparisons", "prefix_choices_changed", "historical_witnesses",
        "native_witnesses", "native_actual_rgb_non_greedy_witnesses", "scope")}, indent=2))


if __name__ == "__main__":
    main()
