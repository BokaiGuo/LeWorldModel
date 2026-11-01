"""Bound-evidence postprocessing: causal margins and root-paired outcomes.

No model inference or simulator execution. Candidate counts are descriptive;
uncertainty resamples roots, never the dependent 136 word cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import choose_model_incumbent, write_new


def signed_counts(first, second):
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("matching finite candidate cost vectors required")
    x, y = a[1:] - a[0], b[1:] - b[0]
    return {"agreements": int(np.sum(x * y > 0)), "reversals": int(np.sum(x * y < 0)),
        "ties": int(np.sum(x * y == 0)), "nonzero_candidates": len(x)}


def paired_difference(first, reference):
    delta = np.asarray(first, np.float64) - np.asarray(reference, np.float64)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("finite root-level paired differences required")
    rng = np.random.default_rng(202643911)
    means = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
    return {"n_roots": len(delta), "mean": float(delta.mean()),
        "ci95_descriptive": np.quantile(means, [.025, .975]).tolist(),
        "wins": int(np.sum(delta < 0)), "losses": int(np.sum(delta > 0)), "ties": int(np.sum(delta == 0)),
        "metric": "wrapped joint distance difference, negative favors first; known roots, not confirmatory"}


def summarize(report):
    if (report.get("schema") != "reacher_d2p_suffix_continuity_audit.v1" or report.get("status") != "completed"
            or not report.get("gates") or not all(value is True for value in report["gates"].values())):
        raise ValueError("complete strict integrity-passing continuity report required")
    seeds = report["sealed_protocol"]["root_seeds"]
    rows = sorted(report["rows"], key=lambda row: row["seed"])
    if len(seeds) != len(set(seeds)) or len(rows) != len(seeds) or [row["seed"] for row in rows] != sorted(seeds):
        raise ValueError("all unique declared roots required")
    margins, distances = [], {name: [] for name in ("cold", "carry", "warm", "incumbent")}
    all_selected = []
    for row in rows:
        cells = sorted(row["cells"], key=lambda cell: cell["proposal_index"])
        if [cell["proposal_index"] for cell in cells] != list(range(17)):
            raise ValueError("all17 distinct word cells required")
        for cell in cells:
            if not cell.get("gates") or not all(value is True for value in cell["gates"].values()):
                raise ValueError("cell integrity gate failed")
            expected = choose_model_incumbent({name: cell["model_costs"][name] for name in ("cold", "warm")})
            if cell["incumbent"] != expected:
                raise ValueError("incumbent was not selected by frozen model cost")
            if any(not np.isfinite(cell["outcomes"][name][metric]) for name in ("cold", "carry", "warm")
                    for metric in ("final_qpos_distance", "final_rgb_cost")):
                raise ValueError("nonfinite physical outcome")
        chosen = cells[row["decision"]["selected_program_index"]]
        if row["selected"] != chosen:
            raise ValueError("selected cell alias changed")
        vectors = {"imagined": [cell["imagined_terminal_cost"] for cell in cells]}
        for name in ("cold", "carry", "warm"):
            vectors["model_" + name] = [cell["model_costs"][name] for cell in cells]
            vectors["actual_" + name] = [cell["outcomes"][name]["final_rgb_cost"] for cell in cells]
            distances[name].append(chosen["outcomes"][name]["final_qpos_distance"])
        vectors["model_incumbent"] = [cell["model_costs"][cell["incumbent"]] for cell in cells]
        vectors["actual_incumbent"] = [cell["outcomes"][cell["incumbent"]]["final_rgb_cost"] for cell in cells]
        distances["incumbent"].append(chosen["outcomes"][chosen["incumbent"]]["final_qpos_distance"])
        comparisons = (("state_transfer_same_tail", "imagined", "model_carry"),
            ("state_and_cold_reoptimization", "imagined", "model_cold"),
            ("carry_terminal_prediction", "model_carry", "actual_carry"),
            ("cold_terminal_prediction", "model_cold", "actual_cold"),
            ("warm_terminal_prediction", "model_warm", "actual_warm"),
            ("incumbent_terminal_prediction", "model_incumbent", "actual_incumbent"))
        margins.append({"seed": row["seed"], "vectors": vectors,
            "counts": {label: signed_counts(vectors[a], vectors[b]) for label, a, b in comparisons}})
        all_selected.append({"seed": row["seed"], "index": chosen["proposal_index"], "incumbent": chosen["incumbent"],
            "model_costs": chosen["model_costs"], "distances": {name: distances[name][-1] for name in distances}})
    means = {name: float(np.mean(value)) for name, value in distances.items()}
    if any(means[name] != report["aggregate"]["mean_selected_distance"][name] for name in ("cold", "carry", "warm")):
        raise ValueError("reported condition means changed")
    if means["incumbent"] != report["aggregate"]["incumbent_mean_selected_distance"]:
        raise ValueError("reported incumbent mean changed")
    return {"roots": len(rows), "candidates": 17 * len(rows), "means": means,
        "paired_vs_cold": {name: paired_difference(value, distances["cold"]) for name, value in distances.items() if name != "cold"},
        "margin_counts": {name: {key: sum(row["counts"][name][key] for row in margins) for key in ("agreements", "reversals", "ties", "nonzero_candidates")}
            for name in margins[0]["counts"]}, "selected": all_selected, "margin_rows": margins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol = report["sealed_protocol"]
    selection_path = args.report.with_suffix(".selection.json")
    if digest(selection_path) != report["selection_sha256"]:
        raise ValueError("global selection changed")
    for row in report["rows"]:
        for cell in row["cells"]:
            path = args.report.with_suffix(".roots") / f"suffix_{row['seed']}_{cell['proposal_index']}.selection.json"
            if digest(path) != cell["suffix_selection_sha256"]:
                raise ValueError("causal suffix selection changed")
            sealed = json.loads(path.read_text())
            if any(sealed[key] != cell[key] for key in ("model_costs", "incumbent")):
                raise ValueError("physical report differs from pre-outcome suffix seal")
    if digest(Path(protocol["reference_path"])) != protocol["reference_sha256"]:
        raise ValueError("historical reference changed")
    summary = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "legend.fontsize": 9, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    styles = (("cold", "Cold recompile", "#0072B2", "o"), ("carry", "Carry planned tail", "#E69F00", "s"),
        ("warm", "Warm recompile", "#CC79A7", "^"), ("incumbent", "Model-only incumbent", "#333333", "x"))
    for offset, (name, label, color, marker) in enumerate(styles):
        ax.scatter(np.arange(1, summary["roots"] + 1) + (offset - 1.5) * .12,
            [row["distances"][name] for row in summary["selected"]], label=label, color=color, marker=marker, s=45)
    ax.set(xlabel="Fixed reference-root ordinal", ylabel="Final wrapped joint distance", xticks=np.arange(1, summary["roots"] + 1))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.19), ncol=2, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    names = ("state_transfer_same_tail", "state_and_cold_reoptimization", "carry_terminal_prediction", "cold_terminal_prediction", "warm_terminal_prediction", "incumbent_terminal_prediction")
    labels = ("State transfer\n(same tail)", "State + cold\nreoptimization", "Carry tail:\nmodel vs RGB", "Cold tail:\nmodel vs RGB", "Warm tail:\nmodel vs RGB", "Incumbent:\nmodel vs RGB")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    bottom = np.zeros(len(names))
    for key, label, color in (("agreements", "Strict agreement", "#0072B2"), ("reversals", "Strict reversal", "#E69F00"), ("ties", "Tie (not agreement)", "#BBBBBB")):
        counts = np.array([summary["margin_counts"][name][key] for name in names])
        ax.bar(np.arange(len(names)), counts, bottom=bottom, label=label, color=color, width=.7)
        for x, base, count in zip(range(len(names)), bottom, counts):
            if count:
                ax.text(x, base + count / 2, str(count), ha="center", va="center", color="white" if key == "agreements" else "black")
        bottom += counts
    ax.set(xticks=np.arange(len(names)), xticklabels=labels, ylabel="Nonneutral candidate margins", ylim=(0, 16 * summary["roots"] + 5))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"schema": "reacher_d2p_suffix_continuity_summary.v1",
        "report_sha256": digest(args.report), "selection_sha256": report["selection_sha256"], **summary})
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {name: digest(args.output_dir / name) for name in ("Figure_1.png", "Figure_1.pdf", "Figure_2.png", "Figure_2.pdf", "summary.json")}})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Fixed-prefix suffix intervention on every reference root. The prefix choice remains sealed and identical. Cold, carried and warm-refined suffixes share the ten-step physical suffix horizon. The incumbent chooses cold or warm by observed-RGB model cost only. Known-root diagnostic; planning compute differs and no P1 admission is implied.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{All nonneutral candidate cost margins relative to the neutral word at the same root. The first comparison isolates imagined-to-observed state transfer with the same tail; subsequent comparisons separate reoptimization and terminal model-to-RGB ranking. The dependent candidate counts are descriptive, not independent statistical samples.}
\end{figure}
""")
    print(json.dumps({key: value for key, value in summary.items() if key != "margin_rows"}, indent=2))


if __name__ == "__main__":
    main()
