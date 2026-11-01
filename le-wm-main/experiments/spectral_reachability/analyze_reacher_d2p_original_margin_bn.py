"""Independently verify and display the complete BN 2x2 intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_original_margin_checkpoint import check_cell, summarize as validate_crossover
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

VARIANTS = ("receipt_params/receipt_bn", "receipt_params/official_bn", "official_params/receipt_bn", "official_params/official_bn")
LABELS = ("Repair parameters / repair BN", "Repair parameters / base BN", "Base parameters / repair BN", "Base parameters / base BN")


def summarize(report, crossover):
    required = {"cuda_only", "all6_fixed_roots", "all4_variants_each_root", "both_native_baselines_bit_exact",
        "only_declared_bn_buffers_changed", "forward_sample_accounting", "inputs_unchanged",
        "crossover_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_original_margin_bn.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict BN diagnostic required")
    validate_crossover(crossover)
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    prior = {r["seed"]: r for r in crossover["rows"]}
    seeds = [r["seed"] for r in rows]
    if (len(seeds) != 6 or len(set(seeds)) != 6 or seeds != sorted(prior)
            or seeds != sorted(report["sealed_protocol"]["root_seeds"]) or report["actual_model_predict_samples"] != 48
            or report["sealed_protocol"]["physical_labels_used_for_planning"] is not False):
        raise ValueError("fixed all-six identity/compute contract required")
    summary = {}
    for row in rows:
        if set(row["variants"]) != set(VARIANTS):
            raise ValueError("all four 2x2 variants required")
        for key, cell in row["variants"].items():
            check_cell(cell)
            for field in ("observed", "observed_zero", "goal"):
                if cell["vectors"][field] != prior[row["seed"]]["checkpoints"]["receipt_checkpoint"]["float32"]["vectors"][field]:
                    raise ValueError("observed successor or goal chart changed")
        for source in ("receipt", "official"):
            cell = row["variants"][source + "_params/" + source + "_bn"]
            if cell["vectors"] != prior[row["seed"]]["checkpoints"][source + "_checkpoint"]["float32"]["vectors"]:
                raise ValueError("native checkpoint baseline not bit exact")
    for variant in VARIANTS:
        cells = [r["variants"][variant] for r in rows]
        summary[variant] = {"order_agreements": sum(c["decomposition"]["predicted_margin"] * c["decomposition"]["observed_margin"] > 0 for c in cells),
            "mean_candidate_latent_l2": float(np.mean([c["candidate_latent_l2"] for c in cells])),
            "mean_zero_latent_l2": float(np.mean([c["zero_latent_l2"] for c in cells]))}
    if summary != report["summary"]:
        raise ValueError("BN summary does not match full stored vectors")
    return {"root_seeds": seeds, "variants": summary, "rows": rows, "scope": report["sealed_protocol"]["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol = report["sealed_protocol"]
    protocol_path = args.report.with_suffix(".protocol.json")
    crossover_path = Path(protocol["crossover_path"])
    if (digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != protocol
            or digest(crossover_path) != protocol["crossover_sha256"]):
        raise ValueError("protocol or crossover seal changed")
    summary = summarize(report, json.loads(crossover_path.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    rows, x = summary["rows"], np.arange(1, 7)
    colors, markers = ("#0072B2", "#009E73", "#E69F00", "#D55E00"), ("o", "D", "^", "s")
    fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
    for j, (variant, label, color, marker) in enumerate(zip(VARIANTS, LABELS, colors, markers)):
        ax.scatter(x + (j - 1.5) * .12, [r["variants"][variant]["decomposition"]["predicted_margin"] for r in rows],
            color=color, marker=marker, s=40, label=label)
    ax.scatter(x, [r["variants"][VARIANTS[0]]["decomposition"]["observed_margin"] for r in rows], color="black", marker="x", s=45, label="Actual RGB margin")
    ax.axhline(0, color=".6", linewidth=.8)
    ax.set_yscale("symlog", linthresh=.001)
    ax.margins(y=.18)
    ax.set(xticks=x, xlabel="Original root ordinal (all six)", ylabel="Candidate minus neutral goal cost")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.26), frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, layout="constrained")
    for ax, branch, title in zip(axes, ("candidate", "zero"), ("Candidate successor", "Neutral successor")):
        for j, (variant, label, color, marker) in enumerate(zip(VARIANTS, LABELS, colors, markers)):
            ax.scatter(x + (j - 1.5) * .12, [r["variants"][variant][branch + "_latent_l2"] for r in rows],
                color=color, marker=marker, s=32, label=label)
        high = max(r["variants"][variant][branch + "_latent_l2"] for r in rows for variant in VARIANTS)
        ax.set(xticks=x, xlabel="Original root ordinal (all six)", ylim=(0, high * 1.18))
        ax.text(.02, .98, title, transform=ax.transAxes, va="top")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Predicted versus actual RGB latent L2 error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=8, loc="outside upper center", frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    (args.output_dir / "notes.txt").write_text(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Full 2-by-2 intervention on learned dynamics parameters and prediction-projection BatchNorm running statistics. All original six roots, goals, actions, and actual successors are retained. The in-memory diagnostic variants are not deployed selectors or P1 admissions.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{All-root prediction errors for candidate and neutral successors under the same four fixed interventions. Only the prediction-head running-statistic buffers are exchanged; model parameters and other buffers are unchanged within each parameter family.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
