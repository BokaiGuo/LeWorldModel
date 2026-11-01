"""Independent stored-vector algebra audit and all-six-root diagnostic plots."""
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

NAMES = ("receipt_checkpoint", "official_checkpoint")


def check_cell(cell):
    v = cell["vectors"]
    keys = ("prediction", "prediction_zero", "observed", "observed_zero", "goal")
    if set(v) != set(keys):
        raise ValueError("all five latent vectors required")
    p, pz, o, oz, g = (np.asarray(v[k], np.float64) for k in keys)
    if any(a.shape != (192,) or not np.isfinite(a).all() for a in (p, pz, o, oz, g)):
        raise ValueError("finite full 192-coordinate vectors required")
    dp, do = p - pz, o - oz
    margin = float(np.mean((p - g) ** 2) - np.mean((pz - g) ** 2))
    actual = float(np.mean((o - g) ** 2) - np.mean((oz - g) ** 2))
    terms = {"predicted_margin": margin, "observed_margin": actual, "margin_error": actual - margin,
        "zero_successor_bias_contribution": float(2 * np.mean((oz - pz) * dp)),
        "response_linear_error_contribution": float(2 * np.mean((oz - g) * (do - dp))),
        "response_quadratic_error_contribution": float(np.mean(do ** 2) - np.mean(dp ** 2))}
    if any(not np.isclose(value, cell["decomposition"][k], rtol=0, atol=1e-11) for k, value in terms.items()):
        raise ValueError("stored decomposition differs from independently recomputed vectors")
    if any(not np.isclose(np.mean((np.asarray(v[k]) - g) ** 2), value, rtol=0, atol=2e-6) for k, value in cell["costs"].items()):
        raise ValueError("stored costs differ from vectors")
    for branch, pred, obs in (("candidate", p, o), ("zero", pz, oz)):
        if not np.isclose(np.linalg.norm(pred - obs), cell[branch + "_latent_l2"], rtol=0, atol=2e-6):
            raise ValueError("stored latent error differs from vectors")
    return terms


def summarize(report):
    required = {"cuda_only", "all6_fixed_roots", "all12_physical_packets_exact_replay", "all_original_four_costs_replay",
        "all_checkpoints_frozen_eval", "forward_sample_accounting", "inputs_unchanged", "implementation_unchanged",
        "action_embedding_reference_fp32_bit_exact"}
    if (report.get("schema") != "reacher_d2p_original_margin_checkpoint.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict checkpoint/precision diagnostic required")
    protocol, rows = report["sealed_protocol"], sorted(report["rows"], key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    if (len(seeds) != 6 or len(set(seeds)) != 6 or seeds != sorted(protocol["root_seeds"])
            or report["actual_model_predict_samples"] != 48
            or protocol["physical_labels_used_for_planning"] is not False):
        raise ValueError("complete root identities, compute and no-label contract required")
    summary = {}
    for name in NAMES:
        for precision in ("float32", "float64"):
            cells = [r["checkpoints"][name][precision] for r in rows]
            values = [check_cell(c) for c in cells]
            summary[name + "/" + precision] = {"order_agreements": sum(t["predicted_margin"] * t["observed_margin"] > 0 for t in values),
                "reversed_roots": [s for s, t in zip(seeds, values) if t["predicted_margin"] * t["observed_margin"] < 0],
                "mean_candidate_latent_l2": float(np.mean([c["candidate_latent_l2"] for c in cells])),
                "mean_zero_latent_l2": float(np.mean([c["zero_latent_l2"] for c in cells]))}
    if summary != report["summary"]:
        raise ValueError("summary does not match all stored root vectors")
    encoder_difference = max(float(np.max(np.abs(np.asarray(r["checkpoints"][NAMES[0]]["float32"]["vectors"][k])
        - np.asarray(r["checkpoints"][NAMES[1]]["float32"]["vectors"][k])))) for r in rows for k in ("observed", "observed_zero", "goal"))
    precision_drift = {name: max(abs(r["checkpoints"][name]["float32"]["decomposition"]["predicted_margin"]
        - r["checkpoints"][name]["float64"]["decomposition"]["predicted_margin"]) for r in rows) for name in NAMES}
    return {"root_seeds": seeds, "checkpoint_precision": summary, "max_cross_checkpoint_visual_chart_abs_difference": encoder_difference,
        "max_dynamics_precision_margin_difference": precision_drift,
        "scope": protocol["scope"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol_path = args.report.with_suffix(".protocol.json")
    if digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]:
        raise ValueError("protocol seal changed")
    for b in report["sealed_protocol"]["inputs"].values():
        if digest(Path(b["path"])) != b["sha256"]:
            raise ValueError("bound original evidence/checkpoint changed")
    summary = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    rows, x = summary["rows"], np.arange(1, 7)
    # Scatter/ablation template: all roots, symmetric scale, no root filtering.
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout="constrained")
    for j, (name, label, color, marker) in enumerate(((NAMES[0], "Receipt checkpoint", "#0072B2", "o"),
            (NAMES[1], "Official checkpoint", "#D55E00", "s"))):
        for k, precision in enumerate(("float32", "float64")):
            ax.scatter(x + (2 * j + k - 1.5) * .11,
                [r["checkpoints"][name][precision]["decomposition"]["predicted_margin"] for r in rows],
                facecolors=color if k == 0 else "none", edgecolors=color, marker=marker, s=45,
                label=label + (" / FP32" if k == 0 else " / FP64 dynamics"))
    ax.scatter(x, [r["checkpoints"][NAMES[0]]["float32"]["decomposition"]["observed_margin"] for r in rows],
        color="#222222", marker="x", s=50, label="Observed RGB margin (receipt chart)")
    ax.axhline(0, color=".6", linewidth=.8)
    ax.set(xticks=x, xlabel="Original root ordinal (all six)", ylabel="Candidate minus neutral goal cost")
    ax.set_yscale("symlog", linthresh=.001)
    ax.margins(y=.18)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.27), frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True, layout="constrained")
    for ax, name, label in zip(axes, NAMES, ("Receipt checkpoint", "Official checkpoint")):
        for j, (term, legend, color) in enumerate((("zero_successor_bias_contribution", "Neutral baseline bias", "#0072B2"),
                ("response_linear_error_contribution", "Response linear error", "#E69F00"),
                ("response_quadratic_error_contribution", "Response quadratic error", "#009E73"))):
            ax.bar(x + (j - 1) * .22, [r["checkpoints"][name]["float32"]["decomposition"][term] for r in rows],
                width=.21, color=color, label=legend)
        ax.scatter(x, [r["checkpoints"][name]["float32"]["decomposition"]["margin_error"] for r in rows],
            color="black", marker="D", s=18, label="Total margin error", zorder=4)
        ax.axhline(0, color=".6", linewidth=.8)
        ax.set(xticks=x, xlabel="Original root ordinal (all six)")
        ax.text(.02, .98, label, transform=ax.transAxes, va="top", fontsize=10)
        ax.set_yscale("symlog", linthresh=.001)
        ax.margins(y=.14)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Observed minus predicted margin: exact terms")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=8, loc="outside upper center", frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    (args.output_dir / "notes.txt").write_text(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All six original roots with unchanged rendered observations, goals, and sealed actions. Checkpoint and dynamics precision are diagnostic interventions, not a new selector. Open markers keep the original FP32 visual chart and evaluate dynamics in FP64. Root ordinals map to the machine-readable summary. Symmetric-log scale retains both signs. No efficacy or P1 admission is claimed.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Exact evaluator-only decomposition of observed minus predicted candidate--neutral margins into baseline bias and linear/quadratic response error. Terms are independently recomputed from the stored full vectors. Neutral successor labels are evaluator-only and cannot be used as a deployed prediction correction.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
