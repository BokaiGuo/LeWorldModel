"""Strict independent all-root displays for the past-only residual test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.spectral_reachability.audit_reacher_d2p_packet_causal_residual import summarize
from experiments.spectral_reachability.analyze_reacher_d2p_packet_rollout_fidelity import summarize as validate_packet
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def load(path):
    report = json.loads(path.read_text())
    required = {"cuda_only", "all8_roots", "all_raw_vectors_bit_exact", "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "predict_sample_accounting", "packet_reference_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_packet_causal_residual.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())
            or report["actual_model_predict_samples"] != 136):
        raise ValueError("complete strict CUDA residual diagnostic required")
    protocol = report["sealed_protocol"]
    if (protocol["coefficient"] != 1. or protocol["coefficient_fitted"] is not False
            or protocol["candidate_successor_used_for_prediction"] is not False or protocol["physical_execution"] is not False
            or protocol["predictions_sealed_before_observation_join"] is not True
            or protocol["variants"] != ["raw", "terminal_unit", "recurrent_unit"] or protocol["goal_contract"] != "joint-feasible-v1"):
        raise ValueError("fixed coefficient past-only no-execution contract required")
    protocol_path, predictions_path = (path.with_suffix(ext) for ext in (".protocol.json", ".predictions.json"))
    if (digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != protocol
            or digest(predictions_path) != report["predictions_sha256"]):
        raise ValueError("residual protocol/prediction seal changed")
    predictions = json.loads(predictions_path.read_text())
    source_path = Path(protocol["packet_reference_path"])
    if predictions["protocol_sha256"] != report["protocol_sha256"] or digest(source_path) != protocol["packet_reference_sha256"]:
        raise ValueError("wrong prediction or upstream reference binding")
    source = json.loads(source_path.read_text())
    original = validate_packet(source)
    if original["root_seeds"] != protocol["root_seeds"]:
        raise ValueError("complete fixed root identity contract required")
    result = summarize(predictions, source)
    if result != report["summary"]:
        raise ValueError("saved residual summary does not match independent arithmetic")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = load(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    styles = (("raw", "Raw", "#0072B2", "o"), ("terminal_unit", "Unit residual at terminal", "#E69F00", "s"),
        ("recurrent_unit", "Unit residual per step", "#009E73", "^"), ("observed", "Actual RGB", "black", "x"))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for ax, row in zip(axes.flat, result["rows"]):
        for key, label, color, marker in styles:
            ax.plot(range(1, 5), row["margins"][key], label=label, color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.text(.03, .97, str(row["seed"]), va="top", transform=ax.transAxes, fontsize=9)
        ax.axhline(0, color=".6", linewidth=.7)
        ax.set(xticks=range(1, 5), xlabel="Predicted packet count")
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Candidate minus neutral goal cost")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for key, label, color, marker in styles[:3]:
        ax.plot(range(1, 5), result["mean_latent_l2_by_packet"][key], label=label, color=color, marker=marker)
    ax.set(xticks=range(1, 5), xlabel="Predicted packet count", ylabel="Mean full-coordinate latent L2 error")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **result})
    (args.output_dir / "notes.txt").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight known roots and unchanged candidate/neutral action plans. Unit residuals are computed only from the already executed transition preceding the root, never candidate successors. Correcting once at the terminal or after every step is a fixed diagnostic, not a learned or selected deployment model.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Prediction error across all sixteen fixed paths. Lower average error under a common residual shift does not guarantee action ranking fidelity; no variant opens P1 or claims physical benefit.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({"reversed_final_roots": result["reversed_final_roots"], "mean_latent_l2_by_packet": result["mean_latent_l2_by_packet"]}), flush=True)


if __name__ == "__main__":
    main()
