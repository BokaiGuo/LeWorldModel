"""Exact NumPy separation of common baseline bias and action-response error.

Uses already measured counterfactual successors for evaluator diagnosis only;
none of these oracle terms is exposed to an action selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_packet_rollout_fidelity import summarize as validate_packet
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def decompose(predicted, predicted_zero, observed, observed_zero, goal):
    values = [np.asarray(v, np.float64) for v in (predicted, predicted_zero, observed, observed_zero, goal)]
    if any(v.shape != (192,) or not np.isfinite(v).all() for v in values):
        raise ValueError("finite full192-coordinate vectors required")
    p, pz, o, oz, g = values
    dp, do = p - pz, o - oz
    predicted_margin = float(np.mean((p - g) ** 2) - np.mean((pz - g) ** 2))
    observed_margin = float(np.mean((o - g) ** 2) - np.mean((oz - g) ** 2))
    bias = float(2 * np.mean((oz - pz) * dp))
    linear = float(2 * np.mean((oz - g) * (do - dp)))
    quadratic = float(np.mean(do ** 2) - np.mean(dp ** 2))
    error = observed_margin - predicted_margin
    identity_error = abs(error - bias - linear - quadratic)
    if identity_error > 1e-12 * max(1., abs(error), abs(bias), abs(linear), abs(quadratic)):
        raise ValueError("exact margin decomposition failed")
    return {"predicted_margin": predicted_margin, "observed_margin": observed_margin, "margin_error": error,
        "baseline_bias": bias, "response_linear": linear, "response_quadratic": quadratic,
        "oracle_zero_anchored_margin": predicted_margin + bias, "identity_abs_error": identity_error,
        "oracle_future_neutral_successor_used": True}


def summarize(report):
    original = validate_packet(report)
    output = []
    for row in sorted(report["rows"], key=lambda r: r["seed"]):
        packets = []
        for i in range(4):
            candidate, neutral = (row["branches"][f]["packets"][i]["vectors"] for f in ("native_dispatch", "zero"))
            packets.append({"packet": i + 1, **{kind: decompose(candidate[kind], neutral[kind], candidate["observed"], neutral["observed"], candidate["goal"])
                for kind in ("free", "teacher_forced")}})
        output.append({"seed": row["seed"], "packets": packets})
    return {"root_seeds": original["root_seeds"], "rows": output,
        "max_identity_abs_error": max(p[k]["identity_abs_error"] for r in output for p in r["packets"] for k in ("free", "teacher_forced")),
        "scope": "All8 known-root evaluator-only exact error attribution; actual neutral future is NOT available to deployed planning or P1 admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol_path = args.report.with_suffix(".protocol.json")
    if digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]:
        raise ValueError("original packet protocol seal changed")
    result = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", sharey=True)
    for ax, kind, label in zip(axes, ("free", "teacher_forced"), ("Free rollout", "Observed-history diagnosis")):
        cells = [r["packets"][-1][kind] for r in result["rows"]]
        x = np.arange(8)
        for shift, key, name, color in ((-.24, "baseline_bias", "Common neutral bias", "#0072B2"),
                (0, "response_linear", "Response linear error", "#E69F00"), (.24, "response_quadratic", "Response quadratic error", "#009E73")):
            ax.bar(x + shift, [c[key] for c in cells], width=.23, label=name, color=color)
        ax.plot(x, [c["margin_error"] for c in cells], "kx", label="Total margin error")
        ax.text(.03, .97, label, va="top", transform=ax.transAxes)
        ax.axhline(0, color=".6", linewidth=.7)
        ax.set(xticks=x, xticklabels=[str(s)[-3:] for s in result["root_seeds"]], xlabel="Root suffix (all eight roots)")
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Observed minus predicted terminal margin")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for ax, row in zip(axes.flat, result["rows"]):
        for field, label, color, marker in (("predicted_margin", "Raw predicted", "#0072B2", "o"),
                ("oracle_zero_anchored_margin", "Oracle neutral anchored", "#E69F00", "s"), ("observed_margin", "Actual RGB", "black", "x")):
            ax.plot(range(1, 5), [p["free"][field] for p in row["packets"]], label=label, color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.text(.03, .97, str(row["seed"]), va="top", transform=ax.transAxes, fontsize=9)
        ax.axhline(0, color=".6", linewidth=.7)
        ax.set(xticks=range(1, 5), xlabel="Predicted packet count")
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Candidate minus neutral goal cost")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **result})
    (args.output_dir / "notes.txt").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Exact full-coordinate margin-error decomposition on every fixed root, separating common neutral-state bias from linear and quadratic action-response errors. Observed-history predictions are an evaluator-only diagnosis.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Replacing the predicted neutral anchor by its true future state isolates common-offset error but is an unavailable oracle, not a deployed correction. All eight roots and all packet boundaries are retained.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({"max_identity_abs_error": result["max_identity_abs_error"], "terminal_rows": [{"seed": r["seed"], **r["packets"][-1]["free"]} for r in result["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
