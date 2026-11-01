"""Independent NumPy packet-error decomposition and complete-root displays."""
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


def summarize(report):
    required = {"cuda_only", "all8_roots", "all16_fixed_prefixes_and64_boundaries_exact", "parameters_unchanged",
        "buffers_unchanged", "all_modules_eval", "predict_sample_accounting", "reference_and_decisions_unchanged",
        "weights_file_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_packet_rollout_fidelity.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict packet fidelity diagnostic required")
    protocol = report["sealed_protocol"]
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    if (len(seeds) != 8 or len(set(seeds)) != 8 or seeds != sorted(protocol["root_seeds"])
            or report["actual_model_predict_samples"] != 256 or protocol["physical_labels_used_for_planning"] is not False
            or protocol["released_rollout_equivalence"] is not True or protocol["packets_per_branch"] != 4
            or protocol["raw_steps_per_packet"] != 5 or protocol["families"] != ["native_dispatch", "zero"]
            or not np.isfinite(protocol["native_rollout_atol"]) or protocol["native_rollout_atol"] != 2e-6):
        raise ValueError("all-eight root, native-rollout, no-label and compute contract required")
    result, errors, native_errors = [], {"free": [], "teacher_forced": []}, []
    for row in rows:
        if set(row["branches"]) != {"native_dispatch", "zero"} or not isinstance(row["native_witness"], bool):
            raise ValueError("both original prefix branches required")
        costs = {}
        for family, branch in row["branches"].items():
            required_branch = {"manual_rollout_terminal_bit_exact", "released_rollout_all4_equivalent", "original_model_cost_replay",
                "original_actual_cost_replay", "all_boundaries_and_actions_exact"}
            if (not required_branch.issubset(branch.get("gates", {})) or not all(v is True for v in branch["gates"].values())
                    or len(branch["packets"]) != 4 or [p["packet"] for p in branch["packets"]] != [1, 2, 3, 4]):
                raise ValueError("all four ordered packet baselines required")
            values = {k: [] for k in ("free", "teacher_forced", "observed")}
            branch_errors = {k: [] for k in errors}
            native = np.asarray(branch["released_rollout_vectors"], np.float64)
            if native.shape != (4, 192) or not np.isfinite(native).all():
                raise ValueError("complete released rollout vectors required")
            branch_native_errors = []
            for i, packet in enumerate(branch["packets"]):
                if set(packet["vectors"]) != {"free", "teacher_forced", "observed", "goal"} or set(packet["costs"]) != set(values):
                    raise ValueError("complete vector/cost fields required")
                vectors = {k: np.asarray(v, np.float64) for k, v in packet["vectors"].items()}
                if any(v.shape != (192,) or not np.isfinite(v).all() for v in vectors.values()):
                    raise ValueError("finite complete latent vectors required")
                if i == 0 and packet["vectors"]["free"] != packet["vectors"]["teacher_forced"]:
                    raise ValueError("first-packet histories must be identical")
                if i and packet["vectors"]["goal"] != branch["packets"][0]["vectors"]["goal"]:
                    raise ValueError("goal changed between packets")
                for kind in values:
                    value = float(np.mean((vectors[kind] - vectors["goal"]) ** 2))
                    if not np.isclose(value, packet["costs"][kind], rtol=0, atol=2e-6):
                        raise ValueError("packet cost is inconsistent with full vectors")
                    values[kind].append(value)
                for kind in errors:
                    branch_errors[kind].append(float(np.linalg.norm(vectors[kind] - vectors["observed"])))
                native_errors.append(float(np.max(np.abs(native[i] - vectors["free"]))))
                branch_native_errors.append(native_errors[-1])
            if any(v > protocol["native_rollout_atol"] for v in native_errors):
                raise ValueError("released rollout changed model recurrence")
            if branch["released_rollout_max_abs_difference"] != max(branch_native_errors):
                raise ValueError("released rollout difference summary changed")
            costs[family] = values
            for kind in errors:
                errors[kind].append(branch_errors[kind])
        if row["branches"]["native_dispatch"]["packets"][0]["vectors"]["goal"] != row["branches"]["zero"]["packets"][0]["vectors"]["goal"]:
            raise ValueError("counterfactual goal charts differ")
        margins = {k: [a - b for a, b in zip(costs["native_dispatch"][k], costs["zero"][k])] for k in values}
        result.append({"seed": row["seed"], "native_witness": row["native_witness"], "margins": margins})
    return {"root_seeds": seeds, "rows": result, "latent_l2_errors": errors,
        "mean_latent_l2_by_packet": {k: np.mean(v, axis=0).tolist() for k, v in errors.items()},
        "released_rollout_max_abs_difference": max(native_errors),
        "reversed_final_roots": {k: [r["seed"] for r in result if r["margins"][k][-1] * r["margins"]["observed"][-1] < 0]
            for k in errors},
        "scope": protocol["scope"] + " Observed-history predictions use evaluator-only future images, not a deployed repair."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol_path = args.report.with_suffix(".protocol.json")
    if digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]:
        raise ValueError("packet fidelity protocol seal changed")
    summary = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for ax, row in zip(axes.flat, summary["rows"]):
        for key, label, color, marker in (("free", "Free rollout", "#0072B2", "o"),
                ("teacher_forced", "Observed-history prediction", "#E69F00", "s"), ("observed", "Actual RGB", "black", "x")):
            ax.plot(range(1, 5), row["margins"][key], label=label, color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.axhline(0, color=".6", linewidth=.7)
        ax.text(.03, .97, str(row["seed"]), transform=ax.transAxes, va="top", fontsize=9)
        ax.set(xticks=range(1, 5), xlabel="Executed packet count")
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Candidate minus neutral goal cost")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True, layout="constrained")
    data = [np.log10(np.maximum(np.asarray(summary["latent_l2_errors"][k]), 1e-12)) for k in ("free", "teacher_forced")]
    lo, hi = min(a.min() for a in data), max(a.max() for a in data)
    labels = [str(seed) + (" / candidate" if i == 0 else " / neutral") for seed in summary["root_seeds"] for i in range(2)]
    for ax, matrix, title in zip(axes, data, ("Free rollout", "Observed-history prediction")):
        im = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=lo, vmax=hi)
        ax.set(xticks=range(4), xticklabels=range(1, 5), yticks=range(16), yticklabels=labels, xlabel="Executed packet count")
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(im, ax=ax, location="top", label=title + " (log10 latent L2 error)")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    (args.output_dir / "notes.txt").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight fixed roots and both sealed four-packet prefixes. Free and actual-history predictions are evaluated against the same goal and real RGB trajectory. Actual-history predictions use evaluator-only future observations; correct ranking is not a deployed selector or P1 admission.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Complete-coordinate prediction error at all 64 physical packet boundaries, retaining every root and branch on one log scale. The official native rollout and explicit free recurrence agree within the sealed tolerance. No roots, parameters, actions or thresholds are selected using outcomes.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("reversed_final_roots", "mean_latent_l2_by_packet", "released_rollout_max_abs_difference", "scope")}, indent=2))


if __name__ == "__main__":
    main()
