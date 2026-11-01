"""Full-coordinate arithmetic/receipt checks and whole16-source figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability import audit_pusht_d2p_live_action_units_cuda as audit
from experiments.spectral_reachability import audit_pusht_d2p_packet_contract as physical
from experiments.spectral_reachability import pusht_d2p_packet_contract as contract
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

GATES = {"cuda_only", "all16_predeclared_episodes", "released_and_explicit_predictions_exact", "predict_accounting",
    "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "contract_report_unchanged",
    "weights_file_unchanged", "implementation_unchanged"}


def summarize(report):
    if (report.get("schema") != "pusht_d2p_live_action_units.v1" or report.get("status") != "completed"
            or set(report.get("gates", {})) != GATES or any(v is not True for v in report["gates"].values())
            or report.get("actual_predict_samples") != 64
            or report["sealed_protocol"]["checkpoint_training_statistics_certified"] is not False
            or report["sealed_protocol"]["physical_actions_unchanged_between_normalizations"] is not True):
        raise ValueError("complete unchanged-action CUDA diagnostic, not training-stat certification, required")
    result = audit.summarize(report["rows"])
    if report["summary"] != result:
        raise ValueError("stored summary differs from independent full-coordinate arithmetic")
    return result


def verify_files(path):
    report = json.loads(path.read_text())
    result = summarize(report)
    pp = path.with_suffix(".protocol.json")
    protocol = report["sealed_protocol"]
    if digest(pp) != report["protocol_sha256"] or json.loads(pp.read_text()) != protocol:
        raise ValueError("sealed neural protocol changed")
    rp = Path(protocol["contract_report_path"])
    physical_report = contract.validate_receipt(json.loads(rp.read_text()))
    if (digest(rp) != protocol["contract_report_sha256"]
            or digest(rp.parent/"protocol.json") != physical_report["protocol_sha256"]
            or digest(rp.parent/"implementation.json") != physical_report["implementation_sha256"]
            or physical_report["simulator"] != "official"):
        raise ValueError("official physical reference seals changed")
    records, source = physical.select_rows(Path(protocol["source"]["source"]["path"]), tuple(range(16)))
    if source != protocol["source"]:
        raise ValueError("bounded source identities changed")
    observed = {r["episode"]: r for r in physical_report["episodes"]}
    source_rows = {r["episode"]: r for r in records}
    if sorted(observed) != list(range(16)) or len(physical_report["episodes"]) != 16:
        raise ValueError("all16 physical source references required")
    for row in report["rows"]:
        episode = row["episode"]
        stage_path = Path(row["prediction_stage_path"])
        if digest(stage_path) != row["prediction_stage_sha256"]:
            raise ValueError("pre-execution forecast seal changed")
        stage = json.loads(stage_path.read_text())
        reference, source = observed[episode], source_rows[episode]
        if (stage["episode"] != episode or stage["protocol_sha256"] != digest(pp)
                or stage["future_observations_used"] is not False or stage["predicted_vectors"] != row["predicted_vectors"]
                or stage["input_boundary_rgb_sha256"] != reference["history"]["boundary_rgb_sha256"][1:4]
                or row["successor_rgb_sha256"] != reference["history"]["boundary_rgb_sha256"][4]
                or row["source_start_row"] != source["offset"]
                or stage["raw_history_packets"] != source["actions"][:15].reshape(3, 5, 2).tolist()
                or stage["raw_future_packet"] != source["actions"][15:20].tolist()):
            raise ValueError("original live RGB, source action or causal frontier changed")
        error = float(np.max(np.abs(np.asarray(row["successor_state"], np.float64)-source["states"][20])))
        if not np.isclose(error, reference["dense"]["max_state_errors"][-1], rtol=0, atol=1e-12):
            raise ValueError("source physical replay changed")
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha:
            raise ValueError("archived neural implementation changed")
    return report, result


def plot(report, result, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(16)
    fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
    for v, label, color in (("extra_div100", "Erroneous extra /100", "#E69F00"), ("live_units", "Live action units", "#0072B2")):
        ax.plot(x, result["per_episode_l2"][v], "o-", label=label, color=color, markersize=4)
    ax.set(xticks=x, xlabel="All16 predeclared source episodes", ylabel="Full192-coordinate prediction L2", ylim=(0, None))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/("Figure_1."+ext), bbox_inches="tight")
    plt.close(fig)
    rows = sorted(report["rows"], key=lambda r: r["episode"])
    errors = {v: np.abs(np.asarray([r["predicted_vectors"][v] for r in rows])-np.asarray([r["observed_vector"] for r in rows])) for v in audit.VARIANTS}
    limit = max(float(a.max()) for a in errors.values())
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    for ax, v, label in zip(axes, audit.VARIANTS, ("Erroneous extra /100", "Live action units"), strict=True):
        im = ax.imshow(errors[v], aspect="auto", cmap="cividis", vmin=0, vmax=limit)
        ax.set(yticks=range(16), ylabel="All16 episodes", xlabel="All192 latent coordinates")
        ax.text(.01, 1.03, label, transform=ax.transAxes)
        fig.colorbar(im, ax=ax, label="Absolute prediction error")
    for ext in ("png", "pdf"):
        fig.savefig(output/("Figure_2."+ext), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, result = verify_files(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot(report, result, args.output_dir)
    write_new(args.output_dir/"summary.json", {"report_sha256": digest(args.report), **result})
    (args.output_dir/"notes.txt").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Frozen-checkpoint prediction error for exactly the same RGB histories and executed five-command packets on all sixteen predeclared source episodes. Only an erroneous extra division by100 differs. Public normalization statistics remain an explicit convention, not certified checkpoint training provenance. This is not P1 or policy efficacy evidence.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Complete coordinatewise absolute error on every source episode with a shared color scale. No latent coordinate or source episode is selected by outcomes; actual successor vectors are used only for evaluation after forecasts are sealed.}
\end{figure}
""")
    write_new(args.output_dir/"manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
