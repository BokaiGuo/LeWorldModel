"""Verify source-linked joint-goal metadata and all32 task-bound actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta
from experiments.spectral_reachability.reacher_d2p_packet_support import receipt_source
from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import summarize as validate_reference


def summarize(report, selection, source_qpos, reference_rows):
    required = {"cuda_only", "all16_roots", "all_exact_replays", "weights_unchanged", "selection_seal_unchanged", "references_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_goal_joint_limits_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict all16 goal repair evidence required")
    protocol = report["sealed_protocol"]
    if (protocol["physical_labels_used_for_planning"] is not False or protocol["all32_plans_sealed_before_terminal_evaluation"] is not True
            or protocol["cem"] != {"horizon": 6, "samples": 128, "steps": 3, "elite": 16, "seed": "root_seed"}):
        raise ValueError("fixed model-only all32-plan protocol required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    choices = {(r["seed"], r["family"]): r for r in selection["rows"]}
    originals = {r["seed"]: r for r in reference_rows}
    if (len(seeds) != 16 or len(set(seeds)) != 16 or [r["seed"] for r in rows] != seeds
            or sorted(source_qpos) != seeds or sorted(originals) != seeds or len(originals) != len(reference_rows)
            or len(choices) != len(selection["rows"]) or set(choices) != {(s, f) for s in seeds for f in ("legacy", "feasible")}):
        raise ValueError("all unique16 source roots and32 decisions required")
    output, means, changed = [], {"legacy": [], "feasible": []}, []
    for row in rows:
        seed, old = row["seed"], originals[row["seed"]]
        if (any(row[k] != old[k] for k in ("source_episode", "source_start_row")) or set(row["methods"]) != set(means)
                or not row.get("gates") or not all(v is True for v in row["gates"].values())
                or row["methods"]["legacy"] != old["methods"]["cem_carry"]):
            raise ValueError("historical source/physical replay changed")
        new, legacy = choices[seed, "feasible"], choices[seed, "legacy"]
        metadata = new["admission"]
        root = np.asarray(source_qpos[seed], dtype=np.float64)
        _, recomputed = feasible_goal_delta(root, protocol["target_delta"], metadata["limited_joints"], metadata["ranges_radians"])
        if recomputed != metadata or metadata != legacy["admission"]:
            raise ValueError("goal metadata not reconstructible from official source qpos")
        # Actual compiled model metadata, not an all-periodic replacement.
        if (metadata["limited_joints"] != [False, True]
                or not np.allclose(metadata["ranges_radians"][1], np.deg2rad([-160., 160.]), rtol=0, atol=1e-12)):
            raise ValueError("actual standard Reacher wrist limit contract required")
        change = bool(metadata["flipped_joint_indices"])
        if row["changed_task"] is not change:
            raise ValueError("task-change classification drift")
        if change:
            changed.append(seed)
        else:
            for k in ("root_rgb_sha256", "goal_rgb_sha256", "history_rgb_sha256", "params", "receipt", "action_witness_raw"):
                if new[k] != legacy[k]:
                    raise ValueError("unchanged task's inputs/plan must be bitwise unchanged")
            if row["methods"]["feasible"] != row["methods"]["legacy"]:
                raise ValueError("unchanged task physical outcome changed")
        for family, record in (("legacy", legacy), ("feasible", new)):
            method = row["methods"][family]
            target = metadata["historical_target_qpos"] if family == "legacy" else metadata["feasible_target_qpos"]
            params, action = np.asarray(record["params"]), np.asarray(method["action_witness_raw"])
            if (any(record[k] != row[k] for k in ("source_episode", "source_start_row"))
                    or not np.allclose(record["target_qpos"], target, rtol=0, atol=1e-14)
                    or params.shape != (1, 6, 10) or not np.isfinite(params).all()
                    or action.shape != (30, 2) or not np.isfinite(action).all() or np.abs(action).max() > 1 + 1e-6
                    or record["action_witness_raw"] != method["action_witness_raw"]):
                raise ValueError("sealed finite target/plan/action identity mismatch")
            receipt = record["receipt"]
            if (receipt["selection_uses_physical_labels"] is not False or receipt["proposal_trajectory_modified"] is not False
                    or receipt["forward_rollout_evaluations"] != 384 or len(receipt["trace"]) != 3):
                raise ValueError("unchanged model-only CEM search required")
            physical = method["physical"]
            if not np.isfinite(list(physical.values())).all() or min(physical.values()) < 0:
                raise ValueError("finite nonnegative physical metrics required")
            means[family].append(physical["final_qpos_distance"])
        output.append({"seed": seed, "changed_task": change, "source_wrist": float(root[1]),
            "wrist_limits": metadata["ranges_radians"][1], "legacy_wrist_goal": legacy["target_qpos"][1],
            "feasible_wrist_goal": new["target_qpos"][1], "metrics": {k: v["physical"] for k, v in row["methods"].items()}})
    aggregate = {"changed_task_roots": changed, "physical_branches": 32,
        "mean_distance": {k: float(np.mean(v)) for k, v in means.items()}}
    if aggregate != report["aggregate"] or report["actual_model_predict_samples"] != 32 * 2304:
        raise ValueError("all-root aggregate/compute accounting mismatch")
    return {**aggregate, "rows": output, "root_seeds": seeds, "unchanged_task_roots": len(seeds) - len(changed),
        "actual_model_predict_samples": report["actual_model_predict_samples"], "scope": protocol["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol = report["sealed_protocol"]
    selection_path, protocol_path = (args.report.with_suffix(ext) for ext in (".selection.json", ".protocol.json"))
    selection = json.loads(selection_path.read_text())
    if (digest(selection_path) != report["selection_sha256"] or selection["protocol_sha256"] != digest(protocol_path)
            or json.loads(protocol_path.read_text()) != protocol or digest(args.adapter_artifact) != protocol["adapter_sha256"]):
        raise ValueError("immutable task decisions/protocol/adapter changed")
    original_rows = []
    for entry in protocol["references"]:
        path = Path(entry["path"])
        ref = json.loads(path.read_text())
        p, s = path.with_suffix(".selection.json"), path.with_suffix(".suffix.selection.json")
        if digest(path) != entry["sha256"] or digest(p) != ref["selection_sha256"] or digest(s) != ref["suffix_selection_sha256"]:
            raise ValueError("historical reference/decisions changed")
        validate_reference(ref, json.loads(p.read_text()), json.loads(s.read_text()))
        original_rows.extend(ref["rows"])
    dataset, source_sha = receipt_source(args.adapter_artifact)
    if source_sha != protocol["source_sha256"]:
        raise ValueError("official source receipt changed")
    with h5py.File(dataset, "r", swmr=True) as handle:
        source_qpos = {r["seed"]: np.asarray(handle["qpos"][r["source_start_row"] + 15], dtype=np.float64) for r in original_rows}
    summary = summarize(report, selection, source_qpos, original_rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(1, 17)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for ax, metric in zip(axes, ("final_qpos_distance", "final_rgb_cost")):
        for offset, (family, color, marker, label) in enumerate((("legacy", "#888888", "o", "Legacy periodic goal"),
                ("feasible", "#0072B2", "D", "Joint-feasible goal"))):
            ax.scatter(x + (offset - .5) * .14, [r["metrics"][family][metric] for r in summary["rows"]], color=color, marker=marker, label=label, s=28)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final wrapped joint distance")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xticks=x, xlabel="Original source root ordinal (all sixteen)", ylabel="Terminal observed RGB cost")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    low, high = summary["rows"][0]["wrist_limits"]
    ax.axhspan(low, high, alpha=.1, color="#0072B2", label="Actual wrist range")
    ax.axhline(low, color=".6", linewidth=.8)
    ax.axhline(high, color=".6", linewidth=.8)
    for offset, (field, color, marker, label) in enumerate((("source_wrist", "#E69F00", "^", "Source wrist"),
            ("legacy_wrist_goal", "#888888", "o", "Legacy wrist goal"), ("feasible_wrist_goal", "#0072B2", "D", "Feasible wrist goal"))):
        ax.scatter(x + (offset - 1) * .13, [r[field] for r in summary["rows"]], color=color, marker=marker, label=label, s=28)
    ax.set(xticks=x, xlabel="Original source root ordinal (all sixteen)", ylabel="Wrist angle (radians)")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.24))
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All sixteen original roots under the historical periodic goal generator and an explicit joint-feasible task version. The frozen model and CEM configuration are unchanged; all thirty-two plans precede terminal evaluation. This intervention changes the invalid tasks, not performance on the same tasks, and does not retroactively admit P1.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Source-linked wrist states and both task goals against the actual compiled joint range. Continuous shoulder wrapping is unchanged; bounded wrist offsets retain their full magnitude but reflect direction when necessary. No roots are removed. Unchanged tasks require exact image, search, action and physical-outcome parity.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("changed_task_roots", "unchanged_task_roots", "mean_distance", "actual_model_predict_samples", "scope")}, indent=2))


if __name__ == "__main__":
    main()
