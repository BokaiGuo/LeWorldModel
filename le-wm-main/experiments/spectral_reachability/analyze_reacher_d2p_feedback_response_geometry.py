"""NumPy/SciPy verification of measured geometry; no neural inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference

SCALES = (.1, .25)
METHODS = tuple(f + "_" + m for f in ("native_dispatch", "greedy", "zero", "cem") for m in ("replay", "tracking"))
REQUIRED_GATES = {"cuda_only", "all8_roots", "all8_methods_and5184_physical_futures", "all256_prefix_boundaries_exact",
    "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "predict_sample_accounting", "reference_unchanged",
    "weights_file_unchanged", "implementation_unchanged", "prefix_snapshot_reference_unchanged"}


def finite(value, shape):
    result = np.asarray(value, np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("finite full-coordinate array required")
    return result


def angular(left, right):
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2 * np.pi) - np.pi


def independent_gain(response, residual):
    n = response.shape[1]
    matrix = np.concatenate((response, .1 * np.eye(n)), axis=0)
    target = np.concatenate((residual, np.zeros(n)))
    fit = lsq_linear(matrix, target, bounds=(-1, 1), tol=1e-12, max_iter=1000, lsq_solver="exact")
    if not fit.success or not np.isfinite(fit.x).all() or np.max(np.abs(fit.x)) > 1:
        raise ValueError("independent bounded least-squares solve failed")
    return float(residual @ residual - np.sum((matrix @ fit.x - target) ** 2))


def check_geometry(receipt, response, residual):
    if (receipt.get("physical_labels_used") is not True or receipt.get("deployed") is not False
            or receipt.get("objective_certificate_passed") is not True):
        raise ValueError("evaluator physical-label boundary and QP certificate required")
    gain = independent_gain(response, residual)
    if not np.isclose(gain, receipt["bounded_quadratic_gain"], rtol=1e-8, atol=1e-8):
        raise ValueError("independent physical-response gain mismatch")
    if not np.allclose(np.linalg.svd(response, compute_uv=False), receipt["response_singular_values"], rtol=1e-8, atol=1e-8):
        raise ValueError("full-response singular values mismatch")
    return gain


def check_probe(probe):
    scale = probe["scale"]
    if scale not in SCALES or len(probe["endpoints"]) != 20 or [e["coordinate"] for e in probe["endpoints"]] != list(range(20)):
        raise ValueError("all twenty ordered probes at predeclared scale required")
    columns = {"model_fd": [], "physical_rgb_fd": [], "physical_qpos_fd": []}
    for e in probe["endpoints"]:
        points = {}
        for sign in ("plus", "minus"):
            point = e[sign]
            points[sign] = {k: finite(point[k], shape) for k, shape in (("model", (192,)), ("observed", (192,)), ("qpos", (2,)), ("raw_actions", (10, 2)))}
            expected = np.zeros(20)
            expected[e["coordinate"]] = float(np.float32(scale)) * (1 if sign == "plus" else -1)
            if not np.allclose(points[sign]["raw_actions"].ravel(), expected, rtol=0, atol=2e-6):
                raise ValueError("complete physical action does not match the sealed coordinate probe")
        step = float(np.float32(scale))
        for key, source in (("model_fd", "model"), ("physical_rgb_fd", "observed")):
            columns[key].append((points["plus"][source] - points["minus"][source]) / (2 * step))
        columns["physical_qpos_fd"].append(angular(points["plus"]["qpos"], points["minus"]["qpos"]) / (2 * step))
    matrices = {k: np.stack(v, axis=1) for k, v in columns.items()}
    for key, matrix in matrices.items():
        if not np.array_equal(finite(probe[key], matrix.shape), matrix):
            raise ValueError("stored finite difference does not equal complete endpoints")
    return matrices


def actual_probe_metrics(cell, target):
    points = [cell["center"]] + [e[sign] for p in cell["probes"] for e in p["endpoints"] for sign in ("plus", "minus")]
    if len(points) != 81:
        raise ValueError("all 81 actually executed futures required")
    rgb = [float(np.mean((finite(p["observed"], (192,)) - target) ** 2)) for p in points]
    qpos = [float(np.linalg.norm(angular(cell["target_qpos"][:2], finite(p["qpos"], (2,))))) for p in points]
    return {"center_rgb_cost": rgb[0], "best_measured_rgb_cost": min(rgb), "rgb_cost_improvement_over_center": rgb[0] - min(rgb),
        "center_qpos_distance": qpos[0], "best_measured_qpos_distance": min(qpos), "qpos_improvement_over_center": qpos[0] - min(qpos),
        "scope": "Evaluator-only best of fixed physical probes, not a deployed policy or optimized action proof."}


def summarize(report):
    protocol = report["sealed_protocol"]
    if (report.get("schema") != "reacher_d2p_feedback_response_geometry.v1" or report.get("status") != "completed"
            or not REQUIRED_GATES.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("strict completed CUDA geometry receipt required")
    if (protocol["methods"] != list(METHODS) or protocol["command_scales"] != list(SCALES) or protocol["prefix_packets"] != 4
            or protocol["future_packets"] != 2 or protocol["command_coordinates"] != 20 or protocol["response_gain_damping"] != .01
            or protocol["physical_labels_used_for_planning"] is not False or protocol["common_task_reference"] != "zero_replay center-command future"):
        raise ValueError("unchanged fixed evaluator protocol required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    if len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds:
        raise ValueError("all eight unique root identities required")
    output, future_count, rerender_differences = [], 0, []
    for row in rows:
        if set(row["methods"]) != set(METHODS):
            raise ValueError("all eight branch methods required")
        target = finite(row["goal_vector"], (192,))
        common_rgb = finite(row["common_rgb_residual"], (192,))
        common_qpos = finite(row["common_qpos_residual"], (2,))
        zero = row["methods"]["zero_replay"]
        if (not np.array_equal(common_rgb, target - finite(zero["center"]["observed"], (192,)))
                or not np.array_equal(common_qpos, angular(zero["target_qpos"][:2], zero["center"]["qpos"]))):
            raise ValueError("common task residual changed across branches")
        metrics = {}
        for name in METHODS:
            cell = row["methods"][name]
            if cell["prefix_observation_exact"] is not True or [p["scale"] for p in cell["probes"]] != list(SCALES):
                raise ValueError("exact prefix and both ordered scales required")
            diagnostics = cell["reconstruction_diagnostics"]
            if len(diagnostics) != 4 or [p["packet"] for p in diagnostics] != [1, 2, 3, 4]:
                raise ValueError("all prefix reconstruction diagnostics required")
            for p in diagnostics:
                if p["physical_integration_and_scene_exact"] is not True or p["rerendered_rgb_exact"] != (p["max_pixel_difference"] == 0 and p["changed_components"] == 0):
                    raise ValueError("exact physical-state and consistent pixel diagnostics required")
                if not p["rerendered_rgb_exact"]:
                    rerender_differences.append({"seed": row["seed"], "method": name, **p})
            center_raw = finite(cell["center"]["raw_actions"], (10, 2))
            if np.max(np.abs(center_raw)) > 2e-6:
                raise ValueError("center command must map to near-zero raw torque")
            analytic = finite(cell["model_analytic_response"], (192, 20))
            analytic_gain = check_geometry(cell["common_task_model_analytic_geometry"], analytic, common_rgb)
            per_scale = []
            future_count += 1
            for probe in cell["probes"]:
                matrices = check_probe(probe)
                gains = {k: check_geometry(probe[k + "_common_task_geometry"], v, common_qpos if k == "physical_qpos_fd" else common_rgb) for k, v in matrices.items()}
                per_scale.append({"scale": probe["scale"], "gains": gains,
                    "analytic_vs_model_fd_relative_error": float(np.linalg.norm(analytic - matrices["model_fd"]) / max(np.linalg.norm(analytic), 1e-12)),
                    "analytic_vs_physical_rgb_relative_error": float(np.linalg.norm(analytic - matrices["physical_rgb_fd"]) / max(np.linalg.norm(matrices["physical_rgb_fd"]), 1e-12)),
                    "physical_rgb_fd_singular_values": np.linalg.svd(matrices["physical_rgb_fd"], compute_uv=False).tolist(),
                    "physical_qpos_fd_singular_values": np.linalg.svd(matrices["physical_qpos_fd"], compute_uv=False).tolist()})
                future_count += 40
            metrics[name] = {"analytic_gain": analytic_gain, "scales": per_scale, "actual_probes": actual_probe_metrics(cell, target)}
        output.append({"seed": row["seed"], "methods": metrics})
    if future_count != 5184 or report["physical_futures"] != future_count or report["actual_model_predict_samples"] != 10496:
        raise ValueError("complete future and predict sample accounting required")
    comparisons = {}
    for mode in ("replay", "tracking"):
        for control in ("zero", "greedy", "cem"):
            label = "native_minus_" + control + "_" + mode
            first, second = "native_dispatch_" + mode, control + "_" + mode
            # Positive difference favors native for these gain metrics, unlike distance.
            comparisons[label] = {"positive_favors": "native", "analytic_rgb": paired_difference(
                [r["methods"][first]["analytic_gain"] for r in output], [r["methods"][second]["analytic_gain"] for r in output]), "scales": [],
                "best_measured_rgb_cost_difference": paired_difference([r["methods"][first]["actual_probes"]["best_measured_rgb_cost"] for r in output],
                    [r["methods"][second]["actual_probes"]["best_measured_rgb_cost"] for r in output]),
                "best_measured_qpos_distance_difference": paired_difference([r["methods"][first]["actual_probes"]["best_measured_qpos_distance"] for r in output],
                    [r["methods"][second]["actual_probes"]["best_measured_qpos_distance"] for r in output])}
            for key in ("best_measured_rgb_cost_difference", "best_measured_qpos_distance_difference"):
                comparisons[label][key]["metric"] = "best fixed physical probe outcome difference; negative favors native; evaluator only, descriptive"
            for i, scale in enumerate(SCALES):
                contrasts = {key: paired_difference([r["methods"][first]["scales"][i]["gains"][key] for r in output],
                    [r["methods"][second]["scales"][i]["gains"][key] for r in output]) for key in ("model_fd", "physical_rgb_fd", "physical_qpos_fd")}
                # shared paired_difference labels lower as a win; do not inherit it here.
                for contrast in contrasts.values():
                    contrast["positive_count"], contrast["negative_count"] = contrast.pop("losses"), contrast.pop("wins")
                    contrast["metric"] = "bounded quadratic response gain difference; positive favors native; descriptive known-root audit"
                comparisons[label]["scales"].append({"scale": scale, **contrasts})
            analytic_summary = comparisons[label]["analytic_rgb"]
            analytic_summary["positive_count"], analytic_summary["negative_count"] = analytic_summary.pop("losses"), analytic_summary.pop("wins")
            analytic_summary["metric"] = "bounded quadratic response gain difference; positive favors native; descriptive known-root audit"
    return {"root_seeds": seeds, "physical_futures": future_count, "actual_model_predict_samples": 10496, "rows": output,
        "comparisons": comparisons, "rerender_differences_at_exact_physical_state": rerender_differences, "scope": protocol["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new immutable analysis directory required")
    report = json.loads(args.report.read_text())
    if digest(args.report.with_suffix(".protocol.json")) != report["protocol_sha256"] or json.loads(args.report.with_suffix(".protocol.json").read_text()) != report["sealed_protocol"]:
        raise ValueError("protocol seal changed")
    ref = report["sealed_protocol"]
    if digest(Path(ref["prefix_snapshot_reference_path"])) != ref["prefix_snapshot_reference_sha256"]:
        raise ValueError("strict repeated prefix observation/state reference changed")
    reference_path = Path(ref["reference_path"])
    if digest(reference_path) != ref["reference_sha256"] or digest(reference_path.with_suffix(".selection.json")) != ref["initial_selection_sha256"]:
        raise ValueError("upstream sealed report changed")
    original = json.loads(reference_path.read_text())
    original_rows = {r["seed"]: r for r in original["rows"]}
    for row in report["rows"]:
        old = original_rows[row["seed"]]
        if any(row[k] != old[k] for k in ("source_episode", "source_start_row")):
            raise ValueError("root source identity changed")
        for name, cell in row["methods"].items():
            branch = old["methods"][name]
            if cell["prefix_raw_actions"] != branch["action_witness_raw"][:20] or cell["prefix_observed_vector"] != branch["packets"][3]["observed_vector"]:
                raise ValueError("prefix witness or complete observed coordinates changed")
    summary = {"report_sha256": digest(args.report), **summarize(report)}
    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir / "summary.json", summary)
    # Style adapted from figure-generation's line template, with semantic labels.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    x = np.arange(1, 9)
    colors = ("#0072B2", "#D55E00", "#333333")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row_index, mode in enumerate(("replay", "tracking")):
        for column, control in enumerate(("zero", "greedy")):
            ax = axes[row_index, column]
            first, second = "native_dispatch_" + mode, control + "_" + mode
            for i, scale in enumerate(SCALES):
                y = [r["methods"][first]["scales"][i]["gains"]["physical_rgb_fd"] - r["methods"][second]["scales"][i]["gains"]["physical_rgb_fd"] for r in summary["rows"]]
                ax.plot(x, y, "o-", color=colors[i], label=f"Physical RGB, probe {scale}")
            y = [r["methods"][first]["analytic_gain"] - r["methods"][second]["analytic_gain"] for r in summary["rows"]]
            ax.plot(x, y, "s--", color=colors[2], label="Frozen-model analytic")
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set(xlabel=f"Root (all eight; {mode})", ylabel=f"Native minus {control}: response gain", xticks=x)
            ax.grid(alpha=.2)
            if row_index == 0 and column == 0:
                ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + suffix), dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for ax, key, label in zip(axes, ("analytic_vs_model_fd_relative_error", "analytic_vs_physical_rgb_relative_error"),
            ("Model nonlinearity: analytic vs finite difference", "Model vs physical RGB response")):
        for i, scale in enumerate(SCALES):
            y = [float(np.mean([r["methods"][name]["scales"][i][key] for name in METHODS])) for r in summary["rows"]]
            ax.plot(x, y, "o-", color=colors[i], label=f"Probe {scale}")
        ax.set(xlabel="Root (all eight)", ylabel="Relative Frobenius error", xticks=x)
        ax.text(.03, .97, label, transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper right")
        ax.margins(y=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + suffix), dpi=300)
    plt.close(fig)
    (args.output_dir / "notes.txt").write_text("All eight known roots and all eight sealed branches retained. Finite-scale physical RGB and qpos response, fixed common task residual and damping. Positive gain contrast favors native. Physical geometry never enters planning; no P1 admission, theory rejection or compute-matched efficacy. Independent SciPy box solves verify all 448 CUDA gain receipts. Root indices 1-8 follow summary.root_seeds.\n")
    (args.output_dir / "figures.tex").write_text("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n\\caption{Predicted and physically measured task-response gain after replayed and feedback-tracked prefixes. All roots and fixed probe scales are shown; this is an evaluator-only mechanism audit.}\n\\end{figure}\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n\\caption{Finite-scale model nonlinearity and physical response mismatch, averaged over all eight sealed branch types at each root.}\n\\end{figure}\n")
    artifacts = {p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file()}
    write_new(args.output_dir / "manifest.json", {"report_sha256": summary["report_sha256"], "analysis_sha256": digest(Path(__file__)), "artifact_sha256": artifacts})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
