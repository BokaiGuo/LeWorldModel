"""Independent NumPy/SciPy verification of the exact-actuator CUDA pilot.

No neural inference. Finite-scale spectra and linearized box fits are
diagnostics, not a controllability theorem or physically executed controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.reacher_d2p_goal_contract import validate_goal_admission

TASKS = {"full_drive": (0, 1), "shoulder_only": (0,), "wrist_only": (1,)}
SCALES, HORIZONS = (.1, .25), (1, 2)
REQUIRED_GATES = {"cuda_only", "all8_roots", "all_sources_disjoint", "all976_future_branches",
    "all_masks_and_gradient_checks", "predictions_unchanged", "parameters_unchanged", "buffers_unchanged",
    "all_modules_eval", "predict_sample_accounting", "weights_file_unchanged", "implementation_unchanged"}


def finite(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("finite full-coordinate array required")
    return array


def columns(task, horizon):
    # Independent indexing, not the chart producer's helper.
    return [2 * t + joint for t in range(5 * horizon) for joint in TASKS[task]]


def joint_residual(target, actual, limited):
    value = finite(target, (2,)) - finite(actual, (2,))
    limited = np.asarray(limited, bool)
    if limited.shape != (2,):
        raise ValueError("two-joint topology required")
    value[~limited] = (value[~limited] + np.pi) % (2 * np.pi) - np.pi
    return value


def check_endpoint(point, raw, horizon):
    actual = finite(point["raw_actions"], (5 * horizon, 2))
    if not np.array_equal(actual, raw):
        raise ValueError("executed raw torque differs from declared complete probe")
    if len(point["boundaries"]) != horizon:
        raise ValueError("all physical packet boundaries required")
    for i, boundary in enumerate(point["boundaries"]):
        expected = hashlib.sha256(np.ascontiguousarray(raw[5*i:5*(i+1)], dtype=np.float32).tobytes()).hexdigest()
        if boundary["action_sha256"] != expected or len(boundary["root_rgb_sha256"]) != 64:
            raise ValueError("packet action hash or RGB identity missing")
    return finite(point["observed_vector"], (192,)), finite(point["qpos"], (2,))


def probe_matrices(forecast, measured, limited):
    horizon = forecast["horizon"]
    if horizon not in HORIZONS or measured["horizon"] != horizon:
        raise ValueError("both fixed packet horizons required")
    inventory = [(s, i, sign) for s in SCALES for i in range(10 * horizon) for sign in (-1, 1)]
    def index(points):
        result = {(p["scale"], p["coordinate"], p["sign"]): p for p in points}
        if len(result) != len(points) or set(result) != set(inventory):
            raise ValueError("duplicate-free complete signed scale/coordinate probes required")
        return result
    forecasts, observations = index(forecast["probes"]), index(measured["endpoints"])
    center = check_endpoint(measured["center"], np.zeros((5*horizon, 2)), horizon)
    matrices, endpoints = {}, {}
    for scale in SCALES:
        values, points = {k: [] for k in ("model_fd", "physical_rgb_fd", "physical_qpos_fd")}, []
        for coordinate in range(10 * horizon):
            pair = {}
            for sign in (-1, 1):
                key = (scale, coordinate, sign)
                raw = np.zeros(10*horizon)
                raw[coordinate] = float(np.float32(sign * scale))
                raw = raw.reshape(5*horizon, 2)
                if not np.array_equal(finite(forecasts[key]["raw_actions"], raw.shape), raw):
                    raise ValueError("sealed model/physical action witness differs")
                observed, qpos = check_endpoint(observations[key], raw, horizon)
                pair[sign] = {"model": finite(forecasts[key]["predicted_vector"], (192,)), "rgb": observed, "qpos": qpos}
                points.append((coordinate, observed, qpos))
            step = float(np.float32(scale))
            values["model_fd"].append((pair[1]["model"] - pair[-1]["model"]) / (2 * step))
            values["physical_rgb_fd"].append((pair[1]["rgb"] - pair[-1]["rgb"]) / (2 * step))
            values["physical_qpos_fd"].append(joint_residual(pair[1]["qpos"], pair[-1]["qpos"], limited) / (2 * step))
        matrices[scale] = {k: np.stack(v, axis=1) for k, v in values.items()}
        endpoints[scale] = points
    return center, matrices, endpoints


def linear_geometry(response, residual, bound):
    """A box-constrained linear-response diagnostic, not a physical outcome."""
    n = response.shape[1]
    matrix = np.concatenate((response, .1 * np.eye(n)))
    target = np.concatenate((residual, np.zeros(n)))
    fit = lsq_linear(matrix, target, bounds=(-bound, bound), tol=1e-12, max_iter=1000, lsq_solver="exact")
    if not fit.success or not np.isfinite(fit.x).all() or np.max(np.abs(fit.x)) > bound:
        raise ValueError("independent bounded quadratic solve failed")
    remainder = residual - response @ fit.x
    baseline = float(residual @ residual)
    singular = np.linalg.svd(response, compute_uv=False)
    return {"singular_values": singular.tolist(), "bounded_regularized_gain": float(baseline - np.sum((matrix @ fit.x - target)**2)),
        "linear_remaining_fraction": float(remainder @ remainder / max(baseline, 1e-12)),
        "raw_command_bound": bound, "action_penalty": .01, "evaluator_only": True,
        "physically_executed_fit": False}


def validate_protocol(protocol):
    if (protocol["tasks"] != {k: list(v) for k, v in TASKS.items()} or protocol["horizons"] != list(HORIZONS)
            or protocol["probe_scales"] != list(SCALES) or protocol["fd_epsilon"] != .001
            or protocol["fd_directions_per_task"] != 4 or protocol["fd_median_relative_error_max"] != .1
            or protocol["target_delta"] != [.65, -.65] or protocol["goal_contract"] != "joint-feasible-v1"
            or protocol["damping"] != .01 or any(protocol[k] is not True for k in
                ("inactive_raw_torques_literal_zero", "no_post_training", "all_predictions_sealed_before_any_future_execution"))):
        raise ValueError("unchanged full-drive and exact-mask protocol required")


def summarize(report, predictions):
    if (report.get("schema") != "reacher_d2p_masked_task_preflight.v1" or report.get("status") != "completed"
            or not REQUIRED_GATES.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("all strict completed CUDA integrity gates required")
    protocol = report["sealed_protocol"]
    validate_protocol(protocol)
    seeds = protocol["root_seeds"]
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("all eight distinct roots required")
    def roots(rows):
        indexed = {r["seed"]: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(seeds):
            raise ValueError("root-identity join requires duplicate-free set equality")
        return indexed
    measured, forecast, source = roots(report["rows"]), roots(predictions["rows"]), roots(protocol["source_roots"])
    excluded = set(protocol["excluded_source_episodes"])
    episodes = [r["source_episode"] for r in source.values()]
    if len(set(episodes)) != 8 or set(episodes) & excluded or len(excluded) != 192:
        raise ValueError("eight episodes disjoint from all 192 historical episodes required")
    output, physical_futures = [], 0
    for seed in sorted(seeds):
        actual, predicted = measured[seed], forecast[seed]
        if any(actual[k] != source[seed][k] or predicted[k] != source[seed][k] for k in ("source_episode", "source_start_row")):
            raise ValueError("source root identity changed")
        validate_goal_admission(protocol["goal_contract"], predicted["goal_admission"])
        admission = predicted["goal_admission"]
        limited, target_qpos = admission["limited_joints"], admission["feasible_target_qpos"]
        target_rgb = finite(predicted["goal_vector"], (192,))
        if [h["horizon"] for h in actual["horizons"]] != list(HORIZONS) or [h["horizon"] for h in predicted["horizons"]] != list(HORIZONS):
            raise ValueError("both ordered horizons required")
        result = []
        for f, m in zip(predicted["horizons"], actual["horizons"]):
            horizon = f["horizon"]
            center, matrices, endpoints = probe_matrices(f, m, limited)
            finite(f["center_predicted_vector"], (192,))
            residuals = {"model_fd": target_rgb - center[0], "physical_rgb_fd": target_rgb - center[0],
                "physical_qpos_fd": joint_residual(target_qpos, center[1], limited)}
            if set(f["tasks"]) != set(TASKS):
                raise ValueError("every predeclared task retained")
            full = finite(f["tasks"]["full_drive"]["model_jacobian"], (192, 10*horizon))
            tasks = {}
            for task in TASKS:
                indices = columns(task, horizon)
                cell = f["tasks"][task]
                derivative = finite(cell["model_jacobian"], (192, len(indices)))
                error = float(np.max(np.abs(derivative - full[:, indices])))
                if (cell["active_columns"] != indices or error > 2e-6
                        or not np.isclose(error, cell["full_column_max_error"], rtol=0, atol=1e-12)
                        or len(cell["fd_directions"]) != 4):
                    raise ValueError("exact reduced derivative/active columns changed")
                fd_errors = []
                for p in cell["fd_directions"]:
                    direction = finite(p["direction"], (len(indices),))
                    fd = (finite(p["positive_vector"], (192,)) - finite(p["negative_vector"], (192,))) / .002
                    analytic = derivative @ direction
                    relative = float(np.linalg.norm(fd - analytic) / max(np.linalg.norm(analytic), 1e-12))
                    if not np.isclose(relative, p["relative_error"], rtol=5e-4, atol=2e-6):
                        raise ValueError("independent directional finite difference mismatch")
                    fd_errors.append(relative)
                if np.median(fd_errors) > .1 or not np.isclose(np.median(fd_errors), cell["fd_median_relative_error"], rtol=5e-4, atol=2e-6):
                    raise ValueError("model finite difference admission failed")
                scales = []
                for scale in SCALES:
                    geom = {k: linear_geometry(v[:, indices], residuals[k], scale) for k, v in matrices[scale].items()}
                    geom["model_ad"] = linear_geometry(derivative, residuals["model_fd"], scale)
                    admissible = [(center[0], center[1])] + [(rgb, q) for i, rgb, q in endpoints[scale] if i in indices]
                    if len(admissible) != 1 + 2*len(indices):
                        raise ValueError("masked physical outcome reuses an inactive-coordinate probe")
                    rgb_costs = [float(np.mean((target_rgb-rgb)**2)) for rgb, _ in admissible]
                    distances = [float(np.linalg.norm(joint_residual(target_qpos, q, limited))) for _, q in admissible]
                    scales.append({"scale": scale, "geometry": geom,
                        "ad_vs_model_fd_relative_error": float(np.linalg.norm(derivative - matrices[scale]["model_fd"][:, indices]) / max(np.linalg.norm(derivative), 1e-12)),
                        "ad_vs_physical_rgb_relative_error": float(np.linalg.norm(derivative - matrices[scale]["physical_rgb_fd"][:, indices]) / max(np.linalg.norm(matrices[scale]["physical_rgb_fd"][:, indices]), 1e-12)),
                        "actual_admissible_probe_count": len(admissible), "center_rgb_cost": rgb_costs[0], "best_executed_probe_rgb_cost": min(rgb_costs),
                        "center_qpos_distance": distances[0], "best_executed_probe_qpos_distance": min(distances),
                        "scope": "Best of actually executed fixed admissible probes, evaluator only; no composed or optimized physical future inferred."})
                tasks[task] = {"active_columns": indices, "independent_fd_errors": fd_errors, "scales": scales}
            result.append({"horizon": horizon, "tasks": tasks})
            physical_futures += 1 + len(m["endpoints"])
        output.append({"seed": seed, "source_episode": actual["source_episode"], "goal_admission": admission, "horizons": result})
    if physical_futures != 976 or report["physical_futures"] != 976 or report["actual_model_predict_samples"] != 2272:
        raise ValueError("complete physical future and CUDA predict sample accounting required")
    return {"root_seeds": sorted(seeds), "physical_futures": physical_futures, "actual_model_predict_samples": 2272,
        "rows": output, "scope": protocol["scope"], "P1_admission": False,
        "interpretation": "Exact actuator-task premise diagnosis only; no task selection, physical controllability theorem, reconfiguration benefit or theory rejection."}


def verify_files(path):
    report = json.loads(path.read_text())
    protocol_path, prediction_path = path.with_suffix(".protocol.json"), path.with_suffix(".predictions.json")
    if digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]:
        raise ValueError("protocol seal changed")
    predictions = json.loads(prediction_path.read_text())
    if digest(prediction_path) != report["predictions_sha256"] or predictions["protocol_sha256"] != report["protocol_sha256"]:
        raise ValueError("pre-execution prediction seal changed")
    implementation = path.with_suffix(".implementation")
    for key, sha in report["sealed_protocol"]["implementation_sha256"].items():
        if digest(implementation / (key + ".py")) != sha:
            raise ValueError("bound implementation copy changed")
    for row in report["rows"]:
        root_path = path.with_suffix(".roots") / f"root_{row['seed']}.json"
        if json.loads(root_path.read_text()) != row:
            raise ValueError("physical root sidecar differs")
    return {"report_sha256": digest(path), **summarize(report, predictions)}


def plot(summary, output):
    # Reuse figure-generation line-template conventions, all roots retained.
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#333333", "#0072B2", "#D55E00")
    labels = ("Full drive", "Shoulder only", "Wrist only")
    x = np.arange(1, 9)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for h, horizon in enumerate(HORIZONS):
        for s, scale in enumerate(SCALES):
            ax = axes[h, s]
            for task, label, color in zip(TASKS, labels, colors):
                y = [r["horizons"][h]["tasks"][task]["scales"][s]["geometry"]["physical_qpos_fd"]["singular_values"][-1] for r in summary["rows"]]
                ax.plot(x, y, "o-", label=label, color=color)
            ax.set(xlabel=f"Root (all eight); {horizon} packet(s), probe {scale}", ylabel="Smallest joint-response singular value\nMeasured finite-scale response", xticks=x, yscale="log")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("Figure_1." + suffix), dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for h, horizon in enumerate(HORIZONS):
        for s, scale in enumerate(SCALES):
            ax = axes[h, s]
            for task, label, color in zip(TASKS, labels, colors):
                y = [r["horizons"][h]["tasks"][task]["scales"][s]["geometry"]["physical_qpos_fd"]["linear_remaining_fraction"] for r in summary["rows"]]
                ax.plot(x, y, "o-", label=label, color=color)
            ax.set(xlabel=f"Root (all eight); {horizon} packet(s), probe {scale}", ylabel="Remaining joint-goal residual fraction\nBounded linear fit, not physical success", xticks=x, ylim=(-.02, 1.02))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("Figure_2." + suffix), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new immutable analysis directory required")
    summary = verify_files(args.report)
    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir / "summary.json", summary)
    plot(summary, args.output_dir)
    (args.output_dir / "notes.txt").write_text("All eight fresh roots, three exact masks, two horizons and two fixed scales retained. Official original frozen checkpoint, host CUDA only; all predictions sealed before future execution. Full physical probes reused only when their actuated coordinate is active. Joint topology respected. All spectra and bounded response solves are independent NumPy/SciPy diagnostics; no optimized fit was physically executed and no P1 or theoretical verdict follows. Figures show all roots, not selected successful tasks.\n")
    (args.output_dir / "figures.tex").write_text("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n\\caption{Measured finite-scale joint response under exact actuator masks. All roots, horizons and probe scales are retained; spectra are not a controllability theorem.}\n\\end{figure}\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n\\caption{Remaining task residual in independent bounded linear response fits. These fits are diagnostic, not physically executed planning outcomes.}\n\\end{figure}\n")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "analysis_sha256": digest(Path(__file__)),
        "artifact_sha256": {p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file()}})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
