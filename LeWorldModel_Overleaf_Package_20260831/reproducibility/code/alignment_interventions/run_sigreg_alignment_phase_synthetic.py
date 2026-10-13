"""SIGReg--alignment phase diagram in a learned synthetic pixel world.

This experiment targets the main unresolved confound in alignment
regularization: a commutator can decrease because SIGReg makes the latent
covariance more isotropic, rather than because the predictor learns a more
useful covariance-relative geometry.  The benchmark reuses the learned
pixel-to-latent world-model protocol, fixes generated splits and initialization
for each seed, and sweeps SIGReg and alignment weights jointly.

The artifact is CPU-only synthetic evidence.  It is a mechanism and boundary
experiment, not end-to-end LeWM, PushT, online-control, or GPU evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

try:
    from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
        PixelBenchmarkConfig,
        run_one,
    )
except ModuleNotFoundError:  # direct invocation from the project root/package snapshot
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    try:
        from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
            PixelBenchmarkConfig,
            run_one,
        )
    except ModuleNotFoundError:  # standalone package execution
        from run_pixel_world_model_synthetic import PixelBenchmarkConfig, run_one


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {"mean": mean, "std": std}


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        key = (
            f"{record['sigreg_weight']}:{record['mode']}"
            f":{record['alignment_weight']}"
        )
        groups.setdefault(key, []).append(record)

    output: dict[str, object] = {}
    for key, rows in groups.items():
        def values(path: tuple[str, ...]) -> list[float]:
            result: list[float] = []
            for row in rows:
                value: object = row
                for item in path:
                    value = value[item]  # type: ignore[index]
                result.append(float(value))
            return result

        output[key] = {
            "n_seeds": len(rows),
            "test_one_step_mse": _mean_std(values(("test_one_step_mse",))),
            "rollout_h8_mse": _mean_std(values(("rollout_mse", "8"))),
            "composition_error": _mean_std(
                values(("action_composition_error_mean",))
            ),
            "global_commutator": _mean_std(
                values(("diagnostics", "global_commutator"))
            ),
            "conditional_commutator": _mean_std(
                values(("diagnostics", "conditional_commutator"))
            ),
            "stable_rank": _mean_std(values(("diagnostics", "stable_rank"))),
            "best_epoch": _mean_std(values(("best_epoch",))),
        }
    return output


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _write_figure(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    summary = report["summary"]
    sigreg_weights = [float(value) for value in report["protocol"]["sigreg_weights"]]  # type: ignore[index]
    alignment_weights = [float(value) for value in report["protocol"]["alignment_weights"]]  # type: ignore[index]
    colors = {0.0: "#1b9e77", 0.01: "#d95f02", 0.1: "#7570b3", 1.0: "#e7298a"}
    markers = {"global": "o", "conditional": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [
        ("global_commutator", "Pooled commutator", True),
        ("conditional_commutator", "Conditional commutator", True),
        ("stable_rank", "Latent stable rank", False),
        ("rollout_h8_mse", "Random-sequence $h=8$ MSE", True),
    ]
    for axis, (metric, title, log_y) in zip(axes.flat, panels):
        for sigreg_weight in sigreg_weights:
            color = colors.get(sigreg_weight, None)
            baseline = summary.get(f"{sigreg_weight}:none:0.0")  # type: ignore[union-attr]
            if baseline is not None:
                axis.axhline(
                    float(baseline[metric]["mean"]),  # type: ignore[index]
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    label=f"SIGReg={sigreg_weight:g}, none",
                )
            for mode in ("global", "conditional"):
                xs: list[float] = []
                ys: list[float] = []
                for alignment_weight in alignment_weights:
                    key = f"{sigreg_weight}:{mode}:{alignment_weight}"
                    row = summary.get(key)  # type: ignore[union-attr]
                    if row is None:
                        continue
                    xs.append(alignment_weight)
                    ys.append(float(row[metric]["mean"]))  # type: ignore[index]
                if xs:
                    axis.plot(
                        xs,
                        ys,
                        color=color,
                        marker=markers[mode],
                        linewidth=1.2,
                        label=f"SIGReg={sigreg_weight:g}, {mode}",
                    )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("alignment weight")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("normalized value")
    axes[0, 1].set_ylabel("normalized value")
    axes[1, 0].set_ylabel("dimension-equivalent rank")
    axes[1, 1].set_ylabel("MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8,
    )
    fig.suptitle("SIGReg × alignment phase diagnostics", y=1.12)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# SIGReg × alignment phase diagnostics",
        "",
        "This is a CPU-only learned synthetic pixel-world experiment. It tests whether alignment changes remain distinct from SIGReg-induced isotropy; it is not nonlinear LeWM, PushT, online-control, or GPU evidence.",
        "",
        f"All rows use the same generated protocol and {report['protocol']['n_seeds']} seed(s) per condition; checkpoints are selected by held-out one-step MSE. Table entries are means; standard deviations are retained in JSON.",
        "",
        "| SIGReg | mode | alignment weight | one-step MSE | random h=8 MSE | composition | global C | conditional C | stable rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        sigreg, mode, weight = key.split(":")

        def fmt(name: str) -> str:
            return f"{values[name]['mean']:.6g} +/- {values[name]['std']:.2g}"  # type: ignore[index]

        lines.append(
            f"| {float(sigreg):.6g} | {mode} | {float(weight):.6g} | "
            f"{fmt('test_one_step_mse')} | {fmt('rollout_h8_mse')} | "
            f"{fmt('composition_error')} | {fmt('global_commutator')} | "
            f"{fmt('conditional_commutator')} | {fmt('stable_rank')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: a lower commutator is not sufficient evidence of better prediction. Stable rank is a diagnostic for covariance flattening, not a causal measure of representation quality. The experiment is designed to identify whether SIGReg and alignment produce separable or confounded geometry changes.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sigreg-weights", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    modes = ["global", "conditional"]
    base_config = PixelBenchmarkConfig(epochs=args.epochs)
    records: list[dict[str, object]] = []
    start = time.time()
    for sigreg_weight in args.sigreg_weights:
        config = replace(base_config, sigreg_weight=float(sigreg_weight))
        for mode in ["none", *modes]:
            weights = [0.0] if mode == "none" else args.alignment_weights
            for alignment_weight in weights:
                for seed in args.seeds:
                    print(
                        f"[sigreg-alignment-phase] sigreg={sigreg_weight} "
                        f"mode={mode} alignment={alignment_weight} seed={seed}: "
                        "separating isotropy from covariance-dynamics alignment"
                    )
                    record = run_one(mode, alignment_weight, seed, config)
                    record["sigreg_weight"] = float(sigreg_weight)
                    records.append(record)
    expected_records = len(args.sigreg_weights) * len(args.seeds) * (
        1 + len(modes) * len(args.alignment_weights)
    )
    if len(records) != expected_records:
        raise RuntimeError(
            f"incomplete phase matrix: expected {expected_records} records, got {len(records)}"
        )
    expected_groups = len(args.sigreg_weights) * (
        1 + len(modes) * len(args.alignment_weights)
    )
    summary = _summary(records)
    if len(summary) != expected_groups:
        raise RuntimeError(
            f"incomplete phase groups: expected {expected_groups}, got {len(summary)}"
        )
    if any(int(values["n_seeds"]) != len(args.seeds) for values in summary.values()):
        raise RuntimeError("at least one phase condition has an incorrect seed count")
    report = {
        "protocol": {
            **asdict(base_config),
            "sigreg_weight": "swept",
            "device": "cpu",
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
            "sigreg_weights": args.sigreg_weights,
            "alignment_weights": args.alignment_weights,
            "modes": ["none", *modes],
            "matched_splits_and_initialization": True,
            "validation_mse_checkpoint_selection": True,
            "synthetic_pixel_only": True,
            "phase_question": "Does alignment remain distinct from SIGReg-induced covariance isotropy?",
            "elapsed_seconds": time.time() - start,
        },
        "records": records,
        "summary": summary,
    }
    if not _finite(report):
        raise FloatingPointError("phase report contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    _write_figure(report, args.output.with_name("phase_diagnostics.png"))
    args.output.with_name("README.md").write_text(
        "# SIGReg × alignment phase diagnostics\n\n"
        "This directory contains the matched CPU-only synthetic pixel-world phase experiment. "
        f"It sweeps SIGReg and global/conditional alignment weights with identical data splits "
        f"and {len(args.seeds)} initialization seed(s) per condition. Stable rank is recorded to detect the "
        "trivial isotropy route to a lower commutator. The result is a mechanism boundary, "
        "not LeWM, PushT, online-control, causal-control, or GPU evidence.\n"
    )
    args.output.with_name("notes.txt").write_text(
        f"Protocol: SIGReg weights, alignment weights, and {len(args.seeds)} seed(s) are fully crossed; "
        "validation-MSE checkpoint selection is shared across conditions.\n"
        "Primary readouts: one-step MSE, random-sequence h=8 MSE, action composition, "
        "global/conditional commutator, and latent stable rank.\n"
        "Interpretation: stable-rank movement is a covariance-isotropy diagnostic; it does "
        "not by itself establish useful geometry or causal planning benefit.\n"
        "Environment boundary: CPU-only synthetic pixel world; no CUDA, PushT, online control, "
        "or nonlinear LeWM claim.\n"
    )
    args.output.with_name("visual_audit.md").write_text(
        "# Visual audit\n\n"
        "The phase diagnostic figure contains four panels for pooled commutator, conditional "
        "commutator, stable rank, and random-sequence h=8 MSE across alignment weights. "
        "The figure is a diagnostic summary; exact values and per-seed standard deviations "
        "remain in the JSON artifact.\n"
    )
    print(json.dumps({"records": len(records), "summaries": len(report["summary"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
