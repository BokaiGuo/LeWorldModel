"""Summarize directional audit, planning alignment, and eval metrics."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="result directories")
    parser.add_argument("--output", default=None, help="optional markdown summary path")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def parse_eval_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    marker = "metrics:"
    idx = text.rfind(marker)
    if idx < 0:
        return {}
    line = text[idx:].splitlines()[0][len(marker) :].strip()
    try:
        value = ast.literal_eval(line)
        return value if isinstance(value, dict) else {"metrics": value}
    except Exception:
        return {"raw_metrics": line}


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def best_by_state_distance(
    costs: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    scored = [
        (name, metrics, finite_float(metrics.get("spearman_cost_state_distance")))
        for name, metrics in costs.items()
    ]
    scored = [(name, metrics, score) for name, metrics, score in scored if score is not None]
    if not scored:
        return None
    name, metrics, _ = max(scored, key=lambda item: item[2])
    return name, metrics


def summarize_run(run: Path) -> dict[str, Any]:
    lift = read_json(run / "lift_metrics.json")
    empirical = read_json(run / "empirical_spectral_audit.json")
    align = read_json(run / "planning_alignment.json")
    eval_metrics = read_json(run / "eval_metrics.json") or parse_eval_metrics(run / "pusht_results.txt")

    weighted = lift.get("weighted_mean", {})
    empirical_weighted = empirical.get("weighted_mean", {})
    empirical_lifts = empirical_weighted.get("lifts", {})
    dyn = weighted.get("dynamics_lift", {}) or empirical_lifts.get("dynamics", {})
    pred = weighted.get("predicted_dynamics_lift", {}) or empirical_lifts.get("predicted_dynamics", {})
    plan = weighted.get("planning_lift", {}) or empirical_lifts.get("planning", {})
    resid = weighted.get("prediction_residual_lift", {}) or empirical_lifts.get("prediction_residual", {})
    nuis = weighted.get("nuisance_lift", {}) or empirical_lifts.get("nuisance", {})
    spectral = align.get("spectral_planner_costs", {})
    best_spectral = best_by_state_distance(spectral)
    planner_costs = {"latent_cost": align, **spectral}
    best_planner = best_by_state_distance(planner_costs)
    vanilla_state = finite_float(align.get("spearman_cost_state_distance"))
    best_spectral_state = (
        finite_float(best_spectral[1].get("spearman_cost_state_distance"))
        if best_spectral
        else None
    )

    return {
        "run": str(run),
        "dtl@32": dyn.get("32"),
        "pred_dtl@32": pred.get("32"),
        "planning_lift@32": plan.get("32"),
        "residual_lift@32": resid.get("32"),
        "nuisance_lift@32": nuis.get("32"),
        "pred_cos": (
            lift.get("prediction_tangent_alignment", {}).get("mean_cosine")
            or empirical_weighted.get("prediction_spectral_error", {}).get("mean_cosine")
        ),
        "spearman_cost_success": align.get("spearman_cost_success"),
        "spearman_cost_state_distance": align.get("spearman_cost_state_distance"),
        "best_spectral_cost": best_spectral[0] if best_spectral else None,
        "best_spectral_success_spearman": (
            best_spectral[1].get("spearman_cost_success") if best_spectral else None
        ),
        "best_spectral_state_spearman": (
            best_spectral[1].get("spearman_cost_state_distance") if best_spectral else None
        ),
        "best_spectral_state_delta_vs_latent": (
            best_spectral_state - vanilla_state
            if best_spectral_state is not None and vanilla_state is not None
            else None
        ),
        "best_planner_cost": best_planner[0] if best_planner else None,
        "best_planner_success_spearman": (
            best_planner[1].get("spearman_cost_success") if best_planner else None
        ),
        "best_planner_state_spearman": (
            best_planner[1].get("spearman_cost_state_distance") if best_planner else None
        ),
        "success": eval_metrics.get("success") or eval_metrics.get("success_rate"),
        "score": eval_metrics.get("score") or eval_metrics.get("return"),
        "num_alignment_points": align.get("num_points"),
    }


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "dtl@32",
        "pred_dtl@32",
        "planning_lift@32",
        "residual_lift@32",
        "nuisance_lift@32",
        "pred_cos",
        "spearman_cost_success",
        "spearman_cost_state_distance",
        "best_spectral_cost",
        "best_spectral_success_spearman",
        "best_spectral_state_spearman",
        "best_spectral_state_delta_vs_latent",
        "best_planner_cost",
        "best_planner_success_spearman",
        "best_planner_state_spearman",
        "success",
        "score",
        "num_alignment_points",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(scalar(row.get(h)) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [summarize_run(Path(run)) for run in args.runs]
    md = to_markdown(rows)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
