"""Summarize lifted-contact runs without pooling incompatible protocols.

The summary is intentionally descriptive. It never converts a local ranking
gain into a closure claim, and it reports each run's root count, selection
metric, temporal-history flag, and contact-loss rate before pooled totals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads((path / "lifted_contact_adapter.json").read_text())


def run_row(path: Path) -> dict[str, Any]:
    report = load_report(path)
    protocol = report["protocol"]
    closure = report["closed_loop"]
    traces = [trace for record in closure["rows"] for trace in record["trace"]]
    contact_loss = sum(float(item["contact_distance_after"]) > 40.0 for item in traces)
    fallback_steps = sum(bool(item.get("contact_recovery_fallback", False)) for item in traces)
    success = sum(bool(record["block_pose_success"]) for record in closure["rows"])
    return {
        "artifact": str(path),
        "seed": int(protocol.get("train_seed", 0)) - 1000,
        "selection_mode": protocol.get("selection_mode"),
        "data_protocol": protocol.get("data_protocol"),
        "pose_xy_weight": protocol.get("pose_projection_metric", {}).get("xy_weight", 1.0),
        "pose_angle_weight": protocol.get("pose_projection_metric", {}).get("angle_weight", 1.0),
        "temporal_history_verified": bool(report.get("temporal_history_verified", False)),
        "train_roots": int(protocol["train_roots"]),
        "held_out_roots": int(protocol["held_out_roots"]),
        "ranking": report["held_out_ranking"]["metrics"],
        "closure": closure["metrics"],
        "static_comparison": report.get("static_comparison"),
        "closure_successes": int(success),
        "closure_cases": len(closure["rows"]),
        "contact_loss_steps": int(contact_loss),
        "trace_steps": len(traces),
        "contact_loss_rate": float(contact_loss / len(traces)) if traces else None,
        "fallback_steps": int(fallback_steps),
    }


def mean(rows: list[dict[str, Any]], section: str, key: str) -> float | None:
    values = [float(row[section][key]["estimate"]) for row in rows]
    return sum(values) / len(values) if values else None


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = sum(row["closure_cases"] for row in rows)
    successes = sum(row["closure_successes"] for row in rows)
    traces = sum(row["trace_steps"] for row in rows)
    contact_loss = sum(row["contact_loss_steps"] for row in rows)
    return {
        "runs": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "closure_successes": successes,
        "closure_cases": cases,
        "closure_rate": successes / cases if cases else None,
        "mean_direction_accuracy": mean(rows, "ranking", "direction_accuracy"),
        "mean_learned_regret": mean(rows, "ranking", "learned_regret"),
        "mean_final_xy": mean(rows, "closure", "final_xy"),
        "mean_final_angle": mean(rows, "closure", "final_angle"),
        "contact_loss_rate": contact_loss / traces if traces else None,
    }


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = json.dumps(
            [row["selection_mode"], row["pose_xy_weight"], row["data_protocol"]],
            separators=(",", ":"),
        )
        groups.setdefault(key, []).append(row)
    return groups


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lifted contact adapter pooled audit",
        "",
        "This report keeps local ranking and closed-loop pose closure separate. "
        "Runs with different selection metrics or pulse horizons are not pooled together.",
        "",
        "| run | seed | mode | xy weight | temporal history | direction acc. | regret | closure | final XY | angle | contact-loss |",
        "|---|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {Path(row['artifact']).name} | {row['seed']} | {row['selection_mode']} | "
            f"{row['pose_xy_weight']:.1f} | {row['temporal_history_verified']} | "
            f"{row['ranking']['direction_accuracy']['estimate']:.3f} | "
            f"{row['ranking']['learned_regret']['estimate']:.4f} | "
            f"{row['closure_successes']}/{row['closure_cases']} | "
            f"{row['closure']['final_xy']['estimate']:.2f} | "
            f"{row['closure']['final_angle']['estimate']:.3f} | "
            f"{row['contact_loss_steps']}/{row['trace_steps']}"
        )
    lines += ["", "## Pooled groups", ""]
    for key, group in sorted(group_rows(rows).items()):
        summary = pooled(group)
        lines.append(
            f"- {key}: {summary['closure_successes']}/{summary['closure_cases']} closure "
            f"({summary['closure_rate']:.3f}), direction accuracy "
            f"{summary['mean_direction_accuracy']:.3f}, mean final XY "
            f"{summary['mean_final_xy']:.2f}, contact-loss rate "
            f"{summary['contact_loss_rate']:.3f}."
        )
    lines += ["", "## Static-lift comparison recorded inside each temporal run", ""]
    for row in rows:
        static = row.get("static_comparison")
        if static is None:
            continue
        static_ranking = static["ranking"]["metrics"]
        static_closure = static["closure"]["metrics"]
        lines.append(
            f"- {Path(row['artifact']).name}: static direction "
            f"{static_ranking['direction_accuracy']['estimate']:.3f}, static regret "
            f"{static_ranking['learned_regret']['estimate']:.4f}, static closure "
            f"{sum(bool(item['block_pose_success']) for item in static['closure']['rows'])}/"
            f"{len(static['closure']['rows'])}, static final XY "
            f"{static_closure['final_xy']['estimate']:.2f}."
        )
    lines += [
        "",
        "## Evidence boundary",
        "",
        "The temporal lift improves or preserves local held-out action ranking in "
        "some root-disjoint runs, but the pooled closure rate remains low and "
        "contact-loss rates remain high. These artifacts therefore support a "
        "representation/interface diagnosis, not stable learned posture closure, "
        "causal dynamics, visual planning, or a universal world model.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    rows = [run_row(Path(value)) for value in args.inputs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {key: pooled(group) for key, group in group_rows(rows).items()}
    (output / "pooled.json").write_text(json.dumps({"runs": rows, "groups": groups}, indent=2, sort_keys=True))
    (output / "pooled.md").write_text(markdown(rows))
    print(json.dumps({"output": str(output), "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
