"""Non-confirmatory goal-geometry sensitivity replay for a sealed V6 panel.

This diagnostic keeps the frozen source/context/candidate panel and model
architecture fixed, changes only the declared task goal, recomputes the
identity-bound calibration costs from already stored terminal states, and
reruns selection plus outcome scoring.  It must not be used as independent
evidence because the panel's outcome roots have already been opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel import validate_goal_qpos


def _rewrite_goal(context: Path, calibration_labels: Path, output: Path, goal: np.ndarray) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    panel = json.loads(context.read_text(encoding="utf-8"))
    labels = json.loads(calibration_labels.read_text(encoding="utf-8"))
    for row in panel["records"]:
        row["goal_raw"] = goal.tolist()
    for value in labels.values():
        terminal = np.asarray(value["realized_terminal_state"], dtype=np.float64)
        value["real_task_cost"] = float(np.linalg.norm(terminal[:2] - goal))
    context_out = output / "goal_sensitivity_context.json"
    labels_out = output / "goal_sensitivity_calibration_outcomes.json"
    context_out.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    labels_out.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return context_out, labels_out


def run(panel_dir: Path, output: Path, *, goal_qpos: tuple[float, float] = (0.8, 0.8), device: str = "cuda", epochs: int = 100) -> dict[str, Any]:
    panel_dir = panel_dir.resolve()
    context_payload = json.loads((panel_dir / "temporal_ood_v6_context.json").read_text(encoding="utf-8"))
    schema = context_payload.get("metadata", {}).get("schema")
    if schema != v6.CONTEXT_SCHEMA:
        raise ValueError(
            f"goal sensitivity requires current {v6.CONTEXT_SCHEMA} artifact; "
            f"received historical {schema!r}"
        )
    output.mkdir(parents=True, exist_ok=True)
    goal = validate_goal_qpos(goal_qpos)
    context_out, labels_out = _rewrite_goal(
        panel_dir / "temporal_ood_v6_context.json",
        panel_dir / "temporal_ood_v6_calibration_outcomes.json",
        output,
        goal,
    )
    frozen = v6.calibrate(
        panel_dir / "temporal_ood_v6_source_dataset.json",
        panel_dir / "temporal_ood_v6_source_arrays.npz",
        context_out,
        labels_out,
        output / "temporal_ood_v6_goal_sensitivity_frozen_selection.json",
        epochs=epochs,
        device=device,
    )
    report = v6.score(
        panel_dir / "temporal_ood_v6_source_dataset.json",
        panel_dir / "temporal_ood_v6_source_arrays.npz",
        context_out,
        output / "temporal_ood_v6_goal_sensitivity_frozen_selection.json",
        output / "score",
        device=device,
    )
    result = {
        "schema": "pointmaze_temporal_ood_v6_goal_sensitivity.v1",
        "panel": str(panel_dir),
        "goal_qpos": goal.tolist(),
        "historical_outcome_roots_reused": True,
        "confirmatory": False,
        "reason": "goal-only replay on an already-opened V6 panel",
        "frozen_selection": frozen,
        "score": report,
    }
    (output / "goal_sensitivity_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-qpos", type=float, nargs=2, default=[0.8, 0.8])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
