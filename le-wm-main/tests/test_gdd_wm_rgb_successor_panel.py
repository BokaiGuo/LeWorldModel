import json

import numpy as np
import pytest

from experiments.gdd_wm.aggregate_rgb_successor_panel import aggregate


def _report(path, shift):
    rows = []
    for root in ("r0", "r1"):
        for candidate in range(2):
            source = np.array([1.0, 0.0])
            realized = np.array([0.0, 0.0])
            predicted = np.array([shift, 0.0])
            rows.append(
                {
                    "instance_id": root,
                    "candidate_id": candidate,
                    "split": "outcome",
                    "source_projection": source.tolist(),
                    "realized_terminal_projection": realized.tolist(),
                    "predicted_terminal_projection": predicted.tolist(),
                }
            )
    path.write_text(
        json.dumps(
            {
                "gates": {
                    "renderer_consistency": True,
                    "realized_rgb_latents_available": True,
                    "pca_projected_successor_comparison": True,
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_rgb_successor_aggregate_uses_outcome_roots_and_gates(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _report(first, 0.2)
    _report(second, 0.4)
    output = tmp_path / "aggregate.json"
    report = aggregate(
        [f"a|{first}", f"b|{second}"], output, samples=200, seed=7
    )
    result = report["aggregate"]["relative_l2_improvement"]
    assert result["all_case_points_positive"] is True
    assert result["two_level_case_and_root_bootstrap"]["point"] == pytest.approx(0.7)
    assert report["case_summary"]["a"]["relative_l2_improvement"]["root_count"] == 2
    assert output.exists()
