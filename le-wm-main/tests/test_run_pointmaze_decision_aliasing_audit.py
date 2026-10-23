from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit import root_table, run


def _row(root: str, candidate: int, history: list[float]) -> dict:
    return {
        "instance_id": root, "candidate_id": candidate, "candidate_bank_hash": "held-bank",
        "source_history_model": history, "source_latent_model": history,
    }


def test_root_table_centers_realized_costs_and_requires_candidate_invariant_history() -> None:
    records = [_row("r0", 1, [0.1, 0.2]), _row("r0", 0, [0.1, 0.2])]
    labels = {
        "r0/0": {"real_task_cost": 2.0},
        "r0/1": {"real_task_cost": 0.0},
    }
    representation, contrast, roots, signature, bank = root_table(records, labels, representation_key="source_history_model")
    assert representation.tolist() == [[0.1, 0.2]]
    assert contrast.tolist() == [[1.0, -1.0]]
    assert roots.tolist() == ["r0"]
    assert signature == ("0", "1")
    assert bank == "held-bank"


def test_root_table_rejects_candidate_conditioned_history_or_missing_labels() -> None:
    records = [_row("r0", 0, [0.0]), _row("r0", 1, [1.0])]
    with pytest.raises(ValueError, match="candidate-invariant"):
        root_table(records, {"r0/0": {"real_task_cost": 0.0}, "r0/1": {"real_task_cost": 1.0}}, representation_key="source_history_model")
    records = [_row("r0", 0, [0.0]), _row("r0", 1, [0.0])]
    with pytest.raises(ValueError, match="missing finite realized label"):
        root_table(records, {"r0/0": {"real_task_cost": 0.0}}, representation_key="source_history_model")


def test_sidecar_outcome_labels_are_used_only_when_explicitly_requested(tmp_path: Path) -> None:
    records = []
    for root, split, history in (("cal0", "calibration", [0.0]), ("cal1", "calibration", [1.0]), ("out", "outcome", [0.2])):
        records.extend([{**_row(root, candidate, history), "split": split} for candidate in (0, 1)])
    calibration = {
        "cal0/0": {"real_task_cost": 0.0}, "cal0/1": {"real_task_cost": 1.0},
        "cal1/0": {"real_task_cost": 1.0}, "cal1/1": {"real_task_cost": 0.0},
    }
    sidecar = {"labels": {"out/0": {"real_task_cost": 1.0}, "out/1": {"real_task_cost": 0.0}}}
    context_path, calibration_path, report_path, sidecar_path, output = (
        tmp_path / "context.json", tmp_path / "calibration.json", tmp_path / "report.json",
        tmp_path / "sidecar.json", tmp_path / "audit.json",
    )
    for path, payload in ((context_path, {"records": records}), (calibration_path, calibration), (report_path, {}), (sidecar_path, sidecar)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = run(
        context=context_path, calibration_labels=calibration_path, outcome_report=report_path,
        outcome_labels=sidecar_path, output=output, representation_key="source_history_model", max_reported_pairs=1,
    )
    assert result["provenance"]["outcome_labels"]["kind"] == "post_execution_label_sidecar"
    assert len(result["audit"]["top_pairs"]) == 1
