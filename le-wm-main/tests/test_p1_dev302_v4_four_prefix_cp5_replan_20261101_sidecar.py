import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import (
    p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar as subject,
)


def write_json(path, value):
    path.write_bytes(subject.canonical(value) + b"\n")
    return subject.sha256(path)


def write_physical_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    directory = tmp_path / "results" / "dev302_v4_four_prefix_physical"
    directory.mkdir(parents=True)
    v4_report = subject.read_json(subject.V4_REPORT)
    v4_rows = {row["index"]: row for row in v4_report["rows"]}
    rows = []
    for order, role in enumerate(subject.ROLES):
        expected = v4_rows[v4_report[subject.ROLE_INDEX_FIELDS[role]]]
        pixels = np.full((224, 224, 3), order * 17, np.uint8)
        filename = f"{role}_prefix_05.png"
        Image.fromarray(pixels).save(directory / filename)
        actions = np.asarray(expected["actions"], np.float32)
        rows.append(
            {
                "role": role,
                "candidate_index": expected["index"],
                "candidate_label": expected["label"],
                "status": "prefix_completed",
                "error": None,
                "plan_returned": 5,
                "planned_actions_sha256": subject.action_sha(actions),
                "transitions": [
                    {"step": index + 1, "action": action.tolist()}
                    for index, action in enumerate(actions)
                ],
                "observations": [
                    {
                        "phase": "prefix",
                        "step": 5,
                        "file": filename,
                        "file_sha256": subject.sha256(directory / filename),
                        "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    }
                ],
            }
        )
    events = []
    previous = None
    for index, kind in enumerate(("run_started", "run_finished")):
        body = {"index": index, "previous": previous, "kind": kind}
        row = {**body, "sha256": subject.digest(body)}
        events.append(row)
        previous = row["sha256"]
    journal = directory / "journal.jsonl"
    journal.write_bytes(b"".join(subject.canonical(row) + b"\n" for row in events))
    report = {
        "schema": subject.physical_v4.NAME + ".report.v1",
        "status": "recorded_development_v3_prefixes",
        "prepare_sha256": subject.PHYSICAL_PREP_SHA256,
        "planner_prep_sha256": subject.V4_PREP_SHA256,
        "planner_report_sha256": subject.V4_REPORT_SHA256,
        "errors": [],
        "rows": rows,
        "journal_sha256": subject.sha256(journal),
        "journal_tail": previous,
        "post_prefix_outcomes_used_for_planning": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    report_sha = write_json(directory / "report.json", report)
    return directory, report_sha


def test_protocol_binds_four_actual_v4_prefixes_and_equal_budget(tmp_path, monkeypatch):
    directory, report_sha = write_physical_fixture(tmp_path, monkeypatch)
    value = subject.protocol(directory, report_sha)
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert [row["candidate_index"] for row in value["roles"]] == [5, 0, 4, 7]
    assert all(len(row["actual_prefix"]) == 5 for row in value["roles"])
    assert all(len(row["old_v4_tail"]) == 2 for row in value["roles"])
    assert value["algorithm"]["remaining_steps"] == 10
    assert value["equal_cem_and_robust_budget"] is True
    assert value["post_cp5_physical_outcomes_used"] is False
    assert value["physical_execution"] is False
    assert value["P1_admission"] is False
    assert value["per_arm_axis_caps"]["predict"] == {"calls": 10, "samples": 3200}


def test_existing_v2_selected_physical_report_is_rejected(monkeypatch):
    old_root = Path(
        "/home/gbk/code/LeWorldModel/le-wm-main/results/"
        "p1_dev302_mixed_prefix_physical_20261101_codex_RUN_MAIN01"
    )
    monkeypatch.setattr(subject, "ROOT", old_root.parents[1])
    with pytest.raises(ValueError, match="V4 four-prefix physical boundary|V4 physical prefix identity"):
        subject.protocol(old_root, subject.sha256(old_root / "report.json"))


class FakeJournal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


def test_axis_meter_is_exact_and_fail_closed():
    journal = FakeJournal()
    caps = {"predict": {"calls": 2, "samples": 8}}
    meter = subject.AxisMeter("selected", journal, caps)
    assert meter.call("predict", 4, lambda: 1) == 1
    assert meter.call("predict", 4, lambda: 2) == 2
    ledger = meter.settle()
    assert ledger["predict"] == {
        "attempted_calls": 2,
        "completed_calls": 2,
        "attempted_samples": 8,
        "completed_samples": 8,
    }
    with pytest.raises(ValueError, match="axis cap"):
        meter.call("predict", 1, lambda: None)


def test_axis_meter_does_not_complete_failed_call():
    journal = FakeJournal()
    meter = subject.AxisMeter(
        "selected", journal, {"predict": {"calls": 1, "samples": 4}}
    )
    with pytest.raises(RuntimeError):
        meter.call("predict", 4, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ValueError, match="unsettled exact axis"):
        meter.settle()
    assert [event["kind"] for event in journal.events] == ["resource_attempt"]
