import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as consumer


def test_journal_rejects_changed_event(tmp_path):
    event = {"index": 0, "previous_sha256": None, "kind": "run_finished"}
    event["event_sha256"] = consumer.runner.native.source.digest(event)
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps(event) + "\n")
    assert consumer.events(path, event["event_sha256"]) == [event]
    event["kind"] = "changed"
    path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        consumer.events(path, event["event_sha256"])


def test_fixed_real_report_and_rejects_changed_action():
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/p1_backtrack_physical_runner_20261030_MAIN02"
    prepared = root / "results/p1_backtrack_physical_runner_20261030_prepare_MAIN02.json"
    if not (directory / "report.json").is_file() or not prepared.is_file():
        pytest.skip("fixed physical artifact not available")
    report = json.loads((directory / "report.json").read_text())
    prep = json.loads(prepared.read_text())
    assert consumer.check(directory, report, prep)["rows"] == 16
    report["rows"][0]["plan"]["raw15_float32_hex"] = "00" * 120
    with pytest.raises(ValueError, match="row and task identity"):
        consumer.check(directory, report, prep)
