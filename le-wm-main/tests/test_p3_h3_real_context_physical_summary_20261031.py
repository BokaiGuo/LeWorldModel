import copy
import json

import pytest

from experiments.spectral_reachability import p3_h3_real_context_physical_summary_20261031 as summary


@pytest.fixture(scope="module")
def first_batch():
    path = summary.ROOT / "results/p3_h3_real_context_physical_20261031_batch00_RUN01"
    p = json.loads((path / "prepare.json").read_text())
    report = json.loads((path / "report.json").read_text())
    events = summary.source.batch.kernel.journal(path / "journal.jsonl", report["journal_chain_tail"])
    return p, report, events


def test_actual_batch_guard_and_pairing(first_batch):
    p, report, events = first_batch
    summary.validate_guards(p, report, events)
    rows = summary.pair_batch(p, report)
    assert len(rows) == 3
    assert all(row["old_returned"] >= 15 and row["new_returned"] >= 15 for row in rows)


def test_guard_order_and_snapshot_tampering_rejected(first_batch):
    p, report, events = copy.deepcopy(first_batch)
    idx = next(i for i, e in enumerate(events) if e["kind"] == "cp15_guard_passed")
    events[idx]["kind"] = "tampered"
    with pytest.raises(ValueError, match="cp15 guard ordering"):
        summary.validate_guards(p, report, events)
    p, report, events = copy.deepcopy(first_batch)
    report["rows"][0]["checkpoints"][-2]["snapshot"] = {}
    with pytest.raises(ValueError, match="saved cp15"):
        summary.validate_guards(p, report, events)


def test_padding_and_duplicate_disagreement_are_not_silent(first_batch):
    rows = summary.pair_batch(first_batch[0], first_batch[1])
    template = rows[0]
    items = []
    for i in range(54):
        row = copy.deepcopy(template)
        row["identity"] = {"logical_pair": i, "source_variant": "old", "padding": i >= 52}
        items.append(row)
    result = summary.aggregate(items)
    assert result["fixed_pairs"] == 52 and result["padding_excluded"] == 2
    assert result["unique_action_pairs"] == 1
    items[1]["new_status"] = "native_success"
    with pytest.raises(ValueError, match="duplicate outcome mismatch"):
        summary.aggregate(items)
