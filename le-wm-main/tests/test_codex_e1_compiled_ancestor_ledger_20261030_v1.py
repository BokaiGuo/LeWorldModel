import copy
import gzip
import sys

import pytest

from experiments.spectral_reachability import codex_e1_compiled_ancestor_ledger_20261030_v1 as audit


@pytest.fixture(scope="module")
def sources():
    return audit.load()


def test_real_sources(sources):
    result = audit.analyze(*sources)
    assert result["status"] == "incomplete"
    assert not result["equal_budget_certified"]
    assert not result["ranking_authorized"]
    assert not result["scientific_admission"]
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert set(row["event_axes"]) == {*audit.RESOURCES, "predictor_samples"}
        for axis in row["event_axes"].values():
            assert (axis["attempted"], axis["completed"]) == (144, 144)
            assert len(axis["event_set_sha256"]) == 64
    assert result == audit.analyze(*sources)


@pytest.mark.parametrize("mutation", ["report", "selection", "target", "root", "task"])
def test_binding_mutations_fail_closed(sources, mutation):
    old, new, report, events = sources
    old, new, report = copy.deepcopy(old), copy.deepcopy(new), copy.deepcopy(report)
    if mutation == "report":
        report["rows"][0]["resources"]["predictor"]["attempted"] += 1
    elif mutation == "selection":
        old["compiled"][0]["selection_sha256"] = "0" * 64
    elif mutation == "target":
        new["rows"][2]["target_f32_le_base64"] = "AAAA"
    elif mutation == "root":
        new["rows"][2]["root_slot"] = 1
    else:
        old["slots"][0]["task_id"] = "wrong"
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit.analyze(old, new, report, events)


@pytest.mark.parametrize("mutation", ["missing_attempt", "bad_return", "bad_attempt", "task"])
def test_event_mutations_fail_closed(sources, mutation):
    old, new, report, original = sources
    events = list(original)
    attempt = next(i for i, (item, _) in enumerate(events)
                   if item["event"]["kind"] == "dispatch_attempted")
    returned = next(i for i, (item, _) in enumerate(events)
                    if item["event"]["kind"] == "dispatch_returned")
    if mutation == "missing_attempt":
        events.pop(attempt)
    else:
        index = returned if mutation == "bad_return" else attempt
        item, digest = events[index]
        item = copy.deepcopy(item)
        if mutation == "bad_return":
            item["event"]["receipt"]["samples"] += 1
        elif mutation == "bad_attempt":
            item["event"]["receipt"]["index"] += 1
        else:
            item["task_id"] = "wrong"
        events[index] = (item, digest)
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit.analyze(old, new, report, events)


def test_truncated_gzip_fails():
    with pytest.raises(ValueError):
        audit.events_from_gzip(gzip.compress(b'{}\n')[:-4])


def test_pinned_source_hash_fails(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    path.write_text("{}")
    monkeypatch.setattr(audit, "OLD", path)
    with pytest.raises(ValueError):
        audit.load()


def test_output_is_exclusive(tmp_path, monkeypatch, sources):
    output = tmp_path / "result.json"
    monkeypatch.setattr(audit, "OUTPUT", output)
    monkeypatch.setattr(audit, "load", lambda: sources)
    monkeypatch.setattr(sys, "argv", ["audit"])
    assert audit.main() == 0
    first = output.read_bytes()
    assert audit.main() == 2
    assert output.read_bytes() == first
