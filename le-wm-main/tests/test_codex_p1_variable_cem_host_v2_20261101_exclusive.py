from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_p1_variable_cem_host_v2_20261101_exclusive as v2
from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import ProposalJournal


def test_closed_journal_anchor_detects_changes_and_truncation(tmp_path):
    path = tmp_path / "full_cem_events.jsonl"
    journal = ProposalJournal(path, run_id="s0", arm="cem", proposal_kind="cem")
    journal.append({"type": "identity"})
    journal.append({"type": "done"})
    journal.close()
    anchor = v2.anchor_journal(journal)
    assert anchor["status"] == "closed"
    assert anchor["event_count"] == 2
    v2.verify_anchor(path, anchor)
    original = path.read_bytes()
    path.write_bytes(original.replace(b"done", b"gone"))
    with pytest.raises(ValueError):
        v2.verify_anchor(path, anchor)
    path.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError):
        v2.verify_anchor(path, anchor)


def test_open_poisoned_and_missing_journal_fail_closed(tmp_path):
    path = tmp_path / "full_cem_events.jsonl"
    journal = ProposalJournal(path, run_id="s0", arm="cem", proposal_kind="cem")
    journal.append({"type": "identity"})
    with pytest.raises(ValueError):
        v2.anchor_journal(journal)
    journal.close()
    journal.poisoned = True
    with pytest.raises(ValueError):
        v2.anchor_journal(journal)
    with pytest.raises(ValueError):
        v2.anchor_journal(None)


def test_recomputed_chain_cannot_bypass_report_anchor(tmp_path):
    path = tmp_path / "full_cem_events.jsonl"
    journal = ProposalJournal(path, run_id="s0", arm="cem", proposal_kind="cem")
    journal.append({"type": "identity"})
    journal.close()
    anchor = v2.anchor_journal(journal)
    path.unlink()
    journal = ProposalJournal(path, run_id="s0", arm="cem", proposal_kind="cem")
    journal.append({"type": "changed"})
    journal.close()
    with pytest.raises(ValueError):
        v2.verify_anchor(path, anchor)


def test_report_contains_closed_anchor_and_missing_writer_demotes(tmp_path):
    path = tmp_path / "full_cem_events.jsonl"
    journal = ProposalJournal(path, run_id="s0", arm="cem", proposal_kind="cem")
    journal.append({"type": "identity"})
    journal.close()
    result = v2.finalize_report({"status": "recorded_known_root_not_admitted", "errors": []},
                                [SimpleNamespace(journal=journal)], tmp_path)
    assert result["cem_journal"]["sha256"]
    assert result["cem_journal"]["status"] == "closed"
    failed = v2.finalize_report({"status": "recorded_known_root_not_admitted", "errors": []},
                                [], tmp_path)
    assert failed["status"] == "partial_or_failed"
    assert failed["cem_journal"]["status"] == "unavailable_or_invalid"
    assert failed["errors"]
