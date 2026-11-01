import copy

import pytest

from experiments.spectral_reachability import p1_uphill_cp5_ledger_pairs_20261030 as audit
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec


def journal(kinds):
    previous = None
    lines = []
    for index, (kind, payload) in enumerate(kinds):
        body = {"index": index, "previous": previous, "kind": kind, "payload": payload}
        previous = spec.digest(body)
        lines.append(spec.canonical({**body, "sha256": previous}))
    return b"\n".join(lines) + b"\n"


def test_phase_counts_and_unfinished_rejected():
    header = ("header", {})
    intent = ("intent", {"phase": "common_suffix", "kind": "predict"})
    done = ("return", {})
    terminal = ("terminal", {})
    assert audit.journal_counts(journal([header, intent, done, terminal])) == [
        {"phase": "common_suffix", "operation": "predict", "returned_dispatches": 1}
    ]
    with pytest.raises(ValueError):
        audit.journal_counts(journal([header, intent, terminal]))
    with pytest.raises(ValueError):
        audit.journal_counts(journal([header, done, terminal]))


def test_real_saved_ledgers_pair_and_identity_guard():
    sources = [audit.read_source(source) for source in audit.SOURCES]
    report = audit.paired(sources)
    assert len(report["rows"]) == 24
    assert all(row["zero"]["returned_dispatches"] == row["warm"]["returned_dispatches"]
               for row in report["rows"])
    assert report["equal_budget_certified"] is False
    assert report["coverage_verified"] is False
    altered = copy.deepcopy(sources)
    altered[1][2][0]["identity"]["cp5_pixel_sha256"] = "different"
    with pytest.raises(ValueError):
        audit.paired(altered)
