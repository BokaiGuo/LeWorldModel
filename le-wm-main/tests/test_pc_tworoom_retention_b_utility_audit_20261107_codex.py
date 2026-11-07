import json

import numpy as np
import pytest

from experiments import pc_tworoom_retention_b_utility_audit_20261107_codex as audit


def test_useful_prefix_recall_distinguishes_success_before_and_after_prefix():
    term = np.zeros((8, 8, 25), dtype=bool)
    term[0, 0, 4] = True
    term[1, 1, 12] = True
    term[2, 2, 3] = True
    term[2, 2, 15] = True
    result = audit.utility_metrics(term, [0, 1])
    assert result["ever"]["useful_prefixes"] == 3
    assert result["ever"]["retained_useful_prefixes"] == 2
    assert result["ever"]["recall"] == pytest.approx(2 / 3)
    assert result["post_prefix"]["useful_prefixes"] == 1
    assert result["post_prefix"]["retained_success"]
    assert result["post_prefix"]["recall"] == 1


def test_no_positive_and_invalid_input():
    term = np.zeros((8, 8, 25), dtype=bool)
    assert audit.utility_metrics(term, [0, 1])["ever"]["recall"] is None
    with pytest.raises(ValueError):
        audit.utility_metrics(term, [0, 0])
    with pytest.raises(ValueError):
        audit.utility_metrics(term.astype(int), [0, 1])


def test_audit_matches_sealed_termination_bank():
    if not audit.OUT.exists():
        pytest.skip("audit has not been run")
    record = json.loads(audit.OUT.read_text())
    assert record["plan_sha256"] == audit.file_hash(audit.q.PLAN)
    assert record["physical_sha256"] == audit.file_hash(audit.source.PHYS)
    assert [r["root"] for r in record["rows"]] == list(range(10, 50))
    with np.load(audit.source.PHYS, allow_pickle=False) as data:
        for row in record["rows"]:
            term = data[f"r{row['root']}_term"].reshape(8, 8, 25)
            for arm in audit.q.METHODS:
                entry = row["arms"][arm]
                assert entry["metrics"] == audit.utility_metrics(term, entry["retained"])
