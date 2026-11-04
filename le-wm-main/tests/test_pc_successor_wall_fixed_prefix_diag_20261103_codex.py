import json

import numpy as np
import pytest

from experiments import pc_successor_wall_fixed_prefix_diag_20261103_codex as diagnostic


def test_sealed_candidate_lineage_and_oracle_separation():
    source = json.loads(diagnostic.prior.PLANS.read_text())
    task, candidates = diagnostic.candidates_for_root(source["rows"], 0)
    assert len(candidates) == 32
    assert sum(x["source"] == "privileged_feasibility" for x in candidates) == 2
    assert all(x["group"] == "oracle_diagnostic" for x in candidates[-2:])
    assert len(task["goal"]) == 2


def test_candidate_nested_budgets_keep_prefix():
    source = json.loads(diagnostic.prior.PLANS.read_text())
    task, candidates = diagnostic.candidates_for_root(source["rows"], 0)
    result = diagnostic.run_candidate(task, 0, candidates[0])
    assert len(result["batches"]) == 32
    assert result["checkpoints"]["384"]["score"] <= result["checkpoints"]["192"]["score"]
    prefix = np.asarray(candidates[0]["prefix"])
    for batch in result["batches"]:
        actions = np.asarray(batch["actions"])
        np.testing.assert_array_equal(actions[:, :len(prefix)], np.broadcast_to(prefix, (12, *prefix.shape)))


def test_sealed_run_audit_when_present():
    if not diagnostic.AUDIT.exists():
        pytest.skip("diagnostic has not yet been audited")
    audit = json.loads(diagnostic.AUDIT.read_text())
    assert audit["runs_sha256"] == diagnostic.sha(diagnostic.PLANS)
    assert len(audit["candidates"]) == 8 * 32
    assert all(x["samples"] == 384 for x in audit["candidates"])
