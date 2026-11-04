import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_fixed_prefix_diag_20261103_codex as diag


def test_sealed_prefixes_and_checkpoints():
    if not diag.PLANS.exists() or not diag.AUDIT.exists():
        pytest.skip("diagnostic artifacts unavailable")
    data = json.loads(diag.PLANS.read_text())
    audit = json.loads(diag.AUDIT.read_text())
    original = json.loads(diag.prior.PLANS.read_text())
    assert data["dependencies"] == diag.dependencies()
    assert audit["plans_sha256"] == diag.prior.sha(diag.PLANS)
    assert len(data["rows"]) == len(original["rows"]) == 24
    assert len(audit["rows"]) == 144
    for row, source in zip(data["rows"], original["rows"], strict=True):
        assert row["root"] == source["root"]
        assert row["task"] == source["task"]
        witness, item = diag.witness.witness(row["task"])
        np.testing.assert_array_equal(row["witness"], witness)
        assert row["witness_steps"] == item["steps_needed"]
        bank = source["result"]["bank"]
        assert len(bank) == 4 and len(row["runs"]) == 6
        for run, candidate in zip(row["runs"][:4], bank, strict=True):
            assert run["name"] == candidate["name"]
            np.testing.assert_array_equal(run["prefix"], candidate["prefix"])
        for run, length in zip(row["runs"][4:], (5, 10), strict=True):
            assert run["name"] == f"witness_first{length}"
            np.testing.assert_array_equal(run["prefix"], witness[:length])
        for run in row["runs"]:
            actions = np.concatenate([np.asarray(b["actions"]) for b in run["trace"]])
            scores = np.concatenate([np.asarray(b["scores"]) for b in run["trace"]])
            assert len(actions) == len(scores) == 384
            for count in (192, 384):
                index = int(np.argmin(scores[:count]))
                checkpoint = run["checkpoints"][str(count)]
                assert checkpoint["score"] == scores[index]
                np.testing.assert_array_equal(checkpoint["chain"], actions[index])
