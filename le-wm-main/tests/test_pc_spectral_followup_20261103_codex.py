import json
from pathlib import Path

import numpy as np

from experiments import pc_spectral_followup_20261103_codex as run
from experiments import pc_spectral_followup_analysis_20261103_codex as analysis
from experiments import pc_spectral_followup_physical_20261103_codex as physical


def test_evidence_roster_and_execution_binding():
    sha = run.old.prior.sha
    source = json.loads(run.base.OUTPUT.read_text())
    old = json.loads(run.old.DECISIONS.read_text())
    plans = json.loads(run.DECISIONS.read_text())
    receipt = json.loads(run.RECEIPT.read_text())
    assert plans["runner_sha256"] == sha(Path(run.__file__))
    assert plans["source_sha256"] == sha(run.base.OUTPUT)
    assert plans["protocol_sha256"] == sha(run.PROTOCOL)
    assert plans["context_sha256"] == sha(run.CONTEXT)
    assert receipt["decision_sha256"] == sha(run.DECISIONS)
    assert receipt["arrays_sha256"] == sha(run.IMAGES)
    assert len(plans["rows"]) == 20
    assert len(receipt["rows"]) == 40
    mean, std = np.asarray(old["action_mean"]), np.asarray(old["action_std"])
    with np.load(run.CONTEXT, allow_pickle=False) as contexts, np.load(run.IMAGES, allow_pickle=False) as images:
        for row in plans["rows"]:
            key = f"root{row['root']}_{row['role']}"
            program = dict(run.roster(source["rows"][row["root"]]))[row["role"]]
            np.testing.assert_allclose(row["prefix"], program["prefix"])
            np.testing.assert_allclose(row["old_suffix"], program["suffix"])
            np.testing.assert_allclose(row["history_actions"], np.concatenate((run.old.WARMUP[10:15], row["prefix"])))
            np.testing.assert_allclose(np.asarray(row["history_normalized"]).reshape(10, 2)*std+mean,
                                       row["history_actions"], atol=1e-6)
            assert run.old.digest_image(contexts[key][20]) == row["cp5_rgb_sha256"]
            assert row["counts"] == {"batches": 8, "samples": 100}
            assert row["best_index"] == int(np.argmin([x["score"] for x in row["evaluations"]]))
            for kind in ("original", "replanned"):
                actual = next(x for x in receipt["rows"] if x["root"] == row["root"] and x["role"] == row["role"] and x["kind"] == kind)
                np.testing.assert_allclose(actual["prefix"], row["prefix"])
                np.testing.assert_allclose(actual["suffix"], row["old_suffix"] if kind == "original" else row["suffix"])
                assert [run.old.digest_image(x) for x in images[key+"_"+kind]] == actual["rgb_sha256"]


def test_physics_replay_and_candidate_census():
    sha = run.old.prior.sha
    census = json.loads(analysis.OUTPUT.read_text())
    output = json.loads(physical.OUTPUT.read_text())
    assert census["source_sha256"] == sha(run.base.OUTPUT)
    assert len(census["rows"]) == 60
    assert output["decision_sha256"] == sha(run.DECISIONS)
    assert output["execution_sha256"] == sha(run.RECEIPT)
    assert len(output["rows"]) == 160
    for row in output["rows"]:
        assert len(row["trace"]) == 31
        for tick, name in ((15, "initial_distance"), (20, "prefix_distance"), (30, "terminal_distance")):
            np.testing.assert_allclose(np.linalg.norm(np.asarray(row["trace"][tick]["state"][:2])-[1.8, 2.8]), row[name], atol=1e-12)
