import json
from pathlib import Path

import numpy as np

from experiments import pc_spectral_bank_rgb_dev_20261103_codex as run
from experiments import pc_spectral_bank_rgb_physical_20261103_codex as physical
from experiments import pc_spectral_bank_cp5_20261103_codex as cp5
from experiments import pc_spectral_bank_cp5_physical_20261103_codex as cp5physical


def test_open_loop_binding_and_budget():
    sha = run.old.prior.sha
    decision = json.loads(run.OUTPUT.read_text())
    execution = json.loads(run.EXECUTION.read_text())
    assert decision["runner_sha256"] == sha(Path(run.__file__))
    assert decision["core_sha256"] == sha(Path(run.core.__file__))
    assert decision["protocol_sha256"] == sha(run.PROTOCOL)
    assert execution["decision_sha256"] == sha(run.OUTPUT)
    assert execution["arrays_sha256"] == sha(run.IMAGES)
    assert not decision["full_cost_certified"]
    with np.load(run.IMAGES, allow_pickle=False) as images:
        for row in decision["rows"]:
            programs = row["plan"]["programs"]
            assert row["arms"]["selected"]["name"] == min(programs, key=lambda x: x["terminal_score"])["name"]
            assert row["arms"]["greedy"]["name"] == min(row["plan"]["prefix_scores"], key=lambda x: x["prefix_score"])["name"]
            assert row["arms"]["no_spectral"]["source"] != "rgb_model"
            assert row["counts"]["prediction_samples"] == 20 + 101*len(programs) + 1520
            for role, program in row["arms"].items():
                actual = next(x for x in execution["rows"] if x["root"] == row["root"] and x["role"] == role)
                np.testing.assert_allclose(actual["actions"], program["prefix"] + program["suffix"])
                assert [run.old.digest_image(x) for x in images[f"root{row['root']}_{role}"]] == actual["rgb_sha256"]


def test_cp5_and_physical_bindings():
    sha = run.old.prior.sha
    plans = json.loads(cp5.DECISION.read_text())
    execution = json.loads(cp5.EXECUTION.read_text())
    physical_rows = json.loads(physical.OUTPUT.read_text())["rows"]
    cp5_result = json.loads(cp5physical.OUTPUT.read_text())
    assert plans["source_sha256"] == sha(run.OUTPUT)
    assert plans["context_sha256"] == sha(cp5.CONTEXT)
    assert execution["decision_sha256"] == sha(cp5.DECISION)
    assert execution["arrays_sha256"] == sha(cp5.IMAGES)
    assert cp5_result["decision_sha256"] == sha(cp5.DECISION)
    with np.load(cp5.CONTEXT, allow_pickle=False) as contexts, np.load(cp5.IMAGES, allow_pickle=False) as images:
        for row in plans["rows"]:
            root, role = row["root"], row["role"]
            key = f"root{root}_{role}"
            assert run.old.digest_image(contexts[key][20]) == row["cp5_rgb_sha256"]
            np.testing.assert_allclose(row["history_actions"], np.concatenate((run.old.WARMUP[10:15], row["prefix"])))
            assert row["prediction_samples"] == 100
            assert row["best_index"] == int(np.argmin([x["score"] for x in row["evaluations"]]))
            actual = next(x for x in execution["rows"] if x["root"] == root and x["role"] == role)
            np.testing.assert_allclose(actual["prefix"], row["prefix"])
            np.testing.assert_allclose(actual["suffix"], row["suffix"])
            assert [run.old.digest_image(x) for x in images[key]] == actual["rgb_sha256"]
            shared = next(x for x in physical_rows if x["root"] == root and x["kind"] == "common"
                          and x["role"] == role and x["suffix_index"] == 0)
            cp = next(x for x in cp5_result["rows"] if x["root"] == root and x["role"] == role)
            np.testing.assert_allclose(shared["trace"][20]["state"], cp["trace"][20]["state"], atol=1e-12)
