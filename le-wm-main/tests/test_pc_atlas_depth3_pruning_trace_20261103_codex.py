import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.pc_atlas_depth3_pruning_trace_20261103_codex import (
    NAMES, ROOT, audit_case, make_actions, sha,
)
from experiments.spectral_reachability.pc_atlas_unicycle_smoke_20261103 import UnicycleBackend


SOURCE = Path(__file__).resolve().parents[1] / "results/pc_atlas_oracle_injection_20261103.json"


@pytest.fixture(scope="module")
def sample():
    return json.loads(SOURCE.read_text())["rows"][0]


@pytest.fixture(scope="module")
def cases(sample):
    return {variant: audit_case(sample, variant) for variant in ("early", "delayed")}


def test_actual_intermediate_beam_both_sides_of_cutoff(cases):
    early, delayed = cases["early"], cases["delayed"]
    assert early["intermediate_rank"] == 1
    assert early["intermediate_retained_in_actual_beam"]
    assert delayed["intermediate_rank"] > 2
    assert not delayed["intermediate_retained_in_actual_beam"]
    for case in cases.values():
        assert [trace["depth"] for trace in case["trace"]] == [2, 3]
        for trace in case["trace"]:
            retained = [n for n in trace["expanded"] if n["retained"]]
            assert len(retained) == len(trace["beam"]) == 2
            assert sorted(n["stable_rank"] for n in retained) == [1, 2]


def test_actual_named_continuation_and_duplicate_identity(cases):
    early, delayed = cases["early"], cases["delayed"]
    assert early["exact_named_terminal_offered"]
    assert not delayed["exact_named_terminal_offered"]
    assert delayed["context_injection_visits"][2] == 0
    terminal = early["trace"][1]["expanded"]
    injected = next(n for n in terminal if n["names"] == list(NAMES))
    duplicates = [n for n in terminal if n["context_sha256"] == injected["context_sha256"]]
    assert len(duplicates) >= 2
    assert injected["stable_rank"] == 3 and not injected["retained"]
    assert any(n["retained"] and n["names"] != injected["names"] for n in duplicates)


def test_constructed_witness_split_binding_and_historical_prefix(sample, cases):
    backend = UnicycleBackend()
    states = []
    for variant in ("early", "delayed"):
        case = cases[variant]
        actions = make_actions(sample, variant)
        first = backend.rollout(ROOT, actions[:1])
        second = backend.rollout(first, actions[1:2])
        third = backend.rollout(second, actions[2:])
        direct = backend.rollout(ROOT, actions)
        assert digest(direct) == digest(third) == case["witness_output_sha256"]
        assert digest(second) == case["witness_input_context_sha256"][2]
        assert case["external_witness_rollouts"] == 5
        assert np.all(actions >= [-1., -np.pi / 2])
        assert np.all(actions <= [1., np.pi / 2])
        if variant == "early":
            assert digest(second) == sample["direct_sha256"]
        states.append(direct["latent_history"][-1])
    np.testing.assert_allclose(states[0], states[1], rtol=0, atol=1e-15)


def test_deterministic_replay(sample, cases):
    assert audit_case(sample, "delayed") == cases["delayed"]


def test_rejects_invalid_witness(sample):
    bad = {**sample, "gradient_actions": [[0., 0.], [1., 0.]]}
    with pytest.raises(ValueError, match="bounds"):
        make_actions(bad, "delayed")
    with pytest.raises(ValueError, match="unknown variant"):
        make_actions(sample, "invalid")


def test_saved_full_experiment_is_source_bound_and_records_all_cases():
    path = SOURCE.with_name("pc_atlas_depth3_pruning_trace_20261103_codex.json")
    report = json.loads(path.read_text())
    assert report["input_sha256"] == sha(SOURCE)
    assert [(c["index"], c["variant"]) for c in report["cases"]] == [
        (i, variant) for i in range(8) for variant in ("early", "delayed")]
    for case in report["cases"]:
        assert case["intermediate_retained_in_actual_beam"] == (case["variant"] == "early")
        assert case["exact_named_terminal_offered"] == (case["variant"] == "early")
        assert case["compiler_counts"]["rollout_attempted"] == case["compiler_counts"]["rollout_returned"]
        assert case["compiler_counts"]["jacobian_attempted"] == case["compiler_counts"]["jacobian_returned"]
