import json

import numpy as np
import pytest

from experiments.a0_pointmaze_physical_cem_mpc_20261103 import (
    ARRAYS, OUTPUT, SOURCE, SOURCE_SHA, image_hash, sequences, sha256_file,
)
from experiments.a0_pointmaze_physical_cem_warm_tail_20261103 import (
    ARRAYS as WARM_ARRAYS, OUTPUT as WARM_OUTPUT, warm_cem,
)
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical


def test_packet_expansion_and_bounds():
    expected = np.array([[[1., -1.]] * 5 + [[0.5, 0.2]] * 5], dtype=np.float32)
    np.testing.assert_array_equal(sequences([[[1, -1], [0.5, 0.2]]]), expected)
    with pytest.raises(ValueError):
        sequences([[[1.1, 0]]])


def test_retained_candidate_cannot_be_lost():
    noise = np.ones((3, 16, 1, 2), np.float32)
    retained = np.zeros((1, 2), np.float32)
    chosen, cost, trace = warm_cem(noise, retained, lambda p: np.sum(p ** 2, axis=(1, 2)))
    np.testing.assert_array_equal(chosen, retained)
    assert cost == trace[0]["scores"][0] == 0


def test_sealed_physical_cem_outputs_and_rescoring():
    if not OUTPUT.exists() or not WARM_OUTPUT.exists():
        pytest.skip("physical execution incomplete")
    report, warm = json.loads(OUTPUT.read_text()), json.loads(WARM_OUTPUT.read_text())
    original = json.loads(SOURCE.read_text())
    assert report["source_sha256"] == SOURCE_SHA == sha256_file(SOURCE)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert warm["arrays_sha256"] == sha256_file(WARM_ARRAYS)
    assert warm["source_sha256"] == sha256_file(OUTPUT)
    assert len(report["rows"]) == len(warm["rows"]) == len(original["rows"]) == 4
    with np.load(ARRAYS, allow_pickle=False) as arrays, np.load(WARM_ARRAYS, allow_pickle=False) as warm_arrays:
        for row, wrow, old in zip(report["rows"], warm["rows"], original["rows"]):
            root = row["root"]
            assert root == wrow["root"] == old["root"]
            assert row["candidate_cost"] == {"candidate_rollouts": 96, "candidate_high_level_steps": 720}
            assert wrow["candidate_cost"] == {"candidate_rollouts": 48, "candidate_high_level_steps": 240}
            assert image_hash(arrays[f"r{root}_rgb10"]) == row["rgb10_sha256"]
            assert image_hash(warm_arrays[f"r{root}_rgb10"]) == wrow["rgb10_sha256"]
            np.testing.assert_array_equal(
                np.asarray(wrow["trace"][0]["proposals"][0], np.float32),
                np.asarray(row["initial_plan"][1:2], np.float32))
            initial, cp5, goal = map(np.asarray, (row["initial_state_privileged"],
                                                  row["cp5_state_privileged"],
                                                  row["goal_position_privileged"]))
            for trace, state in ((row["initial_trace"], initial), (row["tail_trace"], cp5),
                                 (wrow["trace"], cp5)):
                for generation in trace:
                    seq = sequences(generation["proposals"])
                    endpoints = physical.replay_position_endpoints(np.repeat(state[None], 16, axis=0), seq)
                    costs = np.linalg.norm(endpoints[:, :2] - goal, axis=1)
                    np.testing.assert_allclose(costs, generation["scores"], rtol=0, atol=1e-12)
            assert wrow["chosen_cost"] <= wrow["retained_cost"] + 1e-12
