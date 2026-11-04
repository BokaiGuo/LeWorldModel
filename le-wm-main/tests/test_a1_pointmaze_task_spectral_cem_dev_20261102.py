import json

import numpy as np
import pytest

from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import (
    ARRAYS, OUTPUT, ROOTS, cem, image_hash,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_cem_is_bounded_and_reproducible():
    noise = np.random.default_rng(5).standard_normal((3, 16, 2, 2)).astype(np.float32)
    scoring = lambda x: ((x - .2) ** 2).sum(axis=(1, 2))
    first = cem(noise, scoring)
    second = cem(noise, scoring)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    for step in first[2]:
        assert np.max(np.abs(step["proposals"])) <= 1
        assert np.min(step["std"]) >= .1 - 1e-6
    with pytest.raises(ValueError):
        cem(np.full((3, 16, 2, 2), np.nan), scoring)


def test_saved_traces_replay_and_images_bind():
    if not OUTPUT.exists():
        pytest.skip("development run absent")
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert [r["root"] for r in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for row in report["rows"]:
            root = row["root"]
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}"
                assert image_hash(arrays[f"{key}_rgb5"]) == branch["rgb5_sha256"]
                assert image_hash(arrays[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                for kind, noise in (("initial", arrays[f"r{root}_initial_noise"]),
                                    ("tail", arrays[f"r{root}_tail_noise"])):
                    trace = branch[f"{kind}_trace"]
                    counter = iter(trace)
                    def frozen_scores(proposals):
                        saved = next(counter)
                        np.testing.assert_allclose(proposals, saved["proposals"], rtol=0, atol=1e-7)
                        return saved["scores"]
                    inject = row["injected_proposals"] if kind == "initial" and branch["method"] == "spectral_cem" else None
                    plan, cost, _ = cem(noise, frozen_scores, inject)
                    np.testing.assert_allclose(plan, branch[f"{kind}_plan"], rtol=0, atol=1e-7)
                    assert cost == pytest.approx(branch[f"{kind}_cost"], abs=1e-12)
                distance = np.linalg.norm(np.array(branch["endpoint_eval_only"][:2]) -
                                          np.array(row["goal_position_eval_only"]))
                assert distance == pytest.approx(branch["distance_eval_only"], abs=1e-12)
