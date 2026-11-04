"""Analytic retrieval fixture against the pinned upstream observation bank."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory


UPSTREAM = (Path(__file__).resolve().parents[2] / "external" /
            "anchored-planning-984327424ca7" / "code" / "ap" /
            "anchored_planning" / "observation_bank.py")


def test_retrieval_matches_official_for_unique_nonboundary_queries():
    assert UPSTREAM.is_file(), "pinned official AP source is required for parity test"
    spec = importlib.util.spec_from_file_location("ap_upstream_bank", UPSTREAM)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = np.random.default_rng(812)
    sequences = [rng.normal(size=(13, 4)).astype(np.float32),
                 rng.normal(size=(16, 4)).astype(np.float32)]
    memory = ObservationMemory({"0": sequences[0], "1": sequences[1]}, ["query"])
    bank = upstream.ObservationBank(
        {"lengths": np.array([13, 16]), "offsets": np.array([0, 13])},
        np.concatenate(sequences), np.array([0, 1]), 5,
        np.ones(3, dtype=np.float32), torch.device("cpu"),
    )
    for horizon in (5, 7, 10):
        for current, goal in ((sequences[0][2], sequences[0][2 + horizon]),
                              (sequences[1][0], sequences[1][horizon])):
            found = memory.retrieve(current, goal, horizon, 0)
            index, distance = bank.nearest_topk(current[None], goal[None], horizon, 1)
            flattened = found.start + (0 if found.episode_id == "0" else 13)
            assert index[0, 0] == flattened
            np.testing.assert_array_equal(
                found.target, sequences[int(found.episode_id)][found.start + 5]
            )
            # The implementations use different dtypes and distance formulas.
            assert abs(found.squared_distance - float(distance[0, 0])) < 2e-4
