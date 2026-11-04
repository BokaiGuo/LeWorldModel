import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_atlas_root_source_20261103_codex import (
    ARMS, RootSourceAtlas, run_arm,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import (
    GOAL, HIGH, LOW, ROOT, Backend, replay,
)
from experiments.spectral_reachability.d2p_atlas_search_v2 import AtlasConfig
from experiments.spectral_reachability.d2p_local_compiler import CompilerConfig, LocalCompiler


def test_replay_all_programs_and_matched_costs():
    group = [run_arm(20261103, arm) for arm in ARMS]
    for row in group:
        assert row["backend"] == group[0]["backend"]
        assert row["compiler_counts"] == group[0]["compiler_counts"]
        assert row["root_bank"][:2] == group[0]["root_bank"][:2]
        for program, candidate in zip(row["programs"], row["root_bank"]):
            actions = np.asarray(program["actions"])
            assert actions.shape == (16, 2)
            assert np.all(actions >= LOW) and np.all(actions <= HIGH)
            np.testing.assert_array_equal(actions[:4], candidate["actions"])
            checked = replay(actions)
            np.testing.assert_allclose(
                checked["trajectory"][-1], program["output_context"]["latent_history"][-1],
                rtol=0, atol=1e-12,
            )
            assert program["terminal_cost"] == pytest.approx(
                checked["terminal_error"] ** 2 / 3, abs=1e-12
            )
        costs = [program["terminal_cost"] for program in row["programs"]]
        assert row["selected_index"] == min(range(len(costs)), key=lambda i: (costs[i], i))


def test_spectral_pair_and_invalid_source():
    row = run_arm(20261103, "spectral")
    minus = np.asarray(row["root_bank"][2]["actions"])
    plus = np.asarray(row["root_bank"][3]["actions"])
    np.testing.assert_array_equal(minus, -plus)
    assert np.max(np.abs(plus)) == pytest.approx(.3)
    compiler = LocalCompiler(
        Backend(), goal=GOAL, sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[.3, np.pi / 2], low=LOW, high=HIGH,
        config=CompilerConfig(packet_steps=4),
    )
    with pytest.raises(ValueError, match="unknown source"):
        RootSourceAtlas(compiler, AtlasConfig(), "invalid")
    assert ROOT["latent_history"][-1] == [-.5, 0., 0.]
