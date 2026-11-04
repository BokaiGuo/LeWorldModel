import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_atlas_bound_normalized_20261103_codex import run_arm
from experiments.spectral_reachability.pc_wall_atlas_root_source_20261103_codex import run_arm as old_run_arm
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import HIGH, replay


def test_bound_scale_and_witness_replay():
    row = run_arm(20261103, "spectral")
    plus = np.asarray(row["root_bank"][3]["actions"])
    minus = np.asarray(row["root_bank"][2]["actions"])
    np.testing.assert_array_equal(plus, -minus)
    assert np.max(np.abs(plus) / HIGH) == pytest.approx(.3)
    for candidate, program in zip(row["root_bank"], row["programs"]):
        np.testing.assert_array_equal(np.asarray(program["actions"])[:4], candidate["actions"])
        result = replay(program["actions"])
        np.testing.assert_allclose(
            result["trajectory"][-1], program["output_context"]["latent_history"][-1],
            atol=1e-12, rtol=0,
        )
        assert program["terminal_cost"] == pytest.approx(result["terminal_error"] ** 2 / 3)


@pytest.mark.parametrize("arm", ["random", "ordinary"])
def test_controls_unchanged(arm):
    new, old = run_arm(20261103, arm), old_run_arm(20261103, arm)
    for key in ("root_bank", "programs", "compiler_counts", "backend", "selected_index"):
        assert new[key] == old[key]
