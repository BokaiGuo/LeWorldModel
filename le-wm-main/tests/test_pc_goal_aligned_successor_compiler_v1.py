import numpy as np

from experiments.spectral_reachability import pc_goal_aligned_successor_compiler_v1 as compiler
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def test_goal_alignment_rotates_valid_response_subspace():
    matrix = np.zeros((2, 10))
    matrix[0, 0], matrix[1, 1] = 2.0, 1.0

    def predict(actions):
        positions = np.asarray(actions).reshape(len(actions), 10) @ matrix.T
        return positions, np.linalg.norm(positions, axis=1)

    nominal = np.zeros((5, 2))
    args = lambda: (Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0}), np.random.default_rng(4))
    meter, rng = args()
    bank, result = compiler.directions("goal", np.array([1.0, 1.0]), predict, meter, nominal, rng)
    directions = np.asarray(result["proposal_directions"])
    expected = matrix.T @ np.array([1.0, 1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(directions[0], expected, atol=1e-12)
    np.testing.assert_allclose(directions @ directions.T, np.eye(2), atol=1e-12)
    assert len(bank) == 5 and meter.counts["prediction_samples"] == 21
    meter, rng = args()
    _, changed = compiler.directions("goal", np.array([1.0, -1.0]), predict, meter, nominal, rng)
    assert not np.allclose(changed["proposal_directions"][0], directions[0])
    meter, rng = args()
    _, svd = compiler.directions("svd", np.array([1.0, -1.0]), predict, meter, nominal, rng)
    np.testing.assert_allclose(svd["proposal_directions"], svd["right_directions"], atol=1e-12)


def test_rank_zero_and_one_do_not_invent_response_directions():
    for rank in (0, 1):
        matrix = np.zeros((2, 10))
        matrix[0, 0] = rank

        def predict(actions):
            positions = np.asarray(actions).reshape(len(actions), 10) @ matrix.T
            return positions, np.zeros(len(actions))

        meter = Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0})
        bank, evidence = compiler.directions("goal", np.array([1.0, 1.0]), predict, meter,
                                             np.zeros((5, 2)), np.random.default_rng(5))
        assert evidence["retained_rank"] == rank
        assert len(bank) == 1 + 2 * rank
