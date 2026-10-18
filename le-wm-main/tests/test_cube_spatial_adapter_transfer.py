import numpy as np

from experiments.evaluate_cube_spatial_adapter_transfer import (
    _fit_local_probe_map,
    _pairwise_agreement,
    _probe_and_held_indices,
    _score_rows,
)


def test_score_projects_effect_onto_candidate_action():
    actions = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    effects = np.asarray([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(_score_rows(actions, effects), [2.0, 2.0])


def test_pairwise_agreement_is_one_for_same_ordering():
    assert _pairwise_agreement(np.asarray([1.0, 2.0, 3.0]), np.asarray([0.1, 0.2, 0.3])) == 1.0


def test_local_probe_map_requires_two_action_directions():
    mapping, report = _fit_local_probe_map(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[2.0, 0.0], [0.0, 3.0]]),
    )
    np.testing.assert_allclose(mapping, [[2.0, 0.0], [0.0, 3.0]])
    assert report["status"] == "ok"


def test_eight_action_protocol_leaves_unprobed_candidates_for_ranking():
    bank = np.asarray(
        [[np.cos(i * np.pi / 4), np.sin(i * np.pi / 4)] for i in range(8)]
    )
    probe, held = _probe_and_held_indices(bank)
    np.testing.assert_array_equal(probe, [0, 2, 4])
    np.testing.assert_array_equal(held, [1, 3, 5, 6, 7])
    assert not set(probe.tolist()).intersection(held.tolist())
