import numpy as np

from experiments.evaluate_cube_spatial_adapter_transfer import _fit_local_probe_map, _pairwise_agreement, _score_rows


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
