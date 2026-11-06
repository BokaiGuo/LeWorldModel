import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_recursive_response_v17 as v17


def test_score_augmentation_adds_response_without_extra_prediction_calls():
    traces = []

    def predict(actions):
        traces.append(actions.copy())
        return np.zeros((len(actions), 1)), ((actions - .4) ** 2).mean(axis=(1, 2))

    config = v13.Config(action_dim=1, segment_length=5, segments=1,
        nominals=(tuple((.2,) for _ in range(5)),), token_cap=200, seed=8, global_samples=4)
    result = v17.compile_program(predict, [0.], config, score_scale=1., score_target=0.)
    response = np.asarray(result["child"]["layers"][0]["responses"][0]["matrix"])
    assert response.shape == (2, 5)
    assert np.linalg.norm(response[0]) == 0
    assert np.linalg.norm(response[1]) > 0
    assert result["resource"]["prediction_tokens"] == sum(len(batch) for batch in traces)
    selected = np.asarray(result["selected"])
    assert np.isclose(result["selected_score"], ((selected - .4) ** 2).mean())


def test_invalid_score_scale_fails_before_predictor_call():
    config = v13.Config(action_dim=1, segment_length=5, segments=1,
        nominals=(tuple((0.,) for _ in range(5)),))
    with pytest.raises(ValueError, match="geometry"):
        v17.compile_program(lambda _: pytest.fail("unexpected prediction"), [0.], config,
                            score_scale=0., score_target=0.)
