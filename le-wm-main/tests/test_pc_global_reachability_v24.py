import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v24 as compiler


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_global_alternative_is_rescored_and_reaches_shared_refinement(segments):
    nominals = (tuple((0., 0.) for _ in range(5)), tuple((.25, .5) for _ in range(5)))
    config = compiler.Config(action_dim=2, segment_length=5, segments=segments,
                             nominals=nominals, token_cap=1920, seed=23)
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        return actions.sum(axis=1), np.zeros(len(actions))

    result = compiler.compile_program(predict, [0., 0.], config, score_scale=1., score_target=0.)
    proposals = result["global_proposals"]
    assert len(proposals["warm_indices"]) == 1
    warm = np.asarray(proposals["actions"][proposals["warm_indices"][0]], np.float32)
    assert any(np.any(np.all(batch == warm, axis=(1, 2))) for batch in batches[1:]
               if batch.shape[1] == 5 * segments)
    assert any(start["source"] == "global_warm_start"
               for start in result["child"]["refinement_starts"])
    assert result["resource"]["prediction_tokens"] == sum(
        len(batch) * batch.shape[1] // 5 for batch in batches)
    assert result["resource"]["prediction_tokens"] <= config.token_cap


def test_rejects_more_global_seeds_than_refinement_starts():
    config = compiler.Config(action_dim=2, segment_length=5, segments=1,
                             nominals=(tuple((0., 0.) for _ in range(5)),),
                             global_seed_slots=5)
    with pytest.raises(ValueError):
        compiler.compile_program(None, [0., 0.], config, score_scale=1., score_target=0.)
