import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v11 as v11
from experiments.spectral_reachability import pc_recursive_response_v12 as v12


@pytest.mark.parametrize("horizon", [5, 10, 15])
@pytest.mark.parametrize("extra", [0, 1, 7, 11])
def test_remainder_is_used_and_existing_batches_are_identical(horizon, extra):
    nominal = (tuple((0.,) for _ in range(5)),)
    base = dict(action_dim=1, segment_length=5, segments=horizon // 5,
                nominals=nominal, token_cap=1200 + extra * (horizon // 5),
                seed=42)
    observed = []
    for module in (v11, v12):
        batches = []

        def predict(actions):
            batches.append(actions.copy())
            features = actions.sum(axis=1)
            return features, ((features - .7) ** 2).sum(axis=1)

        result = module.compile_program(predict, [.7], module.Config(**base))
        observed.append((result, batches))
    old, old_batches = observed[0]
    new, new_batches = observed[1]
    for old_batch, new_batch in zip(old_batches[:-1], new_batches, strict=False):
        np.testing.assert_array_equal(old_batch, new_batch)
    packets = horizon // 5
    available = (base["token_cap"] - old["resource"]["prediction_tokens"]) // packets
    assert 0 <= available < 12
    assert new["resource"]["prediction_tokens"] == (
        old["resource"]["prediction_tokens"] + available * packets)
    assert new["resource"]["prediction_tokens"] <= base["token_cap"]
    assert new["selected_score"] <= old["selected_score"] + 1e-12
    assert sum(len(b) * b.shape[1] // 5 for b in new_batches) == new["resource"]["prediction_tokens"]
    if available:
        assert new_batches[-2].shape == (available, horizon, 1)
        assert new["refinement"][-1]["batch_size"] == available
    else:
        assert len(old_batches) == len(new_batches)


def test_single_remainder_does_not_duplicate_an_incumbent():
    results = []
    for cap in range(100, 200):
        seen = []

        def predict(actions):
            seen.append(actions.copy())
            feature = actions.sum(axis=1)
            return feature, ((feature - .37) ** 2).sum(axis=1)

        config = v12.Config(action_dim=1, segment_length=5, segments=1,
            nominals=(tuple((0.,) for _ in range(5)),), token_cap=cap, seed=7)
        result = v12.compile_program(predict, [.37], config)
        if result["refinement"] and result["refinement"][-1]["batch_size"] == 1:
            results.append((result, seen))
            break
    assert results
    _, batches = results[0]
    candidate = batches[-2][0]
    earlier = np.concatenate(batches[:-2], axis=0)
    assert not np.any(np.all(earlier == candidate, axis=(1, 2)))
