import pytest

from experiments.pc_rgb_gradient_endpoint_decomposition_20261107_codex import pairwise


def test_pairwise_agreement_disagreement_and_ties():
    assert pairwise([0, 1, 2], [0, 2, 1]) == {"agree": 2, "disagree": 1, "ties": 0}
    assert pairwise([0, 0, 2], [0, 1, 2]) == {"agree": 2, "disagree": 0, "ties": 1}


@pytest.mark.parametrize("a,b", [([1], [1]), ([1, 2], [1]), ([1, float("nan")], [1, 2])])
def test_pairwise_rejects_invalid(a, b):
    with pytest.raises(ValueError):
        pairwise(a, b)
