import numpy as np
import pytest

from experiments import pc_tworoom_wide_offset_failure_decomposition_20261107_codex as audit
from experiments import pc_tworoom_packet_margin_sensitivity_20261107_codex as sensitivity


def test_categories_for_each_possible_pair():
    physical = [0., 1.]
    assert audit.pair_categories([0., 1.], [0., 1.], physical)[0].tolist() == [1, 0, 0, 0]
    assert audit.pair_categories([1., 0.], [0., 1.], physical)[0].tolist() == [0, 1, 0, 0]
    assert audit.pair_categories([0., 1.], [1., 0.], physical)[0].tolist() == [0, 0, 1, 0]
    assert audit.pair_categories([1., 0.], [1., 0.], physical)[0].tolist() == [0, 0, 0, 1]


def test_ties_and_invalid_inputs():
    counts, ties = audit.pair_categories([0., 0.], [0., 1.], [0., 1.])
    assert counts.sum() == 0 and ties == 1
    with pytest.raises(ValueError):
        audit.pair_categories([0., float("nan")], [0., 1.], [0., 1.])
    with pytest.raises(ValueError):
        audit.pair_categories([0.], [0., 1.], [0., 1.])


def test_root_bootstrap_is_reproducible():
    assert audit.summarize([0., .5, 1.]) == audit.summarize([0., .5, 1.])


def test_physical_margin_excludes_close_pairs():
    counts, excluded = sensitivity.categories([0., 1., 2.], [0., 1., 2.], [0., .5, 3.], 1.)
    assert counts.tolist() == [2, 0, 0, 0]
    assert excluded == 1
