import numpy as np

from experiments import pc_conditional_rgb_geometry_baseline_20261107_codex as target


def test_generator_abstains_when_image_invalid():
    chain, error = target.generate(np.zeros((10, 10, 3), dtype=np.uint8),
                                   np.zeros((10, 10, 3), dtype=np.uint8))
    assert chain is None
    assert error


def test_pairwise_counts_include_abstention():
    rows = [
        {"success": True, "comparators": {"other": False}},
        {"success": False, "abstained": True, "comparators": {"other": True}},
        {"success": False, "comparators": {"other": False}},
    ]
    assert target.compare(rows, "other") == {"wins": 1, "losses": 1, "ties": 1}
