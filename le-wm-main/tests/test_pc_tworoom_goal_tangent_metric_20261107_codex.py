import numpy as np
import pytest

from experiments.pc_tworoom_goal_tangent_metric_20261107_codex import make_goal_probes, tangent_basis


def test_red_disk_is_translated_on_uniform_background():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    y, x = np.indices((224, 224))
    image[(x - 112) ** 2 + (y - 112) ** 2 <= 10 ** 2] = [255, 0, 0]
    probes, info = make_goal_probes(image)
    assert info["valid"] and probes.shape == (4, 224, 224, 3)
    for probe, expected in zip(probes, [(116, 112), (108, 112), (112, 116), (112, 108)], strict=True):
        mask = probe[:, :, 0].astype(int) - probe[:, :, 1].astype(int) > 100
        assert (round(x[mask].mean()), round(y[mask].mean())) == expected


def test_goal_probe_abstains_on_missing_marker_and_tangent_on_rank_deficiency():
    probes, info = make_goal_probes(np.full((224, 224, 3), 255, dtype=np.uint8))
    assert probes is None and info["reason"] == "red_support"
    basis, _ = tangent_basis(np.zeros((4, 16)))
    assert basis is None
    with pytest.raises(ValueError):
        tangent_basis(np.zeros((3, 16)))
