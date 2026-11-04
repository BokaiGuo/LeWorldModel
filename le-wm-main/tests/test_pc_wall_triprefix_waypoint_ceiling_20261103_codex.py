import json

import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_triprefix_waypoint_ceiling_20261103_codex import (
    SOURCE, build, run,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import (
    HIGH, LOW, replay,
)


def test_waypoint_builder_bounds_and_validation():
    for point in ([0., 0.], [.5, 1.5], [-.5, -1.5]):
        tail = build([-.86, 0., 0.], point)
        assert tail.shape == (12, 2)
        assert np.all(tail >= LOW) and np.all(tail <= HIGH)
    with pytest.raises(ValueError):
        build([0., 0., np.nan], [0., 0.])


def test_paired_samples_and_replay():
    prefixes = json.loads(SOURCE.read_text())["prefixes"]
    selected = run(prefixes["selected"], 20261103)
    zero = run(prefixes["zero"], 20261103)
    assert selected["waypoints"] == zero["waypoints"]
    assert selected["backend"]["transitions"] == 3072
    assert selected["backend"]["rollouts"] == 256
    assert selected["prefix_backend"]["transitions"] == 4
    assert selected["selected_index"] == int(np.argmin(selected["scores"]))
    np.testing.assert_array_equal(np.asarray(selected["replay"]["actions"])[:4],
                                  prefixes["selected"])
    assert replay(selected["replay"]["actions"]) == selected["replay"]
    assert selected["replay"]["terminal_error"] == pytest.approx(
        selected["scores"][selected["selected_index"]]
    )
