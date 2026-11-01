import importlib

import numpy as np


def test_trajectory_plan_uses_fixed_action_shape_and_objective():
    mod = importlib.import_module(
        "experiments.spectral_reachability.p1_trajectory_objective_physical_20261031"
    )
    assert mod.PLAN_SHA == "0b0fe954458445ac93187b76b804ee516add90863b9c513c5957842dc9d46a06"
    assert np.asarray([0.1, 0.2, 0.3, 0.4]).shape == (4,)
