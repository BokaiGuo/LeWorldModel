from types import SimpleNamespace

import numpy as np
import pytest

from experiments import pc_reset_render_observation_dev_20261106_codex as module


@pytest.mark.parametrize("mode", ["dataset", "render"])
def test_only_pixels_change(monkeypatch, mode):
    actor = object()
    env = SimpleNamespace(unwrapped=actor)
    world = SimpleNamespace(num_envs=1, envs=SimpleNamespace(envs=[env]),
                            infos={"pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
                                   "goal": np.ones((1, 1, 2, 2, 3), dtype=np.uint8),
                                   "state": np.array([[7.]])})
    wrapper = SimpleNamespace(_get_pixels=lambda: ({"pixels": np.full((2, 2, 3), 12, dtype=np.uint8)}, 0))
    monkeypatch.setattr(module, "pixels_wrapper", lambda _: wrapper)
    monkeypatch.setattr(module, "snapshot", lambda _: np.array([4.]))
    records = module.replace_initial(world, mode)
    assert world.infos["pixels"].max() == (12 if mode == "render" else 0)
    assert world.infos["goal"].min() == 1
    assert world.infos["state"][0, 0] == 7.
    assert records[0]["pixel_max_abs"] == 12


def test_render_changing_physical_state_is_rejected(monkeypatch):
    env = SimpleNamespace(unwrapped=object())
    world = SimpleNamespace(num_envs=1, envs=SimpleNamespace(envs=[env]),
                            infos={"pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
                                   "goal": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)})
    wrapper = SimpleNamespace(_get_pixels=lambda: ({"pixels": np.zeros((2, 2, 3), dtype=np.uint8)}, 0))
    values = iter([np.array([1.]), np.array([2.])])
    monkeypatch.setattr(module, "pixels_wrapper", lambda _: wrapper)
    monkeypatch.setattr(module, "snapshot", lambda _: next(values))
    with pytest.raises(ValueError, match="render changed"):
        module.replace_initial(world, "render")
