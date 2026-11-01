"""Non-neural translated-root reset and binding integration."""
from copy import deepcopy
import hashlib

import gymnasium as gym
import numpy as np
import pytest
import stable_worldmodel  # noqa: F401

from experiments.spectral_reachability import p1_centered_h4_parfit_20261101_launcher as launch


def test_true_reset_warmup_and_rebind(monkeypatch):
    exp = launch.setup("structured")
    runner, moving = exp.runner, exp.runner.rgb.moving
    prepared = runner.build()
    data = runner.p3.source_data(prepared["source"])
    report, target, _, images, _ = launch.source_inputs()
    bad = deepcopy(data)
    bad["references"][0]["raw_history"][0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="warmup action identity"):
        launch.rebind(deepcopy(prepared), bad, runner, report, target, images)
    ref = data["references"][0]
    previous_dynamics = deepcopy(ref["dynamics"])
    binding = launch.rebind(prepared, data, runner, report, target, images)
    assert ref["dynamics"] is None
    assert np.array_equal(ref["history"], images)
    assert binding["normalized_action_history_sha256"]
    assert prepared["source"]["tasks"][0]["root_context"]["latent_history"] == target["history_latents"]

    class Meter:
        region = None

        def call(self, _kind, fn):
            return fn()

    class Store:
        def __init__(self):
            self.saved = {}

        def emit(self, *args, **kwargs):
            pass

        def put(self, name, value):
            self.saved[name] = value

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
    row = dict(episode_id="fixture", warmup_returned=0)
    store = Store()

    def capture(env, obs, row, store, phase, step):
        image = np.asarray(env.render()).copy()
        return image, dict(pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                           snapshot=moving.old.kernel.moving.snapshot(env.unwrapped))

    monkeypatch.setattr(moving.old, "capture", capture)
    try:
        assert moving.setup(env, ref, data["plan"], row, Meter(), store)
    finally:
        env.close()
    assert row["warmup_returned"] == 10
    assert row["initial_exact"]
    assert len(ref["dynamics"]) == 3
    assert ref["dynamics"] != previous_dynamics
    assert store.saved["fixture_translated_history_binding.json"]["pixel_sha256"] == target["history_pixel_sha256"]
