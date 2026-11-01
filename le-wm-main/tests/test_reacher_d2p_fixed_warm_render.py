import copy

import numpy as np
import pytest

from experiments.spectral_reachability.reacher_d2p_fixed_warm_render import FixedWarmRenderer, POLICY, REQUIRED, valid_receipts


class Env:
    def __init__(self, mutate=None):
        self.unwrapped = self
        self.calls = 0
        self.state = {k: [0.] for k in REQUIRED}
        self.mutate = mutate
    def render(self):
        self.calls += 1
        if self.calls == self.mutate: self.state["qpos"] = [1.]
        return np.full((2,2,3), self.calls, np.uint8)


@pytest.mark.parametrize("seeds", [[], [3,3], [True], [3.0]])
def test_empty_or_duplicate_or_noninteger_roots_rejected(seeds):
    assert not valid_receipts([], seeds)


def test_exactly_two_fresh_draws_always_returns_second_without_cache_or_rounding():
    env = Env()
    r = FixedWarmRenderer(env, capture=lambda e: copy.deepcopy(e.state), touch_gl=lambda: True)
    assert np.all(env.render() == 2)
    assert np.all(env.render() == 4)
    assert env.calls == 4 and r.stats["requests"] == 2
    assert r.stats["warmup_retained_differences"] == 2
    assert valid_receipts([{"seed": 3, "stats": r.stats}], [3])
    r.restore()
    assert np.all(env.render() == 5)


@pytest.mark.parametrize("mutate", [1, 2])
def test_any_draw_changing_physics_is_rejected(mutate):
    env = Env(mutate)
    r = FixedWarmRenderer(env, capture=lambda e: copy.deepcopy(e.state), touch_gl=lambda: True)
    with pytest.raises(RuntimeError): env.render()
    r.restore()


def test_partial_state_or_unrelated_renderer_replacement_cannot_pass():
    env = Env()
    r = FixedWarmRenderer(env, capture=lambda e: {"qpos": [0.]}, touch_gl=lambda: True)
    with pytest.raises(ValueError): env.render()
    env.render = lambda: None
    with pytest.raises(RuntimeError): r.restore()


@pytest.mark.parametrize("bad", ["seed", "boolean", "passes", "state_checks", "policy", "dither"])
def test_complete_ordered_positive_receipts_required(bad):
    s = {"policy": POLICY, "requests": 2, "raw_render_passes": 4, "exact_state_checks": 4,
        "warmup_retained_differences": 0, "max_warmup_retained_pixel_error": 0, "dither_state_changes": 0}
    seed = 3
    if bad == "seed": seed = 4
    elif bad == "boolean": s["requests"] = True
    elif bad == "passes": s["raw_render_passes"] = 3
    elif bad == "state_checks": s["exact_state_checks"] = 3
    elif bad == "policy": s["policy"] = "cache"
    else: s["dither_state_changes"] = 1
    assert not valid_receipts([{"seed": seed, "stats": s}], [3])
