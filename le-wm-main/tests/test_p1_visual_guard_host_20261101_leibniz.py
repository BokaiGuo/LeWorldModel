from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_visual_guard_host_20261101_leibniz as host


def test_structured_real_episode_uses_isolated_moving():
    guard = object()
    episode = host.structured_episode(guard)
    assert episode.__code__ is host.rgb.episode.__code__
    assert episode.__globals__["moving"].guard is guard
    assert host.rgb.episode.__globals__["moving"] is host.rgb.moving


def test_cem_real_runner_rgb_redirect_is_isolated():
    guard = object()
    runner = SimpleNamespace(rgb=host.rgb, p3="p3", old="old")
    wrapped = host.cem_runner(guard, runner)
    assert wrapped.rgb.moving.guard is guard
    assert runner.rgb is host.rgb and wrapped.p3 == runner.p3
    assert wrapped.rgb.moving.setup is host.rgb.moving.setup


def test_boundary_redirect_and_no_recursion(monkeypatch):
    calls = []
    guard = object()
    monkeypatch.setattr(host.visual_guard, "encode_boundary",
                        lambda *args: calls.append(args) or "encoded")
    assert host.MovingProxy(guard).encode_boundary("handler", "runtime") == "encoded"
    assert calls == [(guard, "handler", "runtime")]


def test_full_host_always_fails_closed():
    with pytest.raises(ValueError, match="fail-closed"):
        host.require_full_host_ready()
