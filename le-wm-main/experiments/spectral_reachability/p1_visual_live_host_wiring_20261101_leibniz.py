"""Development-only wiring of the existing visual guard into actual hosts.

The real execution entry points remain locked until the independent total-cost
readiness gate is implemented. Private wiring helpers support boundary tests;
they are not an isolation or security boundary against arbitrary Python code.
"""
from contextlib import contextmanager
from threading import Lock

from . import p1_visual_guard_host_20261101_leibniz as visual
from . import codex_p1_common_structured_cp10_host_20261101_exclusive as structured
from . import codex_p1_full_cem_host_20261101_exclusive as cem

_configure_lock = Lock()
_original_episode = visual.rgb.episode


def _captured(function, name):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))
    if name not in cells:
        raise ValueError("fail-closed: structured capture missing")
    return cells[name].cell_contents


@contextmanager
def _structured_wiring(guard):
    """Configure actual cp10 host while its episode capture is guarded."""
    host = structured.host
    if not _configure_lock.acquire(blocking=False):
        raise ValueError("fail-closed: concurrent structured configuration")
    try:
        if host.rgb.episode is not _original_episode or _original_episode.__module__ != visual.rgb.__name__:
            raise ValueError("fail-closed: unexpected structured episode")
        # configure mutates these shared module attributes. Restore even on error.
        saved_host = {name: getattr(host, name) for name in ("NAME", "build", "protocol", "execute")}
        saved_rgb = {name: getattr(host.rgb, name) for name in ("roster", "episode")}
        original = host.rgb.episode
        guarded = visual.structured_episode(guard)
        try:
            host.rgb.episode = guarded
            structured.configure()
            configured_episode = host.rgb.episode
            if _captured(configured_episode, "old_episode") is not guarded:
                raise ValueError("fail-closed: structured episode capture bypass")
            if guarded.__code__ is not original.__code__ or guarded.__globals__["moving"].guard is not guard:
                raise ValueError("fail-closed: structured proxy mismatch")
            yield host
            if host.rgb.episode is not configured_episode:
                raise ValueError("fail-closed: structured episode changed during scope")
        finally:
            for name, value in saved_host.items():
                setattr(host, name, value)
            for name, value in saved_rgb.items():
                setattr(host.rgb, name, value)
    finally:
        _configure_lock.release()


def _cem_execute_guarded(guard, prepared, data, store, handler, runtime, factory, runner, rebind, root):
    """Pass a local proxy into the actual CEM execute_one, never the raw runner."""
    if runner.rgb.moving is not visual.rgb.moving:
        raise ValueError("fail-closed: unexpected CEM moving source")
    wrapped = visual.cem_runner(guard, runner)
    if (wrapped is runner or wrapped.rgb is runner.rgb
            or not isinstance(wrapped.rgb.moving, visual.MovingProxy)
            or wrapped.rgb.moving.guard is not guard):
        raise ValueError("fail-closed: CEM proxy mismatch")
    return cem.execute_one(prepared, data, store, handler, runtime, factory, wrapped, rebind, root)


def run_structured(*args, **kwargs):
    """No real host execution without complete total-cost readiness."""
    visual.require_full_host_ready()
    raise ValueError("fail-closed: structured total-host integration not implemented")


def run_cem(*args, **kwargs):
    """No real host execution without complete total-cost readiness."""
    visual.require_full_host_ready()
    raise ValueError("fail-closed: CEM total-host integration not implemented")
