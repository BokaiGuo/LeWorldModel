"""Owned native construction and call-boundary guards; no physics on import."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from experiments.spectral_reachability import p3_six_owned_constructor_20261030_v1 as owner
from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as factory

original = owner.original
native = original.native
need = original.need


def tracked_construct(meter, make, spec, metered_class):
    """Retain the base instance before its __init__, including gym wrapper failure.

    Cannot recover resources held exclusively by an unreturned wrapper or an
    allocation that never became reachable from the base instance.
    """
    captured = []

    class Tracked(metered_class):
        def __init__(self, *args, **kwargs):
            need(not captured, "one native base instance")
            self._p3_close_attempted = False
            self._p3_close_returned = False
            self._p3_constructing = True
            captured.append(self)
            super().__init__(*args, **kwargs)

        def close(self):
            need(not self._p3_close_attempted, "one native base close attempt")
            self._p3_close_attempted = True
            def close_base():
                result = super(Tracked, self).close()
                self._p3_close_returned = True
                return result
            if self._p3_constructing:
                return meter.cleanup(close_base)
            return close_base()

    try:
        env = make(replace(spec, entry_point=Tracked), render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
        need(len(captured) == 1 and getattr(env, "unwrapped", None) is captured[0],
             "captured native base identity")
        captured[0]._p3_constructing = False
        return env
    except BaseException as exc:
        if captured:
            instance = captured[0]
            if not instance._p3_close_attempted:
                try:
                    instance.close()
                except BaseException as cleanup:
                    exc.add_note("captured base cleanup failed: " + repr(cleanup))
            if not instance._p3_close_returned:
                exc.add_note("captured base cleanup not confirmed; wrapper resources unknown")
        else:
            exc.add_note("no native base captured; constructor resources unknown")
        raise


def construct(meter):
    """Original frozen registration lineage, with early base-object ownership."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from stable_worldmodel.envs.pusht.env import PushT
    import inspect
    need(Path(inspect.getsourcefile(PushT)).resolve() == native.kernel.moving.ENV and
         native.sha(native.kernel.moving.ENV) == native.kernel.moving.ENV_SHA, "frozen native base source")
    spec = gym.spec("swm/PushT-v1")
    need(spec.entry_point == "stable_worldmodel.envs.pusht.env:PushT" and
         spec.max_episode_steps is None and spec.additional_wrappers == (), "known native lineage")
    return tracked_construct(meter, gym.make, spec, native.metered_class(PushT, meter))


def _around(guard, label, call):
    guard(label + "_before")
    try:
        result = call()
    except BaseException as exc:
        try:
            guard(label + "_after_failure")
        except BaseException as secondary:
            exc.add_note("post-failure guard: " + repr(secondary))
        raise
    guard(label + "_after")
    return result


def adapters(guard, *, construct_fn=construct):
    """Factory-compatible adapters; explicit injection only for CPU tests."""
    class Meter(original.Meter):
        def begin(self, resource, units=1):
            if resource != "close":
                guard("physical_" + resource + "_before")
            return super().begin(resource, units)

        def end(self, token):
            resource = self.pending[token][0]
            if resource != "close":
                guard("physical_" + resource + "_after")
            return super().end(token)

        def fail(self, token, exc):
            resource = self.pending[token][0]
            if resource != "close":
                try:
                    guard("physical_" + resource + "_after_failure")
                except BaseException as secondary:
                    exc.add_note("post-failure guard: " + repr(secondary))
            return super().fail(token, exc)

    underlying = owner.module(construct=construct_fn)
    meter_module = SimpleNamespace(Meter=Meter, native_factory=underlying.native_factory)
    class Session(factory.native.Session):
        def encode(self, boundary, rgb):
            return _around(guard, "encoding", lambda: super(Session, self).encode(boundary, rgb))
    return meter_module, SimpleNamespace(setup=factory.native.setup, Session=Session)


def install(target, guard):
    """Attach adapters to a fresh, exact native Factory; do not construct a slot."""
    need(type(target) is factory.Factory and target.next_slot == 0 and not target.active and
         not target.poisoned, "fresh exact factory")
    target.meter_module, target.native_module = adapters(guard)
