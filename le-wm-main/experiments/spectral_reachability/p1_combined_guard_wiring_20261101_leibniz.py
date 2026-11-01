"""Development-only, serialized physical/visual guard composition."""
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

from . import p1_constructor_close_guard_20261101_leibniz as constructor
from . import p1_visual_live_host_wiring_20261101_leibniz as host
from . import p1_visual_sink_guard_20261101_leibniz as visual

p, g = constructor.physical, constructor.physical.total


class Combined:
    def __init__(self, physical, meter, store):
        self.physical, self.guard, self.meter, self.store = physical, physical.guard, meter, store
        self.active = False
        self.raw_begin, self.raw_end = meter.begin, meter.end

    @contextmanager
    def scope(self):
        paths = (Path(__file__).resolve(), Path(constructor.__file__).resolve())
        if not all(str(path) in self.guard.contract["source_sha256"] for path in paths):
            self.physical.violation("combined source closure")
        g.check_sources(self.guard.contract)
        if self.active:
            self.physical.violation("combined scope already active")
        with constructor.guarded_class_factory(self.physical, self.meter) as factory:
            self.active = True
            try:
                yield factory
            finally:
                self.active = False

    def _visual(self, operation, meter, store):
        if not self.active or meter is not self.meter or store is not self.store or self.physical.stack:
            self.physical.violation("combined meter/store/scope mismatch")
        # The visual adapter takes exclusive ownership of begin/end only inside
        # its existing serialized scope. The outer physical meter.call stays
        # installed, so a physical call in this phase must not be accepted.
        with patch.object(meter, "begin", self.raw_begin), patch.object(meter, "end", self.raw_end):
            return visual.guarded_boundary(self.guard, meter, store, operation)

    def encode_boundary(self, handler, runtime, task, layer, row, obs, body, meter, store, step):
        return self._visual(lambda: visual.moving.encode_boundary(
            handler, runtime, task, layer, row, obs, body, meter, store, step), meter, store)


class MovingProxy(host.visual.MovingProxy):
    def __init__(self, combined):
        super().__init__(combined.guard)
        self.combined = combined

    def __getattr__(self, name):
        if name == "encode_boundary":
            return self.combined.encode_boundary
        return super().__getattr__(name)


@contextmanager
def structured_wiring(combined):
    """Exercise real cp10 configure capture with the combined moving proxy."""
    if not combined.active:
        combined.physical.violation("combined structured scope inactive")
    def episode(guard):
        if guard is not combined.guard:
            combined.physical.violation("combined structured guard mismatch")
        old = host.visual.rgb.episode
        env = dict(old.__globals__)
        env["moving"] = MovingProxy(combined)
        return FunctionType(old.__code__, env, "combined_episode", old.__defaults__, old.__closure__)
    with patch.object(host.visual, "structured_episode", episode):
        with host._structured_wiring(combined.guard) as configured:
            yield configured


def cem_execute(combined, prepared, data, store, handler, runtime, factory, runner, rebind, root):
    """Pass a combined proxy to the actual CEM execute_one (development only)."""
    if not combined.active or store is not combined.store or runner.rgb.moving is not visual.moving:
        combined.physical.violation("combined CEM runner/store/scope mismatch")
    copied_rgb = SimpleNamespace(**{**vars(runner.rgb), "moving": MovingProxy(combined)})
    wrapped = SimpleNamespace(**{**vars(runner), "rgb": copied_rgb})
    return host.cem.execute_one(prepared, data, store, handler, runtime, factory, wrapped, rebind, root)


def run_full_host(*args, **kwargs):
    host.visual.require_full_host_ready()
    raise ValueError("fail-closed: combined total-host integration not implemented")
