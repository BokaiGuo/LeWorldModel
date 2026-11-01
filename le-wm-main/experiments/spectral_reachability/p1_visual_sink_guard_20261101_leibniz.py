"""Strict adapter for the real legacy visual Sink; no host or CUDA execution."""
from contextlib import ExitStack
from pathlib import Path
import threading
from unittest.mock import patch

from . import p1_prospective_total_guard_20261101_leibniz as total
from . import p1_moving_live_feedback_physical_20261027_v1 as visual
from . import p1_constructive_blind_encode_cuda_20261027_v1 as encoding
from . import agent_p4_moving_multiboundary_20261028_v1 as moving

NAME = "p1_visual_sink_guard_20261101_leibniz"
_LOCK = threading.Lock()
SEQUENCE = ("encoder", "projector")


def check_sources(guard):
    paths = (Path(__file__).resolve(), Path(visual.__file__).resolve(), Path(encoding.__file__).resolve(),
             Path(moving.__file__).resolve())
    total.require(all(str(p) in guard.contract["source_sha256"] for p in paths), "visual adapter source closure")
    total.check_sources(guard.contract)


class MeterProxy:
    """Only a visual Sink may use begin/end, with a fixed no-retry order."""
    def __init__(self, guard, meter):
        self.guard, self.meter = guard, meter
        self.position, self.pending, self.authorized = 0, None, False

    def check(self):
        return self.meter.check()

    def _fail(self, reason):
        self.guard.fail(reason)

    def begin(self, resource, units=1):
        if self.pending is not None or self.position >= len(SEQUENCE) or resource != SEQUENCE[self.position] \
                or type(units) is not int or units != 1:
            self._fail("visual begin order/resource/units")
        stack = self.guard.stack
        if not stack or stack[-1]["expected_hooks"] != list(SEQUENCE) or \
                stack[-1]["observed_hooks"] != list(SEQUENCE[:self.position + 1]):
            self._fail("visual module hook before begin mismatch")
        self.authorized = True
        try:
            ident = self.meter.begin(resource, units)
        finally:
            self.authorized = False
        self.pending = (ident, resource)
        return ident

    def end(self, ident):
        if self.pending is None or type(ident) is not int or ident != self.pending[0]:
            self._fail("visual end token mismatch")
        if self.guard.stack[-1]["observed_hooks"] != list(SEQUENCE[:self.position + 1]):
            self._fail("visual module hook before end mismatch")
        self.authorized = True
        try:
            self.meter.end(ident)
        finally:
            self.authorized = False
        self.pending = None
        self.position += 1

    def complete(self):
        if self.position != len(SEQUENCE) or self.pending is not None:
            self._fail("visual begin/end incomplete")


def guarded_boundary(guard, meter, store, operation):
    """Call actual encode_boundary under one paid source_encode reservation.

    The zero-argument operation must be the original encode_boundary call; it
    must instantiate exactly one real Sink with the supplied meter and store.
    The patches are process-local, scoped and require serialized invocation.
    """
    total.require(not guard.failed and not guard.closed, "active visual guard")
    if not callable(operation) or not isinstance(meter, visual.Meter):
        guard.fail("visual operation or meter type")
    try:
        check_sources(guard)
    except (OSError, ValueError) as exc:
        guard.fail("visual source binding: " + str(exc)[:300])
    if not _LOCK.acquire(blocking=False):
        guard.fail("concurrent visual adapter")
    original_sink, original_begin, original_end = visual.Sink, meter.begin, meter.end
    proxy = MeterProxy(guard, meter)
    created = []

    def begin(resource, units=1):
        if not proxy.authorized:
            guard.fail("direct meter.begin bypass")
        return original_begin(resource, units)

    def end(ident):
        if not proxy.authorized:
            guard.fail("direct meter.end bypass")
        return original_end(ident)

    def sink_factory(actual_store, actual_meter, episode, stage):
        if actual_store is not store or actual_meter is not meter or stage != "encode" or created:
            guard.fail("visual sink identity or roster")
        sink = original_sink(actual_store, proxy, episode, stage)
        created.append(sink)
        return sink

    def execute():
        with ExitStack() as patches:
            patches.enter_context(patch.object(meter, "begin", begin))
            patches.enter_context(patch.object(meter, "end", end))
            patches.enter_context(patch.object(visual, "Sink", sink_factory))
            value = operation()
            if len(created) != 1:
                guard.fail("visual sink not constructed exactly once")
            created[0].complete()
            proxy.complete()
            return value
    try:
        value = guard.dispatch("feedback", "source_encode", 1, execute)
        check_sources(guard)
        return value
    except BaseException:
        guard.failed = True
        raise
    finally:
        _LOCK.release()


def encode_boundary(guard, handler, runtime, task, layer, row, obs, body, meter, store, step):
    """Exact legacy encode_boundary ABI plus explicit prospective guard."""
    total.require(moving.old is visual, "visual module identity")
    return guarded_boundary(
        guard, meter, store,
        lambda: moving.encode_boundary(handler, runtime, task, layer, row, obs, body, meter, store, step))
