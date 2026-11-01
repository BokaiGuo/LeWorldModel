"""Exclusive constructor-failure close adapter for the legacy physical meter."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from . import p1_physical_callback_guard_20261101_leibniz as physical


@contextmanager
def guarded_class_factory(guard, meter):
    """Install meter scope and yield a factory for a guarded legacy subclass."""
    own = Path(__file__).resolve()
    if str(own) not in guard.guard.contract["source_sha256"]:
        guard.violation("constructor adapter source closure")
    physical.total.check_sources(guard.guard.contract)
    state = {"active": True, "close_depth": 0}
    try:
        with physical.physical_meter_callbacks(guard, meter):
            original_call = meter.call

            def tracked_call(resource, fn):
                if not state["active"]:
                    guard.violation("constructor adapter inactive")
                if resource != "close":
                    return original_call(resource, fn)
                state["close_depth"] += 1
                try:
                    return original_call(resource, fn)
                finally:
                    state["close_depth"] -= 1

            def class_factory(base):
                legacy_class = physical.legacy.metered_class(base, meter)

                class Guarded(legacy_class):
                    def close(self):
                        if not state["active"]:
                            guard.violation("constructor adapter inactive")
                        if state["close_depth"]:
                            return super().close()
                        return meter.call("close", lambda: super(Guarded, self).close())

                return Guarded

            with patch.object(meter, "call", tracked_call):
                yield class_factory
    finally:
        state["active"] = False
