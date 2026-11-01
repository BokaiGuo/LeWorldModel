"""Native constructor ownership adapter; no environment is created on import."""
from types import SimpleNamespace
from pathlib import Path

from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as original

Meter = original.Meter


def source_contract():
    """Extend the six-slot factory seal; not a timestamp or authority."""
    from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as factory
    previous = factory.source_contract()
    paths = [Path(__file__).resolve(),
             Path(__file__).with_name("p3_six_durable_transport_20261030_v1.py").resolve()]
    # The preceding contract already reconstructs and checks the prior closure.
    from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
    files = {}
    for path in paths:
        original.need(path.is_file() and not path.is_symlink(), "adapter source")
        files[str(path)] = host.source.host.sha(path)
    return dict(schema="p3_six_owned_constructor_20261030_v1.source",
                previous=previous, added_sha256=files, host_runner_ready=False,
                full_saved_strict=False, scientific_pass=False)


def verify_source_contract(value):
    original.need(value == source_contract(), "owned constructor source drift")
    return value


def module(*, construct=None, wrap=None):
    """Factory-compatible module. Injection exists solely for controlled tests."""
    construct = construct if construct is not None else original.native.native_factory
    wrap = wrap if wrap is not None else original.Env

    def native_factory(meter):
        original.need(isinstance(meter, Meter), "original native meter required")
        token = meter.begin("constructor")
        raw = None
        try:
            raw = construct(meter)
        except BaseException as exc:
            try:
                meter.fail(token, exc)
            except BaseException as secondary:
                exc.add_note("constructor failure ledger: " + repr(secondary))
            raise
        try:
            # Retain raw until the wrapper is safely handed off, even when
            # the durable constructor return or wrapper creation fails.
            meter.end(token)
            return wrap(raw, meter)
        except BaseException as exc:
            try:
                meter.cleanup(raw.close)
            except BaseException as cleanup:
                exc.add_note("constructor-owned cleanup failed: " + repr(cleanup))
            raise
    return SimpleNamespace(Meter=Meter, native_factory=native_factory)
