"""Source-bracketed six-slot injected producer; not a HOST authorization."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as transport
from experiments.spectral_reachability import p3_six_lifecycle_20261030_v1 as lifecycle
from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as native_factory
from experiments.spectral_reachability import p3_six_strict_rebuild_20261030_v1 as strict
from experiments.spectral_reachability import p3_six_terminal_20261030_v1 as terminal

need = transport.need
NAME = "p3_six_bound_producer_20261030_v1"


def own_sha():
    path = Path(__file__).resolve()
    need(path.is_file() and not path.is_symlink(), "producer source")
    return terminal.file_sha(path)


def _equal(a, b):
    """Exact source comparison including array dtype, shape and raw bytes."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and
                a.shape == b.shape and a.tobytes() == b.tobytes())
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def binding(contract, sha):
    need(strict.core.digest(contract) == sha, "strict source contract pin")
    return dict(source_contract_sha256=sha, task_roots=deepcopy(contract["roots"]),
                expected_model_hash=contract["expected_model_hash"], precision=deepcopy(contract["precision"]))


def profile(store):
    return dict(limits=deepcopy(store.limits), cap=store.cap, reserve=store.reserve,
                floor=store.floor, wall=store.wall)


class BoundStore(transport.Store):
    """Insert own producer pin after denominator, without changing old schema."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.producer_binding = None

    def event(self, origin, kind, resource, token, detail):
        if resource == "six_denominator":
            need(self.index == 1 and self.producer_binding is not None and
                 origin == "physical" and kind == "record", "bound denominator position")
            result = super().event(origin, kind, resource, token, detail)
            super().event("physical", "record", "six_producer_binding", self.index,
                          deepcopy(self.producer_binding))
            return result
        return super().event(origin, kind, resource, token, detail)


def execute(store, factory, backend_factory, *, source_sha256, producer_sha256, permit_ram=False):
    """Run injected lifecycle with source checks; caller owns all execution authority.

    Failure is re-raised; a best-effort failure record and terminal snapshot are
    retained. A failed source recheck never produces an end binding.
    """
    need(type(store) is BoundStore and not store.sealed and not store.poisoned and store.index == 0 and
         not list(store.output.iterdir()), "fresh exclusive bound store")
    try:
        need(type(factory) is native_factory.Factory and factory.store is store and
             factory.next_slot == 0 and not factory.active and not factory.poisoned, "fresh six-slot factory")
        need(type(permit_ram) is bool and own_sha() == producer_sha256, "producer source pin")
        contract, prepared, data = strict.source_material()
        detail = binding(contract, source_sha256)
        need(_equal(factory.prepared["tasks"], prepared["tasks"]) and
             _equal(factory.source["roots"], data["roots"]) and
             _equal(factory.source["source_plan"], data["source_plan"]), "factory source material")
        store.producer_binding = dict(schema=NAME + ".binding", producer_sha256=producer_sha256,
                                      source_contract_sha256=source_sha256, profile=profile(store))
        store.event("physical", "record", "six_strict_source_begin", store.index, detail)
        result = lifecycle.execute(prepared["tasks"], store, factory, backend_factory, permit_ram=permit_ram)
        after, after_prepared, after_data = strict.source_material()
        need(after == contract and strict.core.digest(after) == source_sha256 and own_sha() == producer_sha256 and
             _equal(factory.prepared["tasks"], after_prepared["tasks"]) and
             _equal(factory.source["roots"], after_data["roots"]) and
             _equal(factory.source["source_plan"], after_data["source_plan"]), "final source drift")
        store.event("physical", "record", "six_strict_source_end", store.index, detail)
        published = terminal.publish(store)
        return dict(status="lifecycle_returned_not_certified", rows=result["rows"],
                    source_contract_sha256=source_sha256, producer_sha256=producer_sha256,
                    terminal=published, physical_certified=False, scientific_pass=False)
    except BaseException as exc:
        try:
            if not store.poisoned and not store.sealed:
                store.event("physical", "record", "six_producer_failure", store.index,
                            dict(type=type(exc).__name__, message=str(exc)[:2048]))
        except BaseException as secondary:
            exc.add_note("failure journal unavailable: " + repr(secondary))
        try:
            if not store.sealed:
                hashes = terminal.publish(store)
                exc.add_note("partial terminal hashes: " + repr(hashes))
        except BaseException as secondary:
            exc.add_note("partial terminal publication failed: " + repr(secondary))
        raise


def audit_binding(events, report, contract, source_sha256, producer_sha256):
    """Check both strict source brackets and separate producer/profile binding."""
    strict.check_binding(events, contract, source_sha256)
    records = [e for e in events if e["origin"] == "physical" and e["resource"] == "six_producer_binding"]
    need(len(records) == 1 and records[0]["index"] == 2 and records[0]["kind"] == "record" and
         type(producer_sha256) is str and own_sha() == producer_sha256, "pinned producer source")
    expected_profile = {k: report[k] for k in ("limits", "cap", "reserve", "floor", "wall")}
    need(records[0]["detail"] == dict(schema=NAME + ".binding", producer_sha256=producer_sha256,
         source_contract_sha256=source_sha256, profile=expected_profile), "producer and profile binding")


def consume(output, report_sha256, completion_sha256, source_sha256, producer_sha256):
    """Strict math consumer plus independent producer/profile binding check."""
    receipt = strict.consume(output, report_sha256, completion_sha256, source_sha256)
    directory = Path(output).absolute()
    report = strict.native_saved._pinned(directory / terminal.REPORT, report_sha256)
    events = transport.inspect(directory / "events.jsonl", report["journal_sha256"], report["cap"])["events"]
    contract, _, _ = strict.source_material()
    audit_binding(events, report, contract, source_sha256, producer_sha256)
    return dict(schema=NAME + ".consume", source_math=receipt, producer_binding_checked=True,
                producer_sha256=producer_sha256, full_saved_strict=False,
                physical_certified=False, scientific_pass=False)
