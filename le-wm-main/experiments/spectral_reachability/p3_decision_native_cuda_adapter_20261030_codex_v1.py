"""Native CUDA/feedback adapters for four-decision P3; not a launcher or certificate.

Caller supplies an already-loaded frozen CUDA model, two previously allocated
stores, and externally reviewed limits. This module never constructs a model.
"""
from copy import deepcopy
from pathlib import Path
import re

from experiments.spectral_reachability import p3_decision_equal_work_control_20261030_codex_v1 as control
from experiments.spectral_reachability import p3_decision_journal_20261030_codex_v1 as decision_journal
from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as native_transport
from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as native_factory
from experiments.spectral_reachability import p3_six_guarded_native_20261030_v1 as guarded
from experiments.spectral_reachability import p3_six_numeric_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import p3_six_host_identity_bridge_20261030_v1 as identity
from experiments.spectral_reachability import p3_six_strict_rebuild_20261030_v1 as strict

NAME = "p3_decision_native_cuda_adapter_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_P3_DECISION_NATIVE_ADAPTER"
need = control.old.need
POLICY_MAP = {"fixed_h1": "fixed_order", "signal": "signal_order", "reverse_signal": "reverse_signal_order"}


def file_sha(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink(), "regular source file")
    return decision_journal.sha(p.read_bytes())


def checked_sources(expected_adapter_sha, expected_source_sha):
    need(file_sha(__file__) == expected_adapter_sha, "adapter source pin")
    decision_journal.source_pins()
    actual = control.build_prepare()
    prepared_bytes = decision_journal.PREP.read_bytes()
    need(control.core.canonical(actual) == prepared_bytes.rstrip(b"\n"), "decision prepare/source closure")
    contract, prepared, source = strict.source_material()
    need(control.core.digest(contract) == expected_source_sha, "native source contract pin")
    need(len(prepared["tasks"]) == len(source["roots"]) == 2, "two known roots")
    return contract, prepared, source


class TrackedBackend:
    """Latch a backend failure even if outer callback lifecycle catches it."""
    def __init__(self, adapter, backend):
        self.adapter, self.backend, self.kind, self.slot = adapter, backend, backend.kind, adapter.active_slot
        need(self.kind == "HOST_CUDA", "native CUDA backend")

    def invoke(self, name, *args):
        need(not self.adapter.poisoned and not self.adapter.closed and self.adapter.active_slot == self.slot,
             "adapter poisoned or stale backend")
        try:
            return getattr(self.backend, name)(*args)
        except BaseException:
            self.adapter.poisoned = True
            self.adapter.factory.poisoned = True
            raise

    def signal(self, request, ledger):
        return self.invoke("signal", request, ledger)

    def replay(self, context, actions, segmented, ledger):
        return self.invoke("replay", context, actions, segmented, ledger)


class Session:
    def __init__(self, adapter, session):
        self.adapter, self.session, self.episode_id, self.slot = (
            adapter, session, session.episode_id, adapter.active_slot)

    def _call(self, name, *args):
        need(not self.adapter.poisoned and not self.adapter.closed and self.adapter.active_slot == self.slot,
             "adapter poisoned or stale session")
        try:
            self.adapter.guard("session_" + name + "_before")
            result = getattr(self.session, name)(*args)
            self.adapter.guard("session_" + name + "_after")
            return result
        except BaseException:
            self.adapter.poisoned = True
            self.adapter.factory.poisoned = True
            raise

    def packet(self, start, raw):
        return self._call("packet", start, raw)

    def encode(self, boundary, rgb):
        return self._call("encode", boundary, rgb)


class Adapter:
    """Provide session_factory and backend_factory for control.execute_roster.

    Instrumented execution is not a proof of native IO or hardware. The caller
    owns source sealing, memory/disk/deadline admission and final consumption.
    """
    def __init__(self, *, model, torch, device, decision_store, native_store, gpu_uuid,
                 allocated_cap, reserved_cap, authorization, adapter_sha256, source_sha256):
        need(authorization == AUTH, "main-thread authorization")
        need(getattr(device, "type", None) == "cuda" and type(getattr(device, "index", None)) is int
             and device.index >= 0 and type(gpu_uuid) is str and
             re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                          gpu_uuid) is not None,
             "explicit physical CUDA device")
        need(type(allocated_cap) is int and type(reserved_cap) is int and
             0 < allocated_cap <= reserved_cap, "explicit CUDA peaks")
        need(type(decision_store) is decision_journal.Store and
             isinstance(native_store, native_transport.Store) and not decision_store.finished
             and not decision_store.poisoned and not native_store.poisoned, "separate live durable stores")
        contract, prepared, source = checked_sources(adapter_sha256, source_sha256)
        self.model, self.torch, self.device = model, torch, device
        self.contract, self.prepared, self.source = contract, prepared, source
        self.decision_store, self.native_store = decision_store, native_store
        self.gpu_uuid, self.allocated_cap, self.reserved_cap = gpu_uuid, allocated_cap, reserved_cap
        self.adapter_sha256, self.source_sha256 = adapter_sha256, source_sha256
        self.poisoned, self.closed, self.next_slot, self.active_slot, self.arm = False, False, 0, None, None
        self.counts = {}
        self.hooks = identity.host.hook_snapshot(model)
        self.before = identity.snapshot(model, torch, device, prepared, contract)
        need(self.before["device"]["uuid"] == gpu_uuid, "physical UUID identity")
        meter_module, native_module = guarded.adapters(self.guard)
        self.factory = native_factory.Factory(prepared, source, native_store, model, torch, device,
                                              meter_module=meter_module, native_module=native_module)
        self.guard("entry")
        self._anchor("begin", None)

    def _anchor(self, phase, slot, **extra):
        try:
            self.native_store.event("physical", "record", NAME + "_" + phase, self.native_store.index,
                                    dict(slot=slot, decision_index=self.decision_store.index,
                                         decision_tail=self.decision_store.previous,
                                         decision_prepare_sha256=decision_journal.PREP_SHA,
                                         adapter_sha256=self.adapter_sha256, source_sha256=self.source_sha256,
                                         **extra))
        except BaseException:
            self.poisoned = True
            if hasattr(self, "factory"):
                self.factory.poisoned = True
            raise

    def guard(self, label):
        try:
            need(not self.poisoned and not self.closed, "adapter unavailable")
            need(not self.decision_store.poisoned and not self.decision_store.finished, "decision store unavailable")
            self.native_store.check("numerical_events", 0)
            need(identity.host.prior.checked_precision(self.torch) == self.contract["precision"], "precision drift")
            need(str(self.torch.cuda.get_device_properties(self.device.index).uuid) == self.gpu_uuid,
                 "CUDA UUID drift")
            allocated = int(self.torch.cuda.max_memory_allocated(self.device))
            reserved = int(self.torch.cuda.max_memory_reserved(self.device))
            need(0 <= allocated <= self.allocated_cap and 0 <= reserved <= self.reserved_cap, "CUDA peak cap")
            modules = list(self.model.modules())
            parameters = list(self.model.named_parameters())
            buffers = list(self.model.named_buffers())
            need(modules and parameters and all(not m.training for m in modules) and
                 all(not p.requires_grad and p.device == self.device for _, p in parameters) and
                 all(b.device == self.device for _, b in buffers), "frozen CUDA model drift")
            self.counts[label] = self.counts.get(label, 0) + 1
        except BaseException:
            self.poisoned = True
            if hasattr(self, "factory"):
                self.factory.poisoned = True
            raise

    def session_factory(self, slot, root, policy):
        need(not self.poisoned and not self.closed and self.active_slot is None and
             self.next_slot < len(control.ROSTER) and
             (slot, root, policy) == control.ROSTER[self.next_slot], "ordered new roster")
        self.next_slot += 1
        self.guard("slot_before")
        self._anchor("slot_begin", slot, root=root, policy=policy, native_policy=POLICY_MAP[policy])
        try:
            session, prior, close = self.factory(slot, root, POLICY_MAP[policy])
        except BaseException:
            self.poisoned = True
            self.factory.poisoned = True
            raise
        self.active_slot = slot
        try:
            chunks = bridge.Bridge(self.native_store, slot)
            self.arm = identity.host.prior.ArmSink(chunks, slot)
        except BaseException as exc:
            self.poisoned = True
            self.factory.poisoned = True
            self.active_slot = None
            try:
                close()
            except BaseException as cleanup_error:
                exc.add_note("native cleanup failed: " + repr(cleanup_error))
            raise
        def cleanup():
            need(self.active_slot == slot, "single cleanup")
            try:
                result = close()
            except BaseException as exc:
                self.poisoned = True
                self.factory.poisoned = True
                try:
                    self._anchor("slot_end", slot, root=root, policy=policy, cleanup="failed")
                except BaseException as secondary:
                    exc.add_note("cleanup anchor failed: " + repr(secondary))
                raise
            finally:
                self.active_slot = None
                self.arm = None
            # No guard before cleanup: cleanup must remain possible after a failure.
            self._anchor("slot_end", slot, root=root, policy=policy, cleanup="returned")
            return result
        return Session(self, session), prior, cleanup

    def backend_factory(self, task, sink):
        need(not self.poisoned and self.active_slot is not None and sink is self.decision_store,
             "active slot and exact decision sink")
        root = control.ROSTER[self.active_slot][1]
        need(task == control.design.parent.parent_task(self.prepared["tasks"][root], task["root_context"]),
             "task derived from active bound root")
        self.guard("backend_construct")
        # Native tensor/IO trace uses separate chunked store; decision ledger stays
        # in decision_store via the original control's ledger sink.
        try:
            tree = identity.host.prior.NativeTree(self.model, self.device, task, self.arm)
            backend = identity.host.prior.RecordingBackend(tree, task, self.arm, lambda: self.guard("numerical_call"))
            return TrackedBackend(self, backend)
        except BaseException:
            self.poisoned = True
            self.factory.poisoned = True
            raise

    def finish(self):
        need(not self.poisoned and not self.closed and self.active_slot is None and
             self.next_slot == len(control.ROSTER) and not self.factory.active and not self.factory.poisoned,
             "incomplete or failed adapter")
        self.guard("exit")
        need(identity.host.hooks_restored(self.hooks), "model hooks changed")
        after = identity.snapshot(self.model, self.torch, self.device, self.prepared, self.contract)
        need(after == self.before, "model identity changed")
        checked_sources(self.adapter_sha256, self.source_sha256)
        self._anchor("end", None)
        self.closed = True
        return dict(schema=NAME + ".receipt", before=self.before, after=after,
                    counts=deepcopy(self.counts), source_sha256=self.source_sha256,
                    adapter_sha256=self.adapter_sha256, full_saved_strict=False,
                    physical_certified=False, scientific_pass=False)
