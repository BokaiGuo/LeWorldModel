"""Guarded six-slot preloaded-model entry; no launcher or production authority."""
from pathlib import Path
import shutil

from experiments.spectral_reachability import p3_six_host_driver_20261030_v1 as old
from experiments.spectral_reachability import p3_six_guarded_native_20261030_v1 as native

need, producer, identity, base = old.need, old.producer, old.identity, old.base
NAME = "p3_six_guarded_host_20261030_v1"


def file_sha(module):
    path = Path(module.__file__).resolve()
    need(path.is_file() and not path.is_symlink(), "guarded source file")
    return producer.terminal.file_sha(path)


def run(directory, model, torch, device, plan, *, authorization, plan_sha256, source_sha256,
        producer_sha256, bridge_sha256, driver_sha256, native_sha256, host_sha256):
    need(authorization == old.AUTH, "explicit main-thread authorization")
    plan = old.validate_plan(plan)
    need(base.digest(plan) == plan_sha256, "resource plan pin")
    need(old.own_sha() == driver_sha256 and identity.own_sha() == bridge_sha256 and
         producer.own_sha() == producer_sha256 and file_sha(native) == native_sha256 and
         file_sha(__import__(__name__, fromlist=["x"])) == host_sha256, "guarded source pins")
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and not root.exists() and root.parent.is_dir(),
         "exclusive canonical outer directory")
    need(shutil.disk_usage(root.parent).free >= plan["cap"] + plan["outer_cap"] + plan["floor"],
         "combined output cap and disk floor")
    root.mkdir(mode=0o700)
    base.syncdir(root.parent)
    begin_sha = None
    try:
        contract, prepared, data = producer.strict.source_material()
        need(producer.strict.core.digest(contract) == source_sha256, "source contract pin")
        before = identity.snapshot(model, torch, device, prepared, contract)
        begin = dict(schema=NAME + ".begin", plan=plan, plan_sha256=plan_sha256,
                     source_sha256=source_sha256, producer_sha256=producer_sha256,
                     bridge_sha256=bridge_sha256, driver_sha256=driver_sha256,
                     native_sha256=native_sha256, host_sha256=host_sha256, identity=before)
        begin_sha = old._record(root, "begin.json", begin, plan)
        store = producer.BoundStore(root / "run", limits=plan["limits"], cap=plan["cap"],
                                    reserve=plan["reserve"], floor=plan["floor"], wall=plan["wall"])
        factory = producer.native_factory.Factory(prepared, data, store, model, torch, device)
        counts = {}

        def guard(label):
            try:
                store.check("physical_events" if label.startswith("physical_") else "encoding_events", 0)
                identity.host.prior.checked_precision(torch)
                need(str(torch.cuda.get_device_properties(device.index).uuid) == before["device"]["uuid"],
                     "runtime device UUID drift")
                need(int(torch.cuda.max_memory_allocated(device)) <= plan["cuda_allocated_bytes"] and
                     int(torch.cuda.max_memory_reserved(device)) <= plan["cuda_reserved_bytes"],
                     "CUDA peak ceiling")
                counts[label] = counts.get(label, 0) + 1
            except BaseException:
                factory.poisoned = True
                raise

        native.install(factory, guard)

        def backend(task, sink):
            guard("numerical_before")
            tree = identity.host.prior.NativeTree(model, device, task, sink)
            return identity.host.prior.RecordingBackend(tree, task, sink, lambda: guard("numerical_call"))

        guard("entry")
        receipt = identity.execute(store, factory, backend, source_sha256=source_sha256,
                                   producer_sha256=producer_sha256, bridge_sha256=bridge_sha256)
        need(not factory.active and not factory.poisoned, "native factory cleanup")
        need(receipt["before"] == before and receipt["after"] == before, "outer identity binding")
        need(0 <= store.clock() - store.started <= store.wall, "post-publication wall ceiling")
        peaks = dict(allocated=int(torch.cuda.max_memory_allocated(device)),
                     reserved=int(torch.cuda.max_memory_reserved(device)))
        need(peaks["allocated"] <= plan["cuda_allocated_bytes"] and
             peaks["reserved"] <= plan["cuda_reserved_bytes"], "final CUDA peak ceiling")
        need(file_sha(native) == native_sha256 and
             file_sha(__import__(__name__, fromlist=["x"])) == host_sha256 and
             old.own_sha() == driver_sha256, "guarded final source drift")
        end = dict(schema=NAME + ".end", begin_sha256=begin_sha, receipt=receipt,
                   guard_counts=counts, peaks=peaks, scientific_pass=False, physical_certified=False,
                   full_saved_strict=False)
        end_sha = old._record(root, "end.json", end, plan)
        return dict(begin_sha256=begin_sha, end_sha256=end_sha, terminal=receipt["terminal"],
                    physical_certified=False, full_saved_strict=False, scientific_pass=False)
    except BaseException as exc:
        try:
            old._record(root, "failure.json", dict(schema=NAME + ".failure", begin_sha256=begin_sha,
                       type=type(exc).__name__, message=str(exc)[:2048],
                       notes=[str(n)[:2048] for n in getattr(exc, "__notes__", [])][:16]), plan)
        except BaseException as secondary:
            exc.add_note("outer failure record unavailable: " + repr(secondary))
        raise
