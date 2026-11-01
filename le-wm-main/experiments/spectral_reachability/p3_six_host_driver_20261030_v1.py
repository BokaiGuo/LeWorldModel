"""Preloaded-model six-slot host adapter; no CLI, loading, or authorization grant."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import p3_six_host_identity_bridge_20261030_v1 as identity

producer = identity.producer
base = producer.transport.base
need = producer.need
NAME = "p3_six_host_driver_20261030_v1"
AUTH = "MAIN_APPROVED_P3_SIX_HOST"
MAX_RECORD = 4 * 1024 * 1024


def own_sha():
    p = Path(__file__).resolve()
    need(p.is_file() and not p.is_symlink(), "driver source")
    return producer.terminal.file_sha(p)


def validate_plan(plan):
    keys = {"limits", "cap", "reserve", "floor", "wall", "outer_cap",
            "cuda_allocated_bytes", "cuda_reserved_bytes", "coordination"}
    need(type(plan) is dict and set(plan) == keys, "explicit resource plan")
    need(type(plan["limits"]) is dict and set(plan["limits"]) == set(base.LIMITS) and
         all(type(v) is int and v >= 0 for v in plan["limits"].values()), "category limits")
    need(all(type(plan[k]) is int and plan[k] > 0 for k in keys - {"limits", "coordination"})
         and plan["reserve"] < plan["cap"] and
         sum(plan["limits"].values()) <= plan["cap"] - plan["reserve"] and
         plan["outer_cap"] >= 3 * MAX_RECORD and
         plan["cuda_allocated_bytes"] <= plan["cuda_reserved_bytes"], "resource ceilings")
    need(type(plan["coordination"]) is str and len(plan["coordination"].strip()) >= 8,
         "coordination record")
    return deepcopy(plan)


def _record(directory, name, value, plan):
    data = base.raw(value)
    need(len(data) <= MAX_RECORD, "outer record cap")
    files = list(directory.iterdir())
    used = sum(p.stat().st_size for p in files if p.is_file() and not p.is_symlink())
    need(used + len(data) <= plan["outer_cap"] and
         shutil.disk_usage(directory).free >= plan["floor"] + len(data), "outer cap or floor")
    path = directory / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        need(stream.write(data) == len(data), "outer complete write")
        stream.flush()
        os.fsync(stream.fileno())
    base.syncdir(directory)
    return base.fsha(data)


def run(directory, model, torch, device, plan, *, authorization, plan_sha256,
        source_sha256, producer_sha256, bridge_sha256, driver_sha256):
    """Execute only with an already-loaded model and explicit host coordination.

    Resource ceilings begin at this call, not at model loading. This adapter
    does not establish authority or perform the CUDA allocation preflight.
    """
    need(authorization == AUTH, "explicit main-thread authorization")
    plan = validate_plan(plan)
    need(base.digest(plan) == plan_sha256, "resource plan pin")
    need(own_sha() == driver_sha256 and identity.own_sha() == bridge_sha256 and
         producer.own_sha() == producer_sha256, "implementation pins")
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and not root.exists() and
         root.parent.is_dir(), "exclusive canonical outer directory")
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
                     identity=before)
        begin_sha = _record(root, "begin.json", begin, plan)
        store = producer.BoundStore(root / "run", limits=plan["limits"], cap=plan["cap"],
                                    reserve=plan["reserve"], floor=plan["floor"], wall=plan["wall"])
        factory = producer.native_factory.Factory(prepared, data, store, model, torch, device)
        counts = {"guard_calls": 0}

        def guard():
            try:
                store.check("numerical_events", 0)
                identity.host.prior.checked_precision(torch)
                need(str(torch.cuda.get_device_properties(device.index).uuid) == before["device"]["uuid"],
                     "runtime device UUID drift")
                need(int(torch.cuda.max_memory_allocated(device)) <= plan["cuda_allocated_bytes"] and
                     int(torch.cuda.max_memory_reserved(device)) <= plan["cuda_reserved_bytes"],
                     "CUDA peak ceiling")
                counts["guard_calls"] += 1
            except BaseException:
                # The lifecycle records slot exceptions and proceeds. Poisoning
                # prevents its next native factory call from opening an env.
                factory.poisoned = True
                raise

        def backend(task, sink):
            guard()
            tree = identity.host.prior.NativeTree(model, device, task, sink)
            return identity.host.prior.RecordingBackend(tree, task, sink, guard)

        guard()
        receipt = identity.execute(store, factory, backend, source_sha256=source_sha256,
                                   producer_sha256=producer_sha256, bridge_sha256=bridge_sha256)
        need(not factory.active and not factory.poisoned, "native factory cleanup")
        need(receipt["before"] == before and receipt["after"] == before, "outer identity binding")
        peaks = dict(allocated=int(torch.cuda.max_memory_allocated(device)),
                     reserved=int(torch.cuda.max_memory_reserved(device)))
        need(peaks["allocated"] <= plan["cuda_allocated_bytes"] and
             peaks["reserved"] <= plan["cuda_reserved_bytes"], "final CUDA peak ceiling")
        end = dict(schema=NAME + ".end", begin_sha256=begin_sha, receipt=receipt,
                   guard_calls=counts["guard_calls"], peaks=peaks,
                   physical_certified=False, full_saved_strict=False, scientific_pass=False)
        end_sha = _record(root, "end.json", end, plan)
        return dict(begin_sha256=begin_sha, end_sha256=end_sha, terminal=receipt["terminal"],
                    physical_certified=False, full_saved_strict=False, scientific_pass=False)
    except BaseException as exc:
        try:
            _record(root, "failure.json", dict(schema=NAME + ".failure", begin_sha256=begin_sha,
                    type=type(exc).__name__, message=str(exc)[:2048],
                    notes=[str(n)[:2048] for n in getattr(exc, "__notes__", [])][:16]), plan)
        except BaseException as error:
            exc.add_note("outer failure record unavailable: " + repr(error))
        raise
