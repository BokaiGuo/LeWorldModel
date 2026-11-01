"""Explicitly admitted P3 host driver. Never run without main-thread approval.

External supervision, GPU free-memory admission and saved-data certification
are separate requirements; this program does not provide those guarantees.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from experiments.spectral_reachability import p3_decision_native_cuda_adapter_20261030_codex_v1 as adapter

NAME = "p3_decision_host_cuda_driver_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_P3_DECISION_HOST_CUDA"
control = adapter.control
journal = adapter.decision_journal
transport = adapter.native_transport
need = control.old.need
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def file_sha(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink(), "regular pinned file")
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(plan):
    keys = {"gpu_uuid", "source_sha256", "files", "native_limits", "native_cap",
            "native_reserve", "decision_cap", "decision_reserve", "outer_cap",
            "floor", "wall", "allocated_cap", "reserved_cap", "coordination"}
    need(type(plan) is dict and set(plan) == keys, "plan keys")
    need(type(plan["gpu_uuid"]) is str and UUID.fullmatch(plan["gpu_uuid"]), "physical UUID")
    need(type(plan["source_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", plan["source_sha256"]),
         "source pin")
    modules = {"driver": __file__, "adapter": adapter.__file__,
               "decision_journal": journal.__file__, "native_transport": transport.__file__}
    need(type(plan["files"]) is dict and set(plan["files"]) == set(modules), "file pins")
    for name, path in modules.items():
        need(plan["files"][name] == file_sha(path), "file pin: " + name)
    limits = plan["native_limits"]
    need(type(limits) is dict and set(limits) == set(transport.base.LIMITS) and
         all(type(v) is int and v >= 0 for v in limits.values()), "native categories")
    numbers = ("native_cap", "native_reserve", "decision_cap", "decision_reserve",
               "outer_cap", "floor", "wall", "allocated_cap", "reserved_cap")
    need(all(type(plan[k]) is int and plan[k] > 0 for k in numbers), "positive limits")
    need(plan["native_reserve"] < plan["native_cap"] and
         sum(limits.values()) <= plan["native_cap"] - plan["native_reserve"] and
         plan["decision_reserve"] < plan["decision_cap"] and
         plan["allocated_cap"] <= plan["reserved_cap"] and
         plan["outer_cap"] >= 1024 * 1024, "consistent limits")
    need(type(plan["coordination"]) is str and len(plan["coordination"].strip()) >= 8,
         "coordination record")
    return adapter.checked_sources(plan["files"]["adapter"], plan["source_sha256"])


def record(root, name, obj, plan):
    data = control.core.canonical(obj) + b"\n"
    used = sum(p.stat().st_size for p in root.iterdir() if p.is_file() and not p.is_symlink())
    need(used + len(data) <= plan["outer_cap"] and
         shutil.disk_usage(root).free >= plan["floor"] + len(data), "outer storage limit")
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        need(f.write(data) == len(data), "short outer write")
        f.flush()
        os.fsync(f.fileno())
    journal.syncdir(root)
    return hashlib.sha256(data).hexdigest()


def inventory(directory):
    result = []
    for p in sorted(directory.iterdir()):
        need(p.is_file() and not p.is_symlink(), "flat regular native artifacts")
        result.append(dict(name=p.name, bytes=p.stat().st_size, sha256=file_sha(p)))
    return result


def run(root, plan, plan_sha256, authorization, *, runtime=None):
    need(authorization == AUTH, "main-thread authorization")
    need(hashlib.sha256(control.core.canonical(plan)).hexdigest() == plan_sha256, "plan pin")
    _, prepared, _ = validate(plan)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"], "exclusive visible UUID")
    p = Path(root)
    need(p.is_absolute() and p.resolve() == p and not p.exists() and not p.is_symlink()
         and p.parent.is_dir(), "exclusive canonical output")
    need(shutil.disk_usage(p.parent).free >= plan["native_cap"] + plan["decision_cap"] +
         plan["outer_cap"] + plan["floor"], "combined disk admission")
    p.mkdir(mode=0o700)
    journal.syncdir(p.parent)
    stage, ds, ns, roster, receipt = "begin", None, None, None, None
    begin = None
    try:
        begin = record(p, "begin.json", dict(schema=NAME + ".begin", plan=plan,
                       plan_sha256=plan_sha256, external_supervision_verified=False), plan)
        stage = "stores"
        ds = journal.Store(p / "decision", max_bytes=plan["decision_cap"],
                           terminal_reserve=plan["decision_reserve"])
        ns = transport.Store(p / "native", limits=plan["native_limits"], cap=plan["native_cap"],
                             reserve=plan["native_reserve"], floor=plan["floor"], wall=plan["wall"])
        stage = "runtime"
        if runtime is None:
            runtime = host_runtime
        receipt, roster, finish_error = runtime(plan, prepared, ds, ns)
        stage = "decision_terminal"
        decision_sha = ds.finish(roster)
        stage = "inventory"
        files = inventory(p / "native")
        events = next((x for x in files if x["name"] == "events.jsonl"), None)
        need(events is not None, "native events missing")
        inspection = transport.inspect(p / "native" / "events.jsonl", events["sha256"],
                                       plan["native_cap"])
        need(len(inspection["events"]) == ns.index and inspection["tail"] == ns.previous and
             len(inspection["pending"]) == len(ns.pending), "native journal state mismatch")
        complete = (finish_error is None and not ns.poisoned and not ns.pending and
                    inspection["transport_complete"] and len(roster["rows"]) == 6 and
                    all(r["status"] == "callback_complete_unverified" and r["cleanup"] == "returned"
                        for r in roster["rows"]))
        stage = "terminal"
        end = dict(schema=NAME + ".end", begin_sha256=begin, decision_sha256=decision_sha,
                   native_files=files, native_index=ns.index, native_tail=ns.previous,
                   native_pending=len(ns.pending), native_poisoned=ns.poisoned, native_used=ns.used,
                   transport_complete=inspection["transport_complete"], adapter_receipt=receipt,
                   finish_error=finish_error, rows=[{k: r[k] for k in
                       ("slot", "root", "policy", "status", "cleanup", "error")} for r in roster["rows"]],
                   candidate_complete=complete, physical_certified=False, full_saved_strict=False,
                   scientific_pass=False)
        record(p, "end.json", end, plan)
        if not complete:
            raise RuntimeError("six-slot execution incomplete")
        return end
    except BaseException as exc:
        try:
            record(p, "failure.json", dict(schema=NAME + ".failure", begin_sha256=begin,
                   stage=stage, type=type(exc).__name__, message=str(exc)[:2048],
                   decision_finished=getattr(ds, "finished", None),
                   decision_poisoned=getattr(ds, "poisoned", None),
                   native_poisoned=getattr(ns, "poisoned", None),
                   native_pending=len(ns.pending) if ns is not None else None,
                   roster_returned=roster is not None, retry_authorized=False), plan)
        except BaseException as error:
            exc.add_note("failure record unavailable: " + repr(error))
        raise


def host_runtime(plan, prepared, ds, ns):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.checkpoint_compat import load_compatible_pretrained

    require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    need(str(torch.cuda.get_device_properties(0).uuid) == plan["gpu_uuid"], "runtime UUID")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.autocast(device_type="cuda", enabled=False), \
         torch.autocast(device_type="cpu", enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        instance = adapter.Adapter(model=model, torch=torch, device=device, decision_store=ds,
                                   native_store=ns, gpu_uuid=plan["gpu_uuid"],
                                   allocated_cap=plan["allocated_cap"], reserved_cap=plan["reserved_cap"],
                                   authorization=adapter.AUTH, adapter_sha256=plan["files"]["adapter"],
                                   source_sha256=plan["source_sha256"])
        roster = control.execute_roster(prepared["tasks"], instance.session_factory,
                                        instance.backend_factory, ds)
        try:
            torch.cuda.synchronize(device)
            receipt, error = instance.finish(), None
        except Exception as exc:
            receipt, error = None, type(exc).__name__ + ": " + str(exc)[:2048]
        return receipt, roster, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True, help="SHA256 of canonical JSON plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    path = Path(args.plan)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and
         path.stat().st_size <= 1024 * 1024, "bounded plan file")
    with path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    run(args.output, plan, args.plan_sha256, args.authorization)


if __name__ == "__main__":
    main()
