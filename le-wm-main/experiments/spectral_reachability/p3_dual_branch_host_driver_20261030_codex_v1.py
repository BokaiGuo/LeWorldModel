"""Independent progress/R host driver. No execution without main-thread approval."""
import argparse
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import p3_decision_dual_branch_20261030_codex_v1 as r8
from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v2 as v2

old, need = v2.old, v2.need
ROOT = v2.ROOT
NAME = "p3_dual_branch_host_driver_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_P3_DUAL_BRANCH_HOST_CUDA_V1"
BASE = ROOT / "results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json"
BASE_SHA = "0fa0ef09f27b7f29558c10014e64cc3d940cd208a2cc0819d56163786953e541"
SOURCE_NAMES = (
    "p3_dual_branch_host_driver_20261030_codex_v1.py",
    "p3_decision_dual_branch_20261030_codex_v1.py",
    "agent_p3_r_native_orchestrator_20261029_v1.py",
    "agent_p3_parent_r_native_components_20261029_v1.py",
    "agent_p3_r_spectral8_bank_20261029_v1.py",
    "agent_p3_r_native_group_cuda_20261029_v1.py",
    "p3_decision_host_cuda_driver_20261030_codex_v2.py",
    "p3_cuda_uuid_bridge_20261030_codex_v2.py",
)


def source_hashes():
    r8.r.spectral.check_sources()
    directory = ROOT / "experiments/spectral_reachability"
    return {n: old.file_sha(directory / n) for n in SOURCE_NAMES}


def build_plan(output):
    base = v2.read_json(BASE)
    need(old.file_sha(BASE) == BASE_SHA, "base plan bytes")
    old.validate(base)
    return dict(schema=NAME + ".plan", output=str(output), base_path=str(BASE),
                base_file_sha256=BASE_SHA, sources=source_hashes(),
                decision_cap=4 * 1024**3, decision_reserve=128 * 1024**2,
                capsule_cap=2 * 1024**3, capsule_reserve=64 * 1024**2,
                min_free_mib=6144, min_total_mib=8192, hard_seconds=7200,
                note="engineering caps, sufficiency and full source closure not established")


def validate(plan, expected):
    need(type(plan) is dict and v2.canonical_sha(plan) == expected, "plan SHA")
    need(plan == build_plan(plan["output"]), "fixed plan and listed source pins")
    out = Path(plan["output"])
    need(out.is_absolute() and out.resolve() == out and out.parent == ROOT / "results"
         and out.name.startswith(NAME + "_RUN_") and not out.exists() and not out.is_symlink(),
         "exclusive output")
    base = v2.read_json(BASE)
    need(shutil.disk_usage(out.parent).free >= base["native_cap"] + plan["decision_cap"] +
         plan["capsule_cap"] + base["outer_cap"] + base["floor"], "disk admission")
    return base, old.validate(base)[1]


def finish_dual(store, rows):
    need(type(store) is old.journal.Store and type(rows) is list and len(rows) == 6,
         "dual-branch six-slot terminal")
    store._append(dict(kind="dual_journal_end", rows=rows, pins=old.journal.source_pins()),
                  terminal=True)
    store.finished = True
    return old.file_sha(store.path)


def capsule_store(path, plan, base):
    limits = dict(output_bytes=plan["capsule_cap"], final_reserve_bytes=plan["capsule_reserve"],
                  disk_floor_bytes=base["floor"], wall_seconds=base["wall"],
                  cuda_allocated_bytes=base["allocated_cap"], cuda_reserved_bytes=base["reserved_cap"])
    return r8.r.Store(path, limits)


def runtime(plan, base, prepared, ds, ns, cs, snapshot):
    need(snapshot["uuid"] == base["gpu_uuid"] and snapshot["free_mib"] >= plan["min_free_mib"]
         and snapshot["total_mib"] >= plan["min_total_mib"], "GPU admission")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.checkpoint_compat import load_compatible_pretrained
    require_cuda_allocation("cuda:0")
    device = torch.device("cuda:0")
    view = v2.bridge.TorchView(torch, base["gpu_uuid"], snapshot["uuid"],
                               os.environ.get("CUDA_VISIBLE_DEVICES"))
    old.record(Path(plan["output"]), "uuid_binding.json", view.binding_evidence, base)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
        stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        model = load_compatible_pretrained(prepared["policy"], device=device).eval().requires_grad_(False)
        instance = old.adapter.Adapter(model=model, torch=view, device=device, decision_store=ds,
            native_store=ns, gpu_uuid=base["gpu_uuid"], allocated_cap=base["allocated_cap"],
            reserved_cap=base["reserved_cap"], authorization=old.adapter.AUTH,
            adapter_sha256=base["files"]["adapter"], source_sha256=base["source_sha256"])
        roster = r8.execute_roster(prepared["tasks"], instance.session_factory,
            r8.host_backend_factory(instance), ds,
            lambda slot, stage, cid, call: cs.capsule(slot * 4 + stage, cid, call),
            guard=lambda: instance.guard("dual_native_call"))
        try:
            torch.cuda.synchronize(device)
            receipt, error = instance.finish(), None
        except Exception as exc:
            receipt, error = None, type(exc).__name__ + ": " + str(exc)[:2048]
        return roster, receipt, error


def run(plan, expected, authorization, *, execute=runtime, snapshotter=v2.gpu_snapshot):
    need(authorization == AUTH, "explicit main-thread authorization")
    base, prepared = validate(plan, expected)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == base["gpu_uuid"], "visible UUID")
    snapshot = snapshotter(base["gpu_uuid"])
    need(snapshot["uuid"] == base["gpu_uuid"] and snapshot["free_mib"] >= plan["min_free_mib"]
         and snapshot["total_mib"] >= plan["min_total_mib"], "GPU admission")
    out = Path(plan["output"])
    out.mkdir(mode=0o700)
    old.journal.syncdir(out.parent)
    stage, ds, ns, cs = "begin", None, None, None
    try:
        old.record(out, "begin.json", dict(schema=NAME + ".begin", plan=plan, plan_sha256=expected,
                   gpu=snapshot, external_supervision_verified=False, scientific_pass=False), base)
        stage = "stores"
        ds = old.journal.Store(out / "decision", max_bytes=plan["decision_cap"],
                               terminal_reserve=plan["decision_reserve"])
        ns = old.transport.Store(out / "native", limits=base["native_limits"], cap=base["native_cap"],
             reserve=base["native_reserve"], floor=base["floor"], wall=base["wall"])
        cs = capsule_store(out / "capsules", plan, base)
        stage = "runtime"
        roster, receipt, error = execute(plan, base, prepared, ds, ns, cs, snapshot)
        rows = [{k: row[k] for k in ("slot", "root", "policy", "status", "cleanup", "error")}
                for row in roster["rows"]]
        need([(x["slot"], x["root"], x["policy"]) for x in rows] == r8.ROSTER, "fixed roster")
        stage = "terminal"
        decision_sha = finish_dual(ds, rows)
        cs.close()
        need(source_hashes() == plan["sources"] and old.file_sha(BASE) == BASE_SHA, "source drift")
        complete = (error is None and not ns.poisoned and not ns.pending and ns.index > 0 and
                    all(row["status"] == "callback_complete_not_certified" and
                        row["cleanup"] == "returned" for row in rows))
        old.record(out, "end.json", dict(schema=NAME + ".end", rows=rows, adapter_receipt=receipt,
                   finish_error=error, decision_sha256=decision_sha,
                   native_files=old.inventory(out / "native"), capsule_files=old.inventory(out / "capsules"),
                   native_index=ns.index, native_tail=ns.previous, native_pending=len(ns.pending),
                   native_poisoned=ns.poisoned,
                   candidate_complete=complete, full_saved_strict=False, physical_certified=False,
                   scientific_pass=False), base)
        need(complete, "incomplete fixed roster")
    except BaseException as exc:
        try:
            old.record(out, "failure.json", dict(schema=NAME + ".failure", stage=stage,
                       type=type(exc).__name__, message=str(exc)[:2048], retry_authorized=False,
                       decision_poisoned=getattr(ds, "poisoned", None),
                       native_poisoned=getattr(ns, "poisoned", None)), base)
        except BaseException as secondary:
            exc.add_note("failure record unavailable: " + repr(secondary))
        raise
    finally:
        if cs is not None:
            cs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    run(v2.read_json(args.plan), args.plan_sha256, args.authorization)


if __name__ == "__main__":
    main()
