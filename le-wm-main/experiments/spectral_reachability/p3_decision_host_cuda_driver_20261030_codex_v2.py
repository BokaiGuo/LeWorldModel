"""Versioned UUID-aware P3 driver; execution requires independent authorization."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v1 as old
from experiments.spectral_reachability import p3_cuda_uuid_bridge_20261030_codex_v2 as bridge

NAME = "p3_decision_host_cuda_driver_20261030_codex_v2"
AUTH = "MAIN_THREAD_APPROVED_P3_DECISION_HOST_CUDA_V2"
ROOT = Path(__file__).resolve().parents[2]
need = old.need


def canonical_sha(value):
    return hashlib.sha256(old.control.core.canonical(value)).hexdigest()


def read_json(path):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and p.is_file() and not p.is_symlink()
         and p.stat().st_size <= 1024 * 1024, "bounded canonical JSON path")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    with p.open(encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=pairs,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid JSON constant")))


def source_pins(plan):
    paths = {"driver_v2": Path(__file__), "bridge": Path(bridge.__file__),
             "legacy_driver": Path(old.__file__), "gpu_preflight": ROOT / "experiments/gpu_preflight.py",
             "checkpoint_compat": ROOT / "experiments/checkpoint_compat.py"}
    need(type(plan["source_files"]) is dict and set(plan["source_files"]) == set(paths), "source file set")
    for key, path in paths.items():
        need(old.file_sha(path) == plan["source_files"][key], "v2 source pin: " + key)


def preflight(plan, expected_sha):
    need(type(plan) is dict and set(plan) == {"schema", "source_files", "base_plan_path",
         "base_plan_file_sha256", "base_plan_sha256", "output", "min_free_mib",
         "min_total_mib", "hard_seconds", "coordination"}, "v2 plan keys")
    need(plan["schema"] == NAME + ".plan" and canonical_sha(plan) == expected_sha, "v2 plan SHA")
    source_pins(plan)
    need(type(plan["coordination"]) is str and len(plan["coordination"].strip()) >= 8, "coordination")
    need(type(plan["min_free_mib"]) is int and type(plan["min_total_mib"]) is int and
         0 < plan["min_free_mib"] <= plan["min_total_mib"] and
         type(plan["hard_seconds"]) is int and 1 <= plan["hard_seconds"] <= 7200, "external limits")
    base_path = Path(plan["base_plan_path"])
    need(base_path == ROOT / "results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json",
         "fixed base plan location")
    need(old.file_sha(base_path) == plan["base_plan_file_sha256"], "base plan bytes")
    base = read_json(base_path)
    need(canonical_sha(base) == plan["base_plan_sha256"], "base plan canonical SHA")
    _, prepared, _ = old.validate(base)
    out = Path(plan["output"])
    need(out.is_absolute() and out.resolve() == out and out.parent == ROOT / "results" and
         out.name.startswith(NAME + "_RUN_") and not out.exists() and not out.is_symlink(),
         "new exclusive v2 output")
    need(shutil.disk_usage(out.parent).free >= base["native_cap"] + base["decision_cap"] +
         base["outer_cap"] + base["floor"], "combined disk admission")
    return base, prepared


def gpu_snapshot(uuid, runner=subprocess.run):
    bridge.canonical_physical(uuid)
    def query(fields):
        result = runner(["nvidia-smi", "-i", uuid, "--query-gpu=" + fields,
                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        check=True, timeout=10)
        lines = result.stdout.strip().splitlines()
        need(len(lines) == 1, "single nvidia-smi result")
        return [s.strip() for s in lines[0].split(",")]
    info, mig = query("uuid,memory.total,memory.free,compute_mode"), query("mig.mode.current")
    need(len(info) == 4 and info[0] == uuid and info[1].isdigit() and info[2].isdigit() and
         info[3] in ("Default", "Exclusive_Process") and mig in (["Disabled"], ["N/A"], ["[N/A]"]),
         "nvidia-smi UUID or mode")
    total, free = int(info[1]), int(info[2])
    need(0 <= free <= total, "nvidia-smi memory values")
    return dict(uuid=uuid, total_mib=total, free_mib=free, compute_mode=info[3], mig_mode=mig[0])


def runtime_with(plan, base, prepared, ds, ns, *, torch, allocation, loader, sdp_context, snapshot):
    """Run with injected host dependencies; real entry constructs these only after admission."""
    need(snapshot["uuid"] == base["gpu_uuid"] and snapshot["free_mib"] >= plan["min_free_mib"]
         and snapshot["total_mib"] >= plan["min_total_mib"], "GPU free memory admission")
    allocation("cuda:0")
    device = torch.device("cuda:0")
    view = bridge.TorchView(torch, base["gpu_uuid"], snapshot["uuid"],
                            os.environ.get("CUDA_VISIBLE_DEVICES"))
    old.record(Path(plan["output"]), "uuid_binding_v2.json",
               dict(schema=NAME + ".uuid_binding", snapshot=snapshot, binding=view.binding_evidence,
                    plan_sha256=canonical_sha(plan), scientific_pass=False), base)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
        stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
        stack.enter_context(sdp_context())
        model = loader(prepared["policy"], device=device).eval().requires_grad_(False)
        instance = old.adapter.Adapter(model=model, torch=view, device=device, decision_store=ds,
            native_store=ns, gpu_uuid=base["gpu_uuid"], allocated_cap=base["allocated_cap"],
            reserved_cap=base["reserved_cap"], authorization=old.adapter.AUTH,
            adapter_sha256=base["files"]["adapter"], source_sha256=base["source_sha256"])
        roster = old.control.execute_roster(prepared["tasks"], instance.session_factory,
                                            instance.backend_factory, ds)
        try:
            torch.cuda.synchronize(device)
            receipt = instance.finish()
            source_pins(plan)
            receipt["uuid_binding_v2"] = view.binding_evidence
            error = None
        except Exception as exc:
            receipt, error = None, type(exc).__name__ + ": " + str(exc)[:2048]
        return receipt, roster, error


def host_runtime(plan, base, prepared, ds, ns):
    source_pins(plan)
    snapshot = gpu_snapshot(base["gpu_uuid"])
    need(snapshot["free_mib"] >= plan["min_free_mib"] and
         snapshot["total_mib"] >= plan["min_total_mib"], "GPU free memory admission")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.checkpoint_compat import load_compatible_pretrained
    return runtime_with(plan, base, prepared, ds, ns, torch=torch, allocation=require_cuda_allocation,
                        loader=load_compatible_pretrained, sdp_context=lambda: sdpa_kernel(SDPBackend.MATH),
                        snapshot=snapshot)


def run(plan, expected_sha, authorization):
    need(authorization == AUTH, "v2 explicit authorization")
    base, _ = preflight(plan, expected_sha)
    def runtime(b, prepared, ds, ns):
        old.record(Path(plan["output"]), "v2_binding.json",
                   dict(schema=NAME + ".binding", plan=plan, plan_sha256=expected_sha,
                        external_supervision_verified=False, scientific_pass=False), b)
        return host_runtime(plan, b, prepared, ds, ns)
    return old.run(plan["output"], base, plan["base_plan_sha256"], old.AUTH, runtime=runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.preflight_only:
        preflight(plan, args.plan_sha256)
        snapshot = gpu_snapshot(read_json(plan["base_plan_path"])["gpu_uuid"])
        need(snapshot["free_mib"] >= plan["min_free_mib"] and
             snapshot["total_mib"] >= plan["min_total_mib"], "GPU free memory admission")
        print(json.dumps(dict(plan_sha256=args.plan_sha256, gpu=snapshot,
                              note="snapshot only; not a reservation"), sort_keys=True))
    else:
        run(plan, args.plan_sha256, args.authorization)


if __name__ == "__main__":
    main()
