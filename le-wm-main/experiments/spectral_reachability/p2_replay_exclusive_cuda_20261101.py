"""Cooperative, exploratory P2 replay launcher; no physical dispatch.

The lock and census do not provide hardware exclusivity. This entry point
requires a separately coordinated device and an externally pinned inventory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from experiments.spectral_reachability import p2_replay_reserved_runtime_20261101 as runtime
from experiments.spectral_reachability import p2_replay_source_seal_20261101 as seal

frontier = runtime.frontier
LOCK = frontier.PROJECT / "results/e3_p2_host_cuda_cooperative.lock"
CAPS = dict(predict=168, action_encoder=168, jacobian=41,
            vjp_directions=7872, svd=42, solve=41)


def record(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def census(uuid):
    def query(fields, command):
        result = subprocess.run(
            ["nvidia-smi", command + "=" + fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
        return [tuple(part.strip() for part in line.split(",")) for line in result.stdout.splitlines() if line.strip()]

    devices = query("uuid,name,memory.used,memory.total", "--query-gpu")
    processes = query("gpu_uuid,pid,used_gpu_memory", "--query-compute-apps")
    matches = [row for row in devices if len(row) == 4 and row[0] == uuid]
    if len(matches) != 1 or any(len(row) != 3 for row in processes):
        raise ValueError("device census incomplete or UUID not unique")
    if any(row[0] == uuid for row in processes):
        raise ValueError("selected device has active compute processes")
    return {"devices": devices, "compute_processes": processes,
            "selected_uuid": uuid, "hardware_exclusive": False}


def execute(*, inventory, inventory_sha256, output, device_uuid, coordinated):
    out = Path(output)
    if not out.is_absolute() or not out.parent.is_dir() or out.parent.resolve() != out.parent:
        raise ValueError("new absolute output with canonical existing parent required")
    out.mkdir(mode=0o700)  # exclusive, including for failed attempts
    stage = "attempt_record"
    model = None
    torch = None
    before = None
    original_error = None
    checks = {}
    try:
        record(out / "attempt.json", {"status": "started", "inventory": str(inventory),
               "inventory_sha256": inventory_sha256, "device_uuid": device_uuid,
               "physical_authorized": False, "P2_admission": False})
        if not coordinated:
            raise ValueError("explicit operator coordination required")
        if not device_uuid.startswith("GPU-"):
            raise ValueError("physical GPU UUID required")
        stage = "source_preflight"
        inv = seal.verify(inventory, inventory_sha256)
        _, protocol, prepared = frontier.load_prepared(inv["prepared"], inv["prepared_sha256"])
        stage = "gpu_lock"
        with locking.gpu_lock(LOCK):
            try:
                stage = "source_locked"
                seal.verify(inventory, inventory_sha256)
                stage = "device_census"
                checks["device_census"] = census(device_uuid)
                record(out / "device_census.json", checks["device_census"])
                # No torch CUDA initialization before both lock and census.
                if "torch" in __import__("sys").modules and __import__("sys").modules["torch"].cuda.is_initialized():
                    raise ValueError("CUDA already initialized")
                os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
                stage = "model_load"
                import torch as torch_module
                torch = torch_module
                from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as loader
                from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise ValueError("selected CUDA device unavailable or ambiguous")
                device = torch.device("cuda:0")
                torch.backends.cuda.matmul.allow_tf32 = False
                model = loader.load_compatible_pretrained(Path(protocol["context"]["policy"]), device=device).eval().requires_grad_(False)
                before = checked.tensor_state_hash(model)
                checks["model_before"] = before
                stage = "compiled_data"
                source = frontier.read(protocol["context"]["compilation"])
                compiled = {row["root_sha256"]: row["compiled"] for row in source["rows"]}
                cases = protocol["context"]["cases"]
                if len(cases) != 4 or len(compiled) != len(source["rows"]) or any(r["root_sha256"] not in compiled for r in cases):
                    raise ValueError("compiled root identity mismatch")
                for ready in cases:
                    stage = "root_" + str(ready["root_index"])
                    runtime.run_native(model=model, device=device, ready=ready,
                        compiled=compiled[ready["root_sha256"]],
                        provenance_by_arm=lambda bank, ready=ready: {
                            arm: frontier.provenance(protocol, prepared, ready, bank, arm)
                            for arm in runtime.base.ARMS},
                        output=out / stage, caps=CAPS)
                    torch.cuda.synchronize(device)
            except BaseException as exc:
                original_error = exc
            finally:
                if torch is not None:
                    try:
                        torch.cuda.synchronize()
                        checks["cuda_cleanup_synchronized"] = True
                    except BaseException as exc:
                        checks["cleanup_error"] = repr(exc)
                        if original_error is None:
                            original_error = exc
                try:
                    seal.verify(inventory, inventory_sha256)
                    checks["sources_unchanged"] = True
                except BaseException as exc:
                    checks["source_postcheck_error"] = repr(exc)
                    if original_error is None:
                        original_error = exc
                try:
                    if model is not None and before is not None:
                        after = checked.tensor_state_hash(model)
                        frozen = all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
                        checks.update(model_after=after, model_unchanged=after == before, frozen_eval=frozen)
                        if after != before or not frozen:
                            raise ValueError("model state or frozen-eval drift")
                except BaseException as exc:
                    checks["model_postcheck_error"] = repr(exc)
                    if original_error is None:
                        original_error = exc
                if torch is not None:
                    try:
                        model = None
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except BaseException as exc:
                        checks["release_error"] = repr(exc)
                        if original_error is None:
                            original_error = exc
                if original_error is not None:
                    raise original_error
        stage = "complete"
        record(out / "complete.json", {"status": "exploratory_complete", "checks": checks,
               "physical_authorized": False, "P1_admission": False, "P2_admission": False})
        return out
    except BaseException as exc:
        try:
            record(out / "failure.json", {"status": "partial_or_failed", "stage": stage,
                   "error_type": type(exc).__name__, "error": str(exc), "checks": checks,
                   "physical_authorized": False, "P2_admission": False, "time_ns": time.time_ns()})
        except BaseException as persist_error:
            exc.add_note("failure record could not be persisted: " + repr(persist_error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--operator-coordinated", action="store_true")
    args = parser.parse_args()
    execute(inventory=args.inventory, inventory_sha256=args.inventory_sha256,
            output=args.output, device_uuid=args.device_uuid, coordinated=args.operator_coordinated)


if __name__ == "__main__":
    main()
