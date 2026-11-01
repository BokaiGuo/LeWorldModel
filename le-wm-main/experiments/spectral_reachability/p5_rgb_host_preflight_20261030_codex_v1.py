"""Read-only CPU preflight. Never initializes CUDA, runs a model, or launches production."""
import argparse
import datetime
import importlib.metadata
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_instrumented_supervision_20261030_codex_v1 as wiring

abi = wiring.abi
NAME = "p5_rgb_host_preflight_20261030_codex_v1"
WIRING_SHA = "f1f3f1d0bc962b6f153ace297f12da0f6ff63eb990252b96a8a63d62d96ffee7"
PREVIOUS = abi.bridge.ROOT / "results/p5_rgb_causal_panel_bridge_20261030_codex_v1_MAIN01.json"
PREVIOUS_SHA = "559c7c12667e39a320e27151e08c9022ba719f19444b78b08e8a392aed1ba324"


def driver_versions(kernel_path, lib_path, run=subprocess.run):
    """Read version metadata only; modinfo does not initialize a GPU."""
    kernel = Path(kernel_path).read_text().strip()
    target = os.readlink(lib_path)
    match = re.fullmatch(r"libnvidia-ml\.so\.(\d+\.\d+(?:\.\d+)?)", target)
    abi.check(re.fullmatch(r"\d+\.\d+(?:\.\d+)?", kernel) and match, "unrecognized driver version")
    result = run(["modinfo", "-F", "version", "nvidia"], capture_output=True, text=True,
                 check=True, timeout=5)
    disk = result.stdout.strip()
    abi.check(re.fullmatch(r"\d+\.\d+(?:\.\d+)?", disk), "unrecognized module version")
    return {"loaded": kernel, "installed_module": disk, "nvml_link": match.group(1),
            "all_equal": kernel == disk == match.group(1)}


def device_nodes(paths):
    result = {}
    for path in paths:
        try:
            s = os.stat(path, follow_symlinks=False)
            result[path] = {"exists": True, "character_device": stat.S_ISCHR(s.st_mode),
                            "read_write_access_check": os.access(path, os.R_OK | os.W_OK)}
        except OSError as exc:
            result[path] = {"exists": False, "character_device": False,
                            "read_write_access_check": False, "error_type": type(exc).__name__}
    return result


def source_check():
    abi.check(abi.bridge.digest(Path(wiring.__file__).read_bytes()) == WIRING_SHA, "wiring source SHA")
    abi.check(PREVIOUS.is_file() and not PREVIOUS.is_symlink() and
              abi.bridge.digest(PREVIOUS.read_bytes()) == PREVIOUS_SHA, "prior RGB bridge SHA")
    # Existing preflight rehashes named sources/checkpoint and reads only plan-4/5 RGB.
    base = wiring.preflight()
    return {"wiring_sha256": WIRING_SHA, "baseline_sha256": wiring.BASE_SHA,
            "prepared_sha256": wiring.runner.PREP_SHA, "previous_bridge_sha256": PREVIOUS_SHA,
            "checkpoint_sha256": base["checkpoint_sha256"],
            "history_rgb_branches_verified": 164,
            "physical_receipt_and_future_labels_reopened": False}


def assess(sources, driver, devices, disk, disk_min_mib, runtime):
    """Fail closed: static checks are necessary, never sufficient for CUDA."""
    blockers = []
    if sources.get("ok") is not True:
        blockers.append("sealed_source_or_history_check_failed")
    if driver.get("ok") is not True or driver.get("value", {}).get("all_equal") is not True:
        blockers.append("driver_versions_unverified_or_mismatched")
    if not all(v.get("exists") is True and v.get("character_device") is True and
               v.get("read_write_access_check") is True for v in devices.values()) or len(devices) != 3:
        blockers.append("device_nodes_unavailable_or_unverified")
    if disk.get("ok") is not True or disk.get("free_mib", -1) < disk_min_mib:
        blockers.append("disk_floor_unmet_or_unverified")
    if runtime.get("ok") is not True:
        blockers.append("python_environment_unverified")
    blockers.extend(["gpu_identity_memory_mig_and_runtime_unverified",
                     "external_watchdog_and_run_authorization_unverified"])
    return blockers


def attempt(fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}


def inspect(disk_min_mib):
    abi.check(type(disk_min_mib) is int and 1 <= disk_min_mib <= 10_000_000, "explicit disk floor")
    src = attempt(source_check)
    driver = attempt(lambda: driver_versions("/sys/module/nvidia/version",
                                              "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"))
    nodes = device_nodes(("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"))
    def disk_check():
        fs = os.statvfs(abi.bridge.ROOT / "results")
        return (fs.f_bavail * fs.f_frsize) // (1024 * 1024)
    free = attempt(disk_check)
    disk = {"ok": free["ok"], "min_free_mib": disk_min_mib}
    if free["ok"]:
        disk["free_mib"] = free["value"]
    else:
        disk.update(error_type=free["error_type"], error=free["error"])
    def environment():
        torch = importlib.metadata.version("torch")
        python = (abi.bridge.ROOT / ".venv/bin/python").resolve(strict=True)
        abi.check(Path(sys.executable).resolve() == python, "expected project Python")
        abi.check("torch" not in sys.modules, "torch unexpectedly imported")
        return {"python": str(python), "torch_distribution_version": torch, "torch_imported": False}
    runtime = attempt(environment)
    blockers = assess(src, driver, nodes, disk, disk_min_mib, runtime)
    return {"schema": NAME, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "blocked_offline_not_launch_authorization", "ready_for_launch": False,
            "P5_admission": False, "source_check": src, "driver_versions": driver,
            "device_nodes": nodes, "disk": disk, "python_environment": runtime,
            "blockers": blockers, "neural_forward_calls": 0, "cuda_initializations": 0,
            "physics_calls": 0, "gpu_or_process_occupancy": "not_queried",
            "boundary": "current sandbox view only; static checks do not establish GPU availability, free memory or permission"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-disk-free-mib", type=int, required=True)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    name = args.output_name
    abi.check(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json"), "output name")
    result = inspect(args.min_disk_free_mib)
    output = abi.bridge.ROOT / "results" / name
    print(abi.bridge.digest(abi.bridge.canonical(result)), result["status"], result["blockers"])
    wiring.gate.write_new(output, result)
    return 2  # Offline checks cannot grant launch readiness.


if __name__ == "__main__":
    sys.exit(main())
