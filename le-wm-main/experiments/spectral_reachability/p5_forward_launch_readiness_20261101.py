"""Read-only P5 launch checklist; never queries or initializes a GPU."""
import argparse
import json
import os
import re
import stat
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_forward_probe_supervision_20261030_codex_v1 as supervisor

host = supervisor.hostgate.host


def check(suffix, disk_floor, runner_seconds, hard_seconds, service_seconds, overhead_seconds):
    errors = []
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}", suffix):
        raise ValueError("invalid suffix")
    if not (1 <= runner_seconds <= 3600 and 1 <= hard_seconds <= 7200
            and 1 <= service_seconds <= 7200 and overhead_seconds >= 1
            and runner_seconds < hard_seconds and hard_seconds + overhead_seconds <= service_seconds):
        errors.append("deadline_hierarchy_or_overhead_invalid")
    supervision, run = supervisor.paths(suffix)
    acceptance = supervisor.RESULTS / (supervisor.NAME + "_accept_" + suffix + ".json")
    preflight = supervisor.RESULTS / (host.NAME + "_" + suffix + ".json")
    targets = (supervision, run, acceptance, preflight)
    if any(p.exists() or p.is_symlink() for p in targets):
        errors.append("output_path_already_exists")
    results = supervisor.RESULTS
    if not os.access(results, os.W_OK | os.X_OK):
        errors.append("results_directory_not_writable")
    try:
        supervisor.preflight()
        source = {"ok": True}
    except Exception as exc:
        source = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}
        errors.append("probe_contract_or_source_preflight_failed")
    static = host.inspect(disk_floor)
    errors.extend(static["blockers"])
    uid = os.getuid()
    bus = Path(f"/run/user/{uid}/bus")
    try:
        bus_ok = stat.S_ISSOCK(bus.stat().st_mode) and os.access(bus, os.W_OK)
    except OSError:
        bus_ok = False
    if uid != os.geteuid() or not bus_ok:
        errors.append("user_manager_bus_or_uid_unavailable")
    return {"status": "blocked_not_authorization", "errors": errors, "source": source,
            "host_static": static, "user_bus_socket_accessible": bus_ok,
            "output_paths": [str(p) for p in targets], "gpu_queried": False,
            "cuda_initialized": False, "P5_admission": False,
            "boundary": "read-only snapshot; GPU occupancy, service cgroup, quotas and authorization remain unverified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    for key in ("disk-floor", "runner-seconds", "hard-seconds", "service-seconds", "overhead-seconds"):
        parser.add_argument("--" + key, type=int, required=True)
    args = parser.parse_args()
    result = check(args.suffix, args.disk_floor, args.runner_seconds, args.hard_seconds,
                   args.service_seconds, args.overhead_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
