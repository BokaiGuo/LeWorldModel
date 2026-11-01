"""Opt-in production launcher with hash-bound, refreshed, fail-closed static preflight."""
import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.spectral_reachability import p5_rgb_host_preflight_20261030_codex_v1 as host

wiring, abi = host.wiring, host.abi
NAME = "p5_rgb_launch_preflight_gate_20261030_codex_v1"
HOST_SHA = "623ac9defbdaa022a6c40140ecff417a364d2b4e3ca907feaeaf4c742b917c96"
AUTH = "MAIN_THREAD_APPROVED_PREFLIGHT_BOUND_RGB_CUDA"
PENDING = ["gpu_identity_memory_mig_and_runtime_unverified",
           "external_watchdog_and_run_authorization_unverified"]
DEVICES = {"/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"}


def duration_seconds(value):
    """Conservatively parse a single-unit systemd duration; reject compound/infinity."""
    match = re.fullmatch(r"([0-9]+)(us|ms|s|min|h)", value)
    abi.check(match is not None, "unsupported systemd duration")
    return int(match.group(1)) * {"us": 1e-6, "ms": 1e-3, "s": 1,
                                  "min": 60, "h": 3600}[match.group(2)]


def watchdog(unit, max_seconds, run=subprocess.run, cgroup_path="/proc/self/cgroup"):
    """Require this process to live in an active bounded systemd service cgroup."""
    abi.check(re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit) and
              type(max_seconds) is int and 1 <= max_seconds <= 7200, "watchdog policy")
    properties = ("ControlGroup", "ActiveState", "SubState", "KillMode", "SendSIGKILL",
                  "RuntimeMaxUSec", "TimeoutStopUSec")
    command = ["/usr/bin/systemctl", "show", unit, "--no-pager"] + ["--property=" + p for p in properties]
    result = run(command, capture_output=True, text=True, check=True, timeout=5)
    pairs = [line.split("=", 1) for line in result.stdout.splitlines()]
    abi.check(all(len(pair) == 2 for pair in pairs) and len(pairs) == len(properties),
              "watchdog properties")
    values = dict(pairs)
    abi.check(set(values) == set(properties) and len(values) == len(pairs), "watchdog property schema")
    lines = Path(cgroup_path).read_text().splitlines()
    paths = [line[3:] for line in lines if line.startswith("0::")]
    abi.check(len(paths) == 1 and values["ControlGroup"] not in ("", "/") and
              paths[0] == values["ControlGroup"] and values["ActiveState"] == "active" and
              values["SubState"] == "running" and values["KillMode"] == "control-group" and
              values["SendSIGKILL"] == "yes", "external watchdog cgroup")
    runtime = duration_seconds(values["RuntimeMaxUSec"])
    stop = duration_seconds(values["TimeoutStopUSec"])
    abi.check(0 < runtime <= max_seconds and 0 < stop <= 60, "external watchdog time limits")
    return {"unit": unit, "control_group": paths[0], "runtime_max_seconds": runtime,
            "timeout_stop_seconds": stop, "kill_mode": values["KillMode"],
            "send_sigkill": True}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def age_ok(report, max_age_seconds, now):
    abi.check(type(max_age_seconds) is int and 1 <= max_age_seconds <= 300, "age limit")
    created = datetime.datetime.fromisoformat(report["created_utc"])
    abi.check(created.tzinfo is not None and created.utcoffset() == datetime.timedelta(0), "UTC timestamp")
    age = (now - created).total_seconds()
    abi.check(0 <= age <= max_age_seconds, "expired or future preflight")
    return age


def static_check(report):
    """Only verifies host preconditions; never claims GPU or external approval."""
    abi.check(report["schema"] == host.NAME and report["status"] == "blocked_offline_not_launch_authorization"
              and report["ready_for_launch"] is False and report["P5_admission"] is False and
              report["neural_forward_calls"] == 0 and report["cuda_initializations"] == 0 and
              report["physics_calls"] == 0, "preflight status")
    sources, driver, nodes, disk, runtime = (report[k] for k in
                                            ("source_check", "driver_versions", "device_nodes",
                                             "disk", "python_environment"))
    abi.check(type(disk["min_free_mib"]) is int and 1 <= disk["min_free_mib"] <= 10_000_000 and
              type(disk.get("free_mib")) is int and set(nodes) == DEVICES, "preflight schema")
    computed = host.assess(sources, driver, nodes, disk, disk["min_free_mib"], runtime)
    abi.check(computed == PENDING and report["blockers"] == computed, "static blockers remain")
    abi.check(sources["value"]["wiring_sha256"] == host.WIRING_SHA and
              sources["value"]["baseline_sha256"] == wiring.BASE_SHA and
              sources["value"]["prepared_sha256"] == wiring.runner.PREP_SHA and
              sources["value"]["previous_bridge_sha256"] == host.PREVIOUS_SHA and
              sources["value"]["checkpoint_sha256"] == abi.bridge.FILES["checkpoint"][1] and
              sources["value"]["history_rgb_branches_verified"] == 164 and
              sources["value"]["physical_receipt_and_future_labels_reopened"] is False and
              runtime["value"]["torch_imported"] is False, "source/runtime identities")
    return disk["min_free_mib"]


def load_report(name, expected_sha):
    abi.check(Path(name).name == name and name.startswith(host.NAME + "_") and name.endswith(".json"),
              "report name")
    path = abi.bridge.ROOT / "results" / name
    abi.check(not path.is_symlink() and path.is_file() and path.stat().st_size <= 1024 * 1024, "report file")
    raw = path.read_bytes()
    abi.check(re.fullmatch(r"[0-9a-f]{64}", expected_sha) and abi.bridge.digest(raw) == expected_sha,
              "external preflight SHA")
    report = abi.bridge.read_json(raw)
    abi.check(abi.bridge.canonical(report) == raw, "canonical report")
    return report


def verify(report, max_age_seconds, clock=now_utc, inspect=None):
    abi.check(abi.bridge.digest(Path(host.__file__).read_bytes()) == HOST_SHA, "preflight source SHA")
    age_ok(report, max_age_seconds, clock())
    floor = static_check(report)
    fresh = (inspect or host.inspect)(floor)
    static_check(fresh)
    for key in ("source_check", "driver_versions", "device_nodes", "python_environment"):
        abi.check(report[key] == fresh[key], "changed host/source condition: " + key)
    abi.check(fresh["disk"]["min_free_mib"] == floor, "disk floor changed")
    age = age_ok(report, max_age_seconds, clock())
    return fresh, age


def launch(args, launch_fn=None, inspect=None, clock=now_utc, watchdog_fn=None):
    abi.check(args.authorization == AUTH and wiring.sup.UUID.fullmatch(args.gpu_uuid) and
              re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_suffix) and
              1 <= args.min_free_mib <= args.min_total_mib and 1 <= args.hard_seconds <= 7200 and
              1 <= args.runner_seconds <= 3600, "reviewed launch arguments")
    report = load_report(args.preflight_name, args.preflight_sha)
    fresh, age = verify(report, args.max_age_seconds, clock=clock, inspect=inspect)
    check_watchdog = watchdog_fn or watchdog
    watch = check_watchdog(args.watchdog_unit, args.watchdog_max_seconds)
    results = abi.bridge.ROOT / "results"
    binding = results / (NAME + "_binding_" + args.run_suffix + ".json")
    abi.check(not (results / (wiring.NAME + "_run_" + args.run_suffix)).exists() and
              not (results / (wiring.NAME + "_supervisor_" + args.run_suffix)).exists(), "run already exists")
    receipt = {"schema": NAME + ".binding", "status": "static_verified_runtime_and_authorization_pending",
               "preflight_sha256": args.preflight_sha, "preflight_name": args.preflight_name,
               "host_preflight_source_sha256": HOST_SHA, "wiring_sha256": host.WIRING_SHA,
               "gate_source_sha256": abi.bridge.digest(Path(__file__).read_bytes()),
               "verified_utc": clock().isoformat(), "preflight_age_seconds": age,
               "max_age_seconds": args.max_age_seconds, "fresh_static": fresh,
               "run_suffix": args.run_suffix, "gpu_uuid": args.gpu_uuid,
               "min_free_mib": args.min_free_mib, "min_total_mib": args.min_total_mib,
               "hard_seconds": args.hard_seconds, "runner_seconds": args.runner_seconds,
               "external_watchdog_static_check": watch, "P5_admission": False}
    wiring.gate.write_new(binding, receipt)
    # Refuse after durable binding if report or source changed, or freshness expired.
    abi.check(load_report(args.preflight_name, args.preflight_sha) == report and
              abi.bridge.digest(Path(host.__file__).read_bytes()) == HOST_SHA and
              wiring.source_sha() == host.WIRING_SHA and
              abi.bridge.digest(Path(__file__).read_bytes()) == receipt["gate_source_sha256"],
              "changed after binding")
    age_ok(report, args.max_age_seconds, clock())
    abi.check(check_watchdog(args.watchdog_unit, args.watchdog_max_seconds) == watch,
              "watchdog changed after binding")
    launch_args = SimpleNamespace(authorization=wiring.AUTH, gpu_uuid=args.gpu_uuid, run_suffix=args.run_suffix,
                                  min_free_mib=args.min_free_mib, min_total_mib=args.min_total_mib,
                                  hard_seconds=args.hard_seconds, runner_seconds=args.runner_seconds)
    return (launch_fn or wiring.launch)(launch_args)


def main():
    ap = argparse.ArgumentParser()
    for k in ("authorization", "gpu-uuid", "run-suffix", "preflight-name", "preflight-sha", "watchdog-unit"):
        ap.add_argument("--" + k, required=True)
    for k in ("min-free-mib", "min-total-mib", "hard-seconds", "runner-seconds", "max-age-seconds",
              "watchdog-max-seconds"):
        ap.add_argument("--" + k, type=int, required=True)
    return launch(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
