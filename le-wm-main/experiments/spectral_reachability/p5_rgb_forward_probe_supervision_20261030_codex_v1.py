"""Host-gated, externally supervised CUDA forward probe and saved-only acceptance."""
import argparse
import datetime
import math
import os
import re
import signal
import subprocess
from pathlib import Path

from experiments.spectral_reachability import p5_rgb_forward_recursive_probe_20261030_codex_v1 as probe
from experiments.spectral_reachability import p5_rgb_launch_preflight_gate_20261030_codex_v1 as hostgate

bridge, runner, wiring = probe.bridge, probe.runner, probe.wiring
sup = wiring.sup
NAME = "p5_rgb_forward_probe_supervision_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_SUPERVISED_FORWARD_PROBE_CUDA"
PROBE_SHA = "ab5b2b3b9e384110ff9578cd84c51613e5cb63289aa2bbd0fd5b949ca6e99e61"
HOSTGATE_SHA = "f23ce5fae14fc0c7d0e379feafb80a574054396e138982015600b665011d4ab5"
SUP_SHA = "bb5063f84fe4c410839a458c3fcc36d5966649172b745e6509a0ccc529faba75"
PREP_NAME = probe.NAME + "_PREP02.json"
PREP_SHA = "f0c3aca32dd7a267c43295e26a455212fe374201a5cdaf4b42ccd24dc48c8c27"
RESULTS = bridge.ROOT / "results"
PROBE_MODULE = "experiments.spectral_reachability." + probe.NAME


def need(value, message):
    if not value:
        raise ValueError(message)


def source_sha():
    return bridge.digest(Path(__file__).read_bytes())


def preflight():
    for module, expected in ((probe, PROBE_SHA), (hostgate, HOSTGATE_SHA), (sup, SUP_SHA)):
        path = Path(module.__file__)
        need(path.is_file() and not path.is_symlink() and bridge.digest(path.read_bytes()) == expected,
             "source pin " + path.name)
    base, prepared = probe.source_preflight()
    path = probe.target(PREP_NAME, ".json")
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024, "contract path")
    raw = path.read_bytes()
    need(bridge.digest(raw) == PREP_SHA and raw == bridge.canonical(probe.manifest(prepared)), "PREP02 binding")
    return base, prepared


def paths(suffix):
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}", suffix) is not None, "suffix")
    need(RESULTS.resolve() == RESULTS and not RESULTS.is_symlink(), "results path")
    return RESULTS / (NAME + "_supervisor_" + suffix), probe.target(probe.NAME + "_RUN_" + suffix, "")


def command(suffix, seconds):
    return [str(bridge.ROOT / ".venv/bin/python"), "-m", PROBE_MODULE, "run",
            "--authorization", probe.AUTH, "--contract-name", PREP_NAME, "--contract-sha", PREP_SHA,
            "--output-name", probe.NAME + "_RUN_" + suffix, "--max-seconds", str(seconds)]


def checked_review(directory, base, prepared, journal_sha=None):
    path = directory / "events.jsonl"
    need(directory.is_dir() and not directory.is_symlink() and path.is_file() and not path.is_symlink()
         and path.stat().st_size <= 8 * 1024 * 1024, "journal path")
    raw = path.read_bytes()
    sha = bridge.digest(raw)
    need(journal_sha is None or sha == journal_sha, "external journal SHA")
    return probe.review(raw, sha, prepared, PREP_SHA, base)


def user_watchdog(unit, max_seconds, run=subprocess.run, cgroup_path="/proc/self/cgroup", uid=None):
    """Check the exact service of this process in the current UID's user manager."""
    uid = os.getuid() if uid is None else uid
    need(type(uid) is int and uid > 0 and re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit) and
         type(max_seconds) is int and 1 <= max_seconds <= 7200, "user watchdog policy")
    properties = ("ControlGroup", "ActiveState", "SubState", "KillMode", "SendSIGKILL",
                  "RuntimeMaxUSec", "TimeoutStopUSec")
    command = ["/usr/bin/systemctl", "--user", "show", unit, "--no-pager"] + [
        "--property=" + p for p in properties]
    runtime_dir = f"/run/user/{uid}"
    env = dict(os.environ, XDG_RUNTIME_DIR=runtime_dir, DBUS_SESSION_BUS_ADDRESS=f"unix:path={runtime_dir}/bus")
    result = run(command, capture_output=True, text=True, check=True, timeout=5, env=env)
    pairs = [line.split("=", 1) for line in result.stdout.splitlines()]
    need(len(pairs) == len(properties) and all(len(pair) == 2 for pair in pairs), "user watchdog properties")
    values = dict(pairs)
    need(len(values) == len(pairs) and set(values) == set(properties), "user watchdog property schema")
    paths = [line[3:] for line in Path(cgroup_path).read_text().splitlines() if line.startswith("0::")]
    prefix = f"/user.slice/user-{uid}.slice/user@{uid}.service/"
    group = values["ControlGroup"]
    need(len(paths) == 1 and group.startswith(prefix) and group.endswith("/" + unit) and
         paths[0] == group and values["ActiveState"] == "active" and values["SubState"] == "running"
         and values["KillMode"] == "control-group" and values["SendSIGKILL"] == "yes",
         "user watchdog identity/cgroup")
    runtime = hostgate.duration_seconds(values["RuntimeMaxUSec"])
    stop = hostgate.duration_seconds(values["TimeoutStopUSec"])
    need(0 < runtime <= max_seconds and 0 < stop <= 60, "user watchdog time limits")
    return {"manager": "user", "uid": uid, "unit": unit, "control_group": group,
            "runtime_max_seconds": runtime, "timeout_stop_seconds": stop, "kill_mode": "control-group",
            "send_sigkill": True}


def static_binding(args):
    report = hostgate.load_report(args.preflight_name, args.preflight_sha)
    fresh, age = hostgate.verify(report, args.max_age_seconds)
    need(os.getuid() == os.geteuid(), "real/effective UID mismatch")
    watch = user_watchdog(args.watchdog_unit, args.watchdog_max_seconds)
    return report, fresh, age, watch


def launch(args):
    need(args.authorization == AUTH and sup.UUID.fullmatch(args.gpu_uuid) and
         1 <= args.min_free_mib <= args.min_total_mib and 1 <= args.hard_seconds <= 7200 and
         1 <= args.runner_seconds <= 3600 and 1 <= args.max_age_seconds <= 300 and
         1 <= args.watchdog_max_seconds <= 7200, "launch arguments")
    supervision, run_dir = paths(args.run_suffix)
    need(not run_dir.exists() and not run_dir.is_symlink(), "run already exists")
    base, prepared = preflight()
    report, fresh, age, watch = static_binding(args)
    code_sha = source_sha()
    supervision.mkdir()
    runner.sync_dir(RESULTS)
    log = supervision / "supervisor.jsonl"
    sup.emit(log, {"event": "created", "producer_sha256": code_sha, "probe_sha256": PROBE_SHA,
                   "contract_sha256": PREP_SHA, "baseline_sha256": wiring.BASE_SHA,
                   "hostgate_sha256": HOSTGATE_SHA, "supervisor_source_sha256": SUP_SHA,
                   "preflight_name": args.preflight_name, "preflight_sha256": args.preflight_sha,
                   "max_age_seconds": args.max_age_seconds, "verified_utc": hostgate.now_utc().isoformat(),
                   "preflight_age_seconds": age, "fresh_static": fresh, "watchdog": watch,
                   "watchdog_max_seconds": args.watchdog_max_seconds, "gpu_uuid": args.gpu_uuid,
                   "run_suffix": args.run_suffix, "min_free_mib": args.min_free_mib,
                   "min_total_mib": args.min_total_mib, "hard_seconds": args.hard_seconds,
                   "runner_seconds": args.runner_seconds, "retry": False})
    runner.sync_dir(supervision)
    attempted, execution = False, None
    try:
        gpu = sup.query_gpu(args.gpu_uuid)
        sup.admit(gpu, args.min_free_mib, args.min_total_mib)
        sup.emit(log, {"event": "admitted", "snapshot": gpu})
        need(source_sha() == code_sha and hostgate.load_report(args.preflight_name, args.preflight_sha) == report,
             "source/report changed")
        hostgate.age_ok(report, args.max_age_seconds, hostgate.now_utc())
        need(os.getuid() == os.geteuid() and user_watchdog(args.watchdog_unit, args.watchdog_max_seconds) == watch,
             "watchdog changed")
        preflight()
        need(source_sha() == code_sha and hostgate.load_report(args.preflight_name, args.preflight_sha) == report,
             "source/report changed after preflight")
        hostgate.age_ok(report, args.max_age_seconds, hostgate.now_utc())
        need(os.getuid() == os.geteuid() and user_watchdog(args.watchdog_unit, args.watchdog_max_seconds) == watch,
             "watchdog changed after preflight")
        cmd = command(args.run_suffix, args.runner_seconds)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(bridge.ROOT))
        sup.emit(log, {"event": "launch_attempt", "command": cmd,
                       "cuda_visible_devices": args.gpu_uuid, "cuda_device_order": "PCI_BUS_ID"})
        attempted = True
        with (supervision / "stdout.log").open("xb") as out, (supervision / "stderr.log").open("xb") as err:
            runner.sync_dir(supervision)
            def interrupt(signum, frame):
                raise InterruptedError(f"signal {signum}")
            old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                execution = sup.supervise(cmd, env, bridge.ROOT, out, err, args.hard_seconds,
                                          lambda pid: sup.emit(log, {"event": "launched", "pid": pid}))
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            for f in (out, err):
                f.flush()
                os.fsync(f.fileno())
        status = {"status": "partial", "execution": execution, "retry": False,
                  "writer_may_be_alive": execution["returncode"] is None}
        if execution["status"] == "exited" and execution["returncode"] == 0:
            need(math.isfinite(execution["elapsed_seconds"]) and 0 <= execution["elapsed_seconds"] <= args.hard_seconds
                 and source_sha() == code_sha, "deadline/source")
            preflight()
            status.update(status="complete_self_report_only", review=checked_review(run_dir, base, prepared))
    except Exception as exc:
        status = {"status": "partial_or_not_started", "retry": False, "error_type": type(exc).__name__,
                  "error": str(exc)[:500], "writer_may_be_alive": bool(attempted and
                    (execution is None or execution["returncode"] is None))}
    sup.emit(log, {"event": "finished", **status})
    print(status["status"])
    return 0 if status["status"] == "complete_self_report_only" else 2


def validate_supervisor(events, suffix, code_sha, report):
    need([e.get("event") for e in events] == ["created", "admitted", "launch_attempt", "launched", "finished"],
         "supervisor sequence")
    created, admitted, call, launched, finished = events
    need(created["producer_sha256"] == code_sha and created["probe_sha256"] == PROBE_SHA and
         created["contract_sha256"] == PREP_SHA and created["baseline_sha256"] == wiring.BASE_SHA and
         created["hostgate_sha256"] == HOSTGATE_SHA and created["supervisor_source_sha256"] == SUP_SHA and
         created["run_suffix"] == suffix and created["retry"] is False and finished["retry"] is False and
         type(created["runner_seconds"]) is int and 1 <= created["runner_seconds"] <= 3600 and
         type(created["hard_seconds"]) is int and 1 <= created["hard_seconds"] <= 7200 and
         sup.UUID.fullmatch(created["gpu_uuid"]) is not None and call["command"] == command(suffix, created["runner_seconds"])
         and call["cuda_visible_devices"] == created["gpu_uuid"] and call["cuda_device_order"] == "PCI_BUS_ID",
         "supervisor identity")
    hostgate.static_check(report)
    hostgate.static_check(created["fresh_static"])
    for key in ("source_check", "driver_versions", "device_nodes", "python_environment"):
        need(created["fresh_static"][key] == report[key], "fresh host mismatch")
    need(created["fresh_static"]["disk"]["min_free_mib"] == report["disk"]["min_free_mib"], "disk floor")
    verified = datetime.datetime.fromisoformat(created["verified_utc"])
    age = hostgate.age_ok(report, created["max_age_seconds"], verified)
    need(abs(age - created["preflight_age_seconds"]) < 5.0, "preflight age")
    watch = created["watchdog"]
    need(set(watch) == {"manager", "uid", "unit", "control_group", "runtime_max_seconds", "timeout_stop_seconds",
                        "kill_mode", "send_sigkill"} and watch["manager"] == "user" and
         type(watch["uid"]) is int and watch["uid"] > 0 and
         re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", watch["unit"]) and
         isinstance(watch["control_group"], str) and
         watch["control_group"].startswith(f"/user.slice/user-{watch['uid']}.slice/user@{watch['uid']}.service/")
         and watch["control_group"].endswith("/" + watch["unit"])
         and watch["kill_mode"] == "control-group" and watch["send_sigkill"] is True
         and type(created["watchdog_max_seconds"]) is int and 1 <= created["watchdog_max_seconds"] <= 7200
         and 0 < watch["runtime_max_seconds"] <= created["watchdog_max_seconds"]
         and 0 < watch["timeout_stop_seconds"] <= 60, "watchdog evidence")
    snapshot = admitted["snapshot"]
    need(snapshot["uuid"] == created["gpu_uuid"] and snapshot["mig_mode"] in ("Disabled", "N/A", "[N/A]")
         and snapshot["compute_mode"] in ("Default", "Exclusive_Process"), "GPU identity")
    sup.admit(snapshot, created["min_free_mib"], created["min_total_mib"])
    execution = finished["execution"]
    need(finished["status"] == "complete_self_report_only" and finished["writer_may_be_alive"] is False and
         execution["status"] == "exited" and execution["returncode"] == 0 and execution["pid"] == launched["pid"]
         and type(execution["elapsed_seconds"]) in (int, float) and math.isfinite(execution["elapsed_seconds"])
         and 0 <= execution["elapsed_seconds"] <= created["hard_seconds"], "completion")
    return finished["review"]


def accept(args):
    supervision, run_dir = paths(args.run_suffix)
    need(supervision.is_dir() and not supervision.is_symlink(), "supervisor path")
    base, prepared = preflight()
    raw = wiring.acc.read_new_file(supervision / "supervisor.jsonl", args.supervisor_sha)
    need(raw.endswith(b"\n"), "supervisor truncated")
    lines = raw.splitlines()
    events = [bridge.read_json(line) for line in lines]
    need(all(bridge.canonical(e).rstrip(b"\n") == line for e, line in zip(events, lines, strict=True)),
         "supervisor canonical")
    need(events and events[0]["preflight_sha256"] == args.preflight_sha, "preflight external SHA")
    report = hostgate.load_report(events[0]["preflight_name"], args.preflight_sha)
    expected = validate_supervisor(events, args.run_suffix, source_sha(), report)
    reviewed = checked_review(run_dir, base, prepared, args.journal_sha)
    need(reviewed == expected, "review mismatch")
    need(re.fullmatch(re.escape(NAME + "_accept_") + r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}\.json",
                      args.output_name), "accept namespace")
    result = {"schema": NAME + ".accept", "status": "saved_self_report_consistency_not_scientific",
              "producer_sha256": source_sha(), "probe_sha256": PROBE_SHA, "contract_sha256": PREP_SHA,
              "preflight_sha256": args.preflight_sha, "supervisor_sha256": args.supervisor_sha,
              "review": reviewed, "P5_admission": False,
              "boundary": "self-reported execution; finite differences not theoretical causality or fresh confirmation"}
    print(wiring.gate.write_new(RESULTS / args.output_name, result))
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    for k in ("authorization", "gpu-uuid", "run-suffix", "preflight-name", "preflight-sha", "watchdog-unit"):
        p.add_argument("--" + k, required=True)
    for k in ("min-free-mib", "min-total-mib", "hard-seconds", "runner-seconds", "max-age-seconds",
              "watchdog-max-seconds"):
        p.add_argument("--" + k, type=int, required=True)
    p = sub.add_parser("accept")
    for k in ("run-suffix", "preflight-sha", "supervisor-sha", "journal-sha", "output-name"):
        p.add_argument("--" + k, required=True)
    args = parser.parse_args()
    return {"launch": launch, "accept": accept}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
