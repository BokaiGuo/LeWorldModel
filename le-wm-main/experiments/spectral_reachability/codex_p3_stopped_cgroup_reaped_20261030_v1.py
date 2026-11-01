"""Saved-only P3 consumer with invocation-bound cgroup-v2 reaped proof.

Run on the host in the same cgroup namespace as PID 1. This establishes the
named service cgroup is absent, not that processes could not have escaped it.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess

from experiments.spectral_reachability import p3_dual_branch_saved_consumer_20261030_codex_v1 as consumer

need = consumer.need
UNIT = "p3-dual-host-20261030-codex-main01.service"
INVOCATION = "d0faca6af1d64dc9af9ac9e3d82705e2"
PLAN_SHA = "9ac5d68ce7c5f92e4d2351366ef8d068a1ea85d53630794db89ad8637f9ce0f6"
RUN = "/home/gbk/code/LeWorldModel/le-wm-main/results/p3_dual_branch_host_driver_20261030_codex_v1_RUN_MAIN01"
END_SHA = "e9ffcd90d91f7fc37af39a951f21048b70b30d6a9017ae78470c29056d961db5"
PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup",
              "Result", "InvocationID", "ExecMainPID", "ExecMainCode", "ExecMainStatus",
              "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic",
              "ExecStart", "Slice", "KillMode")


def fields(unit, properties, runner):
    result = runner(["systemctl", "--user", "show", unit] +
                    ["--property=" + p for p in properties],
                    capture_output=True, text=True, check=True, timeout=10)
    values = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        need(sep and key not in values, "invalid systemd response")
        values[key] = value
    need(set(values) == set(properties), "incomplete systemd response")
    return values


def verify_snapshot(service, slice_info, cgroup_root, pid1_cgroup, mountinfo, pid1_ns, self_ns):
    """Validate systemd evidence against a host cgroup-v2 mount, no process scan."""
    need(pid1_cgroup.strip() in ("0::/", "0::/init.scope") and
         re.fullmatch(r"cgroup:\[\d+\]", pid1_ns) and pid1_ns == self_ns,
         "must use host PID and cgroup namespace")
    mounts = [line for line in mountinfo.splitlines()
              if " - cgroup2 " in line and line.split(" - ", 1)[0].split()[3:5] == ["/", "/sys/fs/cgroup"]]
    need(len(mounts) == 1 and Path(cgroup_root) == Path("/sys/fs/cgroup"), "host cgroup2 root")
    need(service["Id"] == UNIT and service["LoadState"] == "loaded" and
         service["ActiveState"] in ("failed", "inactive") and service["MainPID"] == "0" and
         service["ControlGroup"] == "" and service["InvocationID"] == INVOCATION and
         service["Slice"] == "app.slice" and service["KillMode"] == "control-group" and
         service["Result"] == "exit-code" and service["ExecMainCode"] == "1" and
         service["ExecMainStatus"] == "1" and service["ExecMainPID"].isdigit() and
         int(service["ExecMainPID"]) > 0 and
         service["ExecMainStartTimestampMonotonic"].isdigit() and
         service["ExecMainExitTimestampMonotonic"].isdigit() and
         0 < int(service["ExecMainStartTimestampMonotonic"]) <
         int(service["ExecMainExitTimestampMonotonic"]), "service invocation/exit state")
    command = service["ExecStart"]
    need(" -m experiments.spectral_reachability.p3_dual_branch_host_driver_20261030_codex_v1 " in command
         and " --plan-sha256 " + PLAN_SHA + " " in command and
         "p3_dual_branch_host_driver_20261030_codex_PREP01.json " in command, "service command binding")
    parent = slice_info["ControlGroup"]
    need(slice_info["Id"] == "app.slice" and slice_info["LoadState"] == "loaded" and
         slice_info["ActiveState"] == "active" and
         re.fullmatch(r"/user\.slice/user-[0-9]+\.slice/user@[0-9]+\.service/app\.slice", parent),
         "systemd parent slice identity")
    relative = Path(parent.lstrip("/"))
    path = Path(cgroup_root)
    for component in relative.parts:
        path /= component
        need(path.is_dir() and not path.is_symlink(), "parent cgroup missing or symlink")
    target = path / UNIT
    need(not os.path.lexists(target), "service cgroup still exists")
    return dict(unit=UNIT, invocation=INVOCATION, plan_sha256=PLAN_SHA,
                parent_cgroup=parent, target_cgroup=str(target), target_absent=True,
                result=service["Result"], exit_pid=service["ExecMainPID"],
                exit_status=service["ExecMainStatus"],
                start_monotonic=service["ExecMainStartTimestampMonotonic"],
                exit_monotonic=service["ExecMainExitTimestampMonotonic"],
                limitations="named cgroup absence only; no proof against escaped or unrelated writers")


def stopped(unit, runner=subprocess.run):
    need(unit == UNIT, "exact run unit required")
    def sample():
        service = fields(unit, PROPERTIES, runner)
        slice_info = fields("app.slice", ("Id", "LoadState", "ActiveState", "ControlGroup"), runner)
        result = verify_snapshot(service, slice_info, Path("/sys/fs/cgroup"),
                                 Path("/proc/1/cgroup").read_text(),
                                 Path("/proc/self/mountinfo").read_text(),
                                 os.readlink("/proc/1/ns/cgroup"), os.readlink("/proc/self/ns/cgroup"))
        return service, slice_info, result
    first, second = sample(), sample()
    need(first == second, "service state changed during cgroup verification")
    return first[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--end-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    need(args.directory == RUN and args.end_sha256 == END_SHA, "exact historical run and terminal SHA")
    result = consumer.consume(args.directory, args.end_sha256, UNIT, 4 * 1024**3,
                              2 * 1024**3, check_service=stopped)
    result["stopped_evidence"] = stopped(UNIT)
    result["original_driver_candidate_complete"] = False
    result["original_driver_status"] = "incomplete_fixed_roster"
    result["scientific_pass"] = False
    out = Path(args.output)
    need(out.is_absolute() and out.resolve() == out and not os.path.lexists(out)
         and out.parent.is_dir(), "exclusive receipt")
    raw = consumer.core.canonical(result) + b"\n"
    with out.open("xb") as stream:
        need(stream.write(raw) == len(raw), "receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    consumer.journal.syncdir(out.parent)
    print(hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
