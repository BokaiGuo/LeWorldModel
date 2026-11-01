"""Source-bound cooperative HOST CUDA runner for the matched P2 comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.spectral_reachability import codex_p2_program_independent_prepare_20261030_v1 as prep
from experiments.spectral_reachability import codex_p2_program_library_supervisor_20261030_v1 as supervisor
from experiments.spectral_reachability import codex_p2_program_model_scope_20261030_v1 as scope
from experiments.spectral_reachability import codex_p2_multistart_joint_cuda_20261030_v1 as adapter
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import p2_matched_program_kernel_20261101_parallel as kernel

ROOT = prep.ROOT
NAME = "p2_matched_program_host_20261101_parallel"
CAP = 128 * 1024**2
PARENT = ROOT / "results/codex_p2_program_independent_prepare_20261030_v1_PREP_02.json"
PARENT_SHA = "eb81fecc0508a995321994015780fb7088e7ffa16cfb50957891f91e6802f621"


def material():
    kernel.old.require(
        prep.original.previous.sha(PARENT) == PARENT_SHA, "pinned parent prepare"
    )
    parent = json.loads(PARENT.read_bytes())
    identity.verify(parent["sources"])
    current = prep.material()
    kernel.old.require(
        {key: value for key, value in parent.items() if key != "sources"}
        == {key: value for key, value in current.items() if key != "sources"},
        "fixed parent contract and case reconstruction",
    )
    current_sources = {row["path"]: row for row in current["sources"]}
    kernel.old.require(
        all(current_sources.get(row["path"]) == row for row in parent["sources"]),
        "pinned parent source subset",
    )
    paths = [
        Path(__file__).resolve(),
        Path(kernel.__file__).resolve(),
        Path(scope.__file__).resolve(),
        Path(adapter.__file__).resolve(),
        Path(supervisor.__file__).resolve(),
        ROOT
        / "experiments/spectral_reachability/p2_matched_program_saved_20261101_parallel.py",
        ROOT / "tests/test_p2_matched_program_20261101_parallel.py",
    ]
    return dict(
        schema=NAME + ".prepare",
        parent_prepare=str(PARENT),
        parent_sha256=PARENT_SHA,
        parent=parent,
        protocol=kernel.contract(),
        own_sources=identity.inventory(paths),
        physical=False,
        scientific_pass=False,
    )


def prepare(path):
    path = Path(path)
    kernel.old.require(
        path.is_absolute()
        and path.parent == ROOT / "results"
        and path.name.startswith(NAME + "_PREP_")
        and not os.path.lexists(path),
        "new exclusive prepare",
    )
    return prep.original.previous.write_new(path, material())


def verify(path, expected):
    path = Path(path)
    kernel.old.require(
        path.is_absolute()
        and not path.is_symlink()
        and prep.original.previous.sha(path) == expected,
        "prepare SHA",
    )
    value = json.loads(path.read_bytes())
    identity.verify(value["own_sources"])
    kernel.old.require(
        value == material(), "reconstruct complete fixed preparation"
    )
    return value


def inherited_lock(path):
    st = os.stat(path, follow_symlinks=False)
    for item in Path("/proc/self/fd").iterdir():
        try:
            fd = int(item.name)
            current = os.fstat(fd)
            if fd > 2 and (current.st_dev, current.st_ino) == (
                st.st_dev,
                st.st_ino,
            ):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return dict(device=st.st_dev, inode=st.st_ino, fd=fd)
        except (ValueError, OSError):
            continue
    raise ValueError("no inherited locked cooperative inode")


def child(prepared, expected, output):
    proof = inherited_lock(prep.original.LOCK)
    manifest = verify(prepared, expected)
    plan = manifest["parent"]
    panel = json.loads(Path(plan["panel_path"]).read_bytes())
    order = kernel.contract()["order"]
    rows = [dict(base=base, arm=arm, status="not_started") for base, arm in order]
    out = Path(output)
    events = journal.Journal(out / "child_events.jsonl", CAP, 2 * 1024**2)
    result = dict(
        schema=NAME + ".report",
        status="partial_or_failed",
        prepare_sha256=expected,
        rows=rows,
        lock_proof=proof,
        physical=False,
        P1_admission=False,
        P2_admission=False,
        measured_resource_match=False,
        equal_total_compute=False,
        total_flops=None,
        energy=None,
        dynamic_source_closure="unknown",
        error=None,
    )
    started = time.monotonic()
    try:
        events.event(
            dict(
                kind="child_start",
                prepare_sha256=expected,
                order=order,
                lock_proof=proof,
            )
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
        snapshot = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        matches = [
            line.split(", ", 2)
            for line in snapshot.splitlines()
            if line.startswith(plan["gpu_uuid"] + ", ")
        ]
        kernel.old.require(
            len(matches) == 1
            and len(matches[0]) == 3
            and int(matches[0][2]) >= 6144,
            "actual UUID and available memory before CUDA",
        )
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as hostmodule
        from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v2 as feedback

        host = hostmodule.HostCuda()
        load_started = time.monotonic()
        host.start({"context": {"policy": plan["policy"]}})
        torch.cuda.synchronize(host.device)
        kernel.old.require(
            torch.cuda.device_count() == 1
            and torch.cuda.get_device_name(host.device) == matches[0][1],
            "visible CUDA device matches GPU identity",
        )
        result["load_seconds"] = time.monotonic() - load_started
        result["device"] = dict(
            uuid=plan["gpu_uuid"],
            name=torch.cuda.get_device_name(host.device),
            nvidia_smi_snapshot=matches[0],
        )
        result["model_before"] = feedback.native.loader.enc.tensor_hash(host.model)
        kernel.old.require(
            result["model_before"] == plan["expected_model_hash"],
            "loaded model hash",
        )
        result["loaded_sources"] = identity.loaded_sources(
            [ROOT, Path(torch.__file__).resolve().parents[1]]
        )
        events.event(
            dict(
                kind="model_loaded",
                device=result["device"],
                model_sha256=result["model_before"],
            )
        )
        torch.backends.cudnn.allow_tf32 = False

        def clock():
            torch.cuda.synchronize(host.device)
            kernel.old.require(
                torch.cuda.memory_allocated(host.device) <= 4 * 1024**3
                and torch.cuda.memory_reserved(host.device) <= 6 * 1024**3,
                "CUDA memory cap",
            )
            return time.monotonic()

        with sdpa_kernel(SDPBackend.MATH), torch.autocast(
            device_type="cuda", enabled=False
        ):
            for index, (base, arm) in enumerate(order):
                case = panel["cases"][base]
                try:
                    row = kernel.run_branch(
                        case,
                        arm,
                        lambda meter: adapter.MultistartCudaAdapter(
                            host.model, host.device, meter
                        ),
                        events.event,
                        clock=clock,
                        model_scope=lambda meter: scope.model_scope(host.model, meter),
                    )
                    name = f"branch_{index}.json"
                    digest = prep.original.previous.write_new(out / name, row)
                    rows[index] = dict(
                        base=base,
                        arm=arm,
                        status="complete",
                        file=name,
                        sha256=digest,
                        resource_counts=row["resource_counts"],
                    )
                    events.event(
                        dict(
                            kind="branch_file",
                            index=index,
                            file=name,
                            sha256=digest,
                        )
                    )
                except BaseException as exc:
                    rows[index] = dict(
                        base=base,
                        arm=arm,
                        status="failed",
                        error_type=type(exc).__name__,
                        message=str(exc)[:1000],
                    )
                    events.event(
                        dict(kind="branch_failure", index=index, row=rows[index])
                    )
                    raise
        host.check_frozen()
        result["model_after"] = feedback.native.loader.enc.tensor_hash(host.model)
        kernel.old.require(
            result["model_after"] == result["model_before"], "frozen model drift"
        )
        identity.verify(plan["sources"])
        identity.verify(manifest["own_sources"])
        identity.verify(result["loaded_sources"])
        vectors = [row["resource_counts"]["attempted"] for row in rows]
        kernel.old.require(
            vectors and all(vector == kernel.CAPS for vector in vectors),
            "all rows exact measured vector",
        )
        result["measured_resource_match"] = True
        result["status"] = "exploratory_complete"
    except BaseException as exc:
        result["error"] = dict(
            type=type(exc).__name__,
            message=str(exc)[:2000],
            traceback=traceback.format_exc()[-8000:],
        )
    finally:
        result["wall_seconds"] = time.monotonic() - started
        events.append(
            dict(
                kind="child_terminal",
                status=result["status"],
                error=result["error"],
            ),
            terminal=True,
        )
        result["journal_tail"] = events.tail
        events.close()
        prep.original.previous.write_new(out / "report.json", result)
    return 0 if result["status"] == "exploratory_complete" else 1


def run(prepared, expected, output):
    verify(prepared, expected)
    out = Path(output)
    kernel.old.require(
        out.is_absolute()
        and out.parent == ROOT / "results"
        and out.name.startswith(NAME + "_RUN_")
        and not os.path.lexists(out)
        and shutil.disk_usage(out.parent).free >= CAP + 512 * 1024**2,
        "exclusive output and disk",
    )
    out.mkdir(mode=0o700)
    events = journal.Journal(
        out / "supervisor_events.jsonl", 4 * 1024**2, 128 * 1024
    )
    try:
        command = [
            sys.executable,
            "-B",
            "-m",
            "experiments.spectral_reachability." + NAME,
            "child",
            "--prepared",
            str(prepared),
            "--sha256",
            expected,
            "--output",
            str(out),
        ]
        try:
            code = supervisor.supervise(prep.original.LOCK, command, events.event)
            events.append(
                dict(kind="supervisor_terminal", returncode=code), terminal=True
            )
            return code
        except BaseException as exc:
            events.append(
                dict(
                    kind="supervisor_failure",
                    error_type=type(exc).__name__,
                    message=str(exc)[:1000],
                ),
                terminal=True,
            )
            raise
    finally:
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "child"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        print(
            json.dumps(
                dict(
                    sha256=prepare(args.output),
                    status="prepared_not_run",
                )
            )
        )
    elif args.action == "child":
        sys.exit(child(args.prepared, args.sha256, args.output))
    else:
        sys.exit(run(args.prepared, args.sha256, args.output))


if __name__ == "__main__":
    main()
