"""Known-root fixed panel host CUDA launcher; exploratory, no physical release."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.spectral_reachability import agent_p3_resume_20261027_next as prior
from experiments.spectral_reachability import codex_p3_earlystop_kernel_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity
from experiments.spectral_reachability import codex_p3_earlystop_saved_20261030_v1 as consumer
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock

ROOT = prior.ROOT
NAME = "codex_p3_earlystop_host_20261030_v1"
PREVIOUS = ROOT / "results/agent_p3_resume_20261027_next_prepare01.json"
PREVIOUS_SHA = "47cc8ff5a32ef3ce36b0dd6807b7c1ff68fea3736137d0e948122d050b2e2846"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
CAP = 512 * 1024**2
RESERVE = 2 * 1024**2
FLOOR = 1024**3
WALL = 1800


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(path, value):
    raw = journal.encode(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("short write")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    d = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(d)
    finally:
        os.close(d)
    return hashlib.sha256(raw).hexdigest()


def material():
    p, tasks, _ = prior.verify(PREVIOUS, PREVIOUS_SHA)
    banks = bridge.banks(prior.dependency().old.BALL)
    for bank in banks:
        for candidate in bank:
            prior.dependency().old.legal(candidate["commands"], candidate["logits"])
    model_dir = Path(p["policy"]).resolve()
    explicit = [Path(__file__).resolve(), Path(consumer.__file__).resolve(),
                Path(identity.__file__).resolve(), Path(journal.__file__).resolve(),
                Path(bridge.__file__).resolve(), PREVIOUS,
                ROOT / "tests/test_codex_p3_earlystop_20261030_v1.py",
                ROOT / "innerdocs/codex_p3_earlystop_protocol_20261030_v1.md",
                ROOT / "tests/test_codex_p3_queued_cost_20261030_v1.py",
                ROOT / "innerdocs/codex_p3_queued_cost_20261030_v1.md",
                ROOT / "experiments/spectral_reachability/codex_p3_queued_cost_batch_20261030_v1.py",
                Path(lock.__file__).resolve(),
                model_dir / "weights.pt", model_dir / "config.json"]
    files = {x["path"]: x for x in identity.inventory(explicit)}
    for row in identity.loaded_sources([ROOT]):
        files[row["path"]] = row
    return dict(schema=NAME + ".prepare", predecessor_sha256=PREVIOUS_SHA,
                task_sha256=[prior.digest(t) for t in tasks],
                bank_sha256=[prior.digest(b) for b in banks],
                model_path=str(model_dir), expected_model_hash=p["expected_model_hash"],
                sources=sorted(files.values(), key=lambda r: r["path"]), gpu_uuid=UUID,
                roster=list(bridge.ARMS), budget_points=list(bridge.POINTS), candidates_per_root=30,
                per_arm_caps={str(n): bridge.caps(n) for n in bridge.POINTS},
                signal_cost="online separately for signal and reverse at each point",
                interpretation="post-observation exploration; stop each root after five consecutive non-improving candidates",
                journal_cap=CAP, terminal_reserve=RESERVE, floor_bytes=FLOOR,
                wall_seconds=WALL, allocated_cap=3*1024**3, reserved_cap=4*1024**3,
                dynamic_source_closure="unknown", known_roots=True, physical_release=False,
                scientific_pass=False), tasks, banks


def verify_prepare(path, sha):
    require(prior.sha(path) == sha, "external prepare sha")
    saved = prior.prior.read(path)
    identity.verify(saved["sources"])
    actual, tasks, banks = material()
    require(saved == actual, "prepare reconstruction/source closure")
    return saved, tasks, banks


def gpu_snapshot():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10)
    entries = [line.strip().split(", ") for line in text.splitlines()]
    matches = [row for row in entries if row[0] == UUID]
    require(len(matches) == 1 and int(matches[0][1]) >= 6144 and int(matches[0][2]) >= 8192, "GPU admission")
    return {"uuid": UUID, "free_mib": int(matches[0][1]), "total_mib": int(matches[0][2])}


def output_path(path, prefix):
    p = Path(path)
    require(p.is_absolute() and p.parent == ROOT / "results" and
            p.name.startswith(NAME + prefix) and not os.path.lexists(p), "exclusive output")
    return p


def child(plan, tasks, banks, out, lock_fd, snapshot, point, arm, lock_wait_seconds):
    st, current = os.fstat(lock_fd), os.stat(LOCK, follow_symlinks=False)
    require((st.st_dev, st.st_ino) == (current.st_dev, current.st_ino), "lock inode")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = None
    start = time.monotonic()
    report = dict(schema=NAME + ".report", status="failed", error=None,
                  prepare_sha256=prior.sha(out / "prepare.json"), gpu_snapshot=snapshot,
                  lock_device=st.st_dev, lock_inode=st.st_ino, model_before=None, model_after=None,
                  point=point, arm=arm, lock_wait_seconds=lock_wait_seconds,
                  source_closure="unknown", physical_release=False, scientific_pass=False)
    model = torch = None
    try:
        store = journal.Journal(out / "events.jsonl", CAP, RESERVE)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = UUID
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.spectral_reachability import p3_cuda_uuid_bridge_20261030_codex_v2 as uuid_bridge
        started = time.monotonic()
        host = require_cuda_allocation("cuda:0")
        report["cuda_preflight_seconds"] = time.monotonic() - started
        view = uuid_bridge.TorchView(torch, UUID, snapshot["uuid"], os.environ["CUDA_VISIBLE_DEVICES"])
        report["device_binding"] = view.binding_evidence
        report["cuda_preflight"] = host
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        io_before = io_counters()
        started = time.monotonic()
        model = load_compatible_pretrained(plan["model_path"], device=device).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        report["model_load_seconds"] = time.monotonic() - started
        io_after = io_counters()
        report["model_load_process_io_delta"] = {k: io_after[k] - io_before[k] for k in io_before}
        enc = prior.dependency().old.enc
        enc.frozen(model)
        report["model_before"] = enc.tensor_hash(model)
        require(report["model_before"] == plan["expected_model_hash"], "model hash")
        report["loaded_python_sources"] = identity.loaded_sources([ROOT, Path(torch.__file__).resolve().parents[1]])
        identity.verify(plan["sources"])
        def limits():
            require(time.monotonic() - start < WALL, "wall cap")
            require(shutil.disk_usage(out).free >= FLOOR, "disk floor")
            require(torch.cuda.max_memory_allocated(device) <= plan["allocated_cap"] and
                    torch.cuda.max_memory_reserved(device) <= plan["reserved_cap"], "cuda memory cap")
        bridge.POINTS, bridge.ARMS = (point,), (arm,)
        def timed_event(event):
            store.event(dict(event, wall_mono_ns=time.monotonic_ns()))
        started = time.monotonic()
        with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            bridge.execute_host(model, device, tasks, timed_event,
                                limits, lambda: torch.cuda.synchronize(device))
        torch.cuda.synchronize(device)
        report["algorithm_seconds_including_journal"] = time.monotonic() - started
        enc.frozen(model)
        report["model_after"] = enc.tensor_hash(model)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        require(report["model_after"] == report["model_before"], "model drift")
        identity.verify(plan["sources"])
        identity.verify(report["loaded_python_sources"])
        report["status"] = "panel_completed_unverified"
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048],
                           "traceback": traceback.format_exc()[-12000:]}
        try:
            if store is not None:
                store.append({"kind": "host_failure", "error": report["error"]}, terminal=True)
        except BaseException as failure:
            report["journal_failure"] = repr(failure)
    finally:
        report["journal_tail"] = store.tail if store else None
        report["journal_events"] = store.index if store else 0
        report["elapsed_seconds"] = time.monotonic() - start
        if store is not None:
            store.close()
        report["report_sha256"] = None
        write_new(out / "report.json", report)
    return report


def io_counters():
    with open("/proc/self/io", encoding="ascii") as f:
        values = dict(line.split(":", 1) for line in f)
    return {k: int(values[k]) for k in ("rchar", "read_bytes")}


def launch(plan_path, sha, out, point, arm):
    require(point in (20, 30) and arm in ("uniform", "signal", "reverse"), "fixed slot")
    plan, tasks, banks = verify_prepare(plan_path, sha)
    out = output_path(out, "_RUN_")
    require(shutil.disk_usage(out.parent).free >= CAP + FLOOR, "disk admission")
    # Parent does not initialize CUDA and holds the existing lock through child exit.
    with lock.gpu_lock(LOCK) as lock_wait_seconds:
        snapshot = gpu_snapshot()
        out.mkdir(mode=0o700)
        write_new(out / "prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                # Reopen the inherited locked file description via /proc is not equivalent;
                # lock is held by the parent for our complete lifetime.
                fds = [int(x) for x in os.listdir("/proc/self/fd") if x.isdigit()]
                fd = next(fd for fd in fds if fd > 2 and _same_lock(fd))
                report = child(plan, tasks, banks, out, fd, snapshot, point, arm, lock_wait_seconds)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(status)
    require(code == 0, "child failed; preserve output")
    return out


def _same_lock(fd):
    try:
        a, b = os.fstat(fd), os.stat(LOCK, follow_symlinks=False)
        return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "consume"))
    parser.add_argument("--output")
    parser.add_argument("--prepare")
    parser.add_argument("--sha256")
    parser.add_argument("--run")
    parser.add_argument("--point", type=int, choices=(20, 30))
    parser.add_argument("--arm", choices=("uniform", "signal", "reverse"))
    a = parser.parse_args()
    if a.action == "prepare":
        value, _, _ = material()
        path = output_path(a.output, "_PREP_")
        print(json.dumps({"prepare": str(path), "sha256": write_new(path, value)}))
    elif a.action == "run":
        print(json.dumps({"run": str(launch(a.prepare, a.sha256, a.output, a.point, a.arm))}))
    else:
        out = Path(a.run)
        report = json.loads((out / "report.json").read_bytes())
        require(prior.sha(out / "report.json") == a.sha256, "external report sha")
        require(report["status"] == "panel_completed_unverified", "incomplete report")
        plan, tasks, banks = verify_prepare(out / "prepare.json", report["prepare_sha256"])
        checked = consumer.consume(out / "events.jsonl", report["journal_tail"], CAP, tasks, banks,
                                   report["point"], report["arm"])
        if a.output:
            path = output_path(a.output, "_CONSUME_")
            print(json.dumps({"saved": str(path), "sha256": write_new(path, checked)}))
        else:
            print(json.dumps({"status": checked["status"], "arms": len(checked["rows"])}))


if __name__ == "__main__":
    main()
