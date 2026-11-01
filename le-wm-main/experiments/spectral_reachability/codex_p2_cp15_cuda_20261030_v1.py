"""Host-only CUDA driver for the frozen same-feedback exploratory panel."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from experiments.spectral_reachability import codex_p2_cp15_symmetric_spectral_20261030_v1 as panel
from experiments.spectral_reachability import codex_p2_symmetric_h3_rawproducer_20261030_v1 as native
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity

ROOT = panel.ROOT
NAME = "codex_p2_cp15_cuda_20261030_v1"
PREP = ROOT / "results/codex_p2_cp15_symmetric_spectral_20261030_v1_PREP_01.json"
PREP_SHA = "908e830d2a3354c39dce875479be20f6a2bfdcb62676945d64f37d3420cd6fbd"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
GPU = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
ORDER = [(i, arm) for i in range(4) for arm in panel.ARMS]
LIMIT = {**native.LIMITS, "predictor": 39, "pred_proj": 39}
CAP = 128 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material():
    panel.require(sha(PREP) == PREP_SHA and json.loads(PREP.read_bytes()) == panel.build(), "panel prepare")
    feedback = panel.source.source.read(str(panel.source.PATHS["prepare"]), panel.source.source.PINS["prepare"])
    policy = Path(feedback["policy"]).resolve()
    files = [Path(__file__).resolve(), ROOT / "tests/test_codex_p2_cp15_cuda_20261030_v1.py",
             ROOT / "innerdocs/codex_p2_cp15_cuda_20261030_v1.md", PREP,
             ROOT / "experiments/spectral_reachability/codex_p2_cp15_cuda_saved_20261030_v1.py",
             Path(panel.__file__).resolve(), Path(native.__file__).resolve(),
             Path(native.capture.__file__).resolve(), Path(native.bridge.__file__).resolve(),
             Path(lock.__file__).resolve(), Path(journal.__file__).resolve(),
             Path(identity.__file__).resolve(), policy / "weights.pt", policy / "config.json"]
    sources = {r["path"]: r for r in identity.inventory(files)}
    for row in identity.loaded_sources([ROOT]):
        sources[row["path"]] = row
    return dict(schema=NAME+".prepare", panel_sha256=PREP_SHA, order=[list(x) for x in ORDER],
                policy=str(policy), expected_model_hash=feedback["expected_model_hash"],
                sources=sorted(sources.values(), key=lambda x: x["path"]), gpu_uuid=GPU,
                limits=LIMIT, per_slot_wall_seconds=450, equal_total_compute=False,
                dynamic_source_closure="unknown", physical=False, scientific_pass=False)


def write_new(path, value):
    data = journal.encode(value)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
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
    return hashlib.sha256(data).hexdigest()


def path_new(path, prefix):
    path = Path(path)
    panel.require(path.is_absolute() and path.parent == ROOT / "results" and
                  path.name.startswith(NAME + prefix) and not os.path.lexists(path), "exclusive path")
    return path


def verify(path, expected):
    panel.require(sha(path) == expected, "external prepare SHA")
    saved = json.loads(Path(path).read_bytes())
    identity.verify(saved["sources"])
    panel.require(saved == material(), "prepare reconstruction")
    return saved


class Meter(native.Meter):
    def __init__(self, emit, clock):
        super().__init__(emit, clock)
        self.extra_attempted = dict.fromkeys(("predictor", "pred_proj"), 0)
        self.extra_returned = dict.fromkeys(("predictor", "pred_proj"), 0)

    def call(self, kind, fn, details=None):
        elapsed = self.tick()
        if elapsed >= native.WALL or self.attempted[kind] >= LIMIT[kind]:
            raise RuntimeError("wall or resource limit before " + kind)
        self.attempted[kind] += 1
        ordinal = self.attempted[kind]
        self.emit(dict(kind="attempt", resource=kind, ordinal=ordinal, elapsed=elapsed, details=details))
        try:
            value = fn()
        except BaseException as exc:
            self.emit(dict(kind="failure", resource=kind, ordinal=ordinal, error=type(exc).__name__,
                           message=str(exc)[:1000]))
            raise
        elapsed = self.tick()
        self.returned[kind] += 1
        self.emit(dict(kind="return", resource=kind, ordinal=ordinal, elapsed=elapsed))
        if elapsed > native.WALL:
            raise RuntimeError("in-flight wall overrun " + kind)
        return value

    def call_extra(self, kind, fn):
        elapsed = self.tick()
        if elapsed >= native.WALL or self.extra_attempted[kind] >= LIMIT[kind]:
            raise RuntimeError("wall or resource limit before " + kind)
        self.extra_attempted[kind] += 1
        ordinal = self.extra_attempted[kind]
        self.emit(dict(kind="attempt", resource=kind, ordinal=ordinal, elapsed=elapsed, details=None))
        try:
            result = fn()
        except BaseException as exc:
            self.emit(dict(kind="failure", resource=kind, ordinal=ordinal, error=type(exc).__name__,
                           message=str(exc)[:1000]))
            raise
        elapsed = self.tick()
        self.extra_returned[kind] += 1
        self.emit(dict(kind="return", resource=kind, ordinal=ordinal, elapsed=elapsed))
        if elapsed > native.WALL:
            raise RuntimeError("in-flight wall overrun " + kind)
        return result


@contextmanager
def extra_meter(model, meter):
    restored = []
    try:
        for name in ("predictor", "pred_proj"):
            obj = getattr(model, name)
            original = obj.forward
            restored.append((obj, "forward" in vars(obj), vars(obj).get("forward")))
            def wrapper(*args, _f=original, _name=name, **kwargs):
                return meter.call_extra(_name, lambda: _f(*args, **kwargs))
            obj.forward = wrapper
        yield
    finally:
        for obj, own, value in reversed(restored):
            if own:
                obj.forward = value
            else:
                del obj.forward


def run(plan, out, prep_sha):
    panel_data = json.loads(PREP.read_bytes())
    rows = [dict(base=i, arm=arm, status="not_started") for i, arm in ORDER]
    store = journal.Journal(out / "events.jsonl", CAP, 2*1024**2)
    started = time.monotonic()
    report = dict(schema=NAME+".report", status="partial_or_failed", prepare_sha256=prep_sha,
                  rows=rows, error=None, scientific_pass=False, total_flops=None, energy=None,
                  engine_backward=None, source_closure="unknown", physical=False)
    try:
        store.event(dict(kind="start", order=[list(x) for x in ORDER], prepare_sha256=prep_sha))
        with lock.gpu_lock(LOCK) as wait:
            report["lock_wait_seconds"] = wait
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10)
            matches = [x.strip().split(", ") for x in gpu.splitlines() if x.startswith(GPU)]
            panel.require(len(matches) == 1 and int(matches[0][1]) >= 6144, "GPU admission")
            import torch
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v2 as feedback
            host = native.bridge.host.HostCuda()
            load_started = time.monotonic()
            host.start({"context": {"policy": plan["policy"]}})
            torch.cuda.synchronize(host.device)
            report["load_seconds"] = time.monotonic()-load_started
            report["device"] = dict(uuid=GPU, name=torch.cuda.get_device_name(host.device), snapshot=matches[0])
            report["model_before"] = feedback.native.loader.enc.tensor_hash(host.model)
            panel.require(report["model_before"] == plan["expected_model_hash"], "model identity")
            report["loaded_sources"] = identity.loaded_sources([ROOT, Path(torch.__file__).resolve().parents[1]])
            store.event(dict(kind="model_loaded", model_hash=report["model_before"], device=report["device"]))
            torch.backends.cudnn.allow_tf32 = False
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                for index, (base, arm) in enumerate(ORDER):
                    case = panel_data["cases"][base]
                    packets, candidates = [], []
                    def emit(event):
                        event = dict(event)
                        if event["kind"] == "actual_cuda_raw_packet_before_transition":
                            packets.append(event)
                        if event["kind"] == "candidate":
                            panel.require(len(packets) >= 3 and [p["packet"] for p in packets[-3:]] == [0, 1, 2],
                                          "candidate raw packet sequence")
                            event["raw_hex"] = "".join(p["raw_hex"] for p in packets[-3:])
                            candidates.append(event)
                        store.event(dict(kind="slot_event", base=base, arm=arm, event=event))
                    def clock():
                        torch.cuda.synchronize(host.device)
                        panel.require(torch.cuda.memory_allocated(host.device) <= 4*1024**3 and
                                      torch.cuda.memory_reserved(host.device) <= 6*1024**3,
                                      "CUDA memory cap")
                        return time.monotonic()
                    meter = Meter(emit, clock)
                    store.event(dict(kind="slot_start", base=base, arm=arm,
                                     context_sha256=case["feedback_context_sha256"]))
                    try:
                        backend = native.Backend(host.model, host.device, np.eye(192), meter)
                        with native.model_meter(host.model, meter), extra_meter(host.model, meter):
                            # Native kernel checks its own five metered axes.
                            row = panel.execute_arm(backend, case, arm, meter)
                        panel.require(meter.attempted == meter.returned == native.LIMITS and
                                      meter.extra_attempted == meter.extra_returned ==
                                      {"predictor": 39, "pred_proj": 39}, "full axis census")
                        winners = [e for e in candidates[:9] if e["witness"] == row["winner"]]
                        panel.require(len(winners) >= 1 and all(e["raw_hex"] == winners[0]["raw_hex"] for e in winners),
                                      "winner raw identity")
                        raw = winners[0]["raw_hex"]
                        panel.require(all(e["raw_hex"] == raw for e in candidates[-2:]), "replay raw identity")
                        row.update(base=base, arm=arm, status="complete", winner_raw_hex=raw,
                                   extra_attempted=meter.extra_attempted, extra_returned=meter.extra_returned,
                                   winner_raw_sha256=hashlib.sha256(bytes.fromhex(raw)).hexdigest())
                        rows[index] = row
                        store.event(dict(kind="slot_complete", base=base, arm=arm, row=row))
                    except BaseException as exc:
                        rows[index] = dict(base=base, arm=arm, status="failed", error=type(exc).__name__,
                                           message=str(exc)[:1000], attempted=meter.attempted, returned=meter.returned,
                                           extra_attempted=meter.extra_attempted, extra_returned=meter.extra_returned)
                        store.event(dict(kind="slot_failure", row=rows[index]))
                        raise
            host.check_frozen()
            report["model_after"] = feedback.native.loader.enc.tensor_hash(host.model)
            panel.require(report["model_before"] == report["model_after"], "model drift")
            identity.verify(plan["sources"])
            identity.verify(report["loaded_sources"])
            report["status"] = "exploratory_complete"
    except BaseException as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc)[:2048],
                               traceback=traceback.format_exc()[-8000:])
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        store.append(dict(kind="terminal", status=report["status"], error=report["error"]), terminal=True)
        report["journal_tail"] = store.tail
        store.close()
        write_new(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--output", required=True)
    p.add_argument("--prepare")
    p.add_argument("--sha256")
    a = p.parse_args()
    if a.action == "prepare":
        print(write_new(path_new(a.output, "_PREP_"), material()))
    else:
        plan = verify(a.prepare, a.sha256)
        out = path_new(a.output, "_RUN_")
        panel.require(shutil.disk_usage(out.parent).free >= CAP + 512*1024**2, "disk floor")
        out.mkdir(mode=0o700)
        write_new(out / "prepare.json", plan)
        result = run(plan, out, a.sha256)
        print(json.dumps({"status": result["status"], "error": result["error"]}))
        panel.require(result["status"] == "exploratory_complete", "partial retained; no retry")


if __name__ == "__main__":
    main()
