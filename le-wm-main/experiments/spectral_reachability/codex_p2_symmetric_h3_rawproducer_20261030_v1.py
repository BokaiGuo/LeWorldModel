"""Four-known-root H3 projected symmetric-work exploration, HOST CUDA only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import codex_p2_symmetric_native_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p2_h3_rawcapture_backend_20261030_v1 as capture
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as gpu_lock

NAME = "codex_p2_symmetric_h3_rawproducer_20261030_v1"
ACK = "H3_KNOWN_ROOT_EXPLORATORY_CUDA_NOT_PREREGISTERED"
CAP = .75 * math.sqrt(15)
ALPHAS = (.25, .5, 1., 2.)
WALL = 450.
LIMITS = dict(packet=39, action_encoder=39, predict=39, jacobian=2, svd=2)
ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/"results/e3_p2_host_cuda_cooperative.lock"


def contract():
    return dict(schema=NAME+".contract", horizon=3, root_count=4,
                order=[[i, a] for i in range(4) for a in
                       (("joint", "suffix") if i % 2 == 0 else ("suffix", "joint"))],
                objective="mean((Q @ (terminal-goal))**2)", projection="original per-root Q",
                logit_l2_cap=CAP, candidate_projection="radial inward to joint L2 ball with floating-point inward correction",
                damping=.01, step_l2_cap=1., alphas=list(ALPHAS), iterations=2,
                suffix_masks=[[0, 10], [0, 20]], limits=LIMITS, wall_seconds=WALL,
                cuda_allocated_limit=4*1024**3, cuda_reserved_limit=6*1024**3,
                nominal=1, jacobian=2, svd=2, candidates=8, full_replays=2,
                finite_schedule_no_padding=True, known_roots=True, preregistered=False,
                equal_total_compute=False, total_flops=None, training=False,
                physical=False, P1_admission=False, P2_admission=False,
                actual_raw_capture="each CUDA packet saved before transition",
                cooperative_lock=str(LOCK))


def ball(u):
    a = np.asarray(u, np.float64)
    if a.shape != (30,) or not np.isfinite(a).all():
        raise ValueError("finite 30 logits required")
    n = np.linalg.norm(a)
    a = a.copy() if n <= CAP else a * (np.nextafter(CAP, 0.) / n)
    for _ in range(32):
        if np.linalg.norm(a) <= CAP:
            return a
        a *= np.nextafter(1., 0.)
    raise ValueError("unable to round inward")


def suffix_ball(candidate, cut):
    a = np.asarray(candidate, np.float64).copy()
    remaining = math.sqrt(max(0., CAP**2 - float(a[:cut] @ a[:cut])))
    n = np.linalg.norm(a[cut:])
    if n > remaining:
        a[cut:] *= np.nextafter(remaining, 0.)/n
    for _ in range(32):
        if np.linalg.norm(a) <= CAP:
            return a
        a[cut:] *= np.nextafter(1., 0.)
    raise ValueError("unable to round suffix inward")


def digest_file(path):
    return bridge.host.frontier.file_sha(Path(path))


def source_bindings():
    m, p, cases = bridge.known_roots()
    files = [Path(__file__), Path(bridge.__file__),
             ROOT/"experiments/spectral_reachability/codex_p2_symmetric_budget_20261030_v1.py",
             Path(capture.__file__), Path(gpu_lock.__file__),
             ROOT/"experiments/spectral_reachability/codex_p2_h3_physical_preflight_20261030_v1.py",
             ROOT/"tests/test_codex_p2_h3_rawcapture_backend_20261030_v1.py"]
    return m, p, cases, {str(f.resolve()): digest_file(f) for f in files}


def prepare(path):
    m, p, cases, sources = source_bindings()
    value = dict(contract=contract(), upstream_prepare_sha256=bridge.PREP_SHA,
                 upstream_source_bindings=m["source_bindings"], own_sources=sources,
                 protocol_sha256=bridge.host.frontier.digest(p),
                 roots=[{k: c[k] for k in ("root_index", "root_sha256", "root_id", "context_sha256",
                                          "goal_sha256", "projection_sha256", "action_domain")}
                        for c in cases])
    path = Path(path)
    bridge.host.need(path.is_absolute() and path.name.startswith(NAME+"_") and
                     path.parent.resolve() == path.parent and not os.path.lexists(path), "exclusive prepare")
    bridge.host.bank._write(path, value)
    return digest_file(path)


def verify_prepare(path, sha):
    path = Path(path)
    bridge.host.need(path.is_absolute() and not path.is_symlink() and digest_file(path) == sha, "prepare identity")
    saved = bridge.host.frontier.read(path)
    m, p, cases, sources = source_bindings()
    expected = dict(contract=contract(), upstream_prepare_sha256=bridge.PREP_SHA,
                    upstream_source_bindings=m["source_bindings"], own_sources=sources,
                    protocol_sha256=bridge.host.frontier.digest(p),
                    roots=[{k: c[k] for k in ("root_index", "root_sha256", "root_id", "context_sha256",
                                             "goal_sha256", "projection_sha256", "action_domain")} for c in cases])
    bridge.host.need(saved == expected, "prepare/source drift")
    return p, cases


class Journal:
    def __init__(self, path):
        self.f = Path(path).open("x", encoding="utf-8")
        self.prev = "0"*64
        self.n = 0
    def emit(self, item):
        entry = dict(index=self.n, previous=self.prev, event=item)
        raw = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
        entry["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        self.f.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.prev = entry["sha256"]
        self.n += 1
    def close(self):
        self.f.close()


class Meter:
    def __init__(self, emit, clock):
        self.emit, self.clock = emit, clock
        self.start = self.last = clock()
        self.attempted = dict.fromkeys(LIMITS, 0)
        self.returned = dict.fromkeys(LIMITS, 0)
    def tick(self):
        now = self.clock()
        if not math.isfinite(now) or now < self.last:
            raise ValueError("non-monotone clock")
        self.last = now
        return now-self.start
    def call(self, kind, fn, details=None):
        elapsed = self.tick()
        if elapsed >= WALL or self.attempted[kind] >= LIMITS[kind]:
            raise RuntimeError("wall or resource limit before "+kind)
        self.attempted[kind] += 1
        self.emit(dict(kind="attempt", resource=kind, ordinal=self.attempted[kind],
                       elapsed=elapsed, details=details))
        try:
            value = fn()
        except BaseException as exc:
            self.emit(dict(kind="failure", resource=kind, ordinal=self.attempted[kind],
                           error=type(exc).__name__, message=str(exc)[:1000], elapsed=self.tick()))
            raise
        elapsed = self.tick()
        self.returned[kind] += 1
        self.emit(dict(kind="return", resource=kind, ordinal=self.attempted[kind], elapsed=elapsed))
        if elapsed > WALL:
            raise RuntimeError("in-flight wall overrun "+kind)
        return value


@contextmanager
def model_meter(model, meter):
    targets = [(model.action_encoder, "forward", "action_encoder"), (model, "predict", "predict")]
    restored = []
    try:
        for owner, name, resource in targets:
            original = getattr(owner, name)
            restored.append((owner, name, name in vars(owner), vars(owner).get(name)))
            def wrapped(*args, _original=original, _resource=resource, **kwargs):
                return meter.call(_resource, lambda: _original(*args, **kwargs))
            setattr(owner, name, wrapped)
        yield
    finally:
        for owner, name, existed, old in reversed(restored):
            if existed: setattr(owner, name, old)
            else: delattr(owner, name)


def run_arm(backend, case, arm, meter):
    q = np.asarray(case["projection"], np.float64)
    target = q @ np.asarray(case["goal"], np.float64)
    context = case["context"]
    def evaluate(u, tag):
        contexts, terminal = backend.rollout(context, u)
        score = float(np.mean((terminal-target)**2))
        if not math.isfinite(score): raise ValueError("nonfinite score")
        witness = dict(tag=tag, logits=u.tolist(), contexts=contexts,
                       projected_terminal=terminal.tolist(), cost=score)
        meter.emit(dict(kind="candidate", witness=witness))
        return witness
    best = evaluate(np.zeros(30), "nominal")
    for iteration in range(2):
        u = np.asarray(best["logits"], np.float64)
        j = meter.call("jacobian", lambda: backend.jacobian(context, u),
                       dict(shape=[192,30], mathematical_vjp_rows_requested=192))
        j = np.asarray(j, np.float64)
        if j.shape != (192,30) or not np.isfinite(j).all(): raise ValueError("invalid J")
        mask = np.ones(30)
        if arm == "suffix": mask[:10 if iteration == 0 else 20] = 0
        left, s, right = meter.call("svd", lambda: np.linalg.svd(j*mask, full_matrices=False),
                                    dict(shape=[192,30]))
        step = right.T @ ((s/(s*s+.01)) * (left.T @ (target-np.asarray(best["projected_terminal"]))))
        step *= mask
        norm = np.linalg.norm(step)
        if norm > 1.: step /= norm
        for alpha in ALPHAS:
            # Preserve the suffix prefix exactly; scale only the variable suffix
            # to the remaining radius when the candidate crosses the ball.
            candidate = u + alpha*step
            if arm == "suffix":
                cut = 10 if iteration == 0 else 20
                candidate = suffix_ball(candidate, cut)
            else:
                candidate = ball(candidate)
            bridge.validate_logits(candidate, CAP)
            trial = evaluate(candidate, f"i{iteration}_a{alpha}")
            if trial["cost"] < best["cost"]: best = trial
    replays = [evaluate(np.asarray(best["logits"]), "replay"+str(i)) for i in range(2)]
    if any(r["contexts"] != best["contexts"] or r["projected_terminal"] != best["projected_terminal"]
           for r in replays):
        raise ValueError("complete replay mismatch")
    if meter.attempted != LIMITS or meter.returned != LIMITS:
        raise ValueError("actual operation census mismatch")
    return dict(winner=best, replays=replays, attempted=meter.attempted,
                returned=meter.returned, elapsed_seconds=meter.tick())


class Backend(capture.RawCaptureBackend):
    def __init__(self, model, device, projection, meter):
        self.meter = meter
        super().__init__(model, device, projection, meter.emit, CAP)
        native = self.transition
        self.transition = lambda *args: meter.call("packet", lambda: native(*args))
    def jacobian(self, context, u):
        # Outer run_arm meters the actual jacobian call.
        return super().jacobian(context, u)


def run(prepared, sha, output, ack):
    if ack != ACK: raise ValueError("explicit exploratory authorization required")
    protocol, cases = verify_prepare(prepared, sha)
    output = Path(output)
    bridge.host.need(output.is_absolute() and output.parent.resolve() == output.parent and
                     output.name.startswith(NAME+"_") and not os.path.lexists(output), "exclusive output")
    bridge.host.need(shutil.disk_usage(output.parent).free >= 512*1024**2, "disk floor")
    output.mkdir(mode=0o700)
    journal = Journal(output/"events.jsonl")
    report = dict(schema=NAME+".run", status="failed_or_partial", prepare_sha256=sha,
                  rows=[], equal_total_compute=False, total_flops=None,
                  P1_admission=False, P2_admission=False, physical=False)
    t0 = time.monotonic()
    try:
        journal.emit(dict(kind="start", prepare_sha256=sha, contract=contract()))
        host = bridge.host.HostCuda()
        host.start(protocol)
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.backends.cudnn.allow_tf32 = False
        before = host.model_hash()
        report.update(host=host.host, model_hash_before=before, load_seconds=time.monotonic()-t0)
        journal.emit(dict(kind="model_loaded", model_hash=before))
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for root, arm in contract()["order"]:
                case = cases[root]
                emit = lambda event, root=root, arm=arm: journal.emit(dict(root=root, arm=arm, event=event))
                def clock():
                    torch.cuda.synchronize(host.device)
                    if torch.cuda.memory_allocated(host.device) > 4*1024**3 or torch.cuda.memory_reserved(host.device) > 6*1024**3:
                        raise RuntimeError("CUDA allocator cap")
                    return time.monotonic()
                meter = Meter(emit, clock)
                emit(dict(kind="arm_start", root_sha256=case["root_sha256"],
                          context_sha256=case["context_sha256"], goal_sha256=case["goal_sha256"],
                          projection_sha256=case["projection_sha256"]))
                try:
                    backend = Backend(host.model, host.device, case["projection"], meter)
                    with model_meter(host.model, meter):
                        row = run_arm(backend, case, arm, meter)
                    row.update(root=root, arm=arm, status="complete")
                except BaseException as exc:
                    row = dict(root=root, arm=arm, status="failed", error=type(exc).__name__,
                               message=str(exc)[:1000], attempted=meter.attempted,
                               returned=meter.returned, elapsed_seconds=meter.tick())
                    emit(dict(kind="arm_failure", error=row["error"], message=row["message"]))
                    report["rows"].append(row)
                    raise
                report["rows"].append(row)
                emit(dict(kind="arm_complete", attempted=meter.attempted, returned=meter.returned))
                host.check_frozen()
                if host.model_hash() != before: raise ValueError("model state drift")
        verify_prepare(prepared, sha)
        report["model_hash_after"] = host.model_hash()
        report["status"] = "exploratory_complete"
    except BaseException as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        report["elapsed_seconds"] = time.monotonic()-t0
        try: journal.emit(dict(kind="terminal", status=report["status"]))
        finally: journal.close()
        report["journal_sha256"] = digest_file(output/"events.jsonl")
        bridge.host.bank._write(output/"report.json", report)
    return report


def audit(directory, expected_report_sha):
    directory = Path(directory)
    report_path = directory/"report.json"
    bridge.host.need(digest_file(report_path) == expected_report_sha, "external report hash")
    report = bridge.host.frontier.read(report_path)
    bridge.host.need(report["status"] == "exploratory_complete" and len(report["rows"]) == 8,
                     "complete fixed denominator required")
    bridge.host.need(digest_file(directory/"events.jsonl") == report["journal_sha256"], "journal hash")
    previous = "0"*64
    events = []
    with (directory/"events.jsonl").open() as stream:
        for n, line in enumerate(stream):
            item = json.loads(line)
            h = item.pop("sha256")
            raw = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
            bridge.host.need(item["index"] == n and item["previous"] == previous and
                             hashlib.sha256(raw.encode()).hexdigest() == h, "journal chain")
            previous = h
            events.append(item["event"])
    bridge.host.need(events[0]["kind"] == "start" and events[0]["contract"] == contract()
                     and events[-1] == dict(kind="terminal", status="exploratory_complete"), "journal boundaries")
    bridge.host.need(report["model_hash_before"] == report["model_hash_after"], "model state drift")
    for row, (root, arm) in zip(report["rows"], contract()["order"]):
        bridge.host.need((row["root"], row["arm"], row["status"]) == (root, arm, "complete"), "slot order")
        sub = [e["event"] for e in events if e.get("root") == root and e.get("arm") == arm]
        for kind, cap in LIMITS.items():
            for phase in ("attempt", "return"):
                numbers = [x["ordinal"] for x in sub if x["kind"] == phase and x["resource"] == kind]
                bridge.host.need(numbers == list(range(1, cap+1)), "actual journal operation census")
        bridge.host.need(row["attempted"] == row["returned"] == LIMITS and
                         row["elapsed_seconds"] <= WALL, "row budget")
        candidates = [x["witness"] for x in sub if x["kind"] == "candidate"]
        bridge.host.need(len(candidates) == 11 and candidates[-2:] == row["replays"] and
                         row["winner"] in candidates[:9], "candidate and replay census")
        for x in candidates:
            bridge.validate_logits(x["logits"], CAP)
        bridge.host.need(all(r["contexts"] == row["winner"]["contexts"] and
                             r["projected_terminal"] == row["winner"]["projected_terminal"]
                             for r in row["replays"]), "saved replay equality")
    return dict(status="journal_and_saved_replays_checked", roots=4, slots=8,
                model_hash=report["model_hash_after"], equal_total_compute=False,
                independent_neural_reexecution=False, P2_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "run", "audit"))
    p.add_argument("--output", required=True)
    p.add_argument("--prepared")
    p.add_argument("--sha")
    p.add_argument("--ack")
    a = p.parse_args()
    if a.mode == "prepare": print(json.dumps(dict(sha256=prepare(a.output))))
    elif a.mode == "run":
        with gpu_lock.gpu_lock(LOCK):
            pid = os.fork()
            if pid == 0:
                try:
                    run(a.prepared, a.sha, a.output, a.ack)
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            _, status = os.waitpid(pid, 0)
            bridge.host.need(os.waitstatus_to_exitcode(status) == 0, "CUDA child failed")
        r = bridge.host.frontier.read(Path(a.output)/"report.json")
        print(json.dumps(dict(status=r["status"], rows=len(r["rows"]), error=r.get("error"))))
    else:
        print(json.dumps(audit(a.output, a.sha), sort_keys=True))


if __name__ == "__main__":
    main()
