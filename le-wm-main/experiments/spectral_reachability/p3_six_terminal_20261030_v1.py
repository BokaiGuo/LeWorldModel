"""Exclusive six-slot terminal snapshot; not native/scientific certification."""
import hashlib
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as transport
from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control

base = transport.base
need = base.need
NAME = "p3_six_terminal_20261030_v1"
REPORT = "six_report.json"
COMPLETION = "six_completion.json"


def source_contract():
    """Extend the prior source closure without asserting preregistration."""
    from experiments.spectral_reachability import p3_six_owned_constructor_20261030_v1 as owner
    paths = (Path(__file__).resolve(), Path(__file__).with_name("p3_six_native_saved_20261030_v1.py").resolve())
    need(all(p.is_file() and not p.is_symlink() for p in paths), "terminal source files")
    return dict(schema=NAME + ".source", previous=owner.source_contract(),
                added_sha256={str(p): file_sha(p) for p in paths},
                host_runner_ready=False, full_saved_strict=False, scientific_pass=False)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(directory, cap):
    files = sorted(directory.iterdir())
    need(all(p.is_file() and not p.is_symlink() for p in files), "flat regular inventory")
    need(sum(p.stat().st_size for p in files) <= cap, "inventory cap")
    return [dict(name=p.name, bytes=p.stat().st_size, sha256=file_sha(p)) for p in files]


def roster(events):
    """Recover fixed denominator and actual lifecycle prefix without filling gaps."""
    rows = [dict(slot=s, root=r, policy=p, status="not_started", result_sha256=None,
                 cleanup="not_required", error=None) for s, r, p in control.ROSTER]
    marker = [e for e in events if e["origin"] == "physical" and e["resource"] in
              ("six_denominator", "six_slot_attempt", "six_slot_return")]
    if not marker:
        return rows
    first = marker[0]
    need(first["resource"] == "six_denominator" and first["kind"] == "record" and
         first["detail"] == dict(rows=rows, fixed_denominator=6), "fixed denominator")
    current = None
    next_slot = 0
    for e in marker[1:]:
        need(e["kind"] == "record", "lifecycle record kind")
        if e["resource"] == "six_slot_attempt":
            need(current is None and next_slot < 6 and e["detail"] ==
                 {k: rows[next_slot][k] for k in ("slot", "root", "policy")}, "ordered attempt")
            current = next_slot
            rows[current]["status"] = "attempted_unsettled"
        else:
            need(current is not None, "return without attempt")
            row = e["detail"]
            need(type(row) is dict and set(row) == set(rows[current]) and
                 all(row[k] == rows[current][k] for k in ("slot", "root", "policy")) and
                 row["status"] in ("callback_feedback_complete_not_certified", "failed",
                                   "native_stopped", "native_initial_stop", "engineering_failed") and
                 row["cleanup"] in ("not_required", "returned", "failed") and
                 (row["result_sha256"] is None or
                  (type(row["result_sha256"]) is str and len(row["result_sha256"]) == 64 and
                   all(c in "0123456789abcdef" for c in row["result_sha256"]))) and
                 (row["error"] is None or type(row["error"]) is str), "returned row")
            rows[current] = row
            next_slot += 1
            current = None
    return rows


def _write_terminal(store, name, value):
    data = base.raw(value)
    need(len(data) <= 4 * 1024**2 and store.final_bytes + len(data) <= store.reserve and
         sum(x["bytes"] for x in inventory(store.output, store.cap)) + len(data) <= store.cap and
         shutil.disk_usage(store.output).free >= store.floor + len(data), "terminal reserve and floor")
    path = store.output / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            need(stream.write(data) == len(data), "complete terminal write")
            stream.flush()
            os.fsync(stream.fileno())
        base.syncdir(store.output)
    except BaseException:
        store.poisoned = True
        raise
    store.final_bytes += len(data)
    return base.fsha(data)


def publish(store):
    """Publish one fsynced report and completion; snapshots, never certifies."""
    need(isinstance(store, transport.Store) and not store.sealed, "owned unsealed store")
    need(not (store.output / REPORT).exists() and not (store.output / COMPLETION).exists(), "exclusive terminal")
    store.sealed = True
    files = inventory(store.output, store.cap)
    path = store.output / "events.jsonl"
    if path.exists():
        sha = file_sha(path)
        journal = transport.inspect(path, sha, sum(store.limits[k] for k in
                                    ("physical_events", "numerical_events", "encoding_events")))
    else:
        sha = None
        journal = dict(events=[], tail=None, pending={}, trailing_bytes=0, trailing_sha256=None,
                       transport_complete=False)
    if not store.poisoned:
        need(journal["tail"] == store.previous and len(journal["events"]) == store.index and
             journal["pending"] == store.pending and journal["trailing_bytes"] == 0,
             "live journal state")
    pending = [dict(origin=k[0], token=k[1], resource=v) for k, v in sorted(journal["pending"].items())]
    elapsed = store.clock() - store.started
    need(0 <= elapsed, "nonnegative elapsed")
    report = dict(schema=NAME + ".report", rows=roster(journal["events"]), inventory=files,
                  journal_sha256=sha, event_count=len(journal["events"]), journal_tail=journal["tail"],
                  pending=pending, trailing_bytes=journal["trailing_bytes"],
                  trailing_sha256=journal["trailing_sha256"], store_poisoned=store.poisoned,
                  store_used_provisional=store.used, limits=store.limits, cap=store.cap,
                  reserve=store.reserve, floor=store.floor, wall=store.wall, elapsed_seconds=elapsed,
                  scientific_pass=False, full_saved_strict=False)
    report_sha = _write_terminal(store, REPORT, report)
    completion = dict(schema=NAME + ".completion", report_sha256=report_sha,
                      boundary="after_report_fsync", scientific_pass=False,
                      full_saved_strict=False, complete=False)
    completion_sha = _write_terminal(store, COMPLETION, completion)
    return dict(report_sha256=report_sha, completion_sha256=completion_sha, complete=False)
