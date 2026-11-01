"""Exclusive bounded journal/store adapter. Transport only; not a HOST runner."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub

base = pub.base
need = base.need


class Store(pub.Store):
    """Explicit caller-supplied ceilings, not an admission or sizing decision.

    Any write exception poisons the store: bytes may already have reached disk
    while index/accounting did not advance. Never resume or append after that.
    """
    def __init__(self, path, *, limits, cap, reserve, floor, wall, clock=time.monotonic):
        p = Path(path)
        need(p.is_absolute() and p.resolve() == p and not p.exists() and
             not p.is_symlink() and p.parent.is_dir(), "exclusive canonical output")
        need(type(limits) is dict and set(limits) == set(base.LIMITS) and
             all(type(v) is int and v >= 0 for v in limits.values()) and
             all(type(v) is int and v > 0 for v in (cap, reserve, floor, wall)) and
             reserve < cap and sum(limits.values()) <= cap - reserve, "explicit whole-job ceilings")
        need(shutil.disk_usage(p.parent).free >= cap + floor, "whole cap and disk floor")
        self.limits = deepcopy(limits)
        self.cap, self.reserve, self.floor, self.wall = cap, reserve, floor, wall
        self.clock, self.started = clock, clock()
        p.mkdir()
        base.syncdir(p.parent)
        self.output = p
        self.used = {key: 0 for key in limits}
        self.number = self.index = self.frames = self.failures = self.final_bytes = 0
        self.previous = None
        self.pending = {}
        self.sealed = False
        self.poisoned = False

    def check(self, category, size):
        need(not self.poisoned and not self.sealed, "store unavailable")
        need(category in self.limits and type(size) is int and size >= 0, "category and size")
        elapsed = self.clock() - self.started
        need(0 <= elapsed <= self.wall, "whole-job wall ceiling")
        need(self.used[category] + size <= self.limits[category] and
             sum(self.used.values()) + size <= self.cap - self.reserve and
             shutil.disk_usage(self.output).free >= self.floor + size, "store ceiling or floor")

    def write(self, name, data, category, append=False):
        need(type(name) is str and Path(name).name == name and name not in ("", ".", "..") and
             type(data) is bytes and type(append) is bool, "flat owned bytes")
        need(name not in ("report.json", "completion.json", "publication_overrun.json"),
             "no inherited publication artifacts")
        path = self.output / name
        need(not path.is_symlink() and (name == "events.jsonl" if append else not path.exists()),
             "exclusive file")
        self.check(category, len(data))
        if append:
            need(path.exists() == (self.index > 0), "journal ownership")
        try:
            with path.open("ab" if append else "xb") as stream:
                need(stream.write(data) == len(data), "complete write")
                stream.flush()
                os.fsync(stream.fileno())
            base.syncdir(self.output)
        except BaseException:
            self.poisoned = True
            raise
        self.used[category] += len(data)

    def final(self, *args, **kwargs):
        raise ValueError("four-slot publication is not valid for this store")

    def publish(self, *args, **kwargs):
        raise ValueError("six-slot terminal publication is not implemented")

    def frame(self, *args, **kwargs):
        raise ValueError("use six-slot CaptureStore")


def inspect(path, expected_sha256, max_bytes):
    """Verify full records and retain a torn final fragment, without modifying it.

    A hash supplied by the caller is an integrity binding, not trusted timestamp
    or proof of source. Complete records with bad JSON/chain are rejected.
    """
    need(type(max_bytes) is int and max_bytes > 0, "explicit read limit")
    blob = base.read_bytes(Path(path), max_bytes)
    need(base.fsha(blob) == expected_sha256, "journal byte hash")
    lines = blob.split(b"\n")
    tail = lines.pop()
    previous = None
    pending, counts, events = {}, {}, []
    for index, line in enumerate(lines):
        need(0 < len(line) <= 16384, "bounded nonempty record")
        e = base.decode(line)
        need(type(e) is dict and set(e) == {"index", "previous", "origin", "kind", "resource",
                                           "token", "detail", "sha256"} and
             type(e["index"]) is int and e["index"] == index and e["previous"] == previous and
             e["sha256"] == base.digest({k: v for k, v in e.items() if k != "sha256"}), "journal chain")
        origin, kind, resource, token = (e[k] for k in ("origin", "kind", "resource", "token"))
        need(origin in base.ORIGINS and kind in ("intent", "return", "failure", "record") and
             type(resource) is str and bool(resource) and type(token) is int and token >= 0 and
             type(e["detail"]) is dict, "journal event ABI")
        key = (origin, token)
        count = counts.setdefault(origin + ":" + resource, dict(attempted=0, returned=0, failed=0))
        if kind == "intent":
            need(token == index and key not in pending, "unique intent")
            pending[key] = resource
            count["attempted"] += 1
        elif kind in ("return", "failure"):
            need(pending.pop(key, None) == resource, "matching disposition")
            count["returned" if kind == "return" else "failed"] += 1
        events.append(e)
        previous = e["sha256"]
    need(len(tail) <= 16384, "bounded torn tail")
    return dict(events=events, tail=previous, pending=pending, counts=counts,
                trailing_bytes=len(tail), trailing_sha256=base.fsha(tail) if tail else None,
                transport_complete=bool(events) and not tail and not pending and
                    all(x["failed"] == 0 for x in counts.values()),
                fsync_verified=False, full_saved_strict=False, physical_certified=False, scientific_pass=False)
