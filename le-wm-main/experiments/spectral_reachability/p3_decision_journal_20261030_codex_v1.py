"""Exclusive durable transport for P3 four-decision callback records, not a host driver."""
import hashlib
import os
from pathlib import Path

from experiments.spectral_reachability import p3_decision_equal_work_control_20261030_codex_v1 as control

NAME = "p3_decision_journal_20261030_codex_v1"
PREP_SHA = "df480bc6d9c55daefbeac9bdb79373bf7d3eba735148cc42f0f850d05d16b860"
CONTROL_SHA = "3629fba1abd9bbbee2dc6ed273f65290d17f88588e30ed863646d3cbb4f83500"
PREP = Path(__file__).resolve().parents[2] / "results/p3_decision_equal_work_control_20261030_codex_v1_prepare_PREP01.json"
MAX_RECORD = 32 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_pins():
    for path, wanted in ((Path(control.__file__), CONTROL_SHA), (PREP, PREP_SHA)):
        control.old.need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == wanted,
                         "fixed source/prepare mismatch")
    return {"prepare_sha256": PREP_SHA, "control_sha256": CONTROL_SHA,
            "journal_source_sha256": sha(Path(__file__).read_bytes())}


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Store:
    """One writer, flat append-only journal; byte limit includes reserved terminal space."""
    def __init__(self, directory, *, max_bytes, terminal_reserve):
        p = Path(directory)
        control.old.need(p.is_absolute() and p.resolve() == p and not p.exists() and not p.is_symlink()
                         and p.parent.is_dir(), "exclusive canonical journal directory")
        control.old.need(type(max_bytes) is int and type(terminal_reserve) is int and
                         0 < terminal_reserve < max_bytes, "journal byte limits")
        pins = source_pins()
        p.mkdir()
        syncdir(p.parent)
        self.path = p / "journal.jsonl"
        self.max_bytes, self.reserve, self.used = max_bytes, terminal_reserve, 0
        self.index, self.previous, self.poisoned, self.finished = 0, None, False, False
        self._append({"kind": "journal_begin", "pins": pins, "contract": control.contract()}, terminal=False)

    def _append(self, payload, *, terminal):
        control.old.need(not self.poisoned and not self.finished and type(payload) is dict, "journal unavailable")
        body = {"index": self.index, "previous": self.previous, "payload": payload}
        record = dict(body, sha256=control.core.digest(body))
        data = control.core.canonical(record) + b"\n"
        if not (len(data) <= MAX_RECORD and self.used + len(data) <=
                (self.max_bytes if terminal else self.max_bytes - self.reserve)):
            self.poisoned = True
            raise ValueError("journal byte cap")
        try:
            with self.path.open("xb" if self.index == 0 else "ab") as stream:
                control.old.need(stream.write(data) == len(data), "short journal write")
                stream.flush()
                os.fsync(stream.fileno())
            syncdir(self.path.parent)
        except BaseException:
            self.poisoned = True
            raise
        self.index += 1
        self.previous = record["sha256"]
        self.used += len(data)

    def event(self, payload):
        control.old.need(type(payload) is dict and payload.get("kind") not in
                         ("journal_begin", "journal_end"), "reserved event kind")
        self._append(payload, terminal=False)

    def finish(self, roster):
        control.old.need(type(roster) is dict and roster.get("schema") == control.NAME + ".roster",
                         "roster terminal")
        self._append({"kind": "journal_end", "roster": roster, "pins": source_pins()}, terminal=True)
        self.finished = True
        return sha(self.path.read_bytes())
