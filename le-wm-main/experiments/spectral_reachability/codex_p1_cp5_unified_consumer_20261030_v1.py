"""Independent verification of cp5 encoder and optimizer journals.

This verifies the recorded boundaries, not the completeness of instrumentation
or isolation. It never reads physical state, observations or outcomes.
"""
import hashlib
import json
from pathlib import Path

from . import codex_four_arm_consumer_20261030 as optimizer
from . import codex_p1_suffix_budget_outcome_20261030_v1 as axes


def need(value, message):
    if not value:
        raise ValueError(message)


def consume_encoder(path, sha, caps, wall_seconds):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "encoder journal file")
    raw = path.read_bytes()
    need(len(raw) <= 4 * 1024 * 1024 and raw.endswith(b"\n") and
         hashlib.sha256(raw).hexdigest() == sha, "encoder journal digest or truncation")
    entries = []
    previous = None
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=optimizer.unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        need(type(row) is dict and set(row) == {"seq", "previous", "kind", "payload", "sha256"},
             "encoder event fields")
        body = {k: row[k] for k in ("seq", "previous", "kind", "payload")}
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        need(canonical(row) == line and row["seq"] == len(entries) and row["previous"] == previous
             and row["sha256"] == hashlib.sha256(canonical(body)).hexdigest(), "encoder event chain")
        previous = row["sha256"]
        entries.append(row)
    need(len(entries) >= 2 and entries[0]["kind"] == "header" and entries[-1]["kind"] == "terminal",
         "encoder terminal missing")
    need(entries[0]["payload"] == {"caps": caps, "wall_seconds": wall_seconds} and
         type(caps) is dict and set(caps) == set(axes.AXES) and
         all(type(v) is int and v >= 0 for v in caps.values()), "encoder budget identity")
    counts, pending, failed = dict.fromkeys(axes.AXES, 0), None, False
    for row in entries[1:-1]:
        kind, p = row["kind"], row["payload"]
        need(not failed, "event after encoder failure")
        if kind == "attempt":
            need(pending is None and type(p) is dict and set(p) == {"axis", "units"} and
                 p["axis"] in axes.AXES and type(p["units"]) is int and p["units"] > 0,
                 "encoder attempt")
            counts[p["axis"]] += p["units"]
            need(counts[p["axis"]] <= caps[p["axis"]], "encoder cap exceeded")
            pending = p
        elif kind in ("return", "failure"):
            need(pending is not None and type(p) is dict and
                 {k: p.get(k) for k in ("axis", "units")} == pending and
                 set(p) == ({"axis", "units"} if kind == "return"
                            else {"axis", "units", "error_type", "message"}), "encoder return/failure")
            pending = None
            failed = kind == "failure"
        elif kind == "blocked":
            need(pending is None and type(p) is dict and set(p) == {"axis", "units"} and
                 p["axis"] in axes.AXES and type(p["units"]) is int and p["units"] > 0,
                 "encoder block")
            # Wall time is not recorded in the event, so a block cannot be
            # independently classified as wall versus axis cap.
            failed = True
        else:
            raise ValueError("invalid or nested encoder event")
    terminal = entries[-1]["payload"]
    need(pending is None and terminal == {"status": "failed" if failed else "complete",
                                         "attempted": counts}, "encoder terminal mismatch")
    return {"status": terminal["status"], "attempted": counts, "sha256": sha}


def consume(encoder_path, encoder_sha, caps, wall_seconds, optimizer_path, optimizer_sha,
            contract_sha, arm, level, run_id):
    enc = consume_encoder(encoder_path, encoder_sha, caps, wall_seconds)
    opt = optimizer.consume(optimizer_path, optimizer_sha, contract_sha, arm, level, run_id)
    return {"encoding": enc, "planning": opt,
            "complete": enc["status"] == "complete" and opt["status"] == "closed_unverified",
            "instrumentation_complete": False, "isolation_certified": False,
            "fairness_proven": False, "P1_admission": False}
