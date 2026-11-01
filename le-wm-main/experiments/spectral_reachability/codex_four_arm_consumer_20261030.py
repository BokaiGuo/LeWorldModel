"""Reconstruct a four-arm journal without invoking its driver or callbacks."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec

NAME = "codex_four_arm_durable_20261030"
MAX_BYTES = 32 * 1024 * 1024


def unique(pairs):
    result = {}
    for key, value in pairs:
        spec.require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _parse(raw):
    spec.require(raw and raw.endswith(b"\n"), "truncated journal")
    previous = None
    rows = []
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        spec.require(type(row) is dict and set(row) == {"index", "previous", "kind", "payload", "sha256"}
                     and spec.canonical(row) == line, "canonical event")
        body = {k: row[k] for k in ("index", "previous", "kind", "payload")}
        spec.require(type(row["index"]) is int and row["index"] == len(rows)
                     and row["previous"] == previous and row["sha256"] == spec.digest(body),
                     "event chain/order")
        previous = row["sha256"]
        rows.append(row)
    return rows


def consume(path, expected_sha, contract_sha, arm, level, run_id):
    path = Path(path)
    spec.require(spec.is_hash(expected_sha) and spec.is_hash(contract_sha)
                 and path.is_file() and not path.is_symlink(), "journal and external hashes required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    spec.require(len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == expected_sha, "journal digest/size")
    rows = _parse(raw)
    spec.require(len(rows) >= 2 and rows[0]["kind"] == "header" and rows[-1]["kind"] == "terminal",
                 "missing header or terminal")
    header = rows[0]["payload"]
    spec.require(type(header) is dict and set(header) == {"schema", "contract", "contract_sha256",
                                                         "arm", "level", "run_id"}
                 and header["schema"] == NAME + ".v1"
                 and header["contract_sha256"] == contract_sha and header["arm"] == arm
                 and header["level"] == level and header["run_id"] == run_id, "external binding")
    spec.preflight(header["contract"], contract_sha)
    spec.require(arm in spec.ARMS and type(run_id) is str and run_id and run_id.strip() == run_id,
                 "arm/run")
    levels = [v for v in header["contract"]["levels"] if v["id"] == level]
    spec.require(len(levels) == 1, "level")
    caps = levels[0]["caps"]
    charged = dict.fromkeys(spec.AXES, 0)
    pending = False
    failed = False
    attempts = completed = blocked = 0
    for row in rows[1:-1]:
        kind, p = row["kind"], row["payload"]
        spec.require(not failed, "event after failure or block")
        if kind in ("intent", "blocked"):
            spec.require(not pending and type(p) is dict and set(p) == {"phase", "kind", "reservation"}
                         and p["phase"] in spec.PHASES and p["kind"] in spec.KINDS,
                         "operation event")
            cost = spec.vector(p["reservation"])
            op = p["kind"]
            spec.require(cost[spec.KINDS[op]] > 0
                         and (op != "encode" or cost["encode_samples"] > 0)
                         and (op != "predict" or cost["predict_samples"] > 0), "operation charge")
            over = any(charged[k] + cost[k] > caps[k] for k in spec.AXES)
            if kind == "blocked":
                spec.require(over, "spurious block")
                blocked += 1
                failed = True
            else:
                spec.require(not over, "over-cap intent")
                for k in spec.AXES:
                    charged[k] += cost[k]
                pending = True
                attempts += 1
        elif kind in ("return", "failure"):
            spec.require(p == {} and pending, "orphan terminal event")
            pending = False
            if kind == "failure":
                failed = True
            else:
                completed += 1
        elif kind == "invalid":
            spec.require(p == {} and not pending, "invalid dispatch event")
            failed = True
        else:
            raise ValueError("unknown or misplaced event")
    terminal = rows[-1]["payload"]
    spec.require(not pending and type(terminal) is dict
                 and set(terminal) == {"status", "charged_upper_bounds"}, "incomplete operation or terminal")
    spec.vector(terminal["charged_upper_bounds"])
    spec.require(terminal["status"] == ("failed_or_blocked" if failed else "closed_unverified")
                 and terminal["charged_upper_bounds"] == charged, "terminal reconstruction mismatch")
    return {"journal_sha256": expected_sha, "contract_sha256": contract_sha, "arm": arm,
            "level": level, "run_id": run_id, "status": terminal["status"],
            "charged_upper_bounds": charged, "attempted_events": attempts,
            "completed_events": completed, "blocked_events": blocked,
            "coverage_verified": False, "actual_usage_verified": False,
            "fairness_proven": False, "execution_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--contract-sha256", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    print(json.dumps(consume(a.journal, a.sha256, a.contract_sha256, a.arm, a.level, a.run_id),
                     sort_keys=True))


if __name__ == "__main__":
    main()
