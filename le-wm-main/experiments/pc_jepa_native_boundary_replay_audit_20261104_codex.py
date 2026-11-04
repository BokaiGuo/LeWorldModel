"""Audit saved physical replay against both original and new journals."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1"
NEW = ROOT / "results/pc_jepa_native_boundary_replay_20261104_codex"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(path):
    prior = None
    events = []
    for index, line in enumerate(path.read_bytes().splitlines()):
        row = json.loads(line)
        entry = row.pop("entry_sha256")
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if row["sequence"] != index or row["previous_sha256"] != prior or hashlib.sha256(raw).hexdigest() != entry:
            raise ValueError("journal hash chain invalid")
        prior = entry
        events.append(row["event"])
    return events


def main():
    report = json.loads((OLD / "report.json").read_text())
    result = json.loads((NEW / "result.json").read_text())
    if report["journal_sha256"] != digest(OLD / "journal.jsonl") or result["journal_sha256"] != digest(NEW / "journal.jsonl"):
        raise ValueError("journal digest mismatch")
    if result["bindings"]["prior_report_sha256"] != digest(OLD / "report.json"):
        raise ValueError("prior report drift")
    old_events, new_events = chain(OLD / "journal.jsonl"), chain(NEW / "journal.jsonl")
    old_records = {e["record"]["frame"]: e["record"] for e in old_events
                   if e.get("phase") == "assets_durable" and e.get("role") == "a"}
    new_records = {e["frame"]: e for e in new_events if e.get("phase") == "frame_comparison"}
    names = ("rgb.npy", "state.npy", "proprio.npy", "full_public_state.json", "transition.json")
    rows = []
    for frame in range(11):
        row = {}
        for name in names:
            a = digest(OLD / f"a/frame_{frame:02d}" / name)
            b = digest(NEW / f"frame_{frame:02d}" / name)
            expected = old_records[frame]["files"][name]["sha256"]
            saved = result["comparisons"][frame]
            row[name] = a == b == expected == saved["expected_sha256"][name] == saved["actual_sha256"][name]
            if not row[name] or not new_records[frame]["exact"][name]:
                raise ValueError(f"mismatch frame={frame} field={name}")
        rows.append({"frame": frame, "checks": row})
    record = {"status": "saved_replay_matches_prior_journal", "old_report_sha256": digest(OLD / "report.json"),
              "new_result_sha256": digest(NEW / "result.json"), "old_journal_entries": len(old_events),
              "new_journal_entries": len(new_events), "frames": rows,
              "scope": "saved-data hash and journal audit; no independent physics execution, no private solver-cache proof"}
    with (NEW / "audit.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "frames": len(rows), "checks": len(rows) * len(names)}))


if __name__ == "__main__":
    main()
