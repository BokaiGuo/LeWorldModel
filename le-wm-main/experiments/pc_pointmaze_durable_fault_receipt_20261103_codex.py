"""Persist intentional abrupt-process fault cases as separate negative evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.pc_pointmaze_durable_dispatch_20261103_codex import _sync_dir, sha

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/"results/pc_pointmaze_durable_fault_receipt_20261103_CODEX"
FIXTURE = ROOT/"tests/fixtures/pc_pointmaze_durable_crash_20261103_codex.py"
PROTOCOL = ROOT.parent/"docs/PC_POINTMAZE_SHADOW_DURABLE_CLOSED_LOOP_PROTOCOL_20261103_CODEX.md"


def main():
    DIRECTORY.mkdir()
    _sync_dir(DIRECTORY.parent)
    rows = []
    for stage, expected in (
        ("after_shadow", [("bridge_attempt", "shadow"), ("bridge_returned", "shadow")]),
        ("during_legacy", [("bridge_attempt", "shadow"), ("bridge_returned", "shadow"),
                           ("bridge_attempt", "legacy")]),
    ):
        target = DIRECTORY/stage
        target.mkdir()
        _sync_dir(DIRECTORY)
        result = subprocess.run([sys.executable, str(FIXTURE), str(target), stage],
                                capture_output=True, text=True, timeout=15)
        journal = target/"events.jsonl"
        entries = [json.loads(x) for x in journal.read_text().splitlines()]
        actual = [(entry["event"], entry["bridge"]) for entry in entries]
        if result.returncode != 73 or actual != expected:
            raise RuntimeError(f"fault outcome mismatch: {stage} {result.returncode} {actual}")
        if not (target/"decision.npz").exists() or not (target/"response_shadow.npz").exists():
            raise RuntimeError("missing durable crash evidence")
        if (target/"response_legacy.npz").exists():
            raise RuntimeError("unexpected legacy return")
        rows.append({"stage": stage, "exit_code": result.returncode, "events": actual,
                     "journal_sha256": sha(journal),
                     "snapshot_sha256": sha(target/"decision.npz"),
                     "shadow_response_sha256": sha(target/"response_shadow.npz"),
                     "stderr_tail": result.stderr[-1000:]})
    report = {"scope": "intentional abrupt-exit negative cases, fake bridges, no GPU/physics",
              "rows": rows, "fixture_sha256": sha(FIXTURE), "protocol_sha256": sha(PROTOCOL),
              "script_sha256": sha(Path(__file__))}
    with (DIRECTORY/"report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(DIRECTORY)
    print(f"Saved {DIRECTORY/'report.json'}")


if __name__ == "__main__":
    main()
