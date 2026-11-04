"""Read-only audit of the already completed paired development run."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pc_shadow_spectral_paired_dev_20261103_CODEX"
OUTPUT = ROOT / "results/pc_shadow_spectral_paired_audit_20261103_CODEX.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines()]
    assert [e["index"] for e in events] == list(range(len(events)))
    counts = Counter(e["event"] for e in events)
    assert (counts["bridge_attempt"], counts["bridge_returned"], counts["decision"],
            counts["branch_completed"], counts["completed"], counts["failed"]) == (72, 72, 20, 4, 1, 0)
    assert counts["predict_attempt"] == counts["predict_returned"] == 80
    assert counts["structure_attempt"] == counts["structure_returned"] == 10
    pending = None
    previous_rgb = {}
    returned, decisions = 0, 0
    for event in events:
        kind = event["event"]
        if kind == "bridge_attempt":
            assert pending is None
            path = ROOT / event["snapshot_path"]
            assert sha(path) == event["snapshot_sha256"]
            branch = str(path.parent)
            with np.load(path, allow_pickle=False) as snapshot:
                assert int(snapshot["clock"]) == event["clock"]
                if event["phase"] != "reset":
                    np.testing.assert_array_equal(snapshot["action"], event["action"])
                    np.testing.assert_array_equal(snapshot["current_rgb"], previous_rgb[branch])
                    if event["phase"] == "feedback":
                        np.testing.assert_array_equal(snapshot["action"], snapshot["plan_raw"][0])
                        assert snapshot["noise"].shape == (3, 16, 10)
                        decisions += 1
            pending = (branch, event["snapshot_sha256"], event["clock"])
        elif kind == "bridge_returned":
            assert pending is not None
            branch, digest, clock = pending
            assert event["snapshot_sha256"] == digest and event["clock"] == clock
            path = ROOT / event["rgb_path"]
            assert sha(path) == event["artifact_sha256"]
            with np.load(path, allow_pickle=False) as artifact:
                rgb = artifact["rgb"]
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == event["rgb_sha256"]
            previous_rgb[branch] = rgb
            pending = None
            returned += 1
        elif kind == "branch_completed":
            assert sha(ROOT / event["trajectory_path"]) == event["trajectory_sha256"]
        elif kind == "decision":
            assert event["cem_counts"] == {"samples": 50, "batches": 4}
            expected = (1, 2, 1) if event["arm"] == "spectral" else (0, 0, 0)
            sc = event["structure_counts"]
            assert (sc["response_forward_calls"], sc["reverse_vector_calls"], sc["svd_calls"]) == expected
    assert pending is None and returned == 72 and decisions == 20
    assert events[1]["model_hash"] == events[-1]["model_hash"]
    result = {"schema": "pc_shadow_spectral_paired_audit.v1", "journal_sha256": sha(RUN / "events.jsonl"),
              "event_counts": dict(counts), "verified_returned_snapshots": returned,
              "verified_feedback_actions": decisions, "model_hash": events[-1]["model_hash"],
              "branch_results": [e for e in events if e["event"] == "branch_completed"],
              "scope": "saved artifacts only; not independent or physical-success validation"}
    with OUTPUT.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
