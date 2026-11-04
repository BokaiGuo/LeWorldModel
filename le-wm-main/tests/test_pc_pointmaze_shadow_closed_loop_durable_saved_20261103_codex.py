"""Verify durable smoke event ordering and all saved response bindings."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/pc_pointmaze_shadow_closed_loop_durable_20261103_CODEX.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_each_reset_and_action_has_bound_snapshot_and_individual_returns():
    result = json.loads(OUT.read_text())
    events_path = OUT.with_name(OUT.stem+"_events.jsonl")
    assert sha(events_path) == result["journal_sha256"]
    for path, digest in result["dependencies"].items():
        assert sha(path) == digest
    events = [json.loads(x) for x in events_path.read_text().splitlines()]
    assert [e["index"] for e in events] == list(range(len(events)))
    assert len([e for e in events if e["event"] == "bridge_attempt"]) == 72
    assert len([e for e in events if e["event"] == "bridge_returned"]) == 72
    assert len([e for e in events if e["event"] == "predict_returned"]) == 40
    assert result["model_before"] == result["model_after"]
    assert result["counts"]["predict_samples"] == 500
    for row in result["rows"]:
        assert row["prefix_checks"] == 13
        assert sha(row["trajectory_path"]) == row["trajectory_sha256"]
        with np.load(row["trajectory_path"], allow_pickle=False) as traj:
            assert traj["actions"].shape == (17, 2)
            for clock in range(18):
                matching = [e for e in events if e.get("branch") == row["name"]
                            and e.get("clock") == clock and e["event"] in ("bridge_attempt", "bridge_returned")]
                assert [(e["event"], e["bridge"]) for e in matching] == [
                    ("bridge_attempt", "shadow"), ("bridge_returned", "shadow"),
                    ("bridge_attempt", "legacy"), ("bridge_returned", "legacy")]
                snapshots = {(e["snapshot_path"], e["snapshot_sha256"]) for e in matching}
                assert len(snapshots) == 1
                path, digest = snapshots.pop()
                assert sha(path) == digest
                with np.load(path, allow_pickle=False) as snapshot:
                    assert int(snapshot["clock"]) == clock
                    if clock:
                        np.testing.assert_array_equal(snapshot["action"], traj["actions"][clock-1])
                        for e in matching:
                            np.testing.assert_array_equal(e["action"], traj["actions"][clock-1])
                for e in (matching[1], matching[3]):
                    assert sha(e["rgb_path"]) == e["artifact_sha256"]
                    with np.load(e["rgb_path"], allow_pickle=False) as response:
                        frame = traj["shadow_frames" if e["bridge"] == "shadow" else "legacy_frames"][clock]
                        np.testing.assert_array_equal(response["rgb"], frame)
                        assert int(response["clock"]) == clock
                        assert hashlib.sha256(frame.tobytes()).hexdigest() == e["rgb_sha256"]
