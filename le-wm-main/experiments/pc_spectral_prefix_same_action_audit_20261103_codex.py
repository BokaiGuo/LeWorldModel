"""Audit hashes and action/RGB bindings in the saved 16-branch journal."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pc_spectral_prefix_same_action_physical_20261103_CODEX"
OUTPUT = RUN / "audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads((RUN / "result.json").read_text())
    if result["journal_sha256"] != sha(RUN / "events.jsonl"):
        raise ValueError("journal changed")
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines()]
    if [x["index"] for x in events] != list(range(len(events))):
        raise ValueError("journal sequence mismatch")
    pending, last_rgb, verified, branches = None, {}, 0, 0
    for event in events:
        if event["event"] == "bridge_attempt":
            if pending is not None:
                raise ValueError("overlapping attempts")
            path = Path(event["snapshot_path"])
            if sha(path) != event["snapshot_sha256"]:
                raise ValueError("snapshot hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                if int(data["clock"]) != event["clock"]:
                    raise ValueError("clock mismatch")
                if event["phase"] != "reset":
                    np.testing.assert_array_equal(data["action"], event["action"])
                    np.testing.assert_array_equal(data["current_rgb"], last_rgb[str(path.parent)])
            pending = (str(path.parent), event["snapshot_sha256"], event["clock"])
        elif event["event"] == "bridge_returned":
            if pending is None or pending[1:] != (event["snapshot_sha256"], event["clock"]):
                raise ValueError("unpaired return")
            path = Path(event["rgb_path"])
            if sha(path) != event["artifact_sha256"]:
                raise ValueError("image artifact hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                image = data["rgb"]
                if hashlib.sha256(image.tobytes()).hexdigest() != event["rgb_sha256"]:
                    raise ValueError("image content mismatch")
            last_rgb[pending[0]] = image
            pending = None
            verified += 1
        elif event["event"] == "branch_completed":
            if sha(Path(event["trajectory_path"])) != event["trajectory_sha256"]:
                raise ValueError("trajectory hash mismatch")
            branches += 1
    if pending is not None or verified != 368 or branches != 16 or events[-1]["event"] != "completed":
        raise ValueError("incomplete run")
    with OUTPUT.open("x") as out:
        json.dump({"journal_sha256": sha(RUN / "events.jsonl"), "result_sha256": sha(RUN / "result.json"),
                   "verified_bridge_returns": verified, "verified_trajectories": branches,
                   "scope": "artifact bindings only, not independent confirmation"}, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
