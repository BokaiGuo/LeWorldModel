"""Fixed archive-start identity preflight; no model, simulator, goals or outcomes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from experiments.spectral_reachability.codex_historical_pose_collision_audit_20261030 import (
    INDEX, INDEX_SHA, same_pose,
)

ARCHIVE = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
# Fixed before inspecting these episodes. No replacement, random search or outcome filtering.
ORDINALS = (100, 101, 102, 103)
N_ROWS, N_EPISODES = 2336736, 18685
SCHEMA = {
    "ep_len": ((N_EPISODES,), "int32"),
    "ep_offset": ((N_EPISODES,), "int64"),
    "episode_idx": ((N_ROWS,), "int64"),
    "step_idx": ((N_ROWS,), "int64"),
    "state": ((N_ROWS, 7), "float32"),
}


def stat_binding(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def scan(f, historical, ordinals=ORDINALS, schema=SCHEMA):
    for key, (shape, dtype) in schema.items():
        if key not in f or f[key].shape != shape or f[key].dtype != np.dtype(dtype):
            raise ValueError("archive schema mismatch: " + key)
    if len(set(ordinals)) != len(ordinals) or any(type(i) is not int or not 0 <= i < schema["ep_len"][0][0] for i in ordinals):
        raise ValueError("invalid fixed episode roster")
    hist = historical["observed_prefix_records"]
    if len(hist) != 80 or historical["history_complete"] is not False:
        raise ValueError("historical index mismatch")
    rows = []
    for ep in ordinals:
        start, length = int(f["ep_offset"][ep]), int(f["ep_len"][ep])
        if length <= 0 or start < 0 or start + length > schema["state"][0][0]:
            raise ValueError("invalid episode extent")
        if ep and int(f["ep_offset"][ep - 1]) + int(f["ep_len"][ep - 1]) != start:
            raise ValueError("previous episode boundary mismatch")
        if ep + 1 < schema["ep_len"][0][0] and int(f["ep_offset"][ep + 1]) != start + length:
            raise ValueError("next episode boundary mismatch")
        if int(f["episode_idx"][start]) != ep or int(f["step_idx"][start]) != 0:
            raise ValueError("episode start identity mismatch")
        state = np.asarray(f["state"][start])
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("invalid state")
        pose = [float(x) for x in state[:5]]
        rows.append({
            "episode_ordinal": ep, "offset": start, "episode_length": length,
            "state_dtype": str(state.dtype), "state7": state.tolist(),
            "state_bytes_sha256": hashlib.sha256(state.tobytes(order="C")).hexdigest(),
            "known_prefix_pose_matches": [i for i, h in enumerate(hist) if same_pose(pose, h["physical_pose"])],
            "batch_pose_matches": [],
        })
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            if same_pose(a["state7"][:5], rows[j]["state7"][:5]):
                a["batch_pose_matches"].append(j)
                rows[j]["batch_pose_matches"].append(i)
    return rows


def prepare():
    raw = INDEX.read_bytes()
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != INDEX_SHA:
        raise ValueError("historical index hash or size mismatch")
    historical = json.loads(raw)
    before = stat_binding(ARCHIVE)
    # Register a locally installed filter if available; never install or download it.
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(ARCHIVE, "r") as archive:
        rows = scan(archive, historical)
    if stat_binding(ARCHIVE) != before:
        raise ValueError("archive changed during read")
    return {
        "schema": "codex_p1_archive_source_preflight_20261030.v1",
        "archive_path": str(ARCHIVE), "archive_stat": before,
        "archive_full_sha256": None, "historical_index_sha256": INDEX_SHA,
        "fixed_ordinals_before_identity_read": list(ORDINALS),
        "rows": rows, "denominator": len(ORDINALS),
        "limits": [
            "train archive, not an independently attested holdout",
            "file stat and slice hashes do not establish dataset revision or provenance",
            "historical index incomplete; prefix state is not necessarily a sampled root",
            "archive state7 lacks physical solver hidden state and moving RGB/action history",
            "no evidence of outcome-blind historical exposure or independent sampling",
        ],
        "history_complete": False, "training_disjoint": "unknown",
        "source_independence": "unknown", "global_freshness": "unknown",
        "p1_admitted": False, "source_run_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare()
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"path": str(args.output.resolve()), "sha256": hashlib.sha256(encoded).hexdigest(),
                      "p1_admitted": False}))


if __name__ == "__main__":
    main()
