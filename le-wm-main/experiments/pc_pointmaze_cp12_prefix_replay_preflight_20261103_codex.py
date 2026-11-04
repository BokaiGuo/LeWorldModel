"""Read-only-source cp12 RGB replay gate, no model or suffix planning."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_causal_history_h10_suffix_20261103 as h

ROOT = h.ROOT
SOURCE = ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP12_PREFIX_REPLAY_PREFLIGHT_PROTOCOL_20261103_CODEX.md"
OUT = ROOT / "results/pc_pointmaze_cp12_prefix_replay_preflight_20261103_CODEX.json"
SOURCE_SHA = "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8"


def compare_frame(actual, saved):
    a, b = np.asarray(actual), np.asarray(saved)
    if a.shape != (224, 224, 3) or b.shape != a.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("invalid RGB")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"equal": bool(np.array_equal(a, b)),
            "actual_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "saved_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "different_channels": int(np.count_nonzero(diff)), "max_abs": int(diff.max()),
            "mean_abs": float(diff.mean())}


def main():
    if OUT.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"input changed: {path}")
    source, camera, old = (json.loads(p.read_text()) for p in (SOURCE, h.CAMERA, h.OLD))
    if h.sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise ValueError("source arrays changed")
    if [row["root"] for row in source["rows"]] != [8343, 8345, 8349, 8351]:
        raise ValueError("unexpected root roster")
    rows = []
    print("Preflight: replay four sealed prefixes; no model or suffix selection.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays:
        for i, root in enumerate((8343, 8345, 8349, 8351)):
            branch = source["rows"][i]["branches"][0]
            if (branch["arm"], branch["candidate"]) != ("raw", 2):
                raise ValueError("branch order mismatch")
            name = f"r{root}_a0_p2"
            saved, actions, start = arrays[name + "_history"], arrays[name + "_actions"], arrays["starts"][8*i]
            if (saved.shape != (18, 224, 224, 3) or actions.shape != (17, 2)
                    or start.shape != (4,) or not np.isfinite(actions).all()
                    or np.any(np.abs(actions) > 1) or not np.isfinite(start).all()):
                raise ValueError("invalid source inputs")
            with h.NativeRGBBridge(start.astype(np.float64)) as env:
                frames = [env.initial_rgb]
                for action in actions[:12]:
                    frames.append(env.step(action[None]))
                if env.step_count != 12:
                    raise ValueError("clock mismatch")
            metrics = [compare_frame(a, b) for a, b in zip(frames, saved[:13], strict=True)]
            states = []
            for t in (2, 7, 12):
                now = h.causal_state(frames[t-1], frames[t], camera["metadata"], old["body_offset_from_static_scene"], .05)
                prior = h.causal_state(saved[t-1], saved[t], camera["metadata"], old["body_offset_from_static_scene"], .05)
                states.append({"t": t, "actual": now.tolist(), "saved": prior.tolist(),
                               "difference": (now-prior).tolist()})
            rows.append({"root": root, "branch": name, "frames": metrics, "causal_states": states})
            print(f"{name}: {sum(x['equal'] for x in metrics)}/13 frames exact", flush=True)
    payload = {"schema": "cp12_prefix_replay_preflight.v1", "rows": rows,
               "pass_exact_all": all(f["equal"] for r in rows for f in r["frames"]),
               "source_sha256": h.sha256_file(SOURCE), "source_npz_sha256": h.sha256_file(SOURCE.with_suffix(".npz")),
               "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
               "bridge_sha256": h.sha256_file(ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
               "camera_sha256": h.sha256_file(h.CAMERA), "calibration_sha256": h.sha256_file(h.OLD),
               "counts": {"prefix_trajectories": 4, "physical_prefix_commands": 48,
                          "suffix_commands": 0, "model_calls": 0, "candidate_selections": 0},
               "scope": "engineering replay gate; physical reset only; no planning or control outcome"}
    with OUT.open("x") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"Saved {OUT}; exact gate={payload['pass_exact_all']}; sha256={h.sha256_file(OUT)}", flush=True)


if __name__ == "__main__":
    main()
