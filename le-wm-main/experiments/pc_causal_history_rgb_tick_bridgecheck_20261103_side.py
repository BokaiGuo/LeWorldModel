"""Compare instrumented replay with the unmodified live bridge and saved source."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import marker
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "results/pc_causal_history_rgb_tick_timing_20261103_side"
SOURCE = ROOT / "results/pc_causal_history_full_bank_replan_20261103_DEV01.npz"
OUT = ROOT / "results/pc_causal_history_rgb_tick_bridgecheck_20261103_side.json"
ROOTS = (8343, 8345, 8349, 8351)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(a, b):
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    ca, cb = np.asarray(marker(a)), np.asarray(marker(b))
    return {"exact": bool(np.array_equal(a, b)), "changed_channels": int(np.count_nonzero(delta)),
            "max_channel_error": int(delta.max()), "centroid_a": ca.tolist(),
            "centroid_b": cb.tolist(), "centroid_distance_pixels": float(np.linalg.norm(ca-cb))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = []
    with np.load(STEM.with_suffix(".npz")) as measured, np.load(SOURCE) as saved:
        for i, root in enumerate(ROOTS):
            with NativeRGBBridge(measured["starts"][i]) as bridge:
                images = [bridge.initial_rgb]
                images.extend([bridge.step(np.zeros((1, 2), np.float32)) for _ in range(2)])
            rows.append({"root": root,
                         "live_bridge_vs_instrumented": [comparison(a, b) for a, b in zip(images, measured["images"][i])],
                         "live_bridge_vs_original": [comparison(a, b) for a, b in zip(images, saved[f"r{root}_p0_history"][:3])]})
    report = {"scope": "offline live bridge verification; no model and not planning input",
              "script_sha256": sha(Path(__file__)), "timing_result_sha256": sha(STEM.with_suffix(".json")),
              "source_npz_sha256": sha(SOURCE), "rows": rows}
    with OUT.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(OUT)


if __name__ == "__main__":
    main()
