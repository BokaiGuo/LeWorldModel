"""Execute every sealed history-selected CP5 tail, including unchanged choices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file
from experiments.a0_pointmaze_native_render_probe_20261102 import DATA

SOURCE = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json"
SOURCE_SHA = "8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446"
HISTORY = ROOT / "results/a1_pointmaze_cp5_history_diagnostic_20261102_DEV01.json"
HISTORY_SHA = "040d627cd21c6337c244713c464ce91d2afa89d9923ca13d7f601e3d0e2b2cac"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP5_HISTORY_PHYSICAL_PROTOCOL_20261102.md"
OUTPUT = ROOT / "results/a1_pointmaze_cp5_history_physical_20261102_DEV01.json"
ARRAYS = ROOT / "results/a1_pointmaze_cp5_history_physical_20261102_DEV01.npz"


def main():
    if OUTPUT.exists() or ARRAYS.exists() or sha256_file(SOURCE) != SOURCE_SHA or sha256_file(HISTORY) != HISTORY_SHA:
        raise RuntimeError("output exists or sealed input changed")
    source, history = json.loads(SOURCE.read_text()), json.loads(HISTORY.read_text())
    if history["source_sha256"] != SOURCE_SHA:
        raise RuntimeError("history source mismatch")
    original = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
    if sha256_file(original) != source["source_sha256"]:
        raise RuntimeError("prefix source mismatch")
    if sha256_file(DATA / "states.pth") != json.loads(original.read_text())["states_sha256"]:
        raise RuntimeError("reset states changed")
    bank = np.asarray(source["bank"], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("bad bank")
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    rows, images, started = [], {}, time.monotonic()
    if [r["root"] for r in source["rows"]] != [r["root"] for r in history["rows"]]:
        raise RuntimeError("root mismatch")
    for source_row, history_row in zip(source["rows"], history["rows"]):
        root = source_row["root"]
        start = states[root, 0].astype(np.float64)
        if [b["method"] for b in source_row["branches"]] != [b["method"] for b in history_row["branches"]]:
            raise RuntimeError("branch mismatch")
        branches, sequences = [], []
        for old, choice in zip(source_row["branches"], history_row["branches"]):
            prefix = np.asarray(old["prefix_action"], dtype=np.float32)
            tail_index = choice["history_choice_not_executed"]
            if (prefix.shape != (2,) or not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1)
                    or not isinstance(tail_index, int) or not 0 <= tail_index < 8
                    or tail_index != int(np.argmin(choice["history_costs"]))
                    or old["tail_index"] != choice["executed_tail"]):
                raise ValueError("invalid saved decision")
            with NativeRGBBridge(start) as env:
                rgb5 = env.step(np.repeat(prefix[None], 5, axis=0))
                if image_hash(rgb5) != old["rgb5_sha256"]:
                    raise RuntimeError("prefix replay mismatch")
                rgb10 = env.step(np.repeat(bank[tail_index][None], 5, axis=0))
            if tail_index == old["tail_index"] and image_hash(rgb10) != old["rgb10_sha256"]:
                raise RuntimeError("unchanged action chain does not reproduce RGB")
            images[f"r{root}_{old['method']}_rgb10"] = rgb10
            sequences.append(np.concatenate((np.repeat(prefix[None], 5, axis=0),
                                             np.repeat(bank[tail_index][None], 5, axis=0))))
            branches.append({"method": old["method"], "source_tail": old["tail_index"],
                             "history_tail": tail_index, "rgb5_sha256": image_hash(rgb5),
                             "rgb10_sha256": image_hash(rgb10),
                             "source_distance_eval_only": old["distance_eval_only"]})
        endpoints = physical.replay_position_endpoints(np.repeat(start[None], len(branches), axis=0),
                                                       np.stack(sequences))
        target = states[root, 10, :2].astype(np.float64)
        for branch, endpoint in zip(branches, endpoints):
            branch["endpoint_eval_only"] = endpoint.tolist()
            branch["history_distance_eval_only"] = float(np.linalg.norm(endpoint[:2] - target))
            branch["distance_change_history_minus_single"] = (
                branch["history_distance_eval_only"] - branch["source_distance_eval_only"])
        rows.append({"root": root, "branches": branches})
        print(f"history physical root {root} complete", flush=True)
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **images)
    result = {"schema": "a1_pointmaze_cp5_history_physical.v1", "source_sha256": SOURCE_SHA,
              "history_sha256": HISTORY_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "states_sha256": sha256_file(DATA / "states.pth"), "physical_backend": physical.position_backend_receipt(),
              "scope": "exposed roots; position-only offline replay; no independent P1",
              "wall_seconds": time.monotonic() - started, "rows": rows}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
