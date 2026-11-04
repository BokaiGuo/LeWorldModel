"""Physical replay of every sealed affine-lookahead prefix and optimized tail."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import OUTPUT as SOURCE
from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import BANK_SHA, SOURCE_SHA
from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import OUTPUT as BANK
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import DATA, GOAL, ROOTS, OUTPUT as OBSTACLE
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

SOURCE_DIGEST = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_AFFINE_NONPROGRESS_ORACLE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_affine_nonprogress_oracle_20261103.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, digest in ((SOURCE, SOURCE_DIGEST), (BANK, BANK_SHA), (OBSTACLE, SOURCE_SHA)):
        if sha256_file(path) != digest:
            raise ValueError(f"changed input: {path}")
    source, bank_doc, obstacle = (json.loads(p.read_text()) for p in (SOURCE, BANK, OBSTACLE))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (BANK.with_suffix(".npz"), bank_doc["arrays_sha256"]),
                         (DATA / "states.pth", obstacle["states_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError(f"changed data: {path}")
    if [row["root"] for row in source["rows"]] != list(ROOTS):
        raise ValueError("root roster mismatch")
    mean, std = np.asarray(source["action_mean"], np.float32), np.asarray(source["action_std"], np.float32)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid action normalization")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu").numpy()
    starts, sequences, records = [], [], []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as inputs, \
            np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in source["rows"]:
            root = row["root"]
            prefixes = bank[f"r{root}_bank"]
            tails = inputs[f"r{root}_lookahead_tails"]
            first = inputs[f"r{root}_first_scores"]
            terminal = inputs[f"r{root}_terminal_scores"]
            if (prefixes.shape != (9, 2) or tails.shape != (9, 10) or first.shape != (9,)
                    or terminal.shape != (9, 2) or not np.isfinite(prefixes).all()
                    or not np.isfinite(tails).all() or np.any(np.abs(prefixes) > 1)):
                raise ValueError("invalid bank or model outputs")
            for i in range(9):
                # Match model executor's float32 physical action conversion.
                tail = np.clip(tails[i].reshape(5, 2) * std + mean, -1, 1).astype(np.float32)
                prefix = np.repeat(prefixes[i][None], 5, axis=0).astype(np.float32)
                starts.append(states[root, 40].astype(np.float64))
                sequences.append(np.concatenate((prefix, tail)))
                records.append({"root": root, "bank_index": i, "model_first_score": float(first[i]),
                                "model_terminal_score": float(terminal[i].min()),
                                "model_nonprogress": bool(i in row["eligible"])})
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    print(f"Replay {len(records)} sealed model-optimized chains in physical backend.", flush=True)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp10 = physical.replay_position_endpoints(starts, sequences)
    receipt = physical.position_backend_receipt()
    for i, record in enumerate(records):
        root, idx = record["root"], record["bank_index"]
        source_row = next(r for r in source["rows"] if r["root"] == root)
        for branch in source_row["branches"]:
            if branch["bank_index"] == idx:
                expected = branch["cp5_distance"]
                actual = float(np.linalg.norm(cp5[i, :2] - GOAL[:2]))
                if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                    raise RuntimeError("saved feedback prefix distance mismatch")
        initial_d = float(np.linalg.norm(starts[i, :2] - GOAL[:2]))
        cp5_d = float(np.linalg.norm(cp5[i, :2] - GOAL[:2]))
        record.update({"initial_distance": initial_d, "cp5_distance": cp5_d,
                       "cp10_distance": float(np.linalg.norm(cp10[i, :2] - GOAL[:2])),
                       "physical_nonprogress": bool(cp5_d >= initial_d)})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, sequences=sequences, cp5=cp5, cp10=cp10)
    result = {"schema": "pc_affine_nonprogress_oracle.v1", "rows": records,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "source_sha256": SOURCE_DIGEST, "bank_sha256": BANK_SHA, "obstacle_sha256": SOURCE_SHA,
              "arrays_sha256": sha256_file(ARRAYS), "physical_backend": receipt,
              "trajectories": len(records), "scope": "oracle only; exposed roots; no P1"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Saved {OUTPUT} sha256={sha256_file(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
