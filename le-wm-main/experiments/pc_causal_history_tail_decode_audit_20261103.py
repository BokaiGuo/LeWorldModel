"""Check stored feedback tail decoding against a historical normalizer receipt."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file

EXPECTED_REFERENCE = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("normalizer_reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.normalizer_reference) != EXPECTED_REFERENCE:
        raise ValueError("normalizer reference changed")
    report = json.loads(args.report.read_text())
    reference = json.loads(args.normalizer_reference.read_text())
    mean, std = np.asarray(reference["action_mean"], np.float32), np.asarray(reference["action_std"], np.float32)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("invalid normalizer")
    arrays_path = args.report.with_suffix(".npz")
    if sha256_file(arrays_path) != report["arrays_sha256"]:
        raise ValueError("saved arrays changed")
    differences = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for root in ROOTS:
            for i in range(9):
                base = f"r{root}_p{i}"
                proposals = arrays[f"{base}_feedback3_proposals"]
                scores = arrays[f"{base}_feedback3_scores"]
                if proposals.shape != (1, 2, 1, 10) or scores.shape != (2,):
                    raise ValueError("invalid final optimizer batch")
                winner = int(scores[1] < scores[0])
                decoded = np.clip(proposals[0, winner, 0].reshape(5, 2) * std + mean, -1, 1)
                tail = arrays[f"{base}_actions"][7:]
                if tail.shape != (5, 2):
                    raise ValueError("invalid stored physical tail")
                differences.append(float(np.max(np.abs(decoded - tail))))
    max_difference = max(differences)
    if max_difference > 1e-7:
        raise ValueError(f"decoded feedback tail mismatch: {max_difference}")
    result = {"schema": "pc_causal_history_tail_decode_audit.v1", "tails_checked": len(differences),
              "max_abs_difference": max_difference, "report_sha256": sha256_file(args.report),
              "arrays_sha256": sha256_file(arrays_path), "normalizer_reference_sha256": EXPECTED_REFERENCE,
              "script_sha256": sha256_file(Path(__file__)),
              "scope": "saved-action consistency with historical normalizer, not full provenance proof"}
    with args.output.open("x") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
