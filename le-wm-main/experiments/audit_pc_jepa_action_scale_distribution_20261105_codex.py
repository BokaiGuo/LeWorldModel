"""Audit local PushT action units and distribution; this is not a provenance receipt."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/jepa_wm_osf/extracted/pusht_noise"
OUT = ROOT / "results/pc_jepa_action_scale_distribution_audit_20261105_codex.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"actions": int(x.shape[0]), "coordinates": int(x.size),
            "mean": x.mean(axis=0).tolist(), "std": x.std(axis=0).tolist(),
            "quantiles_abs_coordinate": np.quantile(np.abs(x), [.5, .9, .95, .99, 1]).tolist(),
            "negative_coordinate_fraction": float(np.mean(x < 0)),
            "outside_minus1_plus1_coordinate_fraction": float(np.mean(np.abs(x) > 1)),
            "outside_declared_zero512_action_fraction": float(np.mean(np.any((x < 0) | (x > 512), axis=1)))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = {}
    for split in ("train", "val"):
        action_path, length_path = DATA / split / "rel_actions.pth", DATA / split / "seq_lengths.pkl"
        actions = torch.load(action_path, map_location="cpu", weights_only=True).float().numpy()
        # Local dataset metadata, loaded as in the dataset implementation.
        with length_path.open("rb") as f:
            lengths = np.asarray(pickle.load(f), dtype=np.int64)
        if actions.ndim != 3 or actions.shape[-1] != 2 or len(lengths) != len(actions):
            raise ValueError("unexpected dataset shape")
        if np.any(lengths < 0) or np.any(lengths > actions.shape[1]):
            raise ValueError("invalid sequence length")
        valid = np.arange(actions.shape[1])[None, :] < lengths[:, None]
        values = actions[valid] / np.float32(100)
        if not np.isfinite(values).all():
            raise ValueError("non-finite action")
        rows[split] = {"shape": list(actions.shape), "length_min": int(lengths.min()),
                       "length_max": int(lengths.max()), "action_sha256": sha(action_path),
                       "length_sha256": sha(length_path), "valid_up_to_seq_length": stats(values)}
    banks = {}
    for name in ("pc_jepa_reachable_random_goal_dev_20261105_codex",
                 "pc_jepa_visible_local_bank_dev_20261105_codex"):
        path = ROOT / "results" / name / "root_0/input/bank.npy"
        x = np.load(path, allow_pickle=False)
        banks[name] = {"sha256": sha(path), "shape": list(x.shape), "all_frames": stats(x.reshape(-1, 2))}
    out = {"status": "completed", "auditor_sha256": sha(Path(__file__)),
           "scope": "local data only; no checkpoint-training provenance; seq_lengths interpreted as valid prefix",
           "rows": rows, "candidate_banks": banks}
    with OUT.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
