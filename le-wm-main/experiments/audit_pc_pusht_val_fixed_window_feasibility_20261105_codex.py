"""Fixed-window local dataset feasibility census; not a planning experiment."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/jepa_wm_osf/extracted/pusht_noise/val"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_VAL_FIXED_WINDOW_FEASIBILITY_PROTOCOL_20261105_CODEX.md"
OUT = ROOT / "results/pc_pusht_val_fixed_window_feasibility_20261105_codex.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    actions = torch.load(DATA / "rel_actions.pth", map_location="cpu", weights_only=True).float().numpy()
    with (DATA / "seq_lengths.pkl").open("rb") as f:
        lengths = np.asarray(pickle.load(f), dtype=np.int64)
    if states.ndim != 3 or states.shape[2] != 5 or actions.shape != (*states.shape[:2], 2) or len(lengths) != len(states):
        raise ValueError("unexpected dataset shapes")
    if np.any(lengths < 0) or np.any(lengths > states.shape[1]):
        raise ValueError("invalid sequence lengths")
    rows = []
    for i, length in enumerate(lengths):
        row = {"sequence": i, "length": int(length), "sufficient_length": bool(length >= 26)}
        if length >= 26:
            s, a = states[i, 10:26].astype(np.float64), actions[i, 10:25].astype(np.float64) / 100
            finite = bool(np.isfinite(s).all() and np.isfinite(a).all())
            center_ok = bool(finite and np.all((s[:, :4] >= 40) & (s[:, :4] <= 472)))
            action_ok = bool(finite and np.all(np.abs(a) <= 1))
            displacement = float(np.linalg.norm(s[-1, 2:4] - s[0, 2:4])) if finite else None
            angle = float(abs((s[-1, 4] - s[0, 4] + np.pi) % (2 * np.pi) - np.pi)) if finite else None
            moved = bool(finite and displacement >= 20)
            row.update(finite=finite, center_ok=center_ok, action_ok=action_ok,
                       displacement=displacement, angle_change=angle, moved=moved,
                       candidate=finite and center_ok and action_ok and moved)
        else:
            row.update(finite=None, center_ok=None, action_ok=None, displacement=None,
                       angle_change=None, moved=None, candidate=False)
        rows.append(row)
    output = {"status": "completed", "scope": "fixed local val window, state-only census; not provenance or physical replay",
              "inputs": {name: sha(DATA / name) for name in ("states.pth", "rel_actions.pth", "seq_lengths.pkl")},
              "protocol_sha256": sha(PROTOCOL), "auditor_sha256": sha(Path(__file__)),
              "counts": {"total": len(rows), **{k: sum(row[k] is True for row in rows)
                                                for k in ("sufficient_length", "finite", "center_ok",
                                                          "action_ok", "moved", "candidate")}},
              "rows": rows}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(output["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
