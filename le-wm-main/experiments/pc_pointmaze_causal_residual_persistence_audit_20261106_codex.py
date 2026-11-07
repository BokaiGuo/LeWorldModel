"""Post-seal candidate successor MSE against a persistence reference."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_pointmaze_causal_residual_dev_20261106_codex import (
    AUDIT, PRIVATE, PUBLIC, ROOT, SELECTION,
)
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAUSAL_RESIDUAL_PERSISTENCE_AUDIT_PROTOCOL_20261106_CODEX.md"
OUT = ROOT / "results/pc_pointmaze_causal_residual_persistence_audit_20261106_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    selection = json.loads(SELECTION.read_text())
    audit = json.loads(AUDIT.read_text())
    if (audit["selection_sha256"] != file_hash(SELECTION) or audit["physical_sha256"] != file_hash(PRIVATE)
            or selection["public_sha256"] != file_hash(PUBLIC)):
        raise ValueError("previous selection and physical data mismatch")
    model, _, permission = load_model()
    if model_fingerprint(model) != selection["model_fingerprint"] or permission.fingerprint != selection["input_audit_hash"]:
        raise ValueError("frozen model mismatch")
    rows = []
    print("Post-hoc persistence successor MSE; no model action choice is changed.", flush=True)
    with np.load(PRIVATE, allow_pickle=False) as physical, np.load(PUBLIC, allow_pickle=False) as inputs, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                current = model.encode(rgb_tensor(inputs["current"][i, camera]))[0]
                outcomes = [model.encode(rgb_tensor(physical["images"][3*i+1, a, camera, -1]))[0]
                            for a in range(3)]
                previous = next(r for r in audit["rows"] if r["root"] == 1884+i and r["camera"] == camera)
                rows.append({"root": 1884+i, "camera": camera,
                             "mse": {**previous["latent_mse"],
                                     "persistence": float(torch.stack([(current-z).square().mean()
                                                                       for z in outcomes]).mean())}})
    root_rows = [{"root": root,
                  "mse": {name: float(np.mean([r["mse"][name] for r in rows if r["root"] == root]))
                          for name in ("raw", "corrected", "persistence")}}
                 for root in range(1884, 1892)]
    means = {name: float(np.mean([r["mse"][name] for r in root_rows]))
             for name in ("raw", "corrected", "persistence")}
    output = {"rows": rows, "per_root": root_rows, "mean_mse": means,
              "corrected_better_than_persistence_roots": sum(
                  r["mse"]["corrected"] < r["mse"]["persistence"] for r in root_rows),
              "selection_sha256": file_hash(SELECTION), "previous_audit_sha256": file_hash(AUDIT),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "scope": "post-hoc state-assisted development diagnostic; no new selection"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_mse", "corrected_better_than_persistence_roots")}, indent=2))


if __name__ == "__main__":
    main()
