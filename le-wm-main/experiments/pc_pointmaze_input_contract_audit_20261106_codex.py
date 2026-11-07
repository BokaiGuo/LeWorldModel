"""Audit the actual multimodal encode contract without altering sealed experiments."""
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model

PREFIX = "pc_pointmaze_causal_residual_replication_20261106_codex"
PUBLIC = ROOT / "results" / (PREFIX + "_inputs.npz")
SELECTION = ROOT / "results" / (PREFIX + "_selection.json")
OUT = ROOT / "results/pc_pointmaze_input_contract_audit_20261106_codex.json"


def audit_one(model, dataset, image, state):
    """Compare public raw-state wrapper with a single explicit normalization."""
    visual = rgb_tensor(image)
    raw_state = torch.tensor(state, dtype=torch.float32, device="cuda")[None, None]
    prepared = ((raw_state.cpu() - dataset.proprio_mean) / dataset.proprio_std).cuda()
    multimodal = model.encode({"visual": visual, "proprio": raw_state})
    direct_proprio = model.encode_proprio(prepared)
    visual_only = model.encode(visual)
    proprio_error = float((multimodal["proprio"] - direct_proprio).abs().max())
    visual_error = float((multimodal["visual"] - visual_only).abs().max())
    double = model.encode({"visual": visual, "proprio": prepared})
    double_difference = float((multimodal["proprio"] - double["proprio"]).abs().max())
    return {"max_proprio_single_normalization_error": proprio_error,
            "max_visual_modal_path_error": visual_error,
            "double_normalization_proprio_difference": double_difference}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(SELECTION.read_text())
    if record["public_sha256"] != file_hash(PUBLIC):
        raise ValueError("selection input fingerprint mismatch")
    model, dataset, permission = load_model()
    fingerprint = model_fingerprint(model)
    if fingerprint != record["model_fingerprint"] or permission.fingerprint != record["input_audit_hash"]:
        raise ValueError("model or preprocessor differs")
    rows = []
    print("Auditing wrapper normalization and visual path; no outcome labels or planner changes.", flush=True)
    with np.load(PUBLIC, allow_pickle=False) as data, torch.inference_mode():
        for i in range(8):
            for camera in range(2):
                row = audit_one(model, dataset, data["current"][i, camera], data["current_proprio"][i, camera])
                row.update(root=1892+i, camera=camera)
                rows.append(row)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    p_err = max(r["max_proprio_single_normalization_error"] for r in rows)
    v_err = max(r["max_visual_modal_path_error"] for r in rows)
    if p_err > 1e-6 or v_err > 1e-6:
        raise ValueError("encode path mismatch")
    output = {"rows": rows, "max_proprio_single_normalization_error": p_err,
              "max_visual_modal_path_error": v_err,
              "minimum_double_normalization_proprio_difference": min(r["double_normalization_proprio_difference"] for r in rows),
              "selection_sha256": file_hash(SELECTION), "public_sha256": file_hash(PUBLIC),
              "runner_sha256": file_hash(__file__), "model_fingerprint": fingerprint,
              "input_audit_hash": permission.fingerprint,
              "scope": "state-assisted local-wrapper interface only; not official full equivalence"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("max_proprio_single_normalization_error",
                                              "max_visual_modal_path_error",
                                              "minimum_double_normalization_proprio_difference")}, indent=2))


if __name__ == "__main__":
    main()
