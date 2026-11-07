"""Compare custom latent rollout against native rollout and captured predictions."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT, SOURCE
from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import ROOT

TARGET = ROOT / "results/pc_pusht_conditional_rollout_parity_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_CONDITIONAL_ROLLOUT_PARITY_PROTOCOL_20261107_CODEX.md"


def native_from_embedding(model, initial, actions):
    if initial.ndim != 2 or initial.shape[0] != 1 or actions.ndim != 3:
        raise ValueError("expected one shared initial state and batch of chains")
    pixels = torch.zeros((1, actions.shape[0], 1, 1), device=actions.device)
    def encode(info):
        return {**info, "emb": initial[None]}
    with patch.object(model, "encode", side_effect=encode):
        info = model.rollout({"pixels": pixels}, actions[None])
    return info["predicted_emb"][0, :, 1:]


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    sealed = json.loads(OUT.with_suffix(".json").read_text())
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    if file_hash(OUT.with_suffix(".npz")) != sealed["arrays_sha256"]:
        raise ValueError("sealed scores changed")
    if file_hash(SOURCE.with_suffix(".npz")) != previous["arrays_sha256"]:
        raise ValueError("source embeddings changed")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != sealed["model_sha256"] or fingerprint != previous["model_sha256"]:
        raise ValueError("checkpoint mismatch")
    rows = []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source, \
         np.load(OUT.with_suffix(".npz"), allow_pickle=False) as score_archive, torch.inference_mode():
        for plan in sealed["plans"]:
            task = plan["task"]
            batch_data = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                                     map_location="cpu", weights_only=True) for i in range(5)]
            captured_errors, native_errors, batch_sizes = [], [], []
            for batch in batch_data:
                acts = batch["actions"][0].to("cuda")
                pred = batch["prediction"][0].to("cuda")
                initial = pred[0, :1]
                torch.testing.assert_close(pred[:, 0], initial.expand_as(pred[:, 0]), atol=1e-6, rtol=1e-6)
                manual = conditional_rollout(model, initial, acts, 0)
                native = native_from_embedding(model, initial, acts)
                captured_errors.append((manual - pred[:, 1:]).abs().max().item())
                native_errors.append((manual - native).abs().max().item())
                batch_sizes.append(acts.shape[0])
            idx = [i for i, item in enumerate(previous["rows"]) if item["task"] == task and item["selected"]]
            if len(idx) != 1:
                raise ValueError("selected observation ambiguous")
            initial = torch.as_tensor(source["actual"][idx[0], 1:2], device="cuda")
            goal = torch.as_tensor(source["goal"][idx[0]], device="cuda")
            acts = torch.stack([batch_data[x["call"]]["actions"][0, x["slot"], 2:]
                                for x in plan["candidates"]]).to("cuda")
            manual = conditional_rollout(model, initial, acts, 0)
            native = native_from_embedding(model, initial, acts)
            manual_score = (manual[:, -1] - goal).square().sum(-1)
            native_score = (native[:, -1] - goal).square().sum(-1)
            old = torch.as_tensor(score_archive[f"task_{task}"], device="cuda")
            rows.append({"task": task, "captured_batch_sizes": batch_sizes,
                         "max_captured_prediction_abs_error": max(captured_errors),
                         "max_initial_native_abs_error": max(native_errors),
                         "suffix_candidates": len(acts),
                         "max_suffix_native_abs_error": (manual - native).abs().max().item(),
                         "max_saved_score_abs_error": (manual_score - old).abs().max().item(),
                         "max_native_score_abs_error": (native_score - old).abs().max().item(),
                         "sealed_winner": plan["rescore_winner"],
                         "native_winner": int(native_score.argmin().item())})
            print(json.dumps(rows[-1]), flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with TARGET.open("x") as stream:
        json.dump({"model_sha256": fingerprint, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)),
                   "sealed_json_sha256": file_hash(OUT.with_suffix(".json")),
                   "rows": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
