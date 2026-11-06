"""Check fixed-observation embedding reuse against fresh same-action predictions."""
import json
from pathlib import Path

import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


def expanded(context, n):
    return {k: context[k].to("cuda").expand(1, n, *context[k].shape[2:]).clone()
            for k in ("pixels", "goal")}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/pc_lewm_cache_equivalence_dev_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    print("Compare reused embeddings versus fresh encoding for identical saved actions and contexts.", flush=True)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    records = []
    with torch.inference_mode():
        for task in (15, 25, 35):
            path = root / f"results/pc_branch_second_replan_lookahead_seed42_20261106_codex.capture/task_{task:02d}_replan_call_00.pt"
            batch = torch.load(path, map_location="cpu", weights_only=False)
            actions = batch["actions"][:, :16].to("cuda")
            context = expanded(batch["context"], 16)
            context["action"] = torch.zeros((1, 16, 1, 10), device="cuda")
            first = model.get_cost(context, actions).clone()
            # Scoring again in a new dictionary recomputes embeddings.
            fresh_context = expanded(batch["context"], 16)
            fresh_context["action"] = torch.zeros((1, 16, 1, 10), device="cuda")
            fresh = model.get_cost(fresh_context, actions).clone()
            # Reuse same dictionary and observation, as CEM does between iterations.
            cached = model.get_cost(context, actions).clone()
            records.append({"task": task, "saved_cost_max_abs": float((first.cpu() - batch["cost"][:, :16]).abs().max()),
                            "fresh_cost_max_abs": float((cached - fresh).abs().max()),
                            "fresh_prediction_max_abs": float((context["predicted_emb"] -
                                                                fresh_context["predicted_emb"]).abs().max()),
                            "fresh_goal_max_abs": float((context["goal_emb"] - fresh_context["goal_emb"]).abs().max()),
                            "candidates": 16})
    if fingerprint != model_fingerprint(model):
        raise ValueError("model mutated")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model_sha256": fingerprint, "records": records,
                   "scope": "same context and actions; not cross-context cache validity"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
