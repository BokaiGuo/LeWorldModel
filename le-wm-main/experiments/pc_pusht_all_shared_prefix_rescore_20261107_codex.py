"""Rescore every previously generated tail at the selected shared observation."""
import hashlib
import json
import time

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_scored_first_pool_run_20261107_codex import BASE
from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import ROOT

SOURCE = ROOT / "results/pc_pusht_pool_success_alignment_20261107_codex"
OUT = ROOT / "results/pc_pusht_all_shared_prefix_rescore_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_ALL_SHARED_PREFIX_RESCORING_PROTOCOL_20261107_CODEX.md"


def main():
    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    if file_hash(SOURCE.with_suffix(".npz")) != previous["arrays_sha256"]:
        raise ValueError("previous alignment changed")
    for name, digest in previous["source_sha256"].items():
        if name.rsplit("/", 1)[-1].startswith("physical_"):
            continue
        if file_hash(name) != digest:
            raise ValueError(f"source changed: {name}")
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays:
        actual, goals = arrays["actual"].copy(), arrays["goal"].copy()
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    if model_fingerprint(model) != previous["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    first_fingerprint = model_fingerprint(model)
    plans, score_arrays = [], {}
    print("Rescore the complete shared-prefix set; truth is opened only after choices are sealed.", flush=True)
    with torch.inference_mode():
        for task in range(4):
            indexes = [i for i, x in enumerate(previous["rows"]) if x["task"] == task and x["selected"]]
            if len(indexes) != 1:
                raise ValueError("missing selected observation")
            observation_index = indexes[0]
            chosen_row = previous["rows"][observation_index]
            query_batches = [torch.load(BASE / "capture" / f"call_{task * 5 + call:03}.pt",
                                        map_location="cpu", weights_only=True) for call in range(5)]
            selected_id = chosen_row["action_sha256"]
            candidates, scores, occurrences, seen = [], [], [], set()
            selected = None
            for call, batch in enumerate(query_batches):
                for slot, action in enumerate(batch["actions"][0].numpy()):
                    digest = hashlib.sha256(action.tobytes()).hexdigest()
                    if digest == selected_id:
                        selected = action.copy()
            if selected is None:
                raise ValueError("selected chain missing")
            for call, batch in enumerate(query_batches):
                for slot, action in enumerate(batch["actions"][0].numpy()):
                    key = action.tobytes()
                    if key in seen or not np.array_equal(action[:2], selected[:2]):
                        continue
                    seen.add(key)
                    candidates.append(action.copy())
                    scores.append(float(batch["cost"][0, slot]))
                    occurrences.append((call, slot, hashlib.sha256(key).hexdigest()))
            actions = torch.as_tensor(np.stack(candidates), device="cuda")
            goal = torch.as_tensor(goals[observation_index], device="cuda")
            observed = torch.as_tensor(actual[observation_index, 1:2], device="cuda")
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output = conditional_rollout(model, observed, actions[:, 2:], 0)
            values = (output[:, -1] - goal).square().sum(-1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            values = values.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("nonfinite rescores")
            selected_pos = next(i for i, (_, _, digest) in enumerate(occurrences) if digest == selected_id)
            winner = int(np.argmin(values))
            score_arrays[f"task_{task}"] = values
            plans.append({"task": task, "selected_sha256": selected_id,
                          "original_selected_position": selected_pos,
                          "rescore_winner": winner, "winner_sha256": occurrences[winner][2],
                          "candidates": [{"call": c, "slot": s, "sha256": h, "original_score": cost}
                                         for (c, s, h), cost in zip(occurrences, scores, strict=True)],
                          "model_packet_evaluations": 3 * len(candidates),
                          "predict_calls": 3, "predictor_seconds": elapsed})
            print(f"task={task}, eligible={len(candidates)}, winner={winner}", flush=True)
    if model_fingerprint(model) != first_fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **score_arrays)
    # Freeze decisions before loading physical-label result files.
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump({"model_sha256": first_fingerprint, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "source_json_sha256": file_hash(SOURCE.with_suffix(".json")),
                   "source_npz_sha256": file_hash(SOURCE.with_suffix(".npz")),
                   "arrays_sha256": file_hash(OUT.with_suffix(".npz")), "plans": plans}, stream,
                  indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
