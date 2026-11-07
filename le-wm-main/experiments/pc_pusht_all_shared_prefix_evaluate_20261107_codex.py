"""Evaluate sealed shared-prefix selections against earlier physical replays."""
import hashlib
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT, SOURCE


def evaluate_plan(plan, scores, batches, physical):
    candidates = plan["candidates"]
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid score array")
    if int(np.argmin(scores)) != plan["rescore_winner"]:
        raise ValueError("winner is not argmin")
    selected = candidates[plan["original_selected_position"]]
    winner = candidates[plan["rescore_winner"]]
    if selected["sha256"] != plan["selected_sha256"] or winner["sha256"] != plan["winner_sha256"]:
        raise ValueError("choice identity mismatch")
    actual_actions = []
    for item in candidates:
        action = batches[item["call"]]["actions"][0, item["slot"]].numpy()
        if hashlib.sha256(action.tobytes()).hexdigest() != item["sha256"]:
            raise ValueError("action fingerprint mismatch")
        actual_actions.append(action)
    prefix = actual_actions[plan["original_selected_position"]][:2]
    if not all(np.array_equal(action[:2], prefix) for action in actual_actions):
        raise ValueError("candidate prefix differs")
    if len({x.tobytes() for x in actual_actions}) != len(actual_actions):
        raise ValueError("duplicate candidate")
    records = {item["action_sha256"]: item for item in physical["records"]}
    if len(records) != len(physical["records"]):
        raise ValueError("duplicate physical identity")
    outcomes = [records[item["sha256"]] for item in candidates]
    if outcomes[plan["original_selected_position"]]["index"] != physical["selected_index"]:
        raise ValueError("original selection mismatch")
    order = np.argsort(scores, kind="stable")
    successful = [i for i in order if outcomes[i]["first_success_step"] is not None]
    final = [i for i in order if outcomes[i]["final_success"]]
    ranks = {int(candidate): rank + 1 for rank, candidate in enumerate(order)}
    def outcome(index):
        record = outcomes[index]
        return {"first_success_step": record["first_success_step"], "final_success": record["final_success"]}
    return {"task": plan["task"], "candidates": len(candidates),
            "ever_success_count": len(successful), "final_success_count": len(final),
            "original": outcome(plan["original_selected_position"]),
            "rescore": outcome(plan["rescore_winner"]),
            "best_ever_success_rank": ranks[int(successful[0])] if successful else None,
            "best_final_success_rank": ranks[int(final[0])] if final else None,
            "model_packet_evaluations": plan["model_packet_evaluations"],
            "predictor_seconds": plan["predictor_seconds"]}


def main():
    target = OUT.parent / (OUT.name + ".evaluation.json")
    if target.exists():
        raise FileExistsError(target)
    sealed = json.loads(OUT.with_suffix(".json").read_text())
    if file_hash(OUT.with_suffix(".npz")) != sealed["arrays_sha256"]:
        raise ValueError("sealed scores changed")
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    physical_hashes = {}
    results = []
    with np.load(OUT.with_suffix(".npz"), allow_pickle=False) as arrays:
        for plan in sealed["plans"]:
            task = plan["task"]
            path = BASE / f"physical_{task}.json"
            digest = file_hash(path)
            if previous["source_sha256"].get(str(path)) != digest:
                raise ValueError("physical replay changed")
            physical_hashes[str(path)] = digest
            physical = json.loads(path.read_text())
            batches = [torch.load(BASE / "capture" / f"call_{task * 5 + call:03}.pt",
                                  map_location="cpu", weights_only=True) for call in range(5)]
            results.append(evaluate_plan(plan, arrays[f"task_{task}"], batches, physical))
    with target.open("x") as stream:
        json.dump({"sealed_json_sha256": file_hash(OUT.with_suffix(".json")),
                   "physical_sha256": physical_hashes, "results": results}, stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
