"""Recover branch elites and physical outcomes from fully scored captured batches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def run(mode, task=0, panel=False, offset5=False):
    root = Path(__file__).resolve().parents[1] / "results"
    if offset5 and not panel:
        raise ValueError("offset5 requires panel")
    roster = (5, 15, 25, 35, 45) if offset5 else (0, 10, 20, 30, 40)
    if task not in roster or (not panel and task != 0):
        raise ValueError("task outside predeclared roster")
    stem = (f"pc_branch_panel_offset5_{mode}_seed42_20261106_codex" if offset5 else
            f"pc_branch_panel_{mode}_seed42_20261105_codex" if panel else
            f"pc_branch_full_pool_{mode}_seed42_20261105_codex")
    datafile = f"{stem}.task_{task:02d}.physical.json" if panel else f"{stem}.physical.json"
    data = json.loads((root / datafile).read_text())
    lookup = {}
    for record in data["records"]:
        for call, slot in record["occurrences"]:
            if (call, slot) in lookup:
                raise ValueError("duplicate occurrence")
            lookup[call, slot] = record
    if len(lookup) != 3000:
        raise ValueError("incomplete pool")
    batches = [torch.load(root / f"{stem}.capture" /
                          (f"task_{task:02d}_call_{i:02d}.pt" if panel else f"call_{i:02d}.pt"),
                          map_location="cpu", weights_only=False) for i in range(20)]
    for i, batch in enumerate(batches):
        predicted, goal = batch["prediction"], batch["goal"]
        reconstructed = (predicted[..., -1, :] - goal[..., -1, :].unsqueeze(1)).square().sum(-1)
        if not torch.allclose(reconstructed, batch["cost"], rtol=1e-4, atol=1e-5):
            raise ValueError(f"prediction/cost mismatch at batch {i}")
    rounds = []
    initial = batches[0]["cost"][0]
    top = torch.topk(initial, 30, largest=False).indices.tolist()
    rounds.append({"iteration": 0, "successful_in_pool": sum(
        lookup[0, i]["first_success_step"] is not None for i in range(300)),
        "successful_elites": sum(lookup[0, i]["first_success_step"] is not None for i in top)})
    for step in range(1, 10):
        partial_call, proposal_call = 2 * step - 1, 2 * step
        count = 150 if step < 9 else 149
        partial = batches[partial_call]["cost"][0].reshape(2, count - 2)
        proposal = batches[proposal_call]["cost"][0].reshape(2, 2)
        combined = torch.cat((partial, proposal), dim=1)
        row = {"iteration": step, "branches": []}
        for branch in range(2):
            def identity(slot):
                return ((partial_call, branch * (count - 2) + slot) if slot < count - 2
                        else (proposal_call, branch * 2 + slot - (count - 2)))
            elites = torch.topk(combined[branch], 15, largest=False).indices.tolist()
            row["branches"].append({
                "successful_in_pool": sum(lookup[identity(i)]["first_success_step"] is not None for i in range(count)),
                "successful_elites": sum(lookup[identity(i)]["first_success_step"] is not None for i in elites),
            })
        rounds.append(row)
    final = [lookup[19, slot] for slot in range(2)]
    selected = int(batches[19]["cost"][0].argmin())
    scores = np.array([r["costs"][0] for r in data["records"]])
    success = np.array([r["first_success_step"] is not None for r in data["records"]])
    order = np.argsort(scores, kind="stable")
    success_ranks = np.flatnonzero(success[order])
    return {"mode": mode, "task": task, "unique": len(scores), "successes": int(success.sum()),
            "first_success_score_rank_1based": int(success_ranks[0] + 1) if len(success_ranks) else None,
            "lowest_score_succeeds": bool(success[order[0]]),
            "selected_final_slot": selected, "selected_first_success": final[selected]["first_success_step"],
            "selected_cost": final[selected]["costs"][0],
            "rounds": rounds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lookahead", "effect"), required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--offset5", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.task, args.panel, args.offset5), indent=2, allow_nan=False))
