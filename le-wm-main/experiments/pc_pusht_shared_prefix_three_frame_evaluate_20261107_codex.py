"""Evaluate sealed three-frame choices using prior physical replay labels."""
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_all_shared_prefix_evaluate_20261107_codex import evaluate_plan
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT as PRIOR
from experiments.pc_pusht_shared_prefix_three_frame_20261107_codex import OUT


def main():
    output = OUT.parent / (OUT.name + ".evaluation.json")
    if output.exists():
        raise FileExistsError(output)
    sealed = json.loads(OUT.with_suffix(".json").read_text())
    prior = json.loads(PRIOR.with_suffix(".json").read_text())
    previous_eval = json.loads((PRIOR.parent / (PRIOR.name + ".evaluation.json")).read_text())
    if file_hash(OUT.with_suffix(".npz")) != sealed["arrays_sha256"]:
        raise ValueError("scores changed")
    if file_hash(PRIOR.with_suffix(".json")) != sealed["prior_sha256"]:
        raise ValueError("candidate list changed")
    results = []
    with np.load(OUT.with_suffix(".npz"), allow_pickle=False) as arrays:
        for plan in sealed["plans"]:
            task = plan["task"]
            old = prior["plans"][task]
            if old["task"] != task or len(old["candidates"]) != plan["candidate_count"]:
                raise ValueError("candidate mismatch")
            path = BASE / f"physical_{task}.json"
            if file_hash(path) != previous_eval["physical_sha256"].get(str(path)):
                raise ValueError("physical file changed")
            physical = json.loads(path.read_text())
            batches = [torch.load(BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                                  map_location="cpu", weights_only=True) for i in range(5)]
            updated = {**old, "rescore_winner": plan["winner"], "winner_sha256": plan["winner_sha256"],
                       "predictor_seconds": plan["predictor_seconds"]}
            result = evaluate_plan(updated, arrays[f"task_{task}"], batches, physical)
            result["one_frame_winner"] = old["rescore_winner"]
            result["three_frame_winner"] = plan["winner"]
            results.append(result)
    with output.open("x") as stream:
        json.dump({"sealed_sha256": file_hash(OUT.with_suffix(".json")), "results": results}, stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
