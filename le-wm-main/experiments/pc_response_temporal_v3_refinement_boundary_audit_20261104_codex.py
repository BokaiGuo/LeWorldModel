"""Independently locate the joint-CEM boundary from logged generation batches."""
import json

import numpy as np

from experiments import pc_response_temporal_v3_fixed_terminal_e0_20261104_codex as panel

OUT = panel.PREFIX.with_name(panel.PREFIX.name + "_refinement_boundary_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if json.loads(panel.AUDIT.read_text())["plans_sha256"] != panel.prior.sha(panel.PLANS):
        raise ValueError("panel audit seal mismatch")
    data, rows = json.loads(panel.PLANS.read_text()), []
    if data["dependencies"] != panel.dependencies():
        raise ValueError("dependencies changed")
    for row in data["rows"]:
        if row["arm"] not in ("response", "matched_random"):
            continue
        for item in row["decisions"]:
            result, trace = item["decision"]["result"], item["decision"]["trace"]
            generations = result["refinement"]
            boundary = len(trace) - 1 - len(generations)
            if boundary < 1 or sum(g["size"] for g in generations) != result["refine_samples"]:
                raise ValueError("invalid refinement boundary")
            for generation, batch in zip(generations, trace[boundary:-1], strict=True):
                if generation["size"] != len(batch["scores"]):
                    raise ValueError("generation/batch mismatch")
            horizon = 15 - 5 * item["step"]
            before = [(actions, score) for batch in trace[:boundary]
                      if len(batch["requested"][0]) == horizon
                      for actions, score in zip(batch["requested"], batch["scores"], strict=True)]
            start = result["refinement_start"]
            if (len(before) != start["full_samples_seen"] or not before
                    or start["terminal_score"] != min(score for _, score in before)
                    or not any(np.array_equal(actions, start["chain"]) and score == start["terminal_score"]
                               for actions, score in before)):
                raise ValueError("refinement does not start from previous best")
            rows.append({"root": row["root"], "arm": row["arm"], "step": item["step"],
                         "pre_refinement_complete_samples": len(before), "boundary_batch": boundary})
    if len(rows) != 144:
        raise ValueError("incomplete panel")
    panel.prior.write(OUT, {"plans_sha256": panel.prior.sha(panel.PLANS),
                            "panel_audit_sha256": panel.prior.sha(panel.AUDIT), "rows": rows})
    print("Independently located and verified all 144 structured-arm joint-CEM boundaries.", flush=True)


if __name__ == "__main__":
    main()
