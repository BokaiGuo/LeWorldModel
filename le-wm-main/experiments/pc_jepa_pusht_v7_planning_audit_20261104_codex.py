"""Non-neural recount of the saved JEPA v7/CEM planning requests."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex.json"
OUTPUT = ROOT / "results/pc_jepa_pusht_v7_planning_audit_20261104_codex.json"
EXPECTED = "10a8cfa21c7821cd97b69cfd2c1078c9a0534c706f8251f1276dc9f88a154466"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(SOURCE) != EXPECTED:
        raise ValueError("source SHA differs")
    source = json.loads(SOURCE.read_text())
    report = {}
    for arm, row in source["results"].items():
        samples = tokens = module_calls = full_count = 0
        selected = row["decision"]["selected"]
        selected_action = np.asarray(selected["chain"], dtype=np.float32)
        occurrences, scores = [], []
        for event in row["evaluations"]:
            actions = np.asarray(event["actions"], dtype=np.float32)
            positions = np.asarray(event["positions"])
            values = np.asarray(event["scores"])
            if (actions.ndim != 3 or actions.shape[1] not in (5, 10, 15) or actions.shape[2] != 2
                    or values.shape != (len(actions),) or positions.shape != (len(actions), 2)
                    or not np.isfinite(actions).all() or not np.isfinite(values).all()
                    or np.max(np.abs(actions)) > 1.000001):
                raise ValueError("invalid saved event")
            count, steps = actions.shape[:2]
            samples += count
            tokens += count * steps // 5
            module_calls += math.ceil(count / 4) * (steps // 5)
            if steps == 15:
                full_count += count
                scores.extend(values.tolist())
                occurrences.extend(float(values[i]) for i in range(count)
                                   if np.array_equal(actions[i], selected_action))
        meter = row["meter"]
        if (samples != meter["trajectory_samples"] or tokens != meter["prediction_token_samples"]
                or len(row["evaluations"]) != meter["prediction_calls"] or tokens > meter["token_cap"]
                or module_calls != row["predictor_module_calls"] or not occurrences):
            raise ValueError("recount or selected action binding failed")
        if min(abs(value - selected["terminal_score"]) for value in occurrences) > 1e-6:
            raise ValueError("selected score not found")
        report[arm] = {"calls": len(row["evaluations"]), "samples": samples,
                       "token_samples": tokens, "predictor_module_calls": module_calls,
                       "full_chain_samples": full_count, "min_saved_full_chain_score": min(scores),
                       "selected_score": selected["terminal_score"],
                       "selected_action_occurrences": len(occurrences)}
    v7 = source["results"]["v7_response"]
    pair = v7["decision"]["paired_comparison"]
    last = v7["evaluations"][-1]
    if len(last["scores"]) != 2 or not np.allclose(
        last["scores"], [pair["incumbent_score"], pair["zero_prefix_score"]], rtol=0, atol=1e-7
    ):
        raise ValueError("v7 paired comparison mismatch")
    output = {"scope": "non-neural record recount, not an independent prediction or physical replay",
              "source_sha256": EXPECTED, "arms": report, "paired_final_scores_match": True}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
