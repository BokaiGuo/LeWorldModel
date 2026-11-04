"""Independent arithmetic audit of real v11 full-chain and carry retention."""
import json

import numpy as np

from experiments.pc_jepa_native_v11_feasible_feedback_audit_20261104_codex import OUT, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    rows = []
    for step in range(3):
        base = OUT / f"response_{step}"
        data = json.loads((base / "output/decision.json").read_text())
        compiled = data["decision"]["compiler"]
        horizon = 15 - 5 * step
        batches = data["evaluations"]
        with np.load(base / "input/context.npz", allow_pickle=False) as package:
            carry = package["carry"].copy()
        first = np.asarray(batches[0]["actions"], np.float32)
        require(first.shape == ((1 if step == 0 else 2), horizon, 2), "initial batch")
        require(np.array_equal(first[0], np.zeros((horizon, 2), np.float32)), "zero missing")
        if step:
            require(np.array_equal(first[1], carry), "carry not re-evaluated")
        scored = [(np.asarray(a, np.float32), float(s))
                  for batch in batches[:-1]
                  for a, s in zip(batch["actions"], batch["scores"], strict=True)
                  if len(a) == horizon]
        require(scored, "no full-chain evaluations")
        best_action, best_score = min(scored, key=lambda pair: pair[1])
        incumbent = np.asarray(compiled["incumbent"], np.float32)
        require(np.array_equal(best_action, incumbent), "incumbent is not first global minimum")
        pair_actions = np.asarray(batches[-1]["actions"], np.float32)
        pair_scores = np.asarray(batches[-1]["scores"], float)
        require(np.array_equal(pair_actions[0], incumbent), "incumbent missing from final pair")
        compare = incumbent.copy()
        compare[:5] = 0
        require(np.array_equal(pair_actions[1], compare), "incorrect paired intervention")
        selected_index = int(compiled["abstained"])
        require(selected_index == int(np.array_equal(pair_actions[0], pair_actions[1])
                                      or pair_scores[1] <= pair_scores[0]), "abstention rule")
        rows.append({"step": step, "full_before_pair": len(scored),
                     "best_pre_pair_score": best_score, "paired_scores": pair_scores.tolist(),
                     "incumbent_source": compiled["incumbent_source"],
                     "carry_initial_score": batches[0]["scores"][1] if step else None,
                     "abstained": bool(selected_index)})
    record = {"status": "passed", "rows": rows, "summary_sha256": sha(OUT / "summary.json"),
              "scope": "saved prediction records only; no model or physics replay"}
    with (OUT / "incumbent_audit.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
