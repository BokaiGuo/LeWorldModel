"""Inject a privileged known plan only into the sealed policy's refinement boundary."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as exp

PROTOCOL = exp.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_WITNESS_INJECTION_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_witness_injection_diagnostic.json")


def witness(task):
    candidates = [x for x in exp.exact.wall.feasibility(task)
                  if x["within_horizon"] and x["steps_needed"] <= 15]
    if not candidates:
        raise ValueError("no eligible known witness")
    item = min(candidates, key=lambda x: x["steps_needed"])
    actions = np.asarray(item["actions"], dtype=float)[:15] / exp.exact.wall.HIGH
    if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1 + 1e-12:
        raise ValueError("invalid witness")
    return actions, item


def run_one(task, root):
    actions, source = witness(task)
    original_refine = exp.policy.policy.refine
    captured = {}

    def diagnostic_refine(chain, initial_score, count, rng, predict):
        if count < 1 or captured:
            raise ValueError("unsupported refinement")
        normal_chain, normal_score = original_refine(chain, initial_score, count - 1, rng, predict)
        injected_score = float(predict(actions[None])[0])
        captured.update({"normal_chain": normal_chain.tolist(), "normal_score": normal_score,
                         "witness_score": injected_score, "refinement_samples": count,
                         "regular_refinement_samples": count - 1})
        return (actions.copy(), injected_score) if injected_score < normal_score else (normal_chain, normal_score)

    meter = exp.Budget()
    try:
        exp.policy.policy.refine = diagnostic_refine
        result = exp.policy.decide("successor", root, 0, np.asarray(task["goal"]),
                                   lambda x: meter.predict(task, x), meter)
    finally:
        exp.policy.policy.refine = original_refine
    if not captured:
        raise ValueError("refinement hook not used")
    action32 = actions.astype(np.float32)
    position = exp.exact.endpoints(task, action32[None])[0]
    score = float(np.linalg.norm(position - task["goal"]))
    if abs(score - captured["witness_score"]) > 1e-12:
        raise ValueError("witness score differs")
    if abs(result["selected"]["terminal_score"] -
           min(captured["normal_score"], score, result["final_prefix_comparison"]["comparison_score"])) > 1e-12:
        raise ValueError("final selector did not retain minimum")
    found = any(np.any(np.all(np.asarray(batch["requested_actions"]) == actions, axis=(1, 2)))
                for batch in result["prediction_trace"] if np.asarray(batch["requested_actions"]).shape[1:] == (15, 2))
    if not found:
        raise ValueError("injection not in actual trace")
    return {"root": root, "task": task, "witness": actions.tolist(), "witness_steps": source["steps_needed"],
            "max_normalized_action": float(np.max(np.abs(actions))),
            "float32_max_action_error": float(np.max(np.abs(action32.astype(float) - actions))),
            "witness_position": position.tolist(), "witness_distance": score, "hook": captured,
            "tokens": meter.token_samples, "result": result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(exp.PLANS.read_text())
    if prior["dependencies"] != exp.dependencies() or json.loads(exp.AUDIT.read_text())["plans_sha256"] != exp.sha(exp.PLANS):
        raise ValueError("prior seal mismatch")
    sources = {str(p): exp.sha(p) for p in (Path(__file__), PROTOCOL, exp.PLANS, exp.AUDIT)}
    print("Known-plan injection at refinement boundary; privileged diagnostic only.", flush=True)
    rows = [run_one(row["task"], row["root"]) for row in prior["rows"]]
    if sources != {str(p): exp.sha(p) for p in (Path(__file__), PROTOCOL, exp.PLANS, exp.AUDIT)}:
        raise ValueError("source drift")
    with OUTPUT.open("x") as stream:
        json.dump({"sources": sources, "prior_dependencies": prior["dependencies"], "rows": rows,
                   "scope": "privileged injected diagnostic; not planning performance"},
                  stream, separators=(",", ":"), allow_nan=False)
    print(f"Replayed and injected {len(rows)} known plans.", flush=True)


if __name__ == "__main__":
    main()
