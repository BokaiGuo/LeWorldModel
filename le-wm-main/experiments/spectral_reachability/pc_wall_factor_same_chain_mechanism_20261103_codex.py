"""Post-outcome paired intervention on every actually selected factorial plan."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_retention_factor_dev_20261103_codex as factor

prior = factor.prior
ROOT = factor.ROOT
PROTOCOL = ROOT.parent / "docs/PC_WALL_FACTOR_SAME_CHAIN_MECHANISM_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_wall_factor_same_chain_mechanism_20261103_codex.json"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed = json.loads(factor.SEAL.read_text())
    report = json.loads(factor.OUTPUT.read_text())
    original = factor.load_prior()
    sha = prior.base.digest
    if (report["seal_sha256"] != sha(factor.SEAL)
            or sealed["prior_sha256"] != sha(prior.OUTPUT)
            or sealed["runner_sha256"] != sha(Path(factor.__file__))
            or sealed["protocol_sha256"] != sha(factor.PROTOCOL)):
        raise ValueError("factor source binding mismatch")
    hashes = {str(path): sha(path) for path in (PROTOCOL, Path(__file__), factor.OUTPUT,
              factor.SEAL, Path(factor.__file__), prior.OUTPUT, Path(prior.__file__),
              Path(prior.base.__file__), Path(prior.base.physics.__file__))}
    bank = prior.prefix_bank()
    fixed = prior.probe_family()
    rows = []
    additional_optimization_calls = 0
    intervention_evaluations = 0
    for taskrow, oldrow in zip(report["rows"], original["rows"], strict=True):
        index, task = oldrow["index"], oldrow["task"]
        if index != taskrow["index"]:
            raise ValueError("task mismatch")
        screen = oldrow["screen"]
        initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - task["goal"]))
        for newbudget, oldbudget in zip(taskrow["budgets"], oldrow["budgets"], strict=True):
            cap = newbudget["cap"]
            if cap != oldbudget["cap"]:
                raise ValueError("budget mismatch")
            extra = {}
            for arm, selected in newbudget["arms"].items():
                optimizer = arm.rsplit("_", 1)[1]
                selected_index = selected["selected_index"]
                chain = np.asarray(selected["actions"])
                if not np.array_equal(chain[:6], bank[selected_index]):
                    raise ValueError("selected prefix mismatch")
                indices = {"selected": selected_index, "zero": 0,
                           "greedy": screen["same_bank_greedy"]}
                tails = np.concatenate((fixed, chain[None, 6:]), axis=0)
                controls = {}
                for name, idx in indices.items():
                    if optimizer == "original":
                        plan = oldbudget["census"][idx]
                    else:
                        key = str(idx)
                        if key in newbudget["stratified_plans"]:
                            plan = newbudget["stratified_plans"][key]
                        else:
                            if idx not in extra:
                                extra[idx] = factor.stratified(task, index, bank[idx], screen["probe_costs"][idx], cap)
                                additional_optimization_calls += cap
                            plan = extra[idx]
                    distances = prior.evaluate(task, bank[idx], tails)
                    intervention_evaluations += len(tails)
                    own_replay = prior.evaluate(task, bank[idx], np.asarray(plan["suffix"])[None])[0]
                    intervention_evaluations += 1
                    if not np.isclose(own_replay, plan["terminal_distance"], atol=1e-12, rtol=0):
                        raise ValueError("own-plan replay mismatch")
                    controls[name] = {"index": idx, "prefix": bank[idx].tolist(),
                                      "prefix_distance": screen["prefix_costs"][idx],
                                      "common_suffix_distances": distances.tolist(),
                                      "relative_to_zero_suffix": (distances - distances[0]).tolist(),
                                      "own_optimized_suffix": plan["suffix"],
                                      "own_planned_distance": float(own_replay)}
                chosen = controls["selected"]
                if not np.array_equal(chain[6:], np.asarray(chosen["own_optimized_suffix"])):
                    raise ValueError("selected suffix mismatch")
                nonprogress = chosen["prefix_distance"] >= initial - 1e-12
                active = all(chosen["relative_to_zero_suffix"][-1] <
                             controls[c]["relative_to_zero_suffix"][-1] - 1e-12 for c in ("zero", "greedy"))
                own = all(chosen["own_planned_distance"] < controls[c]["own_planned_distance"] - 1e-12
                          for c in ("zero", "greedy"))
                rows.append({"task": index, "cap": cap, "arm": arm, "initial_distance": initial,
                             "selected_actions": chain.tolist(), "controls": controls,
                             "nonprogress": bool(nonprogress), "active_response": bool(active),
                             "own_plan_advantage": bool(own),
                             "limited_witness": bool(nonprogress and active and own)})
        print(f"audited selected chains for task {index}", flush=True)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("dependency changed during diagnostic")
    summary = {}
    for cap in prior.CAPS:
        summary[str(cap)] = {}
        for arm in sorted({row["arm"] for row in rows}):
            selected = [row for row in rows if row["cap"] == cap and row["arm"] == arm]
            summary[str(cap)][arm] = {key: sum(row[key] for row in selected)
                                      for key in ("nonprogress", "active_response", "own_plan_advantage", "limited_witness")}
    result = {"scope": "post-outcome exact-model development diagnostic, not PC admission",
              "dependencies": hashes, "rows": rows, "summary": summary,
              "additional_optimization_calls": additional_optimization_calls,
              "intervention_evaluations": intervention_evaluations,
              "PC2_pass": False, "PC3_pass": False, "P1_admission": False}
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
