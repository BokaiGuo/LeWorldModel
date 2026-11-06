"""Budget-quality development curves for edits of shared global parent chains."""
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments import pc_good_chain_response_edit_v2_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_good_chain_budget_curve_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_GOOD_CHAIN_BUDGET_CURVE_DEV_PROTOCOL_20261105_CODEX.md"
BUDGETS = (157, 313, 625, 1249)
ARMS = ("response", "random", "cem")


def run_arm(predict, parent, target, arm, seed, budget):
    if arm not in ARMS or budget not in BUDGETS:
        raise ValueError("invalid arm or budget")
    parent = np.asarray(parent, np.float32)
    features, scores = predict(parent[None])
    parent_score = float(scores[0])
    chosen, chosen_score, origin = parent.copy(), parent_score, "parent"
    precompute = {"matrices": [], "denominators": [], "edits": [], "candidate_scores": []}
    if arm == "response":
        dirs, matrices, denoms, probe, score, probe_id = prior.direction_response(
            predict, parent, np.asarray(target, np.float64), features[0])
        precompute["matrices"], precompute["denominators"] = matrices, denoms
        if score < chosen_score:
            chosen, chosen_score, origin = probe, score, probe_id
    elif arm == "random":
        rng = np.random.default_rng(seed + 100000)
        dirs = []
        for _ in range(3):
            d = rng.normal(size=(5, 2))
            dirs.append(d / np.max(np.abs(d)))
    if arm in ("response", "random"):
        candidates, precompute["edits"] = prior.candidates(parent, dirs)
        _, scores = predict(candidates)
        precompute["candidate_scores"] = scores.tolist()
        winner = int(np.argmin(scores))
        if scores[winner] < chosen_score:
            chosen, chosen_score, origin = candidates[winner].copy(), float(scores[winner]), f"edit_{winner}"
    used = {"response": 73, "random": 13, "cem": 1}[arm]
    refined = fixed.optimizer.optimize(lambda actions: predict(actions)[1], np.empty((0, 2), np.float32),
                                        chosen[None], budget-used, seed=seed, batch_size=12, elites=4)
    return {"parent_score": parent_score, "precompute": precompute, "start_origin": origin,
            "start_actions": chosen.tolist(), "start_score": chosen_score,
            "refine_samples": budget-used, "final_actions": refined["actions"].tolist(),
            "final_score": float(refined["score"]), "refinement": refined["starts"]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = prior.OUT
    v2_manifest = json.loads((previous / "manifest.json").read_text())
    audit = json.loads((previous / "audit.json").read_text())
    if audit["status"] != "passed" or audit["rows"] != 192 or audit["summary_sha256"] != prior.digest(previous / "summary.json"):
        raise ValueError("prior V2 audit mismatch")
    if any(prior.digest(Path(p)) != d for p, d in v2_manifest["bindings"].items()):
        raise ValueError("prior source mismatch")
    fixed.prior.prior.prior.seal.verify(v2_manifest["source_dependencies"])
    dependencies = fixed.prior.prior.prior.seal.capture([
        Path(__file__), PROTOCOL, Path(prior.__file__), Path(fixed.optimizer.__file__),
        Path(fixed.prior.prior.prior.journal.__file__), previous / "manifest.json", previous / "audit.json"])
    OUT.mkdir()
    manifest = {"dependencies": dependencies, "tasks": fixed.prior.tasks(), "arms": ARMS, "budgets": BUDGETS,
                "source_dependencies": v2_manifest["source_dependencies"]}
    fixed.prior.prior.prior.journal.write_json(OUT / "manifest.json", manifest)
    rows = []
    for root, task in enumerate(manifest["tasks"]):
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        if (source["root"], source["task"], source["status"]) != (root, task, "completed"):
            raise ValueError("source parent mismatch")
        parent = fixed.starts_from_result(source["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + 2*root + repeat
            for budget in BUDGETS:
                for arm in ARMS:
                    name = f"r{root:02d}_rep{repeat}_n{budget}_{arm}"
                    recorder = fixed.prior.prior.prior.journal.PredictionJournal(
                        OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                        action_dim=2, packet_length=5, token_cap=budget*3)
                    row = {"root": root, "repeat": repeat, "budget": budget, "arm": arm, "seed": seed}
                    start = time.monotonic()
                    try:
                        result = run_arm(recorder, parent, task["goal"], arm, seed, budget)
                        if recorder.tokens != budget*3 or result["final_score"] > result["parent_score"] + 1e-12:
                            raise ValueError("budget or incumbent mismatch")
                        row.update(status="completed", result=result)
                    except Exception:
                        row.update(status="failed", error=traceback.format_exc())
                    row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed_seconds=time.monotonic()-start)
                    rows.append(row)
                    fixed.prior.prior.prior.journal.write_json(OUT / f"{name}.json", row)
                    print(f"{name} {row['status']} tokens={row['tokens']}", flush=True)
    fixed.prior.prior.prior.seal.verify(dependencies)
    fixed.prior.prior.prior.seal.verify(v2_manifest["source_dependencies"])
    fixed.prior.prior.prior.journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Running finite paired budget-quality curves for shared parent chains.", flush=True)
    main()
