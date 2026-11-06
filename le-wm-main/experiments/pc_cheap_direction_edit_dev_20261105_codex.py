"""Small-query directional edits against response, random and CEM controls."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_good_chain_response_edit_v2_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_cheap_direction_edit_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_CHEAP_DIRECTION_EDIT_DEV_PROTOCOL_20261105_CODEX.md"
ARMS = tuple(f"{method}_{budget}" for budget in ("fixed", "total")
             for method in ("cheap", "response", "random", "cem"))


def cheap_arm(predict, parent, target, seed, total):
    parent = np.asarray(parent, np.float32)
    base, values = predict(parent[None])
    chosen, best, source = parent.copy(), float(values[0]), "parent"
    rng = np.random.default_rng(seed + 200000)
    probes, directions = [], []
    for segment in range(3):
        signed = rng.choice((-1., 1.), size=(5, 2))
        pair = np.repeat(parent[None], 2, axis=0)
        sl = slice(5*segment, 5*(segment+1))
        pair[0, sl] = np.clip(parent[sl].astype(np.float64) + .05*signed, -1, 1)
        pair[1, sl] = np.clip(parent[sl].astype(np.float64) - .05*signed, -1, 1)
        probes.extend(pair)
    probes = np.stack(probes)
    features, scores = predict(probes)
    for index, score in enumerate(scores):
        if score < best:
            chosen, best, source = probes[index].copy(), float(score), f"probe_{index}"
    records = []
    for segment in range(3):
        sl = slice(5*segment, 5*(segment+1))
        actual_delta = probes[2*segment, sl].astype(np.float64) - probes[2*segment+1, sl]
        feature_delta = features[2*segment] - features[2*segment+1]
        projection = float(np.dot(feature_delta, np.asarray(target, np.float64) - base[0]))
        direction = actual_delta * projection
        norm = float(np.max(np.abs(direction)))
        directions.append(direction / norm if norm > 1e-12 else np.zeros_like(direction))
        records.append({"actual_delta": actual_delta.tolist(), "feature_delta": feature_delta.tolist(),
                        "projection": projection, "zero_direction": norm <= 1e-12})
    edits, edit_records = prior.candidates(parent, directions)
    _, scores = predict(edits)
    for index, score in enumerate(scores):
        if score < best:
            chosen, best, source = edits[index].copy(), float(score), f"edit_{index}"
    refinements = 138 if total else 84
    result = fixed.optimizer.optimize(lambda actions: predict(actions)[1], np.empty((0, 2), np.float32),
                                      chosen[None], refinements, seed=seed, batch_size=12, elites=4)
    if float(result["score"]) > best + 1e-12:
        raise ValueError("refinement lost previously scored incumbent")
    return {"parent_score": float(values[0]), "direction_records": records,
            "edits": edit_records, "start_origin": source, "refine_start": chosen.tolist(),
            "refine_start_score": best, "refine_samples": refinements,
            "final_actions": result["actions"].tolist(), "final_score": float(result["score"]),
            "refinement": result["starts"]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest = json.loads((prior.OUT / "manifest.json").read_text())
    source_audit = json.loads((prior.OUT / "audit.json").read_text())
    if source_audit["status"] != "passed" or source_audit["rows"] != 192:
        raise ValueError("V2 audit incomplete")
    if source_audit["summary_sha256"] != prior.digest(prior.OUT / "summary.json"):
        raise ValueError("V2 summary mismatch")
    if any(prior.digest(Path(p)) != h for p, h in source_manifest["bindings"].items()):
        raise ValueError("V2 dependency mismatch")
    seal = fixed.prior.prior.prior.seal
    seal.verify(source_manifest["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(fixed.optimizer.__file__),
                         Path(fixed.prior.prior.prior.journal.__file__),
                         prior.OUT / "manifest.json", prior.OUT / "audit.json"])
    OUT.mkdir()
    journal = fixed.prior.prior.prior.journal
    tasks = fixed.prior.tasks()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks, "arms": ARMS, "dependencies": deps,
                                               "source_dependencies": source_manifest["source_dependencies"]})
    rows = []
    for root, task in enumerate(tasks):
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        if (source["root"], source["task"], source["status"]) != (root, task, "completed"):
            raise ValueError("parent source mismatch")
        parent = fixed.starts_from_result(source["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + 2*root + repeat
            for arm in ARMS:
                name = f"root{root:02d}_rep{repeat}_{arm}"
                samples = {"cheap_fixed": 103, "response_fixed": 157, "random_fixed": 97,
                           "cem_fixed": 84}.get(arm, 157)
                rec = journal.PredictionJournal(
                    OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=samples*3)
                row = {"root": root, "repeat": repeat, "arm": arm, "seed": seed}
                started = time.monotonic()
                try:
                    result = (cheap_arm(rec, parent, task["goal"], seed, arm.endswith("_total"))
                              if arm.startswith("cheap") else
                              prior.run_arm(rec, parent, task["goal"], arm, seed, seed+100000))
                    if rec.tokens != samples*3 or result["final_score"] > result["parent_score"] + 1e-12:
                        raise ValueError("budget or incumbent failure")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=rec.tokens, calls=rec.calls, elapsed_seconds=time.monotonic()-started)
                rows.append(row)
                journal.write_json(OUT / f"{name}.json", row)
                print(f"{name} {row['status']} tokens={row['tokens']}", flush=True)
    seal.verify(deps)
    seal.verify(source_manifest["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Testing directional probes with fixed refinement and equal token budgets.", flush=True)
    main()
