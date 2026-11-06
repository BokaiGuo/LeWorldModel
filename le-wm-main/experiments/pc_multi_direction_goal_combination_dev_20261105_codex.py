"""Goal-weighted multisample directions on the already exposed E0 panel."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_good_chain_response_edit_v2_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_multi_direction_goal_combination_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_MULTI_DIRECTION_GOAL_COMBINATION_DEV_PROTOCOL_20261105_CODEX.md"
ARMS = tuple(f"{arm}_{budget}" for budget in ("fixed", "total") for arm in ("goal", "random", "probe"))


def combine_directions(actual_deltas, feature_deltas, residual, coefficients=None):
    actual_deltas = np.asarray(actual_deltas, dtype=np.float64)
    feature_deltas = np.asarray(feature_deltas, dtype=np.float64)
    weights = (feature_deltas @ np.asarray(residual, dtype=np.float64)
               if coefficients is None else np.asarray(coefficients, dtype=np.float64))
    direction = np.tensordot(weights, actual_deltas, axes=1)
    norm = float(np.max(np.abs(direction)))
    return (direction / norm if norm > 1e-12 else np.zeros_like(direction)), weights


def run_arm(predict, parent, target, arm, seed):
    parent = np.asarray(parent, dtype=np.float32)
    base, base_score = predict(parent[None])
    best, chosen, source = float(base_score[0]), parent.copy(), "parent"
    rng = np.random.default_rng(seed + 200000)
    probes = []
    for segment in range(3):
        for _ in range(4):
            signed = rng.choice((-1., 1.), size=(5, 2))
            for sign in (1, -1):
                chain = parent.copy()
                sl = slice(segment * 5, (segment + 1) * 5)
                chain[sl] = np.clip(parent[sl].astype(np.float64) + sign * .05 * signed, -1, 1)
                probes.append(chain)
    probes = np.stack(probes)
    features, scores = predict(probes)
    for index, score in enumerate(scores):
        if float(score) < best:
            best, chosen, source = float(score), probes[index].copy(), f"probe_{index}"
    directions, records = [], []
    random_rng = np.random.default_rng(seed + 400000)
    for segment in range(3):
        sl = slice(segment * 5, (segment + 1) * 5)
        pair = probes[segment * 8:(segment + 1) * 8]
        actual = (pair[0::2, sl].astype(np.float64) - pair[1::2, sl]) / 2
        feature_delta = (features[segment * 8:(segment + 1) * 8][0::2]
                         - features[segment * 8:(segment + 1) * 8][1::2]) / 2
        random_weights = random_rng.normal(size=4)
        direction, weights = combine_directions(
            actual, feature_delta, np.asarray(target) - base[0],
            random_weights if arm.startswith("random") else None)
        directions.append(direction)
        records.append({"actual_deltas": actual.tolist(), "feature_deltas": feature_delta.tolist(),
                        "weights": weights.tolist(), "direction": direction.tolist()})
    edits_record, edit_scores = [], []
    if not arm.startswith("probe"):
        edits, edits_record = prior.candidates(parent, directions)
        _, edit_scores_array = predict(edits)
        edit_scores = np.asarray(edit_scores_array).tolist()
        for index, score in enumerate(edit_scores_array):
            if float(score) < best:
                best, chosen, source = float(score), edits[index].copy(), f"edit_{index}"
    n_refine = (132 if arm.startswith("probe") else 120) if arm.endswith("_total") else 84
    result = fixed.optimizer.optimize(lambda actions: predict(actions)[1], np.empty((0, 2), np.float32),
                                      chosen[None], n_refine, seed=seed, batch_size=12, elites=4)
    if float(result["score"]) > best + 1e-12:
        raise ValueError("refinement lost best scored start")
    return {"parent_score": float(base_score[0]), "probe_scores": np.asarray(scores).tolist(),
            "direction_records": records, "edit_records": edits_record, "edit_scores": edit_scores,
            "start_origin": source, "refine_start": chosen.tolist(), "refine_start_score": best,
            "refine_samples": n_refine, "final_actions": result["actions"].tolist(),
            "final_score": float(result["score"]), "refinement": result["starts"]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((prior.OUT / "manifest.json").read_text())
    audit = json.loads((prior.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["rows"] != 192 or audit["summary_sha256"] != prior.digest(prior.OUT / "summary.json"):
        raise ValueError("source audit incomplete or mismatched")
    if any(prior.digest(Path(p)) != h for p, h in manifest["bindings"].items()):
        raise ValueError("source binding mismatch")
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["source_dependencies"])
    paths = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(fixed.optimizer.__file__),
             prior.OUT / "manifest.json", prior.OUT / "audit.json"]
    dependencies = seal.capture(paths)
    journal = fixed.prior.prior.prior.journal
    tasks = fixed.prior.tasks()
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks, "arms": ARMS, "dependencies": dependencies,
                                               "source_dependencies": manifest["source_dependencies"]})
    rows = []
    for root, task in enumerate(tasks):
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        if (source["root"], source["task"], source["status"]) != (root, task, "completed"):
            raise ValueError("parent source mismatch")
        parent = fixed.starts_from_result(source["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + 2 * root + repeat
            for arm in ARMS:
                n = 157 if arm.endswith("_total") else (109 if arm.startswith("probe") else 121)
                name = f"root{root:02d}_rep{repeat}_{arm}"
                recorder = journal.PredictionJournal(
                    OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=n * 3)
                row = {"root": root, "repeat": repeat, "arm": arm, "seed": seed}
                started = time.monotonic()
                try:
                    result = run_arm(recorder, parent, task["goal"], arm, seed)
                    if recorder.tokens != n * 3:
                        raise ValueError(f"token mismatch {recorder.tokens} != {n * 3}")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed_seconds=time.monotonic() - started)
                rows.append(row)
                journal.write_json(OUT / f"{name}.json", row)
                print(f"{name} {row['status']} tokens={row['tokens']}", flush=True)
    seal.verify(dependencies)
    seal.verify(manifest["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Development only: goal-weighted vs randomized directions, identical probes, fixed and equal total opportunity.", flush=True)
    main()
