"""Fixed-opportunity and equal-token parent-chain edits on exposed E0 tasks."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments import pc_action_effect_fixed_refinement_dev_20261105_codex as fixed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_good_chain_response_edit_v2_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_GOOD_CHAIN_RESPONSE_EDIT_V2_DEV_PROTOCOL_20261105_CODEX.md"
ARMS = ("response_fixed", "random_fixed", "cem_fixed", "response_total", "random_total", "cem_total")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direction_response(predict, parent, target, base_feature):
    directions, matrices, denominators = [], [], []
    best_probe, best_score, best_id = None, float("inf"), None
    for segment in range(3):
        probes = np.repeat(parent[None], 20, axis=0)
        for coordinate in range(10):
            step, axis = divmod(coordinate, 2)
            location = segment * 5 + step
            probes[2 * coordinate, location, axis] = np.clip(float(parent[location, axis]) + .05, -1, 1)
            probes[2 * coordinate + 1, location, axis] = np.clip(float(parent[location, axis]) - .05, -1, 1)
        features, scores = predict(probes)
        winner = int(np.argmin(scores))
        if scores[winner] < best_score:
            best_probe, best_score, best_id = probes[winner].copy(), float(scores[winner]), f"fd_{segment}_{winner}"
        denom = np.array([float(probes[2*k].reshape(-1)[segment*10+k])
                          - float(probes[2*k+1].reshape(-1)[segment*10+k]) for k in range(10)])
        if np.any(denom <= 0):
            raise ValueError("zero actual finite difference")
        matrix = ((features[0::2] - features[1::2]) / denom[:, None]).T
        direction = matrix.T @ (target - base_feature)
        scale = np.max(np.abs(direction))
        directions.append((direction / scale if scale > 1e-12 else np.zeros(10)).reshape(5, 2))
        matrices.append(matrix.tolist())
        denominators.append(denom.tolist())
    return directions, matrices, denominators, best_probe, best_score, best_id


def candidates(parent, directions):
    results, edits = [], []
    for segment, direction in enumerate(directions):
        for amplitude in (-.5, -.25, .25, .5):
            chain = parent.copy()
            sl = slice(segment*5, (segment+1)*5)
            chain[sl] = np.clip(chain[sl].astype(np.float64) + amplitude*direction, -1, 1)
            delta = chain[sl].astype(np.float64) - parent[sl].astype(np.float64)
            edits.append({"segment": segment, "amplitude": amplitude,
                          "actual_l2": float(np.linalg.norm(delta)),
                          "actual_linf": float(np.max(np.abs(delta)))})
            results.append(chain)
    return np.stack(results), edits


def run_arm(predict, parent, target, arm, seed, random_seed):
    parent = np.asarray(parent, np.float32)
    start_feature, start_score = predict(parent[None])
    start_score = float(start_score[0])
    response, random = arm.startswith("response"), arm.startswith("random")
    record = {"parent_score": start_score, "candidate_scores": [], "edits": [],
              "matrices": [], "denominators": [], "selected_source": "parent"}
    chosen, chosen_score = parent.copy(), start_score
    if response or random:
        if response:
            directions, record["matrices"], record["denominators"], probe, probe_score, probe_id = direction_response(
                predict, parent, np.asarray(target, np.float64), start_feature[0])
            if probe_score < chosen_score:
                chosen, chosen_score = probe, probe_score
                record["selected_source"] = probe_id
        else:
            rng = np.random.default_rng(random_seed)
            directions = []
            for _ in range(3):
                d = rng.normal(size=(5, 2))
                directions.append(d / np.max(np.abs(d)))
        edits, record["edits"] = candidates(parent, directions)
        _, scores = predict(edits)
        record["candidate_scores"] = scores.tolist()
        winner = int(np.argmin(scores))
        if scores[winner] < chosen_score:
            chosen, chosen_score = edits[winner].copy(), float(scores[winner])
            record["selected_source"] = f"edit_{winner}"
    is_total = arm.endswith("total")
    if response:
        budget = 84
    elif random:
        budget = 144 if is_total else 84
    else:
        # The parent score above is charged, so avoid double counting it in the total.
        budget = 156 if is_total else 83
    result = fixed.optimizer.optimize(lambda chains: predict(chains)[1], np.empty((0, 2), np.float32),
                                      chosen[None], budget, seed=seed, batch_size=12, elites=4)
    record.update(refine_start=chosen.tolist(), refine_start_score=chosen_score,
                  final_actions=result["actions"].tolist(), final_score=float(result["score"]),
                  refinement=result["starts"], refine_samples=budget)
    return record


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = fixed.prior.OUT
    source_manifest = json.loads((source / "manifest.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    fixed.prior.prior.prior.seal.verify(source_manifest["dependencies"])
    if audit["status"] != "passed" or audit["run_count"] != 80 or source_manifest["tasks"] != fixed.prior.tasks():
        raise ValueError("source audit or tasks mismatch")
    paths = [PROTOCOL, Path(__file__), Path(fixed.__file__), Path(fixed.optimizer.__file__),
             source / "manifest.json", source / "audit.json"]
    paths += [source / f"plan_{root:02d}_random_min.json" for root in range(16)]
    bindings = {str(path): digest(path) for path in paths}
    OUT.mkdir()
    manifest = {"status": "prepared", "bindings": bindings, "source_dependencies": source_manifest["dependencies"],
                "tasks": fixed.prior.tasks(), "arms": ARMS, "repeats": 2}
    fixed.prior.prior.prior.journal.write_json(OUT / "manifest.json", manifest)
    rows = []
    for root, task in enumerate(manifest["tasks"]):
        source_row = json.loads((source / f"plan_{root:02d}_random_min.json").read_text())
        if (source_row["status"], source_row["root"], source_row["task"]) != ("completed", root, task):
            raise ValueError("source row mismatch")
        parent = fixed.starts_from_result(source_row["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + 2*root + repeat
            for arm in ARMS:
                recorder = fixed.prior.prior.prior.journal.PredictionJournal(
                    OUT / f"root{root:02d}_rep{repeat}_{arm}_calls",
                    fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=471)
                start = time.monotonic()
                row = {"root": root, "repeat": repeat, "arm": arm, "seed": seed}
                try:
                    result = run_arm(recorder, parent, task["goal"], arm, seed, seed + 100000)
                    expected = {"response_fixed": 157, "random_fixed": 97, "cem_fixed": 84,
                                "response_total": 157, "random_total": 157, "cem_total": 157}[arm]
                    if recorder.tokens != expected*3 or result["final_score"] > result["parent_score"] + 1e-12:
                        raise ValueError("budget or incumbent mismatch")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed_seconds=time.monotonic()-start)
                rows.append(row)
                fixed.prior.prior.prior.journal.write_json(OUT / f"root{root:02d}_rep{repeat}_{arm}.json", row)
                print(f"root={root} rep={repeat} arm={arm} status={row['status']} tokens={row['tokens']}", flush=True)
    fixed.prior.prior.prior.seal.verify(source_manifest["dependencies"])
    if any(digest(Path(path)) != value for path, value in bindings.items()):
        raise ValueError("dependency changed")
    fixed.prior.prior.prior.journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Comparing response and random edits around shared global parent chains.", flush=True)
    main()
