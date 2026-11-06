"""Matched single versus two-segment complete-chain repair on E0."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_cheap_direction_edit_dev_20261105_codex as old

ROOT, fixed = old.ROOT, old.fixed
OUT = ROOT / "results/pc_paired_segment_repair_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PAIRED_SEGMENT_REPAIR_DEV_PROTOCOL_20261105_CODEX.md"


def make_candidates(parent, seed):
    parent = np.asarray(parent, np.float32)
    rng = np.random.default_rng(seed)
    singles, pairs, segments = [], [], []
    for _ in range(12):
        indices = sorted(rng.choice(3, size=2, replace=False).tolist())
        pair = parent.copy()
        for segment in indices:
            direction = rng.choice((-1., 1.), size=(5, 2))
            sl = slice(segment*5, (segment+1)*5)
            single = parent.copy()
            single[sl] = np.clip(parent[sl].astype(np.float64) + .35*direction, -1, 1)
            pair[sl] = single[sl]
            singles.append(single)
        pairs.append(pair)
        segments.append(indices)
    extras = []
    for _ in range(12):
        segment = int(rng.integers(3))
        direction = rng.choice((-1., 1.), size=(5, 2))
        sl = slice(segment*5, (segment+1)*5)
        single = parent.copy()
        single[sl] = np.clip(parent[sl].astype(np.float64) + .35*direction, -1, 1)
        extras.append(single)
    return np.stack(singles), np.stack(pairs), np.stack(extras), segments


def witnesses(parent_score, single_scores, pair_scores, parent, singles, segments):
    flags = []
    for i, (a, b) in enumerate(segments):
        changed_a = np.any(parent[5*a:5*a+5] != singles[2*i, 5*a:5*a+5])
        changed_b = np.any(parent[5*b:5*b+5] != singles[2*i+1, 5*b:5*b+5])
        flags.append(bool(changed_a and changed_b and pair_scores[i] <= parent_score-.01
                          and parent_score-single_scores[2*i] < .01
                          and parent_score-single_scores[2*i+1] < .01))
    return flags


def run_arm(predict, parent, arm, proposal_seed, optimizer_seed):
    parent = np.asarray(parent, np.float32)
    _, parent_values = predict(parent[None])
    singles, pairs, extras, segments = make_candidates(parent, proposal_seed)
    _, single_scores = predict(singles)
    extra = pairs if arm == "pair" else extras
    _, extra_scores = predict(extra)
    candidates = np.concatenate((parent[None], singles, extra), axis=0)
    scores = np.concatenate((parent_values, single_scores, extra_scores))
    index = int(np.argmin(scores))
    selected = candidates[index].copy()
    result = fixed.optimizer.optimize(lambda a: predict(a)[1], np.empty((0, 2), np.float32),
                                      selected[None], 120, seed=optimizer_seed, batch_size=12, elites=4)
    if result["score"] > scores[index] + 1e-12:
        raise ValueError("scored incumbent lost")
    flags = (witnesses(float(parent_values[0]), single_scores, extra_scores, parent, singles, segments)
             if arm == "pair" else [])
    return {"parent_score": float(parent_values[0]), "single_scores": single_scores.tolist(),
            "extra_scores": extra_scores.tolist(), "segments": segments, "witnesses": flags,
            "start_index": index, "start_score": float(scores[index]), "start_actions": selected.tolist(),
            "final_score": float(result["score"]), "final_actions": result["actions"].tolist()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads((old.OUT / "manifest.json").read_text())
    audit = json.loads((old.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["summary_sha256"] != old.prior.digest(old.OUT / "summary.json"):
        raise ValueError("prior audit mismatch")
    seal, journal = fixed.prior.prior.prior.seal, fixed.prior.prior.prior.journal
    seal.verify(previous["dependencies"])
    seal.verify(previous["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, Path(fixed.optimizer.__file__),
                         old.OUT / "summary.json", old.OUT / "audit.json"])
    OUT.mkdir()
    tasks = previous["tasks"]
    journal.write_json(OUT / "manifest.json", {"tasks": tasks, "dependencies": deps,
                                               "source_dependencies": previous["source_dependencies"]})
    rows = []
    for root, task in enumerate(tasks):
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        if (source["root"], source["task"], source["status"]) != (root, task, "completed"):
            raise ValueError("parent source mismatch")
        parent = fixed.starts_from_result(source["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + root*2 + repeat
            proposal_seed = 2051300000 + root*2 + repeat
            for arm in ("pair", "single"):
                name = f"root{root:02d}_rep{repeat}_{arm}"
                rec = journal.PredictionJournal(
                    OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=471)
                row = {"root": root, "repeat": repeat, "arm": arm,
                       "proposal_seed": proposal_seed, "optimizer_seed": seed}
                started = time.monotonic()
                try:
                    result = run_arm(rec, parent, arm, proposal_seed, seed)
                    if rec.tokens != 471:
                        raise ValueError("budget mismatch")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=rec.tokens, calls=rec.calls, elapsed_seconds=time.monotonic()-started)
                rows.append(row)
                journal.write_json(OUT / f"{name}.json", row)
                print(name, row["status"], flush=True)
    seal.verify(deps)
    seal.verify(previous["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Testing two-segment changes against matched single-segment edits and CEM.", flush=True)
    main()
