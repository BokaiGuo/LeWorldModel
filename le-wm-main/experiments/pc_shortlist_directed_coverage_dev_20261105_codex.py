"""Fixed-bank shortlist and directed-retention development comparison."""
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments import pc_action_effect_ranked_end_to_end_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_response_temporal_search_v1 as suffix_cem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_shortlist_directed_coverage_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_SHORTLIST_DIRECTED_COVERAGE_DEV_PROTOCOL_20261105_CODEX.md"
ARMS = ("rank_min", "rank_directed", "explore_min", "explore_directed")
DIRECTIONS = ("random", "spectral")


def unique_order(candidates):
    order, seen = [], set()
    for index in np.argsort([c["score"] for c in candidates], kind="stable"):
        index = int(index)
        key = np.asarray(candidates[index]["chain"], np.float32).tobytes()
        if key not in seen:
            order.append(index)
            seen.add(key)
    return order


def shortlist(candidates, explore, seed):
    ordered = unique_order(candidates)
    if len(ordered) < 5:
        raise ValueError("fewer than five distinct candidates")
    return ordered[:3] + [int(np.random.default_rng(seed).choice(ordered[3:]))] if explore else ordered[:4]


def choose(indices, scores, rule):
    scores = np.asarray(scores, np.float64)
    if len(indices) != 4 or scores.shape != (4, 9) or len(set(indices)) != 4:
        raise ValueError("invalid probe matrix")
    if rule == "min":
        second = min(range(1, 4), key=lambda i: (float(scores[i].min()), indices[i]))
    elif rule == "directed":
        second = min(range(1, 4), key=lambda i: (-float(np.maximum(scores[0] - scores[i], 0).mean()), indices[i]))
    else:
        raise ValueError("unknown rule")
    return (0, second)


def tails(tail_length):
    rng = np.random.default_rng(2051200010 + tail_length)
    finite = np.clip(rng.normal(0, .35, (8, tail_length, 2)), -.5, .5).astype(np.float32)
    return np.concatenate((np.zeros((1, tail_length, 2), np.float32), finite))


def evaluate_arm(predict, candidates, depth, indices, rule, seed):
    prefixes = [np.asarray(candidates[i]["chain"], np.float32) for i in indices]
    length = (depth + 1)*5
    if any(p.shape != (length, 2) for p in prefixes):
        raise ValueError("invalid candidate prefix")
    bank = tails(15-length)
    chains = np.concatenate([np.concatenate((np.broadcast_to(p, (9, length, 2)), bank), axis=1)
                             for p in prefixes], axis=0)
    _, values = predict(chains)
    values = np.asarray(values).reshape(4, 9)
    selected = choose(indices, values, rule)
    best_index = int(np.argmin(values))
    best = chains[best_index].copy()
    best_score = float(values.reshape(-1)[best_index])
    source = f"probe_{best_index//9}_{best_index%9}"
    refinement = []
    for slot, candidate in enumerate(selected):
        tail_index = int(np.argmin(values[candidate]))
        initial_score = float(values[candidate, tail_index])
        chain, score, generation = suffix_cem.cem(
            bank[tail_index], prefixes[candidate], 120, np.random.default_rng(seed + slot),
            predict, initial_score, 4)
        chain = np.asarray(chain, np.float32)
        if score < best_score:
            best, best_score, source = chain.copy(), float(score), f"cem_{slot}"
        refinement.append({"slot": slot, "candidate": indices[candidate], "tail_index": tail_index,
                           "initial_score": initial_score, "final_score": float(score),
                           "chain": chain.tolist(), "generations": generation})
    return {"shortlist": indices, "retained": [indices[i] for i in selected],
            "probe_scores": values.tolist(), "probe_chains": chains.tolist(), "refinement": refinement,
            "final_actions": best.tolist(), "final_score": best_score, "final_source": source}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = prior.OUT
    manifest_source = json.loads((source / "manifest.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    prior.prior.prior.seal.verify(manifest_source["dependencies"])
    if audit["status"] != "passed" or audit["run_count"] != 80 or manifest_source["tasks"] != prior.tasks():
        raise ValueError("unverified source")
    source_files = [source / f"plan_{root:02d}_{direction}_min.json"
                    for root in range(16) for direction in DIRECTIONS]
    dependencies = prior.prior.prior.seal.capture([
        Path(__file__), PROTOCOL, Path(prior.__file__), Path(suffix_cem.__file__),
        Path(prior.prior.prior.journal.__file__), source / "manifest.json", source / "audit.json", *source_files])
    OUT.mkdir()
    prior.prior.prior.journal.write_json(OUT / "manifest.json", {
        "dependencies": dependencies, "source_dependencies": manifest_source["dependencies"],
        "tasks": prior.tasks(), "directions": DIRECTIONS, "arms": ARMS})
    rows = []
    for root, task in enumerate(prior.tasks()):
        for direction in DIRECTIONS:
            source_path = source / f"plan_{root:02d}_{direction}_min.json"
            saved = json.loads(source_path.read_text())
            if (saved["root"], saved["task"], saved["status"]) != (root, task, "completed"):
                raise ValueError("source row mismatch")
            for depth in (0, 1):
                layer = saved["result"]["child"]["layers"][depth]
                candidates = layer["candidates"]
                for arm in ARMS:
                    name = f"r{root:02d}_{direction}_d{depth}_{arm}"
                    recorder = prior.prior.prior.journal.PredictionJournal(
                        OUT / f"{name}_calls", prior.prior.prior.old.previous.fixed.predictor(task),
                        action_dim=2, packet_length=5, token_cap=828)
                    row = {"root": root, "direction": direction, "depth": depth, "arm": arm}
                    start = time.monotonic()
                    try:
                        dir_index = DIRECTIONS.index(direction)
                        base_seed = 2051201000 + root*100 + dir_index*20 + depth*4
                        indices = shortlist(candidates, arm.startswith("explore"), base_seed)
                        result = evaluate_arm(recorder, candidates, depth, indices,
                                              "directed" if arm.endswith("directed") else "min",
                                              base_seed + 2)
                        if recorder.tokens != 828:
                            raise ValueError("incorrect budget")
                        row.update(status="completed", result=result)
                    except Exception:
                        row.update(status="failed", error=traceback.format_exc())
                    row.update(tokens=recorder.tokens, calls=recorder.calls, seconds=time.monotonic()-start)
                    rows.append(row)
                    prior.prior.prior.journal.write_json(OUT / f"{name}.json", row)
                    print(f"{name} {row['status']} tokens={row['tokens']}", flush=True)
    prior.prior.prior.seal.verify(dependencies)
    prior.prior.prior.seal.verify(manifest_source["dependencies"])
    prior.prior.prior.journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Comparing fixed-bank shortlist and directed retention under common suffix opportunity.", flush=True)
    main()
