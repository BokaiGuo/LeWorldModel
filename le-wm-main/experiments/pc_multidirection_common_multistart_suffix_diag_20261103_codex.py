"""Matched multi-start suffix optimization for sealed mechanism prefixes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as source
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize

exp = source.exp
PROTOCOL = exp.PROTOCOL.parent / "PC_MULTIDIRECTION_COMMON_MULTISTART_SUFFIX_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_common_multistart_suffix_diag.json")
NAMES = ("selected", "zero", "greedy", "random_bridge", "lookahead")
STARTS, SAMPLES = 4, 160


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = exp.sha
    paths = (Path(__file__), PROTOCOL, source.OUTPUT, Path(optimize.__code__.co_filename))
    hashes = {str(p): sha(p) for p in paths}
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != source.dependencies():
        raise ValueError("source seal mismatch")
    print("Matched four-start suffix optimization for every sealed prefix; retrospective diagnostic.", flush=True)
    rows = []
    for row in data["rows"]:
        root, task = row["root"], row["task"]
        goal = np.asarray(task["goal"])
        for name in NAMES:
            prefix = np.asarray(row["controls"][name]["prefix"], dtype=float)
            if prefix.shape != (5, 2) or np.any(np.abs(prefix) > 1):
                raise ValueError("invalid prefix")
            starts, cumulative, best = [], [], float("inf")
            for index in range(STARTS):
                seed = 2027110300 + root * STARTS + index
                counts = {"calls": 0, "chains": 0}

                def predict(actions):
                    model = np.asarray(actions, dtype=np.float32)
                    if model.ndim != 3 or model.shape[1:] != (15, 2) or np.any(np.abs(model) > 1):
                        raise ValueError("invalid search actions")
                    if not np.array_equal(model[:, :5], np.broadcast_to(prefix.astype(np.float32), model[:, :5].shape)):
                        raise ValueError("fixed prefix changed")
                    counts["calls"] += 1
                    counts["chains"] += len(model)
                    return np.linalg.norm(exp.previous.prior.exact.endpoints(task, model) - goal, axis=1)

                chain, score, scores = optimize(prefix, 15, SAMPLES, np.random.default_rng(seed), predict)
                if counts["chains"] != SAMPLES or len(scores) != SAMPLES or score != min(scores):
                    raise ValueError("search accounting mismatch")
                recomputed = float(np.linalg.norm(exp.previous.prior.exact.endpoints(
                    task, chain[None].astype(np.float32))[0] - goal))
                if abs(recomputed - score) > 1e-12:
                    raise ValueError("best chain replay mismatch")
                best = min(best, score)
                cumulative.append(best)
                starts.append({"seed": seed, "scores": scores, "best_chain": chain.tolist(),
                               "best_score": score, "prediction_calls": counts["calls"],
                               "prediction_chains": counts["chains"], "prediction_tokens": 3 * counts["chains"]})
            rows.append({"root": root, "name": name, "prefix": prefix.tolist(),
                         "selected_nonprogress": row["nonprogress"], "cumulative_best": cumulative,
                         "starts": starts})
    if len(rows) != 24 * len(NAMES) or len({(r["root"], r["name"]) for r in rows}) != len(rows):
        raise ValueError("panel mismatch")
    if hashes != {str(p): sha(p) for p in paths} or data["dependencies"] != source.dependencies():
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} equally budgeted prefix searches.", flush=True)


if __name__ == "__main__":
    main()
