"""Reconstruct frozen v23 CEM generations using saved action/score batches."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v23_shared_cem_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.ROOT / "results/pc_v23_cem_generation_reconstruction_20261104_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_V23_CEM_GENERATION_RECONSTRUCTION_PROTOCOL_20261104_CODEX.md"


def reconstruct(initial_actions, initial_scores, batches, starts, seed, batch_size, elites):
    initial_actions = np.asarray(initial_actions, np.float32)
    initial_scores = np.asarray(initial_scores, np.float64)
    if (len(starts) != len(initial_actions) or initial_scores.shape != (len(starts),)
            or len(batches) != sum(len(start["generations"]) for start in starts)):
        raise ValueError("generation or initial batch mismatch")
    result, cursor = [], 0
    for index, start in enumerate(starts):
        if initial_scores[index] != start["seed_score"]:
            raise ValueError("seed score mismatch")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), index]))
        local, local_score = initial_actions[index].copy(), float(initial_scores[index])
        mean = local.astype(np.float64)
        scale = np.full(mean.shape, .35, np.float64)
        records = []
        for generation in start["generations"]:
            actions, scores = batches[cursor]
            cursor += 1
            size = int(generation["samples"])
            if size < 1 or size > batch_size or len(actions) != size:
                raise ValueError("bad generation size")
            expected = np.clip(mean + rng.normal(size=(size, *mean.shape)) * scale, -1, 1).astype(np.float32)
            expected[0] = local
            np.testing.assert_array_equal(actions, expected)
            scores = np.asarray(scores, np.float64)
            winner = int(np.argmin(scores))
            if scores[winner] < local_score:
                local, local_score = actions[winner].copy(), float(scores[winner])
            if local_score != generation["best_score"]:
                raise ValueError("generation incumbent mismatch")
            elite = actions[np.argsort(scores, kind="stable")[:min(size, elites)]].astype(np.float64)
            next_mean, next_scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .07)
            records.append({"samples": size, "mean_before": mean.tolist(), "scale_before": scale.tolist(),
                            "mean_after": next_mean.tolist(), "scale_after": next_scale.tolist(),
                            "best_score": local_score})
            mean, scale = next_mean, next_scale
        if (start["samples"] != 1 + sum(item["samples"] for item in records)
                or local_score != start["best_score"] or len(records) < 2):
            raise ValueError("start accounting or incumbent mismatch")
        result.append({"samples": start["samples"], "generations": records})
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path, plans_path, audit_path = (source.OUT / name for name in
                                             ("manifest.json", "plans.json", "audit.json"))
    manifest, plans, audit = (json.loads(path.read_text()) for path in
                              (manifest_path, plans_path, audit_path))
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(plans_path)
            or plans["manifest_sha256"] != seal.digest(manifest_path)
            or len(plans["rows"]) != 16):
        raise ValueError("source audit or manifest binding mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, manifest_path, plans_path, audit_path,
                         *[source.OUT / row["trace"] for row in plans["rows"]]])
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed" or seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("incomplete source or trace drift")
        result, calls = row["result"]["result"], row["result"]["calls"]
        if row["arm"] == "v23":
            starts = result["child"]["refinement_starts"]
            config = result["config"]
            seed = config["seed"]
        elif row["arm"] == "shared_cem_v23":
            starts = result["starts"]
            config = plans["rows"][2 * row["root"]]["result"]["result"]["config"]
            seed = 2029502000 + row["root"] * 10
        else:
            raise ValueError("unknown arm")
        generations = sum(len(item["generations"]) for item in starts)
        seed_index = len(calls) - generations - 2
        if seed_index < 0:
            raise ValueError("missing seed batch")
        with np.load(source.OUT / row["trace"], allow_pickle=False) as data:
            initial_actions = data[f"actions_{seed_index}"]
            initial_scores = data[f"scores_{seed_index}"]
            batches = [(data[f"actions_{i}"], data[f"scores_{i}"])
                       for i in range(seed_index + 1, len(calls) - 1)]
            rebuilt = reconstruct(initial_actions, initial_scores, batches, starts, seed,
                                  config["refine_batch"], config["refine_elites"])
        rows.append({"root": row["root"], "arm": row["arm"], "seed_batch": seed_index,
                     "samples": sum(item["samples"] for item in rebuilt), "starts": rebuilt})
        print(f"reconstructed root={row['root']} arm={row['arm']} generations={generations}", flush=True)
    seal.verify(deps)
    seal.verify(manifest["dependencies"])
    with OUT.open("x") as handle:
        json.dump({"status": "passed", "scope": "same RNG and recorded scores; not independent replication",
                   "dependencies": deps, "rows": rows}, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
