"""Post-result same-bank ordinary lookahead with equal suffix optimization."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as src
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = src.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V3_SAME_BANK_LOOKAHEAD_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v3_same_bank_lookahead_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
DATA = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def source():
    prior = src.prior
    data = json.loads(src.PLANS.read_text())
    if (json.loads(src.AUDIT.read_text())["plans_sha256"] != prior.base.sha(src.PLANS)
            or json.loads(src.REBUILD.read_text())["intervention_sha256"] != prior.base.sha(src.INTERVENTION)):
        raise ValueError("source seal mismatch")
    return [r for r in data["rows"] if r["arm"] == "response"]


def plan():
    if any(p.exists() for p in (MANIFEST, DATA, AUDIT)):
        raise FileExistsError("existing or partial result")
    rows = source()
    dependencies = seal.capture([Path(__file__), PROTOCOL, src.PLANS, src.AUDIT, src.INTERVENTION,
                                 src.REBUILD, Path(src.prior.base.exact.__file__)])
    src.prior.base.write(MANIFEST, {"seal": dependencies, "source": src.prior.base.sha(src.PLANS),
                                    "scope": "post-result fixed bank diagnosis"})
    print("Equal-budget suffix CEM for every sealed candidate and selected prefix.", flush=True)
    out = []
    for row in rows:
        root, task, result = row["root"], row["task"], row["result"]
        goal = np.asarray(task["goal"])
        selected = np.asarray(result["selected"]["chain"], dtype=np.float32)
        bank = [{"name": item["name"], "prefix": item["prefix"]} for item in result["first_bank"]]
        candidates = bank + [{"name": "selected_final", "prefix": selected[:5].tolist()}]
        for candidate in candidates:
            prefix = np.asarray(candidate["prefix"])
            trace = []

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)
                pos = src.prior.base.exact.endpoints(task, raw)
                scores = np.linalg.norm(pos - goal, axis=1)
                trace.append({"raw": raw.tolist(), "positions": pos.tolist(), "scores": scores.tolist()})
                return pos, scores

            zero = np.concatenate((prefix, np.zeros((10, 2))))[None]
            initial = float(predict(zero)[1][0])
            chain, value, generations = core.cem(np.zeros((10, 2)), prefix, 119,
                np.random.default_rng(2027111400 + root), predict, initial, 4)
            common = np.concatenate((prefix, selected[5:]))[None]
            common_distance = float(predict(common)[1][0])
            candidate.update({"zero_distance": initial, "common_distance": common_distance,
                              "optimized_chain": chain.tolist(), "optimized_distance": value,
                              "generations": generations, "trace": trace})
        lookahead = int(np.argmin([c["optimized_distance"] for c in candidates[:-1]]))
        greedy = int(np.argmin(result["prefix_scores"]))
        out.append({"root": root, "task": task, "lookahead_index": lookahead, "greedy_index": greedy,
                    "selected_prefix_in_bank": any(np.array_equal(np.asarray(x["prefix"], np.float32),
                                                                  selected[:5]) for x in bank),
                    "selected_tail": selected[5:].tolist(), "candidates": candidates})
        print(f"root={root} candidates={len(candidates)}", flush=True)
    seal.verify(dependencies)
    src.prior.base.write(DATA, {"manifest_sha256": src.prior.base.sha(MANIFEST), "rows": out})
    print("Same-bank diagnostic sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    original = {r["root"]: r for r in source()}
    manifest, data = json.loads(MANIFEST.read_text()), json.loads(DATA.read_text())
    seal.verify(manifest["seal"])
    if data["manifest_sha256"] != src.prior.base.sha(MANIFEST):
        raise ValueError("manifest mismatch")
    sample_count = 0
    for row in data["rows"]:
        ref = original[row["root"]]
        result = ref["result"]
        if row["task"] != ref["task"] or len(row["candidates"]) != len(result["first_bank"]) + 1:
            raise ValueError("source bank mismatch")
        chosen = np.asarray(result["selected"]["chain"], dtype=np.float32)
        for index, candidate in enumerate(row["candidates"]):
            expected = result["first_bank"][index]["prefix"] if index < len(result["first_bank"]) else chosen[:5].tolist()
            if candidate["prefix"] != expected:
                raise ValueError("prefix mismatch")
            values, found = [], False
            for batch in candidate["trace"]:
                raw = np.asarray(batch["raw"], dtype=np.float32)
                pos = src.prior.base.exact.endpoints(row["task"], raw)
                score = np.linalg.norm(pos - np.asarray(row["task"]["goal"]), axis=1)
                np.testing.assert_allclose(pos, batch["positions"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(score, batch["scores"], rtol=0, atol=1e-12)
                values.extend(score.tolist())
                sample_count += len(raw)
                found |= any(score[i] == candidate["optimized_distance"]
                             for i in np.flatnonzero(np.all(raw == np.asarray(candidate["optimized_chain"], np.float32),
                                                            axis=(1, 2))))
            if (len(values) != 121 or not found or min(values[:-1]) != candidate["optimized_distance"]
                    or values[0] != candidate["zero_distance"] or values[-1] != candidate["common_distance"]):
                raise ValueError("optimizer selection or sample count mismatch")
        if (row["lookahead_index"] != int(np.argmin([c["optimized_distance"] for c in row["candidates"][:-1]]))
                or row["greedy_index"] != int(np.argmin(result["prefix_scores"]))):
            raise ValueError("selection mismatch")
    if len(data["rows"]) != 24 or {r["root"] for r in data["rows"]} != set(range(24)):
        raise ValueError("incomplete panel")
    seal.verify(manifest["seal"])
    src.prior.base.write(AUDIT, {"plans_sha256": src.prior.base.sha(DATA), "chains": sample_count,
                                "scope": "same exact dynamics, independently recomputed"})
    print(f"Recomputed {sample_count} full-chain evaluations.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
