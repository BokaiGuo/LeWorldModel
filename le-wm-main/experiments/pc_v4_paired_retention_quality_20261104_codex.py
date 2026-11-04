"""Offline paired quality of two sealed retention sets on a shared bank."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as src
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = src.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_PAIRED_RETENTION_QUALITY_PROTOCOL_20261104_CODEX.md"
BASE = ROOT / "results/pc_target_reach_v4_e0_20261104_codex"
ALT = ROOT / "results/pc_target_reach_v4_position_retention_e0_20261104_codex"
OUT = ROOT / "results/pc_v4_paired_retention_quality_20261104_codex"
path = lambda p, suffix: p.with_name(p.name + suffix + ".json")
MANIFEST, DATA, AUDIT = [path(OUT, s) for s in ("_manifest", "_plans", "_audit")]
PLANS = [path(BASE, "_plans"), path(ALT, "_plans")]
AUDITS = [path(BASE, "_audit"), path(ALT, "_audit")]
RULES = path(ALT, "_rules_audit")
sha = src.prior.base.sha
write = src.prior.base.write


def source():
    rules = json.loads(RULES.read_text())
    if rules["plans_sha256"] != [sha(p) for p in PLANS] or rules["verified_roots"] != 24:
        raise ValueError("retention audit source mismatch")
    for p, a in zip(PLANS, AUDITS, strict=True):
        if json.loads(a.read_text())["plans_sha256"] != sha(p):
            raise ValueError("source audit mismatch")
    pairs = list(zip(*(json.loads(p.read_text())["rows"] for p in PLANS), strict=True))
    if len(pairs) != 24 or {a["root"] for a, _ in pairs} != set(range(24)):
        raise ValueError("root panel mismatch")
    for a, b in pairs:
        x, y = a["result"], b["result"]
        if (a["root"] != b["root"] or a["task"] != b["task"]
                or x["first_bank"] != y["first_bank"] or x["prefix_scores"] != y["prefix_scores"]
                or len(x["first_bank"]) != 9):
            raise ValueError("bank mismatch")
        for z in (x, y):
            if len(z["retained"]) != 4 or len(set(z["retained"])) != 4 or not all(
                    0 <= i < 9 for i in z["retained"]):
                raise ValueError("invalid retention")
    return pairs


def run():
    if any(p.exists() for p in (MANIFEST, DATA, AUDIT)):
        raise FileExistsError("result or partial result exists")
    pairs = source()
    dependencies = seal.capture([Path(__file__), PROTOCOL, *PLANS, *AUDITS, RULES,
                                 Path(src.prior.base.exact.__file__)])
    write(MANIFEST, {"seal": dependencies, "scope": "exposed E0 offline fixed-bank diagnostic"})
    print("Paired fixed-bank retention quality, three common seeds per root.", flush=True)
    rows = []
    for a, b in pairs:
        root, task, result = a["root"], a["task"], a["result"]
        goal = np.asarray(task["goal"])
        candidates = []
        for item in result["first_bank"]:
            prefix = np.asarray(item["prefix"])
            repeats = []
            for rep in range(3):
                trace = []

                def predict(actions):
                    raw = np.asarray(actions, dtype=np.float32)
                    pos = src.prior.base.exact.endpoints(task, raw)
                    score = np.linalg.norm(pos - goal, axis=1)
                    trace.append({"raw": raw.tolist(), "positions": pos.tolist(), "scores": score.tolist()})
                    return pos, score

                initial = float(predict(np.concatenate((prefix, np.zeros((10, 2))))[None])[1][0])
                seed = 2027111900 + root * 100 + rep
                chain, value, gens = core.cem(np.zeros((10, 2)), prefix, 119,
                                               np.random.default_rng(seed), predict, initial, 4)
                repeats.append({"seed": seed, "chain": chain.tolist(), "score": value,
                                "generations": gens, "trace": trace})
            candidates.append({"name": item["name"], "prefix": item["prefix"], "repeats": repeats})
        rows.append({"root": root, "task": task, "retained": [result["retained"], b["result"]["retained"]],
                     "greedy": int(np.argmin(result["prefix_scores"])), "candidates": candidates})
        print(f"root={root}", flush=True)
    seal.verify(dependencies)
    write(DATA, {"manifest_sha256": sha(MANIFEST), "rows": rows})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    pairs = source()
    manifest, data = json.loads(MANIFEST.read_text()), json.loads(DATA.read_text())
    seal.verify(manifest["seal"])
    if data["manifest_sha256"] != sha(MANIFEST) or len(data["rows"]) != 24:
        raise ValueError("manifest or row mismatch")
    count = 0
    for row, (a, b) in zip(data["rows"], pairs, strict=True):
        result = a["result"]
        if (row["root"] != a["root"] or row["task"] != a["task"]
                or row["retained"] != [result["retained"], b["result"]["retained"]]
                or row["greedy"] != int(np.argmin(result["prefix_scores"]))
                or len(row["candidates"]) != 9):
            raise ValueError("row mismatch")
        for item, expected in zip(row["candidates"], result["first_bank"], strict=True):
            if item["name"] != expected["name"] or item["prefix"] != expected["prefix"] or len(item["repeats"]) != 3:
                raise ValueError("candidate mismatch")
            for rep, run_result in enumerate(item["repeats"]):
                if run_result["seed"] != 2027111900 + row["root"] * 100 + rep:
                    raise ValueError("seed mismatch")
                if sum(g["size"] for g in run_result["generations"]) != 119:
                    raise ValueError("generation count")
                scores, found = [], False
                for batch in run_result["trace"]:
                    raw = np.asarray(batch["raw"], dtype=np.float32)
                    np.testing.assert_array_equal(raw[:, :5], np.broadcast_to(np.asarray(item["prefix"], np.float32), raw[:, :5].shape))
                    pos = src.prior.base.exact.endpoints(row["task"], raw)
                    values = np.linalg.norm(pos - np.asarray(row["task"]["goal"]), axis=1)
                    np.testing.assert_allclose(pos, batch["positions"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
                    scores.extend(values.tolist())
                    count += len(raw)
                    found |= any(values[i] == run_result["score"] for i in np.flatnonzero(
                        np.all(raw == np.asarray(run_result["chain"], np.float32), axis=(1, 2))))
                first = np.asarray(run_result["trace"][0]["raw"], np.float32)
                if (len(scores) != 120 or first.shape != (1, 15, 2) or np.any(first[0, 5:] != 0)
                        or not found or min(scores) != run_result["score"]):
                    raise ValueError("optimizer trace mismatch")
    seal.verify(manifest["seal"])
    write(AUDIT, {"plans_sha256": sha(DATA), "chains": count,
                  "scope": "same exact dynamics, independently recomputed"})
    print(f"Recomputed {count} chains.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
