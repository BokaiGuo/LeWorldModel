"""Prespecified equal-token multistart whole-action CEM development baselines."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_fixed_terminal_e0_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_MULTISTART_CEM_E0_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_multistart_cem_e0_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
KNOTS = (4, 8, 15)
COUNTS = (160, 160, 160, 159)


def dependencies():
    paths = (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT)
    return {**source.dependencies(), **{str(path.resolve()): source.prior.sha(path) for path in paths}}


def load_source():
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != source.prior.sha(source.PLANS):
        raise ValueError("sealed source mismatch")
    data = json.loads(source.PLANS.read_text())
    if data["dependencies"] != source.dependencies():
        raise ValueError("source dependency drift")
    return {(row["root"], row["arm"]): row for row in data["rows"]}


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("existing or partial baseline panel")
    old = load_source()
    hashes = dependencies()
    source.prior.write(MANIFEST, {"dependencies": hashes, "knots": KNOTS, "counts": COUNTS,
                                  "cap": 1920, "scope": "exposed-root baseline development"})
    print("Multistart CEM baselines, equal prediction token, exposed development roots.", flush=True)
    rows = []
    for root in range(24):
        response = old[root, "response"]
        task = response["task"]
        for knots in KNOTS:
            meter, trace = Meter(1920), []
            goal = np.asarray(task["goal"])

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    position = source.prior.previous.exact.endpoints(task, batch)
                    return position, np.linalg.norm(position - goal, axis=1)

                position, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": position.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw) * raw.shape[1] // 5})
                return position, scores

            zero = np.zeros((15, 2))
            zero_score = float(predict(zero[None])[1][0])
            best, score, restarts = zero.copy(), zero_score, []
            for index, count in enumerate(COUNTS):
                seed = 2027111500 + root * 100 + index
                chain, value, generations = core.cem(zero, [], count, np.random.default_rng(seed),
                                                      predict, zero_score, knots)
                restarts.append({"seed": seed, "samples": count, "score": value, "generations": generations})
                if value < score:
                    best, score = chain, value
            rows.append({"root": root, "task": task, "knots": knots, "selected": best.tolist(),
                         "score": score, "zero_score": zero_score, "restarts": restarts,
                         "counts": meter.as_dict(), "trace": trace,
                         "response_selected": response["decisions"][0]["decision"]["result"]["selected"]})
        print(f"root={root} all knot configurations complete", flush=True)
    if dependencies() != hashes:
        raise ValueError("dependency drift")
    source.prior.write(PLANS, {"manifest_sha256": source.prior.sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("Baselines sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    old, data = load_source(), json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies() or data["manifest_sha256"] != source.prior.sha(MANIFEST):
        raise ValueError("panel seal mismatch")
    seen, findings = set(), []
    for row in data["rows"]:
        key = row["root"], row["knots"]
        response = old[row["root"], "response"]
        if (key in seen or row["knots"] not in KNOTS or row["task"] != response["task"]
                or row["response_selected"] != response["decisions"][0]["decision"]["result"]["selected"]):
            raise ValueError("source mismatch")
        seen.add(key)
        tokens, scores, found = 0, [], False
        chain = np.asarray(row["selected"])
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if not np.array_equal(requested.astype(np.float32), raw):
                raise ValueError("float32 mismatch")
            position = source.prior.previous.exact.endpoints(row["task"], raw)
            values = np.linalg.norm(position - np.asarray(row["task"]["goal"]), axis=1)
            np.testing.assert_allclose(position, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
            if batch["tokens"] != len(raw) * 3:
                raise ValueError("batch token mismatch")
            tokens += batch["tokens"]
            scores.extend(values.tolist())
            found |= any(values[i] == row["score"] for i in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
        if (not found or min(scores) != row["score"] or tokens != 1920
                or tokens != row["counts"]["prediction_token_samples"]
                or len(row["trace"]) != row["counts"]["prediction_calls"]
                or tuple(restart["samples"] for restart in row["restarts"]) != COUNTS):
            raise ValueError("incumbent or resource mismatch")
        findings.append({"root": row["root"], "knots": row["knots"], "score": row["score"],
                         "response_score": row["response_selected"]["terminal_score"], "tokens": tokens})
    if seen != {(root, knots) for root in range(24) for knots in KNOTS}:
        raise ValueError("incomplete panel")
    source.prior.write(AUDIT, {"plans_sha256": source.prior.sha(PLANS), "rows": findings})
    print("All 72 multistart baselines independently recalculated.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
