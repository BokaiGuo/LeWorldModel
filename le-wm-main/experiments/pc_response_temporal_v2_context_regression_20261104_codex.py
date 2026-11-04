"""Replay v2 on sealed v1 contexts; this does not run a feedback trajectory."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_fixed_terminal_e0_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v2 as method

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V2_CONTEXT_REGRESSION_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v2_context_regression_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(method.__file__), source.PLANS, source.AUDIT,
             source.MANIFEST)
    return {**source.dependencies(), **{str(p.resolve()): sha(p) for p in paths}}


def load_source():
    audit = json.loads(source.AUDIT.read_text())
    if audit["plans_sha256"] != sha(source.PLANS):
        raise ValueError("source plan seal mismatch")
    data = json.loads(source.PLANS.read_text())
    if data["dependencies"] != source.dependencies() or data["manifest_sha256"] != sha(source.MANIFEST):
        raise ValueError("source dependency mismatch")
    return data


def plan():
    if any(p.exists() for p in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("new regression artifact already exists")
    data = load_source()
    hashes = dependencies()
    write(MANIFEST, {"dependencies": hashes, "method": method.METHOD, "contexts": 96,
                     "arms": ["response", "matched_random"], "steps": [1, 2], "cap": 1920})
    print("Replaying v2 on sealed v1 contexts, not executing closed-loop trajectories.", flush=True)
    rows = []
    for old in data["rows"]:
        if old["arm"] not in ("response", "matched_random"):
            continue
        for item in old["decisions"][1:]:
            context, carry, step = item["context"], item["carry_in"], item["step"]
            meter, trace = Meter(1920), []

            def predict(actions):
                requested = np.asarray(actions, dtype=float)
                raw = requested.astype(np.float32)

                def evaluate(batch):
                    positions = source.previous.exact.endpoints(context, batch)
                    return positions, np.linalg.norm(positions - np.asarray(context["goal"]), axis=1)

                positions, scores = meter.predict(raw, evaluate)
                trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                              "positions": positions.tolist(), "scores": scores.tolist(),
                              "tokens": len(raw) * raw.shape[1] // 5})
                return positions, scores

            result = method.decide(old["arm"], old["root"], step, context["goal"], predict, meter, carry)
            rows.append({"root": old["root"], "arm": old["arm"], "step": step,
                         "context": context, "carry": carry, "v1_selected": item["decision"]["result"]["selected"],
                         "result": result, "counts": meter.as_dict(), "trace": trace})
        print(f"root={old['root']} arm={old['arm']} contexts complete", flush=True)
    if len(rows) != 96 or dependencies() != hashes:
        raise ValueError("context count or dependency drift")
    write(PLANS, {"manifest_sha256": sha(MANIFEST), "dependencies": hashes, "rows": rows})
    print("Regression plans sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    original = load_source()
    data = json.loads(PLANS.read_text())
    if data["manifest_sha256"] != sha(MANIFEST) or data["dependencies"] != dependencies():
        raise ValueError("new result seal mismatch")
    expected = {(row["root"], row["arm"], item["step"]): (item, row)
                for row in original["rows"] if row["arm"] in ("response", "matched_random")
                for item in row["decisions"][1:]}
    findings, seen = [], set()
    for row in data["rows"]:
        key = (row["root"], row["arm"], row["step"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or unexpected context")
        seen.add(key)
        old, _ = expected[key]
        if (row["context"] != old["context"] or row["carry"] != old["carry_in"]
                or row["v1_selected"] != old["decision"]["result"]["selected"]):
            raise ValueError("sealed context mismatch")
        result, trace = row["result"], row["trace"]
        horizon, count, scores, found, v1_seen = 15 - 5 * row["step"], 0, [], False, False
        selected, v1 = result["selected"], row["v1_selected"]
        for batch in trace:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if not np.array_equal(requested.astype(np.float32), raw):
                raise ValueError("float conversion mismatch")
            positions = source.previous.exact.endpoints(row["context"], raw)
            values = np.linalg.norm(positions - np.asarray(row["context"]["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
            tokens = len(raw) * raw.shape[1] // 5
            if tokens != batch["tokens"]:
                raise ValueError("batch token mismatch")
            count += tokens
            if raw.shape[1] == horizon:
                scores.extend(values.tolist())
                found |= any(abs(values[i] - selected["terminal_score"]) < 1e-12
                             for i in np.flatnonzero(np.all(requested == selected["chain"], axis=(1, 2))))
                v1_seen |= bool(np.any(np.all(requested == v1["chain"], axis=(1, 2))))
        if (not found or abs(min(scores) - selected["terminal_score"]) > 1e-12
                or count != row["counts"]["prediction_token_samples"] or count > 1920
                or len(trace) != row["counts"]["prediction_calls"]
                or trace[0]["requested"] != [row["carry"]]
                or result["carry_score"] != trace[0]["scores"][0]):
            raise ValueError("global selection, carry or budget mismatch")
        incumbent = result["global_incumbent_pre_abstention"]
        comparator = np.asarray(incumbent["chain"]).copy()
        comparator[:5] = 0
        if (comparator.tolist() != result["comparison_chain"]
                or trace[-1]["requested"] != [comparator.tolist()]
                or trace[-1]["scores"][0] != result["comparison_score"]
                or result["abstained"] != (result["comparison_score"] <= incumbent["terminal_score"])):
            raise ValueError("terminal abstention mismatch")
        findings.append({"root": key[0], "arm": key[1], "step": key[2], "tokens": count,
                         "v1_score": v1["terminal_score"], "v2_score": selected["terminal_score"],
                         "same_chain": selected["chain"] == v1["chain"], "v1_chain_scored_by_v2": v1_seen,
                         "v2_abstained": result["abstained"],
                         "v2_changed_v1_internal_selection": selected["chain"] != result["v1_selected_before_global_check"]["chain"]})
    if len(seen) != 96 or seen != set(expected):
        raise ValueError("incomplete panel")
    write(AUDIT, {"plans_sha256": sha(PLANS), "findings": findings})
    print("All 96 contexts independently audited.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
