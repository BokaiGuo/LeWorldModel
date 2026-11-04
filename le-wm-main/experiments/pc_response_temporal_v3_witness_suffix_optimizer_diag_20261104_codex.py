"""Fixed-prefix common-optimizer quality diagnostic on the sealed v3 witness."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as panel
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core

ROOT = panel.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_WITNESS_SUFFIX_OPTIMIZER_DIAG_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_witness_suffix_optimizer_diag_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INDEPENDENT = panel.PREFIX.with_name(panel.PREFIX.name + "_witness_independent_audit.json")
NAMES = ("selected", "zero", "greedy", "random_bridge", "lookahead")
BUDGETS, KNOTS, INITS = (120, 480, 1920), (4, 8), ("zero", "selected_tail")
SEEDS = (2027111600, 2027111601, 2027111602)


def dependencies():
    panel.configure()
    paths = (Path(__file__), PROTOCOL, panel.PLANS, panel.AUDIT, panel.INTERVENTION, panel.REBUILD, INDEPENDENT)
    return {**panel.dependencies(), **{str(p.resolve()): panel.prior.base.sha(p) for p in paths}}


def source_row():
    sha = panel.prior.base.sha
    intervention = json.loads(panel.INTERVENTION.read_text())
    independent = json.loads(INDEPENDENT.read_text())
    if (intervention["dependencies"] != panel.dependencies()
            or independent["plans_sha256"] != sha(panel.PLANS)
            or independent["intervention_sha256"] != sha(panel.INTERVENTION)):
        raise ValueError("source binding mismatch")
    candidates = [row for row in intervention["rows"] if row["root"] == 20]
    if len(candidates) != 1 or not candidates[0]["tail_all_intersection"]:
        raise ValueError("sealed witness mismatch")
    return candidates[0]


def plan():
    if any(path.exists() for path in (MANIFEST, PLANS, AUDIT)):
        raise FileExistsError("existing or partial optimization diagnostic")
    row, hashes = source_row(), dependencies()
    task, selected = row["task"], np.asarray(row["selected_chain_float32"])
    panel.prior.base.write(MANIFEST, {"dependencies": hashes, "root": 20, "task": task,
                                     "budgets": BUDGETS, "seeds": SEEDS, "knots": KNOTS,
                                     "initializations": INITS, "prefixes": NAMES})
    print("Common fixed-prefix optimization quality, not a full-method performance comparison.", flush=True)
    records = []
    for name in NAMES:
        prefix = np.asarray(row["controls"][name]["prefix"])
        for init in INITS:
            base = np.zeros((10, 2)) if init == "zero" else selected[5:]
            for knots in KNOTS:
                for budget in BUDGETS:
                    for seed in SEEDS:
                        meter, trace = Meter(3 * budget), []

                        def predict(actions):
                            requested = np.asarray(actions, dtype=float)
                            raw = requested.astype(np.float32)

                            def evaluate(batch):
                                positions = panel.prior.base.exact.endpoints(task, batch)
                                return positions, np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)

                            positions, scores = meter.predict(raw, evaluate)
                            trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                                          "positions": positions.tolist(), "scores": scores.tolist()})
                            return positions, scores

                        initial = np.concatenate((prefix, base))
                        initial_score = float(predict(initial[None])[1][0])
                        chain, score, generations = core.cem(base, prefix, budget - 1, np.random.default_rng(seed),
                                                              predict, initial_score, knots)
                        records.append({"prefix": name, "init": init, "knots": knots, "budget": budget,
                                        "seed": seed, "initial_score": initial_score, "score": score,
                                        "chain": chain.tolist(), "generations": generations,
                                        "counts": meter.as_dict(), "trace": trace})
        print(f"prefix={name} complete", flush=True)
    if len(records) != 180 or hashes != dependencies():
        raise ValueError("missing configurations or dependency drift")
    panel.prior.base.write(PLANS, {"manifest_sha256": panel.prior.base.sha(MANIFEST),
                                  "dependencies": hashes, "records": records})
    print("Fixed-prefix diagnostic sealed.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    row, data = source_row(), json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies() or data["manifest_sha256"] != panel.prior.base.sha(MANIFEST):
        raise ValueError("diagnostic seal mismatch")
    seen, findings = set(), []
    task = row["task"]
    for entry in data["records"]:
        key = tuple(entry[k] for k in ("prefix", "init", "knots", "budget", "seed"))
        if key in seen or key[0] not in NAMES or key[1] not in INITS or key[2] not in KNOTS or key[3] not in BUDGETS or key[4] not in SEEDS:
            raise ValueError("unexpected configuration")
        seen.add(key)
        count, values, found = 0, [], False
        selected = np.asarray(entry["chain"])
        prefix = np.asarray(row["controls"][key[0]]["prefix"])
        for batch in entry["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], dtype=np.float32)
            if not np.array_equal(requested.astype(np.float32), raw) or not np.all(requested[:, :5] == prefix):
                raise ValueError("prefix or float32 mismatch")
            positions = panel.prior.base.exact.endpoints(task, raw)
            scores = np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            values.extend(scores.tolist())
            count += len(raw)
            found |= any(scores[i] == entry["score"] for i in np.flatnonzero(np.all(requested == selected, axis=(1, 2))))
        if (not found or min(values) != entry["score"] or count != entry["budget"]
                or entry["counts"]["prediction_token_samples"] != 3 * count
                or len(entry["trace"]) != entry["counts"]["prediction_calls"]
                or sum(g["size"] for g in entry["generations"]) != count - 1):
            raise ValueError("optimization or resource mismatch")
        findings.append({k: entry[k] for k in ("prefix", "init", "knots", "budget", "seed", "score", "initial_score")})
    expected = {(name, init, knots, budget, seed) for name in NAMES for init in INITS
                for knots in KNOTS for budget in BUDGETS for seed in SEEDS}
    if seen != expected:
        raise ValueError("missing configuration")
    panel.prior.base.write(AUDIT, {"plans_sha256": panel.prior.base.sha(PLANS), "records": findings})
    print("All 180 optimizations and 151200 full-chain evaluations independently recalculated.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    panel.configure()
    {"plan": plan, "audit": audit}[sys.argv[1]]()
