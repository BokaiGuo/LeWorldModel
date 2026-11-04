"""Evaluate a common suffix optimizer on frozen privileged witness prefixes."""
import json
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as source
from experiments import pc_v22_response_retention_feasibility_posthoc_20261104_codex as witness_source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.ROOT / "results/pc_v22_retention_witness_suffix_posthoc_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V22_RETENTION_WITNESS_SUFFIX_POSTHOC_PROTOCOL_20261104_CODEX.md"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    witness = json.loads(witness_source.OUT.read_text())
    seal.verify(manifest["dependencies"])
    seal.verify(witness["dependencies"])
    if (audit["status"] != "passed" or audit["result_sha256"] != seal.digest(source.OUT / "result.json")
            or witness["source_result_sha256"] != seal.digest(source.OUT / "result.json")
            or witness["source_audit_sha256"] != seal.digest(source.OUT / "audit.json")
            or witness["probe_audit_sha256"] != seal.digest(source.OUT / "probe_input_audit.json")):
        raise ValueError("source panel or witness binding mismatch")
    old = source.prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(source.__file__), Path(witness_source.__file__),
                         Path(source.prior.prior.cem.__file__), Path(old.exact.__file__),
                         Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
                         Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"dependencies": deps, "witness_sha256": seal.digest(witness_source.OUT),
                                       "source_result_sha256": seal.digest(source.OUT / "result.json"),
                                       "scope": "privileged diagnostic, not planner performance"})
    rows = []
    for root, task in enumerate(source.tasks()):
        trace, runs, error = [], [], None
        eligible = [a for a in witness["rows"][root]["attempts"] if a.get("success")]
        try:
            if not eligible or witness["rows"][root]["task"] != task:
                raise ValueError("missing successful witness")
            selected = eligible[0]
            full = np.asarray(selected["actions"], np.float32)
            prefix = full[:5]
            goal = np.asarray(task["goal"])

            def evaluate(actions):
                actions = np.asarray(actions, np.float32)
                if (actions.shape[1:] != (15, 2) or not np.isfinite(actions).all()
                        or np.max(np.abs(actions)) > 1 or not np.all(actions[:, :5] == prefix)):
                    raise ValueError("invalid fixed-prefix batch")
                positions = old.exact.endpoints(task, actions)
                scores = np.linalg.norm(positions - goal, axis=1)
                trace.append((actions.copy(), positions.copy(), scores.copy()))
                return positions, scores

            zero = np.concatenate((prefix, np.zeros((10, 2), np.float32)))
            _, scores = evaluate(zero[None])
            for restart in range(3):
                begin = len(trace)
                chain, best, generations = source.prior.prior.cem.cem(
                    np.zeros((10, 2)), prefix, 120,
                    np.random.default_rng(2029182000 + 10 * root + restart), evaluate,
                    float(scores[0]), 10)
                runs.append({"restart": restart, "first_batch": begin, "last_batch": len(trace),
                             "best": float(best), "chain": np.asarray(chain, np.float32).tolist(),
                             "generations": generations})
        except Exception:
            error = traceback.format_exc()
        trace_path = OUT / f"root{root:02d}.npz"
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed", "error": error,
                     "witness_index": witness["rows"][root]["attempts"].index(eligible[0]) if eligible else None,
                     "witness_distance": eligible[0]["distance"] if eligible else None,
                     "zero_score": float(scores[0]) if error is None else None, "runs": runs,
                     "batches": len(trace), "trace": trace_path.name, "trace_sha256": seal.digest(trace_path)})
    seal.verify(deps)
    old.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    witness = json.loads(witness_source.OUT.read_text())
    seal.verify(manifest["dependencies"])
    if (manifest["witness_sha256"] != seal.digest(witness_source.OUT)
            or result["manifest_sha256"] != seal.digest(OUT / "manifest.json")):
        raise ValueError("source mismatch")
    old = source.prior.prior.prior
    rows = []
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["status"] != "completed" or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("incomplete or changed trace")
        attempt = witness["rows"][root]["attempts"][row["witness_index"]]
        prefix = np.asarray(attempt["actions"], np.float32)[:5]
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            for index in range(row["batches"]):
                actions, pos, scores = (data[f"{name}_{index}"] for name in ("actions", "positions", "scores"))
                if (actions.dtype != np.float32 or actions.shape[1:] != (15, 2)
                        or np.any(np.abs(actions) > 1) or not np.all(actions[:, :5] == prefix)):
                    raise ValueError("prefix or action violation")
                path, _ = old.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], pos, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(pos-row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            np.testing.assert_allclose(data["scores_0"][0], row["zero_score"], atol=1e-12, rtol=0)
            for restart, run_data in enumerate(row["runs"]):
                actions = np.concatenate([data[f"actions_{i}"] for i in range(run_data["first_batch"], run_data["last_batch"])])
                scores = np.concatenate([data[f"scores_{i}"] for i in range(run_data["first_batch"], run_data["last_batch"])])
                if restart != run_data["restart"] or len(actions) != 120:
                    raise ValueError("incorrect suffix allocation")
                np.testing.assert_allclose(min(row["zero_score"], float(scores.min())), run_data["best"], atol=1e-12, rtol=0)
        if len(row["runs"]) != 3:
            raise ValueError("missing restart")
        rows.append({"root": root, "zero_score": row["zero_score"],
                     "witness_distance": row["witness_distance"], "best": [r["best"] for r in row["runs"]],
                     "any_success": any(r["best"] < .05 for r in row["runs"]),
                     "zero_tokens": 3, "suffix_tokens": 1080})
    if len(rows) != len(source.tasks()):
        raise ValueError("missing root")
    seal.verify(manifest["dependencies"])
    old.writer(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
                                   "rows": rows, "scope": "alternate geometry, not independent replication"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
