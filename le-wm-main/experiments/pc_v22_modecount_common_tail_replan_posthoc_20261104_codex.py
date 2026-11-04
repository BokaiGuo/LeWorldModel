"""Equal-opportunity replan from each sealed chain's shared selected tail."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v22_modecount_same_chain_posthoc_20261104_codex as source
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.panel.ROOT / "results/pc_v22_modecount_common_tail_replan_posthoc_20261104_codex"
PROTOCOL = source.panel.ROOT.parent / "docs/PC_V22_MODECOUNT_COMMON_TAIL_REPLAN_POSTHOC_PROTOCOL_20261104_CODEX.md"
NAMES = ("selected", "zero", "greedy", "lookahead", "random")


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    records = json.loads((source.OUT / "interventions.json").read_text())
    predicate = json.loads((source.OUT / "predicate_audit.json").read_text())
    if predicate["result_sha256"] != seal.digest(source.OUT / "interventions.json"):
        raise ValueError("upstream predicate audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(cem.__file__), Path(source.__file__),
        Path(source.mechanism.exact.__file__), Path(source.mechanism.exact.wall.__file__),
        Path(source.mechanism.exact.wall.physics.__file__), source.OUT / "interventions.json",
        source.OUT / "predicate_audit.json"])
    OUT.mkdir()
    source.mechanism.writer(OUT / "manifest.json", {"dependencies": deps, "names": NAMES,
                                                   "scope": "post-hoc shared initialization diagnostic"})
    rows = []
    for record in records["rows"]:
        root, trace, controls, error = record["root"], [], {}, None
        try:
            if record["status"] != "completed":
                raise ValueError("prior intervention incomplete")
            data = record["data"]
            tail = np.asarray(data["selected_chain"], np.float32)[5:]
            goal, task = np.asarray(data["task"]["goal"]), data["task"]
            if tail.shape != (10, 2):
                raise ValueError("invalid selected tail")
            for name in NAMES:
                prefix = np.asarray(data["controls"][name]["prefix"], np.float32)
                runs = []
                for repeat in range(3):
                    begin = len(trace)

                    def predict(actions):
                        actions = np.asarray(actions, np.float32)
                        if (actions.shape[1:] != (15, 2) or not np.isfinite(actions).all()
                                or np.any(np.abs(actions) > 1) or not np.all(actions[:, :5] == prefix)):
                            raise ValueError("invalid fixed-prefix candidate")
                        positions = source.mechanism.exact.endpoints(task, actions)
                        scores = np.linalg.norm(positions-goal, axis=1)
                        trace.append((actions.copy(), positions.copy(), scores.copy()))
                        return positions, scores

                    initial = np.concatenate((prefix, tail))
                    initial_score = float(predict(initial[None])[1][0])
                    chain, score, generations = cem.cem(tail, prefix, 480,
                        np.random.default_rng(2027112400 + root*100 + repeat), predict, initial_score, 4)
                    runs.append({"initial_score": initial_score, "score": float(score),
                                 "chain": np.asarray(chain, np.float32).tolist(),
                                 "first_batch": begin, "last_batch": len(trace), "generations": generations})
                controls[name] = {"prefix": prefix.tolist(), "runs": runs,
                                  "mean_score": float(np.mean([item["score"] for item in runs]))}
        except Exception:
            error = traceback.format_exc()
        path = OUT / f"root{root:02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "status": "failed" if error else "completed", "error": error,
                     "controls": controls, "batches": len(trace), "trace": path.name,
                     "trace_sha256": seal.digest(path)})
        print(f"root={root} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    source.mechanism.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"),
                                                 "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    upstream = json.loads((source.OUT / "interventions.json").read_text())
    seal.verify(manifest["dependencies"])
    if result["manifest_sha256"] != seal.digest(OUT / "manifest.json") or len(result["rows"]) != 8:
        raise ValueError("result binding mismatch")
    summaries = []
    for row, prior in zip(result["rows"], upstream["rows"], strict=True):
        if row["root"] != prior["root"] or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("trace binding mismatch")
        if row["status"] != "completed":
            summaries.append({"root": row["root"], "status": "failed"})
            continue
        task = prior["data"]["task"]
        shared = np.asarray(prior["data"]["selected_chain"], np.float32)[5:]
        with np.load(OUT / row["trace"], allow_pickle=False) as trace:
            for name in NAMES:
                control = row["controls"][name]
                prefix = np.asarray(prior["data"]["controls"][name]["prefix"], np.float32)
                if not np.array_equal(control["prefix"], prefix):
                    raise ValueError("control prefix mismatch")
                for item in control["runs"]:
                    actions = np.concatenate([trace[f"actions_{i}"] for i in range(item["first_batch"], item["last_batch"])])
                    positions = np.concatenate([trace[f"positions_{i}"] for i in range(item["first_batch"], item["last_batch"])])
                    scores = np.concatenate([trace[f"scores_{i}"] for i in range(item["first_batch"], item["last_batch"])])
                    if (len(actions) != 481 or np.any(np.abs(actions) > 1)
                            or not np.all(actions[:, :5] == prefix) or not np.array_equal(actions[0, 5:], shared)):
                        raise ValueError("actions or budget mismatch")
                    independent, _ = source.panel.previous.prior.prior.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(independent[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(np.linalg.norm(positions-task["goal"], axis=1), scores,
                                               atol=1e-12, rtol=0)
                    if abs(float(scores.min()) - item["score"]) > 1e-12:
                        raise ValueError("best scored action mismatch")
        means = {name: row["controls"][name]["mean_score"] for name in NAMES}
        summaries.append({"root": row["root"], "status": "completed", "mean_scores": means,
                          "selected_beats_zero_and_greedy":
                          means["selected"] < min(means["zero"], means["greedy"]) - 1e-10})
    seal.verify(manifest["dependencies"])
    source.mechanism.writer(output, {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
                                    "rows": summaries, "scope": "alternate same-environment geometry"})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
