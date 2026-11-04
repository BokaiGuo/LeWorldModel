"""Analyze already scored v13 panel traces and recorded response retention."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_v13_global_proposals_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_E0_V13_COVERAGE_RETENTION_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUTPUT = source.OUT / "coverage_retention_posthoc.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result_path, audit_path = source.OUT / "result.json", source.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["result_sha256"] != seal.digest(result_path):
        raise ValueError("unverified source")
    result = json.loads(result_path.read_text())
    deps = seal.capture([Path(__file__), PROTOCOL, result_path, audit_path,
                         Path(source.independent.__code__.co_filename),
                         *(source.OUT / row["trace_file"] for row in result["rows"])])
    rows = []
    for row in result["rows"]:
        if seal.digest(source.OUT / row["trace_file"]) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        lengths = {str(i): {"samples": 0, "ever_right": 0, "terminal_right": 0,
                            "blocked": 0, "success": 0, "best_crossing_distance": None}
                   for i in (5, 10, 15)}
        goal = np.asarray(row["task"]["goal"])
        with np.load(source.OUT / row["trace_file"], allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions = data[f"actions_{i}"]
                path, blocked = source.independent(row["task"]["start"], actions)
                pos, scores = path[:, -1, :2], np.linalg.norm(path[:, -1, :2] - goal, axis=1)
                np.testing.assert_allclose(pos, data[f"positions_{i}"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, data[f"scores_{i}"], atol=1e-12, rtol=0)
                item = lengths[str(actions.shape[1])]
                crossing = np.any(path[:, 1:, 0] > 0, axis=1)
                item["samples"] += len(actions)
                item["ever_right"] += int(crossing.sum())
                item["terminal_right"] += int((path[:, -1, 0] > 0).sum())
                item["blocked"] += int(np.any(blocked, axis=1).sum())
                item["success"] += int((scores < .05).sum()) if actions.shape[1] == 15 else 0
                if np.any(crossing):
                    best = float(scores[crossing].min())
                    item["best_crossing_distance"] = min(item["best_crossing_distance"], best) \
                        if item["best_crossing_distance"] is not None else best
        selected_crossed = None
        if row["selected"] is not None:
            path, _ = source.independent(row["task"]["start"],
                                          np.asarray(row["selected"], np.float32)[None])
            selected_crossed = bool(np.any(path[0, 1:, 0] > 0))
        layers = []
        if row["status"] == "completed" and row["arm"] in ("v13_global_response", "v12_response"):
            compiler = row["result"]["child"] if row["arm"] == "v13_global_response" else row["result"]
            for layer in compiler["layers"]:
                actions = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                path, _ = source.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], [x["feature"] for x in layer["candidates"]],
                                           atol=1e-12, rtol=0)
                scores = np.linalg.norm(path[:, -1, :2] - goal, axis=1)
                np.testing.assert_allclose(scores, [x["score"] for x in layer["candidates"]],
                                           atol=1e-12, rtol=0)
                crossing = np.any(path[:, 1:, 0] > 0, axis=1)
                retained = np.asarray(layer["retained"], int)
                if len(np.unique(retained)) != len(retained) or np.any(retained < 0) or np.any(retained >= len(actions)):
                    raise ValueError("invalid retained indices")
                layers.append({"depth": layer["depth"], "candidates": len(actions),
                               "crossing_candidates": int(crossing.sum()),
                               "retained": len(retained),
                               "crossing_retained": int(crossing[retained].sum())})
        rows.append({"root": row["root"], "arm": row["arm"], "status": row["status"],
                     "selected_crossed": selected_crossed, "lengths": lengths, "layers": layers})
    summary = {}
    for arm in source.ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"selected_crossed": sum(r["selected_crossed"] is True for r in arm_rows)}
        for length in ("5", "10", "15"):
            values = [r["lengths"][length] for r in arm_rows]
            summary[arm][length] = {k: sum(x[k] for x in values)
                                     for k in ("samples", "ever_right", "terminal_right", "blocked", "success")}
            best = [x["best_crossing_distance"] for x in values if x["best_crossing_distance"] is not None]
            summary[arm][length]["best_crossing_distance"] = min(best) if best else None
            summary[arm][length]["roots_with_crossing"] = len(best)
        if arm in ("v13_global_response", "v12_response"):
            summary[arm]["layers"] = []
            for depth in range(3):
                values = [x for r in arm_rows for x in r["layers"] if x["depth"] == depth]
                summary[arm]["layers"].append({"depth": depth,
                    **{key: sum(x[key] for x in values) for key in
                       ("candidates", "crossing_candidates", "retained", "crossing_retained")},
                    "roots_with_crossing": sum(x["crossing_candidates"] > 0 for x in values),
                    "roots_retaining_crossing": sum(x["crossing_retained"] > 0 for x in values)})
    seal.verify(deps)
    source.prior.prior.previous.write(OUTPUT, {"dependencies": deps, "rows": rows, "summary": summary,
        "scope": "post-hoc repeated search trajectories on exposed development roots"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
