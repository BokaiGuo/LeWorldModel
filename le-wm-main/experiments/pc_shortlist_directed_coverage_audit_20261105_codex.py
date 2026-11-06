"""Replay the sealed fixed-bank diagnostic; do not change its original records."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_shortlist_directed_coverage_dev_20261105_codex as exp


def load(path):
    return json.loads(path.read_text())


def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-7, atol=1e-8), (a, b)
    else:
        assert a == b, (a, b)


def main():
    out = exp.OUT
    audit_path = out / "audit_replay.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    manifest = load(out / "manifest.json")
    exp.prior.prior.prior.seal.verify(manifest["dependencies"])
    exp.prior.prior.prior.seal.verify(manifest["source_dependencies"])
    rows = load(out / "summary.json")["rows"]
    assert len(rows) == 16 * 2 * 2 * 4
    assert len({(r["root"], r["direction"], r["depth"], r["arm"]) for r in rows}) == len(rows)
    records = {}
    for row in rows:
        root, direction, depth, arm = (row[k] for k in ("root", "direction", "depth", "arm"))
        name = f"r{root:02d}_{direction}_d{depth}_{arm}"
        assert row == load(out / f"{name}.json")
        assert row["status"] == "completed" and row["tokens"] == 828 and row["calls"] == 21
        source = load(exp.prior.OUT / f"plan_{root:02d}_{direction}_min.json")
        candidates = source["result"]["child"]["layers"][depth]["candidates"]
        seed = 2051201000 + root * 100 + exp.DIRECTIONS.index(direction) * 20 + depth * 4
        # Reconstruct ranking, float32 deduplication and fixed random exploration independently.
        seen, ordered = set(), []
        for i in sorted(range(len(candidates)), key=lambda i: (candidates[i]["score"], i)):
            key = np.asarray(candidates[i]["chain"], np.float32).tobytes()
            if key not in seen:
                seen.add(key)
                ordered.append(i)
        assert len(ordered) >= 5
        indices = ordered[:4]
        if arm.startswith("explore"):
            indices = ordered[:3] + [int(np.random.default_rng(seed).choice(ordered[3:]))]
        assert indices == row["result"]["shortlist"]
        folder = out / f"{name}_calls"
        predictor = exp.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        index, total, best_score, best_action = 0, 0, float("inf"), None

        def replay(actions):
            nonlocal index, total, best_score, best_action
            attempt = load(folder / f"{index:06d}_attempt.json")
            returned = load(folder / f"{index:06d}_returned.json")
            action_file, result_file = folder / attempt["actions"], folder / returned["result"]
            assert attempt["index"] == returned["index"] == index
            assert exp.prior.prior.prior.journal.digest(action_file) == attempt["sha256"]
            assert exp.prior.prior.prior.journal.digest(result_file) == returned["sha256"]
            actual_actions = np.asarray(actions, np.float32)
            with np.load(action_file) as saved:
                assert np.array_equal(actual_actions, saved["actions"])
            with np.load(result_file) as saved:
                features, scores = saved["features"].copy(), saved["scores"].copy()
            replay_features, replay_scores = predictor(actual_actions.copy())
            assert np.allclose(features, replay_features, rtol=1e-7, atol=1e-8)
            assert np.allclose(scores, replay_scores, rtol=1e-7, atol=1e-8)
            assert attempt["cost"] == len(actual_actions) * 3
            total += attempt["cost"]
            k = int(np.argmin(scores))
            if scores[k] < best_score:
                best_score, best_action = float(scores[k]), actual_actions[k].copy()
            index += 1
            return features, scores

        regenerated = exp.evaluate_arm(replay, candidates, depth, indices,
                                       "directed" if arm.endswith("directed") else "min", seed + 2)
        compare(regenerated, row["result"])
        assert index == 21 and total == 828
        assert np.isclose(best_score, row["result"]["final_score"], rtol=1e-7, atol=1e-8)
        # Ties can have different actions; require returned action to be an observed minimum.
        result_actions = np.asarray(row["result"]["final_actions"], np.float32)
        assert np.array_equal(best_action, result_actions) or any(
            np.any(np.all(np.load(folder / f"{i:06d}_actions.npz")["actions"] == result_actions,
                          axis=(1, 2)) & np.isclose(np.load(folder / f"{i:06d}_result.npz")["scores"],
                                                    best_score, rtol=1e-7, atol=1e-8))
            for i in range(index))
        records[(root, direction, depth, arm)] = row
        print(f"replayed {name}", flush=True)

    summary = {}
    for arm in exp.ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "mean_score": float(np.mean([r["result"]["final_score"] for r in selected])),
            "successes_of_64": sum(r["result"]["final_score"] < .05 for r in selected),
            "mean_seconds": float(np.mean([r["seconds"] for r in selected])),
        }
    paired = {}
    for left, right in (("rank_min", "explore_min"), ("rank_directed", "explore_directed"),
                        ("rank_min", "rank_directed"), ("explore_min", "explore_directed")):
        task_deltas = []
        changed = 0
        for root in range(16):
            deltas = []
            for direction in exp.DIRECTIONS:
                for depth in (0, 1):
                    a, b = (records[(root, direction, depth, arm)]["result"] for arm in (left, right))
                    deltas.append(b["final_score"] - a["final_score"])
                    changed += a["retained"] != b["retained"]
            task_deltas.append(float(np.mean(deltas)))
        paired[f"{right}_minus_{left}"] = {
            "mean_task_delta": float(np.mean(task_deltas)), "task_deltas": task_deltas,
            "retained_changed_of_64": changed,
        }
    exp.prior.prior.prior.journal.write_json(audit_path, {
        "status": "passed", "rows": len(rows), "calls": len(rows) * 21,
        "cost_tokens_per_row": 828, "summary": summary, "paired": paired,
        "caveat": "Development diagnostic; task roots are not independent confirmation."})
    print(json.dumps({"summary": summary, "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
