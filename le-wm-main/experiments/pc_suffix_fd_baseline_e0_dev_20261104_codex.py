"""Compare CEM and FD suffix optimizers from shared privileged-prefix starts."""
import itertools
import json
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_suffix_global_exploration_e0_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_bounded_fd_suffix_optimizer as fd
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = prior.source.ROOT / "results/pc_suffix_fd_baseline_e0_dev_20261104_codex"
PROTOCOL = prior.source.ROOT.parent / "docs/PC_SUFFIX_FD_BASELINE_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("cem", "fd")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, .19]}
            for y, theta in itertools.product((-.21, .21), (-1.12, 1.12))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    s = prior.source
    previous = (prior.tasks(), s.tasks(), s.v22_panel.tasks(), s.v22_panel.v22_mechanism.tasks(),
                s.v22_panel.v21_mechanism.tasks(), s.v22_panel.v21_panel.tasks(),
                s.v22_panel.v20_panel.tasks(), s.v22_panel.v19_mechanism.tasks(),
                s.v22_panel.v19_panel.tasks(), s.v22_panel.prior_panel.tasks(),
                s.v22_panel.previous.tasks(), s.prior.tasks(), *s.prior.source_panels())
    if any(t in panel for t in tasks() for panel in previous):
        raise ValueError("overlap with checked panels")
    old = s.prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(s.__file__), Path(fd.__file__),
                         Path(s.prior.prior.cem.__file__), Path(old.exact.__file__),
                         Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
                         Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "dependencies": deps,
                                       "scope": "privileged prefix baseline development; provenance unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        witnesses = []
        for attempt in old.exact.wall.feasibility(task):
            if attempt["within_horizon"] and attempt["steps_needed"] <= 15:
                action = (np.asarray(attempt["actions"][:15]) / old.exact.wall.HIGH).astype(np.float32)
                if action.shape != (15, 2) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("invalid witness")
                score = float(np.linalg.norm(old.exact.endpoints(task, action[None])[0]-task["goal"]))
                if score < .05:
                    witnesses.append((action, score))
        if not witnesses:
            rows.append({"root": root, "task": task, "status": "ineligible"})
            continue
        full, witness_score = witnesses[0]
        prefix, goal, trace, labels, runs = full[:5], np.asarray(task["goal"]), [], [], []

        def evaluate(actions):
            actions = np.asarray(actions, np.float32)
            if (actions.ndim != 3 or actions.shape[1:] != (15, 2) or not np.isfinite(actions).all()
                    or np.any(np.abs(actions) > 1) or not np.all(actions[:, :5] == prefix)):
                raise ValueError("invalid fixed-prefix action")
            pos = old.exact.endpoints(task, actions)
            scores = np.linalg.norm(pos-goal, axis=1)
            trace.append((actions.copy(), pos.copy(), scores.copy()))
            return pos, scores

        error = None
        try:
            rng = np.random.default_rng(2030050000 + root)
            tails = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-.5, .5, (2, 10, 2)))).astype(np.float32)
            initial_chains = np.concatenate((np.broadcast_to(prefix, (3, 5, 2)), tails), axis=1)
            _, initial = evaluate(initial_chains)
            for arm in ARMS:
                for start in range(3):
                    begin = len(trace)
                    if arm == "cem":
                        chain, score, history = s.prior.prior.cem.cem(
                            tails[start], prefix, 120, np.random.default_rng(2030060000 + 10*root + start),
                            evaluate, float(initial[start]), 10)
                        used = 120
                    else:
                        result = fd.optimize(evaluate, prefix, tails[start], float(initial[start]),
                                             fd.Config(samples=120))
                        chain, score, history, used = result["chain"], result["score"], result["generations"], result["samples"]
                    labels.append({"arm": arm, "start": start, "first_batch": begin, "last_batch": len(trace)})
                    runs.append({"arm": arm, "start": start, "best": float(score), "samples": used,
                                 "chain": np.asarray(chain, np.float32).tolist(), "generations": history})
        except Exception:
            error = traceback.format_exc()
        path = OUT / f"root{root:02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed", "error": error,
                     "witness": full.tolist(), "witness_score": witness_score,
                     "trace": path.name, "trace_sha256": seal.digest(path), "batches": len(trace),
                     "labels": labels, "runs": runs})
        print(f"root={root} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or result["manifest_sha256"] != seal.digest(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    old, rows = prior.source.prior.prior.prior, []
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["task"] != tasks()[root]:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            rows.append({"root": root, "status": row["status"], "error": row.get("error")})
            continue
        if seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("trace mismatch")
        witness = np.asarray(row["witness"], np.float32)
        path, _ = old.previous.independent(row["task"]["start"], witness[None])
        np.testing.assert_allclose(np.linalg.norm(path[0, -1, :2]-row["task"]["goal"]), row["witness_score"],
                                   atol=1e-12, rtol=0)
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            expected_tails = np.concatenate((np.zeros((1, 10, 2)),
                np.random.default_rng(2030050000 + root).uniform(-.5, .5, (2, 10, 2)))).astype(np.float32)
            np.testing.assert_array_equal(data["actions_0"][:, 5:], expected_tails)
            initial = data["scores_0"]
            for i in range(row["batches"]):
                actions, pos, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if (actions.dtype != np.float32 or actions.shape[1:] != (15, 2)
                        or np.any(np.abs(actions) > 1) or not np.all(actions[:, :5] == witness[:5])):
                    raise ValueError("action or prefix mismatch")
                path, _ = old.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], pos, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(pos-row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            arms = {}
            for arm in ARMS:
                bests, counts = [], []
                for record, label in zip(row["runs"], row["labels"], strict=True):
                    if record["arm"] != arm:
                        continue
                    if label["arm"] != arm or label["start"] != record["start"]:
                        raise ValueError("start mapping mismatch")
                    chunks = [data[f"scores_{i}"] for i in range(label["first_batch"], label["last_batch"])]
                    scores = np.concatenate(chunks) if chunks else np.empty(0)
                    if len(scores) != record["samples"] or len(scores) > 120:
                        raise ValueError("budget mismatch")
                    best = min(float(initial[record["start"]]), float(scores.min()) if len(scores) else float("inf"))
                    np.testing.assert_allclose(best, record["best"], atol=1e-12, rtol=0)
                    bests.append(best)
                    counts.append(len(scores))
                if len(bests) != 3:
                    raise ValueError("missing start")
                arms[arm] = {"bests": bests, "samples": counts,
                             "any_success": any(x < .05 for x in bests),
                             "prediction_tokens": 3 * sum(counts)}
        rows.append({"root": root, "status": "completed", "witness_score": row["witness_score"],
                     "shared_initial_tokens": 9, "arms": arms})
    if len(rows) != len(tasks()):
        raise ValueError("missing roots")
    seal.verify(manifest["dependencies"])
    old.writer(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
                                   "rows": rows, "scope": "alternate same-environment geometry"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
