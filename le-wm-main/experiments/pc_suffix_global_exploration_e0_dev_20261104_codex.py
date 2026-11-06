"""Equal-budget privileged-prefix diagnostic of two suffix optimizers."""
import itertools
import json
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_v22_response_retention_fixedbank_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_suffix_population_optimizer as proposal
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.ROOT / "results/pc_suffix_global_exploration_e0_dev_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_SUFFIX_GLOBAL_EXPLORATION_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("existing", "global_exploration")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, -.22]}
            for y, theta in itertools.product((-.24, .24), (-1.16, 1.16))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior_panels = (source.tasks(), source.v22_panel.tasks(), source.v22_panel.v22_mechanism.tasks(),
                    source.v22_panel.v21_mechanism.tasks(), source.v22_panel.v21_panel.tasks(),
                    source.v22_panel.v20_panel.tasks(), source.v22_panel.v19_mechanism.tasks(),
                    source.v22_panel.v19_panel.tasks(), source.v22_panel.prior_panel.tasks(),
                    source.v22_panel.previous.tasks(), source.prior.tasks(), *source.prior.source_panels())
    if any(task in panel for task in tasks() for panel in prior_panels):
        raise ValueError("overlap with checked panels")
    old = source.prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(source.__file__), Path(proposal.__file__),
                         Path(source.prior.prior.cem.__file__), Path(old.exact.__file__),
                         Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
                         Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "dependencies": deps, "arms": ARMS,
                                       "scope": "privileged prefix development, provenance unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        attempts = old.exact.wall.feasibility(task)
        witnesses = []
        for item in attempts:
            if item["within_horizon"] and item["steps_needed"] <= 15:
                actions = (np.asarray(item["actions"][:15]) / old.exact.wall.HIGH).astype(np.float32)
                if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                    raise ValueError("illegal witness")
                score = float(np.linalg.norm(old.exact.endpoints(task, actions[None])[0] - task["goal"]))
                if score < .05:
                    witnesses.append((actions, score))
        trace, labels, runs, error = [], [], [], None
        if not witnesses:
            rows.append({"root": root, "task": task, "status": "ineligible", "attempts": len(attempts)})
            continue
        full, witness_score = witnesses[0]
        prefix, goal = full[:5], np.asarray(task["goal"])

        def evaluate(actions):
            raw = np.asarray(actions, np.float32)
            if (raw.ndim != 3 or raw.shape[1:] != (15, 2) or not np.isfinite(raw).all()
                    or np.any(np.abs(raw) > 1) or not np.all(raw[:, :5] == prefix)):
                raise ValueError("invalid fixed-prefix batch")
            positions = old.exact.endpoints(task, raw)
            scores = np.linalg.norm(positions-goal, axis=1)
            trace.append((raw.copy(), positions.copy(), scores.copy()))
            return positions, scores

        try:
            zero = np.concatenate((prefix, np.zeros((10, 2), np.float32)))
            _, initial = evaluate(zero[None])
            labels.append({"arm": "shared_zero", "batch": 0})
            for arm in ARMS:
                for restart in range(3):
                    begin = len(trace)
                    seed = 2030020000 + 10*root + restart
                    if arm == "existing":
                        chain, score, history = source.prior.prior.cem.cem(
                            zero[5:], prefix, 120, np.random.default_rng(seed), evaluate, float(initial[0]), 10)
                    else:
                        result = proposal.optimize(evaluate, prefix, zero[5:], float(initial[0]),
                                                   proposal.Config(samples=120), seed=seed)
                        chain, score, history = result["chain"], result["score"], result["generations"]
                    labels.append({"arm": arm, "restart": restart, "first_batch": begin, "last_batch": len(trace)})
                    runs.append({"arm": arm, "restart": restart, "best": float(score),
                                 "chain": np.asarray(chain, np.float32).tolist(), "generations": history})
        except Exception:
            error = traceback.format_exc()
        path = OUT / f"root{root:02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed", "error": error,
                     "witness": full.tolist(), "witness_score": witness_score,
                     "initial_score": float(initial[0]) if error is None else None,
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
    old, summary = source.prior.prior.prior, []
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["task"] != tasks()[root]:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            summary.append({"root": root, "status": row["status"], "error": row.get("error")})
            continue
        if seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        prefix = np.asarray(row["witness"], np.float32)[:5]
        witness_path, _ = old.previous.independent(row["task"]["start"],
                                                    np.asarray(row["witness"], np.float32)[None])
        np.testing.assert_allclose(np.linalg.norm(witness_path[0, -1, :2] - row["task"]["goal"]),
                                   row["witness_score"], atol=1e-12, rtol=0)
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if (actions.dtype != np.float32 or actions.shape[1:] != (15, 2)
                        or np.any(np.abs(actions) > 1) or not np.all(actions[:, :5] == prefix)):
                    raise ValueError("action or prefix mismatch")
                path, _ = old.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions-row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            arms = {}
            for arm in ARMS:
                scores_out = []
                for run_data, label in zip(row["runs"], row["labels"][1:], strict=True):
                    if label["arm"] != arm:
                        continue
                    chunks = [data[f"scores_{i}"] for i in range(label["first_batch"], label["last_batch"])]
                    scores = np.concatenate(chunks)
                    if len(scores) != 120 or run_data["arm"] != arm or run_data["restart"] != label["restart"]:
                        raise ValueError("suffix budget or label mismatch")
                    np.testing.assert_allclose(min(row["initial_score"], float(scores.min())), run_data["best"],
                                               atol=1e-12, rtol=0)
                    scores_out.append(run_data["best"])
                if len(scores_out) != 3:
                    raise ValueError("missing optimizer start")
                arms[arm] = {"bests": scores_out, "any_success": any(s < .05 for s in scores_out),
                             "suffix_tokens": 1080}
        summary.append({"root": root, "status": "completed", "witness_score": row["witness_score"],
                        "zero_score": row["initial_score"], "shared_zero_tokens": 3, "arms": arms})
    if len(summary) != len(tasks()):
        raise ValueError("missing tasks")
    seal.verify(manifest["dependencies"])
    old.writer(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
                                   "rows": summary, "scope": "same-environment alternate geometry"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
