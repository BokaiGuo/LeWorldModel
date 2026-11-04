"""Predeclared E0 waypoint diagnostic with complete prediction trace."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_recursive_response_v10_e0_dev_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_goal_waypoint_cem_dev_20261104_codex as waypoint
from experiments.spectral_reachability import pc_recursive_response_v11 as v11
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_e0_goal_waypoint_search_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_GOAL_WAYPOINT_SEARCH_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("waypoint_cem", "v11_response", "direct_cem")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.18, .19), (-.65, .45), (-.07, .13))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in prior.tasks() or task in prior.previous.tasks() for task in tasks()):
        raise ValueError("overlaps cited sealed panel")
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(waypoint.__file__),
        Path(v11.__file__), Path(core.__file__), Path(prior.previous.exact.__file__),
        Path(prior.previous.exact.wall.__file__), Path(prior.__file__), Path(prior.previous.__file__)])
    OUT.mkdir()
    prior.previous.write(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": 1920,
        "seed_policy": "waypoint=2028120100+root*100; v11=2027113100+root; "
                       "direct=2027112900/2027112800+root*100+restart",
        "dependencies": dependencies, "scope": "development only; no independence or gate claim"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            meter, trace, error, result = Meter(1920), [], None, None
            started = time.monotonic()

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                def evaluate(batch):
                    positions = prior.previous.exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions - goal, axis=1)
                features, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), features.copy(), scores.copy()))
                return features, scores

            try:
                if arm == "waypoint_cem":
                    result = waypoint.plan(predict, meter, goal, 2028120100 + root * 100)
                    selected, score = result["selected"], result["selected_score"]
                elif arm == "v11_response":
                    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                                     for nominal in core.NOMINALS)
                    config = v11.Config(action_dim=2, segment_length=5, segments=3,
                        model_packet_length=5, nominals=nominals, max_modes=1, token_cap=1920,
                        seed=2027113100 + root)
                    result = v11.compile_program(predict, goal, config)
                    selected, score = result["selected"], result["selected_score"]
                    if result["resource"]["prediction_tokens"] != meter.token_samples:
                        raise ValueError("v11 ledger mismatch")
                else:
                    bases = [np.zeros((15, 2))]
                    for restart in (1, 2):
                        rng = np.random.default_rng(2027112900 + root * 100 + restart)
                        bases.append(np.clip(core.interpolate(rng.normal(0, .6, size=(1, 4, 2)), 15)[0], -1, 1))
                    _, initial = predict(np.asarray(bases))
                    choice = int(np.argmin(initial))
                    selected, score, runs = bases[choice], float(initial[choice]), []
                    for restart, count in enumerate((213, 212, 212)):
                        chain, value, generations = core.cem(bases[restart], [], count,
                            np.random.default_rng(2027112800 + root * 100 + restart), predict,
                            float(initial[restart]), 4)
                        runs.append({"restart": restart, "count": count, "score": value,
                                     "generations": generations})
                        if value < score:
                            selected, score = chain, value
                    result = {"selected": np.asarray(selected, np.float32).tolist(), "score": score, "runs": runs}
            except Exception:
                error = traceback.format_exc()
                selected, score = None, None
            path = OUT / f"root{root:02d}_{arm}.npz"
            arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                      for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                         "error": error, "result": result, "selected": np.asarray(selected, np.float32).tolist()
                         if selected is not None else None, "score": float(score) if score is not None else None,
                         "counts": meter.as_dict(), "wall_seconds": time.monotonic() - started,
                         "trace_file": path.name, "trace_sha256": prior.previous.sha(path), "batches": len(trace)})
            print(f"root={root} arm={arm} score={score} tokens={meter.token_samples} error={bool(error)}", flush=True)
    seal.verify(dependencies)
    prior.previous.write(OUT / "result.json", {"manifest_sha256": prior.previous.sha(OUT / "manifest.json"),
                                               "rows": rows})


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError("audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or result["manifest_sha256"] != prior.previous.sha(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    if len(result["rows"]) != 24:
        raise ValueError("missing arm/root")
    count = 0
    for row, (root, arm) in zip(result["rows"], itertools.product(range(8), ARMS), strict=True):
        if row["root"] != root or row["arm"] != arm or row["task"] != tasks()[root]:
            raise ValueError("row identity mismatch")
        path = OUT / row["trace_file"]
        if row["trace_sha256"] != prior.previous.sha(path):
            raise ValueError("trace hash")
        tokens, samples, full = 0, 0, []
        selected_found = False
        with np.load(path, allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, saved_pos, saved_scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or np.max(np.abs(actions)) > 1:
                    raise ValueError("invalid actions")
                pos = prior.previous.exact.endpoints(row["task"], actions)
                scores = np.linalg.norm(pos - np.asarray(row["task"]["goal"]), axis=1)
                np.testing.assert_allclose(pos, saved_pos, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, saved_scores, atol=1e-12, rtol=0)
                tokens += len(actions) * actions.shape[1] // 5
                samples += len(actions)
                if actions.shape[1] == 15:
                    full.extend(scores.tolist())
                    if row["selected"] is not None:
                        selected = np.asarray(row["selected"], np.float32)
                        selected_found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                                      & (np.abs(scores - row["score"]) < 1e-12)))
            if row["status"] == "completed" and (not selected_found or row["score"] > min(full) + 1e-12):
                raise ValueError("selected not a full-chain minimum")
        if tokens > 1920 or tokens != row["counts"]["prediction_token_samples"] or samples != row["counts"]["trajectory_samples"]:
            raise ValueError("budget mismatch")
        count += samples
    seal.verify(manifest["dependencies"])
    prior.previous.write(OUT / "audit.json", {"status": "passed", "result_sha256": prior.previous.sha(OUT / "result.json"),
        "plans": len(result["rows"]), "trajectory_samples": count,
        "scope": "same exact backend; no independent physical environment"})
    print(f"audited {len(result['rows'])} plans and {count} trajectories", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
