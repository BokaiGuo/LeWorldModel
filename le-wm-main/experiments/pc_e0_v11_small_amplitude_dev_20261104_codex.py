"""Fixed v11 amplitude ablation on a fresh exposed E0 development panel."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as prior
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_e0_v11_small_amplitude_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_V11_SMALL_AMPLITUDE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v11_small", "v11_original", "direct_cem")
CAP = 1920


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.19, .21), (-.55, .9), (-.17, .13))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = (prior.tasks(), prior.previous_waypoint.tasks(), prior.prior.tasks(),
           prior.prior.previous.tasks())
    if any(task in panel for task in tasks() for panel in old):
        raise ValueError("task overlaps an explicitly checked panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__),
        Path(prior.v11.__file__), Path(prior.core.__file__),
        Path(prior.prior.previous.exact.__file__), Path(independent.__code__.co_filename)])
    OUT.mkdir()
    writer = prior.prior.previous.write
    writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
        "compiler_seed": "2029111000 + root", "cem_init_seed": "2029112000 + root*100 + restart",
        "cem_seed": "2029113000 + root*100 + restart", "dependencies": deps,
        "scope": "development only; broader historical independence unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            meter, trace, error, result = Meter(CAP), [], None, None
            started = time.monotonic()

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                def evaluate(batch):
                    position = prior.prior.previous.exact.endpoints(task, batch)
                    return position, np.linalg.norm(position - goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            try:
                if arm.startswith("v11_"):
                    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                                    for item in prior.core.NOMINALS)
                    config = prior.v11.Config(action_dim=2, segment_length=5, segments=3,
                        model_packet_length=5, nominals=nominal, max_modes=1, token_cap=CAP,
                        amplitudes=(.1, .2) if arm == "v11_small" else (.25, .5),
                        seed=2029111000 + root)
                    result = prior.v11.compile_program(predict, goal, config)
                    selected, score = result["selected"], result["selected_score"]
                    if result["resource"]["prediction_tokens"] != meter.token_samples:
                        raise ValueError("compiler and meter differ")
                else:
                    bases = [np.zeros((15, 2))]
                    for restart in (1, 2):
                        rng = np.random.default_rng(2029112000 + root * 100 + restart)
                        bases.append(np.clip(prior.core.interpolate(
                            rng.normal(0, .6, size=(1, 4, 2)), 15)[0], -1, 1))
                    _, scores = predict(np.asarray(bases))
                    choice = int(np.argmin(scores))
                    selected, score = bases[choice], float(scores[choice])
                    runs = []
                    for restart, count in enumerate((213, 212, 212)):
                        chain, value, generations = prior.core.cem(
                            bases[restart], [], count,
                            np.random.default_rng(2029113000 + root * 100 + restart),
                            predict, float(scores[restart]), 4)
                        runs.append({"restart": restart, "count": count, "score": value,
                                     "generations": generations})
                        if value < score:
                            selected, score = chain, value
                    result = {"selected": np.asarray(selected, np.float32).tolist(),
                              "score": score, "runs": runs}
            except Exception:
                error = traceback.format_exc()
                selected, score = None, None
            path = OUT / f"root{root:02d}_{arm}.npz"
            arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                      for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            rows.append({"root": root, "task": task, "arm": arm,
                "status": "failed" if error else "completed", "error": error, "result": result,
                "selected": np.asarray(selected, np.float32).tolist() if selected is not None else None,
                "score": float(score) if score is not None else None, "counts": meter.as_dict(),
                "wall_seconds": time.monotonic() - started, "trace_file": path.name,
                "trace_sha256": seal.digest(path), "batches": len(trace)})
            print(f"root={root} arm={arm} score={score} tokens={meter.token_samples} error={bool(error)}",
                  flush=True)
    seal.verify(deps)
    writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError("audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if result["manifest_sha256"] != seal.digest(OUT / "manifest.json") or manifest["tasks"] != tasks():
        raise ValueError("manifest mismatch")
    if len(result["rows"]) != len(tasks()) * len(ARMS):
        raise ValueError("missing result")
    counts = {arm: {"complete": 0, "success": 0, "scores": [], "tokens": []} for arm in ARMS}
    samples = 0
    for row, (root, arm) in zip(result["rows"], itertools.product(range(8), ARMS), strict=True):
        if (row["root"], row["arm"], row["task"]) != (root, arm, tasks()[root]):
            raise ValueError("result identity")
        path = OUT / row["trace_file"]
        if seal.digest(path) != row["trace_sha256"]:
            raise ValueError("trace hash")
        tokens, trajectories, full_scores, found = 0, 0, [], False
        with np.load(path, allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, positions, scores = (data[f"{name}_{i}"]
                                               for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1:
                    raise ValueError("action contract")
                exact = prior.prior.previous.exact.endpoints(row["task"], actions)
                path_values, _ = independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path_values[:, -1, :2], exact, atol=1e-12, rtol=0)
                np.testing.assert_allclose(positions, exact, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(exact - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                tokens += len(actions) * actions.shape[1] // 5
                trajectories += len(actions)
                if actions.shape[1] == 15:
                    full_scores.extend(scores.tolist())
                    if row["selected"] is not None:
                        found |= bool(np.any(np.all(actions == np.asarray(row["selected"], np.float32),
                                                    axis=(1, 2)) & (np.abs(scores-row["score"]) < 1e-12)))
        if tokens > CAP or tokens != row["counts"]["prediction_token_samples"]:
            raise ValueError("token accounting")
        if trajectories != row["counts"]["trajectory_samples"]:
            raise ValueError("sample accounting")
        if row["status"] == "completed":
            if not found or row["score"] > min(full_scores) + 1e-12:
                raise ValueError("complete-chain incumbent discarded")
            counts[arm]["complete"] += 1
            counts[arm]["success"] += int(row["score"] < .05)
            counts[arm]["scores"].append(row["score"])
        counts[arm]["tokens"].append(tokens)
        samples += trajectories
    seal.verify(manifest["dependencies"])
    prior.prior.previous.write(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
        "counts": counts, "trajectory_samples": samples,
        "scope": "same-backend plus independent collision implementation; not independent replication"})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
