"""Fixed v13 first-layer candidates, bounded suffix budget diagnostic."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v13_fixedbank_suffix_budget_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V13_FIXEDBANK_SUFFIX_BUDGET_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, .13]}
            for y, theta in itertools.product((-.23, .21), (-.9, .95))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = (prior.tasks(), prior.previous.tasks(), prior.previous14.tasks(),
           prior.previous.remainder.tasks(), prior.previous.small.tasks(),
           prior.previous.prior.tasks(), prior.previous.prior.previous_waypoint.tasks(),
           prior.previous.prior.prior.tasks(), prior.previous.prior.prior.previous.tasks(),
           prior.old_panel.tasks())
    if any(task in panel for task in tasks() for panel in old):
        raise ValueError("overlap with enumerated stopped panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(cem.__file__),
        Path(prior.compiler.__file__), Path(prior.compiler_child.__file__),
        Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
        Path(prior.exact.wall.physics.__file__), Path(prior.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    prior.writer(OUT / "manifest.json", {"tasks": tasks(), "dependencies": deps,
        "small": 120, "large_restarts": [480, 480, 480],
        "seed": "2029142000+root*10+restart", "scope": "candidate diagnostic; independence unknown"})
    rows = []
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in cem.NOMINALS)
    for root, task in enumerate(tasks()):
        trace = []
        goal = np.asarray(task["goal"])
        meter = Meter(1920)

        def predict(actions):
            raw = np.asarray(actions, np.float32)
            position = prior.exact.endpoints(task, raw)
            score = np.linalg.norm(position - goal, axis=1)
            trace.append((raw.copy(), position.copy(), score.copy()))
            return position, score

        def planning_predict(actions):
            def evaluate(batch):
                position = prior.exact.endpoints(task, batch)
                return position, np.linalg.norm(position - goal, axis=1)

            return meter.predict(actions, evaluate)

        try:
            config = prior.compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, max_modes=1,
                token_cap=1920, seed=2029141000 + root)
            compiled = prior.compiler.compile_program(planning_predict, goal, config)
            layer = compiled["child"]["layers"][0]
            if len(layer["candidates"]) != 10 or meter.token_samples != compiled["resource"]["prediction_tokens"]:
                raise ValueError("candidate bank or generation ledger mismatch")
            candidates = []
            for index, candidate in enumerate(layer["candidates"]):
                prefix = np.asarray(candidate["chain"], np.float32)
                base = np.zeros((10, 2), np.float32)
                _, score = predict(np.concatenate((prefix, base))[None])
                runs = []
                for label, restart, count in [("small", 0, 120), *[("large", k, 480) for k in range(3)]]:
                    begin = len(trace)
                    chain, best, generations = cem.cem(base, prefix, count,
                        np.random.default_rng(2029142000 + root * 10 + restart),
                        predict, float(score[0]), 4)
                    runs.append({"label": label, "restart": restart, "samples": count,
                        "first_batch": begin, "last_batch": len(trace), "best": best,
                        "chain": np.asarray(chain, np.float32).tolist(), "generations": generations})
                candidates.append({"index": index, "prefix": prefix.tolist(), "initial": float(score[0]),
                                   "runs": runs})
            error = None
        except Exception:
            layer, candidates, error = None, [], traceback.format_exc()
        path = OUT / f"root{root:02d}_trace.npz"
        arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                  for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed",
            "error": error, "layer": layer, "candidates": candidates, "generation_cost": meter.as_dict(),
            "batches": len(trace), "trace": path.name, "trace_sha256": seal.digest(path)})
        print(f"root={root}: {rows[-1]['status']} diagnostic batches={len(trace)}", flush=True)
    seal.verify(deps)
    prior.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError("audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or result["manifest_sha256"] != seal.digest(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    if len(result["rows"]) != 4:
        raise ValueError("missing roots")
    summary = []
    total = 0
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["task"] != tasks()[root] or row["status"] != "completed":
            raise ValueError("incomplete root")
        if len(row["candidates"]) != 10 or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("candidate or trace mismatch")
        retained = row["layer"]["retained"]
        small, large = [], []
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or actions.shape[1:] != (15, 2) or np.max(np.abs(actions)) > 1:
                    raise ValueError("invalid action")
                independent, _ = prior.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(independent[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                total += len(actions)
            for index, candidate in enumerate(row["candidates"]):
                prefix = np.asarray(candidate["prefix"], np.float32)
                np.testing.assert_array_equal(prefix, np.asarray(row["layer"]["candidates"][index]["chain"], np.float32))
                runs = candidate["runs"]
                if [(x["label"], x["restart"], x["samples"]) for x in runs] != [
                        ("small", 0, 120), ("large", 0, 480), ("large", 1, 480), ("large", 2, 480)]:
                    raise ValueError("run policy mismatch")
                for run in runs:
                    actions = np.concatenate([data[f"actions_{i}"] for i in range(run["first_batch"], run["last_batch"])])
                    scores = np.concatenate([data[f"scores_{i}"] for i in range(run["first_batch"], run["last_batch"])])
                    if len(actions) != run["samples"] or not np.all(actions[:, :5] == prefix):
                        raise ValueError("fixed prefix or sample count mismatch")
                    if not np.isclose(run["best"], min(candidate["initial"], float(scores.min())), atol=1e-12):
                        raise ValueError("incumbent mismatch")
                for i in range(10):
                    np.testing.assert_array_equal(data[f"actions_{runs[0]['first_batch'] + i}"],
                                                  data[f"actions_{runs[1]['first_batch'] + i}"])
                small.append(runs[0]["best"])
                large.append(min(r["best"] for r in runs[1:]))
        summary.append({"root": root, "small_all": min(small), "large_all": min(large),
                        "small_retained": min(small[i] for i in retained),
                        "large_retained": min(large[i] for i in retained),
                        "candidate_improved": sum(b < a - 1e-12 for a, b in zip(small, large, strict=True))})
    seal.verify(manifest["dependencies"])
    prior.writer(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
        "trajectory_samples": total, "prediction_tokens": total * 3, "summary": summary,
        "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
