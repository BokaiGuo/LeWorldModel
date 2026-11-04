"""Common candidate bank; compare smooth and full-action suffix CEM."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as prior
from experiments import pc_v13_fixedbank_suffix_budget_dev_20261104_codex as budget
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v13_fixedbank_suffix_expression_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V13_FIXEDBANK_SUFFIX_EXPRESSION_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, -.17]}
            for y, theta in itertools.product((-.19, .18), (-.7, .8))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = (budget.tasks(), prior.tasks(), prior.previous.tasks(), prior.previous14.tasks(),
           prior.previous.remainder.tasks(), prior.previous.small.tasks(),
           prior.previous.prior.tasks(), prior.previous.prior.previous_waypoint.tasks(),
           prior.previous.prior.prior.tasks(), prior.previous.prior.prior.previous.tasks(),
           prior.old_panel.tasks())
    if any(task in panel for task in tasks() for panel in old):
        raise ValueError("overlap with enumerated stopped panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(budget.__file__), Path(prior.__file__),
        Path(cem.__file__), Path(prior.compiler.__file__), Path(prior.compiler_child.__file__),
        Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
        Path(prior.exact.wall.physics.__file__), Path(prior.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    prior.writer(OUT / "manifest.json", {"tasks": tasks(), "dependencies": deps,
        "starts": 3, "samples_per_start": 480, "knots": [4, 10],
        "scope": "fixed-bank development diagnostic; independence unknown"})
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item) for item in cem.NOMINALS)
    rows = []
    for root, task in enumerate(tasks()):
        trace, meter = [], Meter(1920)
        goal = np.asarray(task["goal"])

        def evaluate(actions):
            raw = np.asarray(actions, np.float32)
            pos = prior.exact.endpoints(task, raw)
            score = np.linalg.norm(pos - goal, axis=1)
            trace.append((raw.copy(), pos.copy(), score.copy()))
            return pos, score

        def planning_predict(actions):
            return meter.predict(actions, evaluate)

        try:
            config = prior.compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, max_modes=1,
                token_cap=1920, seed=2029151000 + root)
            compiled = prior.compiler.compile_program(planning_predict, goal, config)
            generation_batches = len(trace)
            if meter.token_samples != compiled["resource"]["prediction_tokens"]:
                raise ValueError("generation ledger mismatch")
            layer, candidates = compiled["child"]["layers"][0], []
            if len(layer["candidates"]) != 10:
                raise ValueError("invalid candidate bank")
            for index, candidate in enumerate(layer["candidates"]):
                prefix = np.asarray(candidate["chain"], np.float32)
                base = np.zeros((10, 2), np.float32)
                initial_batch = len(trace)
                _, score = evaluate(np.concatenate((prefix, base))[None])
                runs = []
                for knots in (4, 10):
                    for restart in range(3):
                        begin = len(trace)
                        chain, best, generations = cem.cem(base, prefix, 480,
                            np.random.default_rng(2029152000 + root * 10 + restart),
                            evaluate, float(score[0]), knots)
                        runs.append({"knots": knots, "restart": restart, "first_batch": begin,
                            "last_batch": len(trace), "best": best, "chain": np.asarray(chain, np.float32).tolist(),
                            "generations": generations})
                candidates.append({"index": index, "prefix": prefix.tolist(),
                                   "initial_batch": initial_batch, "initial": float(score[0]), "runs": runs})
            error = None
        except Exception:
            compiled, generation_batches, candidates, error = None, None, [], traceback.format_exc()
        path = OUT / f"root{root:02d}_trace.npz"
        arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                  for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed", "error": error,
                     "compiler": compiled, "generation_batches": generation_batches,
                     "generation_cost": meter.as_dict(), "candidates": candidates, "batches": len(trace),
                     "trace": path.name, "trace_sha256": seal.digest(path)})
        print(f"root={root} status={rows[-1]['status']} batches={len(trace)}", flush=True)
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
    summary, gen_total, suffix_total = [], 0, 0
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["task"] != tasks()[root] or row["status"] != "completed":
            raise ValueError("incomplete root")
        if seal.digest(OUT / row["trace"]) != row["trace_sha256"] or len(row["candidates"]) != 10:
            raise ValueError("trace or candidate mismatch")
        layer = row["compiler"]["child"]["layers"][0]
        retained, scores_by_arm = layer["retained"], {4: [], 10: []}
        generation_tokens = 0
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, pos, scores = (data[f"{key}_{i}"] for key in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1:
                    raise ValueError("invalid action batch")
                replay, _ = prior.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(replay[:, -1, :2], pos, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(pos - row["task"]["goal"], axis=1), atol=1e-12, rtol=0)
                tokens = len(actions) * actions.shape[1] // 5
                if i < row["generation_batches"]:
                    generation_tokens += tokens
                else:
                    if actions.shape[1:] != (15, 2):
                        raise ValueError("suffix horizon mismatch")
                    suffix_total += len(actions)
            if generation_tokens != row["generation_cost"]["prediction_token_samples"]:
                raise ValueError("generation token mismatch")
            gen_total += generation_tokens
            for index, candidate in enumerate(row["candidates"]):
                prefix = np.asarray(candidate["prefix"], np.float32)
                np.testing.assert_array_equal(prefix, np.asarray(layer["candidates"][index]["chain"], np.float32))
                ib = candidate["initial_batch"]
                np.testing.assert_array_equal(data[f"actions_{ib}"][0, :5], prefix)
                if candidate["initial"] != float(data[f"scores_{ib}"][0]):
                    raise ValueError("initial score mismatch")
                runs = candidate["runs"]
                if [(r["knots"], r["restart"]) for r in runs] != [(k, r) for k in (4, 10) for r in range(3)]:
                    raise ValueError("run mapping mismatch")
                for run in runs:
                    actions = np.concatenate([data[f"actions_{i}"] for i in range(run["first_batch"], run["last_batch"])])
                    scores = np.concatenate([data[f"scores_{i}"] for i in range(run["first_batch"], run["last_batch"])])
                    if len(actions) != 480 or not np.all(actions[:, :5] == prefix):
                        raise ValueError("prefix or count mismatch")
                    if not np.isclose(run["best"], min(candidate["initial"], float(scores.min())), atol=1e-12):
                        raise ValueError("incumbent mismatch")
                for knots in (4, 10):
                    scores_by_arm[knots].append(min(r["best"] for r in runs if r["knots"] == knots))
        summary.append({"root": root, **{
            f"knots{k}_{group}": min(scores_by_arm[k][i] for i in indexes)
            for k in (4, 10) for group, indexes in (("all", range(10)), ("retained", retained))
        }, "candidate_comparisons": scores_by_arm})
    seal.verify(manifest["dependencies"])
    prior.writer(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
        "generation_tokens": gen_total, "suffix_samples": suffix_total, "suffix_tokens": suffix_total * 3,
        "summary": summary, "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
