"""Fixed-bank successor-response retention diagnostic on new exact-Wall tasks."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v15_portfolio_e0_dev_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_retention_e0 as retention
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_response_retention_fixedbank_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_RETENTION_FIXEDBANK_E0_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, -.21]}
            for y, theta in itertools.product((-.23, .23), (-1., 1.))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in panel for task in tasks() for panel in (prior.tasks(), *prior.source_panels())):
        raise ValueError("task overlaps cited panel")
    old = prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(retention.__file__),
        Path(old.compiler.__file__), Path(old.compiler_child.__file__), Path(prior.prior.cem.__file__),
        Path(old.exact.__file__), Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
        Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "dependencies": deps,
        "scope": "fixed bank development; broader provenance unknown"})
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in prior.prior.cem.NOMINALS)
    rows = []
    for root, task in enumerate(tasks()):
        trace, meter, goal = [], Meter(1920), np.asarray(task["goal"])

        def evaluate(actions):
            raw = np.asarray(actions, np.float32)
            if raw.ndim != 3 or raw.shape[1] % 5 or not np.isfinite(raw).all() or np.abs(raw).max() > 1:
                raise ValueError("invalid actions")
            pos = old.exact.endpoints(task, raw)
            score = np.linalg.norm(pos - goal, axis=1)
            trace.append((raw.copy(), pos.copy(), score.copy()))
            return pos, score

        def planning_predict(actions):
            return meter.predict(actions, evaluate)

        try:
            config = old.compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, token_cap=1920, seed=2029181000 + root)
            compiled = old.compiler.compile_program(planning_predict, goal, config)
            generation_batches = len(trace)
            if meter.token_samples != compiled["resource"]["prediction_tokens"]:
                raise ValueError("generation accounting mismatch")
            layer = compiled["child"]["layers"][0]
            if len(layer["candidates"]) != 10:
                raise ValueError("unexpected candidate count")
            axis = np.eye(20).reshape(20, 10, 2)
            probes = np.concatenate((np.zeros((1, 10, 2)), .05 * axis, -.05 * axis))
            candidates, responses, values, lookahead = [], [], [], []
            for index, candidate in enumerate(layer["candidates"]):
                prefix = np.asarray(candidate["chain"], np.float32)
                actions = np.concatenate((np.broadcast_to(prefix, (41, 5, 2)), probes), axis=1)
                response_batch = len(trace)
                positions, scores = evaluate(actions)
                matrix = ((positions[1:21] - positions[21:41]) / .1).T
                value = retention.ridge_residual(matrix, goal - positions[0])
                responses.append(matrix)
                values.append(value)
                lookahead_rng = np.random.default_rng(2029184000 + root * 100 + index)
                lookahead_actions = np.concatenate((
                    np.broadcast_to(prefix, (24, 5, 2)),
                    lookahead_rng.uniform(-1., 1., (24, 10, 2))), axis=1)
                lookahead_batch = len(trace)
                _, lookahead_scores = evaluate(lookahead_actions)
                lookahead_score = min(float(scores[0]), float(lookahead_scores.min()))
                lookahead.append(lookahead_score)
                runs = []
                for restart in range(3):
                    begin = len(trace)
                    chain, best, generations = prior.prior.cem.cem(
                        np.zeros((10, 2)), prefix, 120,
                        np.random.default_rng(2029182000 + 10 * root + restart), evaluate,
                        float(scores[0]), 10)
                    runs.append({"restart": restart, "first_batch": begin, "last_batch": len(trace),
                                 "best": float(best), "chain": np.asarray(chain, np.float32).tolist(),
                                 "generations": generations})
                candidates.append({"index": index, "prefix": prefix.tolist(), "response_batch": response_batch,
                                   "response": matrix.tolist(), "ridge": value,
                                   "zero_tail_score": float(scores[0]), "lookahead_batch": lookahead_batch,
                                   "lookahead_score": lookahead_score, "runs": runs})
            selected = retention.select([x["score"] for x in layer["candidates"]],
                [x["feature"] for x in layer["candidates"]], responses, values, lookahead, 2029183000 + root)
            error = None
        except Exception:
            compiled, generation_batches, candidates, selected, error = None, None, [], None, traceback.format_exc()
        path = OUT / f"root{root:02d}_trace.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed", "error": error,
            "compiler": compiled, "generation_batches": generation_batches, "generation_cost": meter.as_dict(),
            "candidates": candidates, "selected": selected, "batches": len(trace),
            "trace": path.name, "trace_sha256": seal.digest(path)})
        print(f"root={root} status={rows[-1]['status']} batches={len(trace)}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    out = OUT / "audit.json"
    if out.exists():
        raise FileExistsError(out)
    manifest, result = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "result.json"))
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or result["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or len(result["rows"]) != 4):
        raise ValueError("manifest mismatch")
    old = prior.prior.prior
    summary = []
    for root, row in enumerate(result["rows"]):
        if row["root"] != root or row["task"] != tasks()[root] or row["status"] != "completed":
            summary.append({"root": root, "status": "failed", "error": row["error"]})
            continue
        if seal.digest(OUT / row["trace"]) != row["trace_sha256"] or len(row["candidates"]) != 10:
            raise ValueError("trace mismatch")
        layer = row["compiler"]["child"]["layers"][0]
        bests, responses, values, lookahead = [], [], [], []
        generation_tokens = response_tokens = lookahead_tokens = suffix_tokens = 0
        with np.load(OUT / row["trace"], allow_pickle=False) as data:
            response_batches = {candidate["response_batch"] for candidate in row["candidates"]}
            lookahead_batches = {candidate["lookahead_batch"] for candidate in row["candidates"]}
            for i in range(row["batches"]):
                actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1:
                    raise ValueError("action contract")
                replay, _ = old.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(replay[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                tokens = len(actions) * actions.shape[1] // 5
                if i < row["generation_batches"]:
                    generation_tokens += tokens
                elif i in response_batches:
                    response_tokens += tokens
                elif i in lookahead_batches:
                    lookahead_tokens += tokens
                else:
                    suffix_tokens += tokens
            if generation_tokens != row["generation_cost"]["prediction_token_samples"]:
                raise ValueError("generation ledger mismatch")
            for index, candidate in enumerate(row["candidates"]):
                prefix = np.asarray(candidate["prefix"], np.float32)
                np.testing.assert_array_equal(prefix, np.asarray(layer["candidates"][index]["chain"], np.float32))
                i = candidate["response_batch"]
                actions, positions = data[f"actions_{i}"], data[f"positions_{i}"]
                if actions.shape != (41, 15, 2) or not np.all(actions[:, :5] == prefix):
                    raise ValueError("response probe mapping")
                matrix = ((positions[1:21] - positions[21:41]) / .1).T
                np.testing.assert_allclose(matrix, candidate["response"], atol=1e-12, rtol=0)
                value = retention.ridge_residual(matrix, np.asarray(row["task"]["goal"]) - positions[0])
                np.testing.assert_allclose(value, candidate["ridge"], atol=1e-12, rtol=0)
                responses.append(matrix)
                values.append(value)
                j = candidate["lookahead_batch"]
                actions, scores = data[f"actions_{j}"], data[f"scores_{j}"]
                expected_rng = np.random.default_rng(2029184000 + root * 100 + index)
                expected = expected_rng.uniform(-1., 1., (24, 10, 2)).astype(np.float32)
                if actions.shape != (24, 15, 2) or not np.all(actions[:, :5] == prefix):
                    raise ValueError("lookahead mapping")
                np.testing.assert_array_equal(actions[:, 5:], expected)
                lookahead_score = min(candidate["zero_tail_score"], float(scores.min()))
                np.testing.assert_allclose(lookahead_score, candidate["lookahead_score"], atol=1e-12, rtol=0)
                lookahead.append(lookahead_score)
                run_bests = []
                for restart, run in enumerate(candidate["runs"]):
                    actions = np.concatenate([data[f"actions_{j}"] for j in range(run["first_batch"], run["last_batch"])])
                    scores = np.concatenate([data[f"scores_{j}"] for j in range(run["first_batch"], run["last_batch"])])
                    if run["restart"] != restart or len(actions) != 120 or not np.all(actions[:, :5] == prefix):
                        raise ValueError("suffix mapping")
                    best = min(candidate["zero_tail_score"], float(scores.min()))
                    np.testing.assert_allclose(best, run["best"], atol=1e-12, rtol=0)
                    run_bests.append(best)
                if len(run_bests) != 3:
                    raise ValueError("missing restart")
                bests.append(min(run_bests))
        chosen = retention.select([x["score"] for x in layer["candidates"]],
            [x["feature"] for x in layer["candidates"]], responses, values, lookahead, 2029183000 + root)
        if (chosen != row["selected"] or response_tokens != 10 * 41 * 3
                or lookahead_tokens != 10 * 24 * 3 or suffix_tokens != 10 * 3 * 120 * 3):
            raise ValueError("selector or diagnostic budget mismatch")
        baseline = min(bests)
        summary.append({"root": root, "status": "completed", "all_bank_best": baseline,
            "rules": {name: {"indices": indices, "best": min(bests[i] for i in indices),
                             "gap": min(bests[i] for i in indices) - baseline,
                             "success": min(bests[i] for i in indices) < .05}
                      for name, indices in chosen.items()},
            "generation_tokens": generation_tokens, "response_tokens": response_tokens,
            "lookahead_tokens": lookahead_tokens, "suffix_tokens": suffix_tokens})
    seal.verify(manifest["dependencies"])
    old.writer(out, {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
        "rows": summary, "scope": "same-environment alternate geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
