"""Prospective bounded exact-Wall portfolio development comparison."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_v13_new_feasible_wall_dev_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v15 as v15
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v15_portfolio_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V15_PORTFOLIO_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("response", "global_cem", "portfolio")
CAP = 2409


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, .21]}
            for y, theta in itertools.product((-.21, .21), (-.9, .9))]


def source_panels():
    p = prior.prior
    return (prior.tasks(), prior.budget.tasks(), prior.expression.tasks(), p.tasks(),
            p.previous.tasks(), p.previous14.tasks(), p.previous.remainder.tasks(),
            p.previous.small.tasks(), p.previous.prior.tasks(),
            p.previous.prior.previous_waypoint.tasks(), p.previous.prior.prior.tasks(),
            p.previous.prior.prior.previous.tasks(), p.old_panel.tasks())


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in panel for task in tasks() for panel in source_panels()):
        raise ValueError("task overlaps cited panel")
    p = prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(p.__file__),
        Path(v15.__file__), Path(p.compiler.__file__), Path(p.compiler_child.__file__),
        Path(prior.cem.__file__), Path(p.exact.__file__), Path(p.exact.wall.__file__),
        Path(p.exact.wall.physics.__file__), Path(p.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    p.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
        "dependencies": deps, "scope": "development; broader provenance unknown"})
    rows = []
    nominals = tuple(tuple(tuple(float(x) for x in row) for row in nominal)
                     for nominal in prior.cem.NOMINALS)
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            trace, meter = [], Meter(CAP)

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                if raw.ndim != 3 or raw.shape[1] % 5 or not np.isfinite(raw).all() or np.abs(raw).max() > 1:
                    raise ValueError("invalid raw action batch")
                def evaluate(batch):
                    position = p.exact.endpoints(task, batch)
                    return position, np.linalg.norm(position - goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            config = p.compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominals, max_modes=1,
                token_cap=CAP, seed=2029171000 + root)
            started = time.monotonic()
            try:
                if arm == "response":
                    details = p.compiler.compile_program(predict, goal, config)
                elif arm == "portfolio":
                    details = v15.compile_program(predict, goal, config)
                else:
                    bank = p.compiler.proposal_bank(config)
                    _, scores = predict(bank)
                    index = int(np.argmin(scores))
                    count = (CAP - meter.token_samples - 6) // 3
                    chain, _, generations = prior.cem.cem(bank[index], [], count,
                        np.random.default_rng(config.seed + 6200), predict, float(scores[index]), 15)
                    zero = np.asarray(chain, np.float32).copy()
                    zero[:5] = 0
                    pair = np.stack((chain, zero)).astype(np.float32)
                    _, final_scores = predict(pair)
                    choice = int(np.argmin(final_scores))
                    details = {"selected": pair[choice].tolist(), "selected_score": float(final_scores[choice]),
                               "choice": choice, "bank_choice": index, "generations": generations}
                selected = np.asarray(details["selected"], np.float32)
                score = float(details["selected_score"])
                error = None
            except Exception:
                selected, score, details, error = None, None, None, traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{name}_{i}": value
                    for i, triple in enumerate(trace)
                    for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
            rows.append({"root": root, "task": task, "arm": arm,
                "status": "failed" if error else "completed", "error": error,
                "selected": None if selected is None else selected.tolist(), "score": score,
                "details": details, "counts": meter.as_dict(), "wall_seconds": time.monotonic() - started,
                "batches": len(trace), "trace": path.name, "trace_sha256": seal.digest(path)})
            print(f"root={root} arm={arm} score={score} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    p.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS)
            or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or len(plans["rows"]) != len(tasks()) * len(ARMS)):
        raise ValueError("manifest mismatch")
    result = []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            matches = [row for row in plans["rows"] if row["root"] == root and row["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("missing or duplicate arm")
            row = matches[0]
            if row["status"] != "completed":
                result.append({"root": root, "arm": arm, "status": "failed", "error": row["error"]})
                continue
            if row["task"] != task or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("trace binding mismatch")
            count, best, found = 0, float("inf"), False
            selected = np.asarray(row["selected"], np.float32)
            with np.load(OUT / row["trace"], allow_pickle=False) as data:
                banks.append(data["actions_0"].copy())
                for i in range(row["batches"]):
                    actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                    if (actions.dtype != np.float32 or actions.shape[1] % 5
                            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1):
                        raise ValueError("action contract")
                    replay, _ = prior.prior.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(replay[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1), atol=1e-12, rtol=0)
                    count += len(actions) * actions.shape[1] // 5
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores - row["score"]) < 1e-12)))
            if (count != row["counts"]["prediction_token_samples"] or count > CAP
                    or not found or row["score"] > best + 1e-12):
                raise ValueError("token or incumbent mismatch")
            result.append({"root": root, "arm": arm, "status": "completed", "score": row["score"],
                           "success": row["score"] < .05, "tokens": count})
        if len(banks) > 1:
            for bank in banks[1:]:
                np.testing.assert_array_equal(bank, banks[0])
    seal.verify(manifest["dependencies"])
    prior.prior.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
        "rows": result, "scope": "alternate same-environment geometry, not independent replication"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
