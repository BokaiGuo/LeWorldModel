"""Seal v13 vs matched CEM plans before any privileged feasibility witness."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as prior
from experiments import pc_v13_fixedbank_suffix_budget_dev_20261104_codex as budget
from experiments import pc_v13_fixedbank_suffix_expression_dev_20261104_codex as expression
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v13_new_feasible_wall_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V13_NEW_FEASIBLE_WALL_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v13", "global_cem")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, .19]}
            for y, theta in itertools.product((-.26, .26), (-1.2, 1.2))]


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = (budget.tasks(), expression.tasks(), prior.tasks(), prior.previous.tasks(), prior.previous14.tasks(),
           prior.previous.remainder.tasks(), prior.previous.small.tasks(), prior.previous.prior.tasks(),
           prior.previous.prior.previous_waypoint.tasks(), prior.previous.prior.prior.tasks(),
           prior.previous.prior.prior.previous.tasks(), prior.old_panel.tasks())
    if any(task in panel for task in tasks() for panel in old):
        raise ValueError("overlap with cited panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(cem.__file__),
        Path(prior.compiler.__file__), Path(prior.compiler_child.__file__), Path(prior.exact.__file__),
        Path(prior.exact.wall.__file__), Path(prior.exact.wall.physics.__file__),
        Path(prior.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    prior.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": 1920,
        "dependencies": deps, "scope": "new development panel; provenance independence unknown"})
    rows = []
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item) for item in cem.NOMINALS)
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            trace, meter = [], Meter(1920)

            def predict(actions):
                raw = np.asarray(actions, np.float32)

                def evaluate(batch):
                    pos = prior.exact.endpoints(task, batch)
                    return pos, np.linalg.norm(pos - goal, axis=1)

                pos, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), pos.copy(), scores.copy()))
                return pos, scores

            try:
                config = prior.compiler.Config(action_dim=2, segment_length=5, segments=3,
                    model_packet_length=5, nominals=nominal, max_modes=1,
                    token_cap=1920, seed=2029161000 + root)
                if arm == "v13":
                    details = prior.compiler.compile_program(predict, goal, config)
                    selected, score = details["selected"], details["selected_score"]
                else:
                    bank = prior.compiler.proposal_bank(config)
                    _, scores = predict(bank)
                    choice = int(np.argmin(scores))
                    samples = (1920 - meter.token_samples - 6) // 3
                    chain, _, generations = cem.cem(bank[choice], [], samples,
                        np.random.default_rng(2029162000 + root), predict, float(scores[choice]), 15)
                    comparison = np.asarray(chain).copy()
                    comparison[:5] = 0
                    pair = np.stack((chain, comparison))
                    _, pair_scores = predict(pair)
                    selected = pair[int(np.argmin(pair_scores))]
                    score = float(np.min(pair_scores))
                    details = {"bank_choice": choice, "samples": samples,
                               "generations": generations, "pair_scores": pair_scores.tolist()}
                error = None
            except Exception:
                selected, score, details, error = None, None, None, traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            arrays = {f"{name}_{i}": v for i, triple in enumerate(trace)
                      for name, v in zip(("actions", "positions", "scores"), triple, strict=True)}
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                "error": error, "selected": None if selected is None else np.asarray(selected, np.float32).tolist(),
                "score": score, "details": details, "counts": meter.as_dict(), "batches": len(trace),
                "trace": path.name, "trace_sha256": seal.digest(path)})
            print(f"root={root} arm={arm} status={rows[-1]['status']} score={score}", flush=True)
    seal.verify(deps)
    prior.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    if (OUT / "plans_audit.json").exists():
        raise FileExistsError("audit already exists")
    manifest, plans = (json.loads((OUT / p).read_text()) for p in ("manifest.json", "plans.json"))
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    if len(plans["rows"]) != 8:
        raise ValueError("missing plans")
    summary, total = [], 0
    for root in range(4):
        banks = []
        for arm in ARMS:
            row = next(r for r in plans["rows"] if r["root"] == root and r["arm"] == arm)
            if row["task"] != tasks()[root] or row["status"] != "completed" or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("invalid plan or trace")
            count, best, found = 0, float("inf"), False
            selected = np.asarray(row["selected"], np.float32)
            with np.load(OUT / row["trace"], allow_pickle=False) as data:
                banks.append(data["actions_0"].copy())
                for i in range(row["batches"]):
                    actions, pos, scores = (data[f"{key}_{i}"] for key in ("actions", "positions", "scores"))
                    if actions.dtype != np.float32 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1:
                        raise ValueError("action contract")
                    replay, _ = prior.previous.independent(row["task"]["start"], actions)
                    np.testing.assert_allclose(replay[:, -1, :2], pos, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(pos - row["task"]["goal"], axis=1), atol=1e-12, rtol=0)
                    count += len(actions) * actions.shape[1] // 5
                    total += len(actions)
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2)) & (abs(scores-row["score"]) < 1e-12)))
            if count != row["counts"]["prediction_token_samples"] or count > 1920 or not found or row["score"] > best + 1e-12:
                raise ValueError("budget or complete-chain incumbent mismatch")
            summary.append({"root": root, "arm": arm, "score": row["score"], "tokens": count})
        np.testing.assert_array_equal(banks[0], banks[1])
    seal.verify(manifest["dependencies"])
    prior.writer(OUT / "plans_audit.json", {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
        "trajectory_samples": total, "summary": summary, "scope": "alternate same-environment geometry"})
    print(json.dumps(summary, indent=2))


def witness():
    if (OUT / "witness.json").exists():
        raise FileExistsError("witness exists")
    pa = json.loads((OUT / "plans_audit.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if pa["status"] != "passed" or pa["plans_sha256"] != seal.digest(OUT / "plans.json"):
        raise ValueError("plans not sealed and audited")
    rows = []
    for root, task in enumerate(tasks()):
        attempts = []
        for item in prior.exact.wall.feasibility(task):
            record = {"height": item["height"], "steps_needed": item["steps_needed"],
                      "eligible": bool(item["within_horizon"] and item["steps_needed"] <= 15)}
            if record["eligible"]:
                raw = (np.asarray(item["actions"][:15], np.float64) / prior.exact.wall.HIGH).astype(np.float32)
                if raw.shape != (15, 2) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
                    raise ValueError("invalid witness actions")
                pos = prior.exact.endpoints(task, raw[None])
                path, blocked = prior.previous.independent(task["start"], raw[None])
                np.testing.assert_allclose(path[:, -1, :2], pos, atol=1e-12, rtol=0)
                distance = float(np.linalg.norm(pos[0] - task["goal"]))
                record.update({"actions": raw.tolist(), "path": path[0].tolist(),
                               "blocked": blocked[0].tolist(), "distance": distance,
                               "success": distance < .05})
            attempts.append(record)
        rows.append({"root": root, "task": task, "witnesses": attempts})
    seal.verify(manifest["dependencies"])
    prior.writer(OUT / "witness.json", {"plans_sha256": seal.digest(OUT / "plans.json"),
        "plans_audit_sha256": seal.digest(OUT / "plans_audit.json"), "rows": rows,
        "scope": "privileged diagnostic only; not method performance"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit", "witness"):
        raise SystemExit("choose plan, audit, or witness")
    {"plan": plan, "audit": audit, "witness": witness}[sys.argv[1]]()
