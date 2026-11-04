"""Paired multi-nominal exact-Wall development experiment."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_v15_portfolio_e0_dev_20261104_codex as prior
from experiments import pc_response_retention_fixedbank_e0_dev_20261104_codex as fixed
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v16 as v16
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v16_multinominal_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V16_MULTINOMINAL_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v13", "v16", "global_cem")
CAP = 4095


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, -.23]}
            for y, theta in itertools.product((-.27, .27), (-1.1, 1.1))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in panel for task in tasks() for panel in (prior.tasks(), fixed.tasks(), *prior.source_panels())):
        raise ValueError("overlap with cited panels")
    old = prior.prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(v16.__file__), Path(prior.__file__), Path(fixed.__file__),
        Path(old.compiler.__file__), Path(old.compiler_child.__file__), Path(prior.prior.cem.__file__),
        Path(old.exact.__file__), Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
        Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
        "dependencies": deps, "scope": "development; broader provenance unknown"})
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in prior.prior.cem.NOMINALS)
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            trace, meter = [], Meter(CAP)

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                if raw.ndim != 3 or raw.shape[1] % 5 or not np.isfinite(raw).all() or np.abs(raw).max() > 1:
                    raise ValueError("invalid action")
                def evaluate(batch):
                    positions = old.exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions - goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            config = old.compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, max_modes=1,
                token_cap=CAP, seed=2029191000 + root)
            started = time.monotonic()
            try:
                if arm == "v13":
                    details = old.compiler.compile_program(predict, goal, config)
                elif arm == "v16":
                    details = v16.compile_program(predict, goal, config, nominal_seed=2029193000 + root)
                else:
                    bank = old.compiler.proposal_bank(config)
                    _, scores = predict(bank)
                    index = int(np.argmin(scores))
                    count = (CAP - meter.token_samples - 6) // 3
                    chain, _, generations = prior.prior.cem.cem(bank[index], [], count,
                        np.random.default_rng(2029192000 + root), predict, float(scores[index]), 15)
                    zero = np.asarray(chain, np.float32).copy()
                    zero[:5] = 0
                    pair = np.stack((chain, zero)).astype(np.float32)
                    _, scores = predict(pair)
                    choice = int(np.argmin(scores))
                    details = {"selected": pair[choice].tolist(), "selected_score": float(scores[choice]),
                               "choice": choice, "bank_choice": index, "generations": generations}
                selected, score = np.asarray(details["selected"], np.float32), float(details["selected_score"])
                error = None
            except Exception:
                selected, score, details, error = None, None, None, traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                    for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                "error": error, "selected": None if selected is None else selected.tolist(), "score": score,
                "details": details, "counts": meter.as_dict(), "wall_seconds": time.monotonic() - started,
                "batches": len(trace), "trace": path.name, "trace_sha256": seal.digest(path)})
            print(f"root={root} arm={arm} status={rows[-1]['status']} score={score}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "plans.json"))
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS)
            or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or len(plans["rows"]) != 12):
        raise ValueError("manifest mismatch")
    old, summary = prior.prior.prior, []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            matches = [x for x in plans["rows"] if x["root"] == root and x["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("missing or duplicated arm")
            row = matches[0]
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": "failed", "error": row["error"]})
                continue
            if row["task"] != task or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("trace binding mismatch")
            if arm == "v16":
                expected = v16.extra_nominals(5, 2, 2029193000 + root)
                np.testing.assert_array_equal(row["details"]["extra_nominals"], expected)
            count, best, found = 0, float("inf"), False
            selected = np.asarray(row["selected"], np.float32)
            with np.load(OUT / row["trace"], allow_pickle=False) as data:
                banks.append(data["actions_0"].copy())
                for i in range(row["batches"]):
                    actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                    if (actions.dtype != np.float32 or actions.shape[1] % 5
                            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1):
                        raise ValueError("action contract")
                    path, _ = old.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1), atol=1e-12, rtol=0)
                    count += len(actions) * actions.shape[1] // 5
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores - row["score"]) < 1e-12)))
            if count != row["counts"]["prediction_token_samples"] or count > CAP or not found or row["score"] > best + 1e-12:
                raise ValueError("budget or full-chain incumbent mismatch")
            summary.append({"root": root, "arm": arm, "status": "completed", "score": row["score"],
                "success": row["score"] < .05, "tokens": count,
                "svd_calls": row["details"].get("resource", {}).get("svd_calls", 0)})
        for bank in banks[1:]:
            np.testing.assert_array_equal(bank, banks[0])
    seal.verify(manifest["dependencies"])
    old.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
        "rows": summary, "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
