"""v17/v13 against shared-bank multi-start CEM on an E0 development panel."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_v17_e0_same_chain_mechanism_dev_20261104_codex as prior
from experiments import pc_v16_multinominal_e0_dev_20261104_codex as v16
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_global_multistart_cem as baseline
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v17_e0_multistart_baseline_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V17_E0_MULTISTART_BASELINE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v13", "v17", "multistart_cem")
CAP = 4095


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.29, .29), (-1.3, 1.3), (-.29, .29))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    panels = (prior.tasks(), prior.v15.tasks(), v16.tasks(), prior.fixed.tasks(), *prior.v15.source_panels())
    if any(task in panel for task in tasks() for panel in panels) or len({str(t) for t in tasks()}) != len(tasks()):
        raise ValueError("overlap with cited panel")
    old = prior.prior
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(baseline.__file__),
        Path(prior.compiler.__file__), Path(old.compiler.__file__), Path(old.compiler_child.__file__),
        Path(prior.v15.prior.cem.__file__), Path(old.exact.__file__), Path(old.exact.wall.__file__),
        Path(old.exact.wall.physics.__file__), Path(old.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
        "dependencies": deps, "scope": "development, broader provenance unknown"})
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in old.previous.prior.core.NOMINALS)
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            trace, meter = [], Meter(CAP)

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                if raw.ndim != 3 or raw.shape[1] % 5 or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
                    raise ValueError("invalid actions")
                def evaluate(batch):
                    pos = old.exact.endpoints(task, batch)
                    return pos, np.linalg.norm(pos - goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            config = old.compiler.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                nominals=nominals, token_cap=CAP, seed=2029211000 + root)
            started = time.monotonic()
            try:
                if arm == "v13":
                    result = old.compiler.compile_program(predict, goal, config)
                elif arm == "v17":
                    result = prior.compiler.compile_program(predict, goal, config, score_scale=1., score_target=0.)
                else:
                    result = baseline.plan(predict, config, seed=2029212000 + root * 10)
                tokens = result["prediction_tokens"] if arm == "multistart_cem" else result["resource"]["prediction_tokens"]
                if tokens != meter.token_samples:
                    raise ValueError("meter mismatch")
                error = None
            except Exception:
                result, error = None, traceback.format_exc()
            path = OUT / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                    for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
            rows.append({"root": root, "task": task, "arm": arm, "status": "failed" if error else "completed",
                "error": error, "result": result, "counts": meter.as_dict(), "wall_seconds": time.monotonic()-started,
                "batches": len(trace), "trace": path.name, "trace_sha256": seal.digest(path)})
            print(f"root={root} arm={arm} status={rows[-1]['status']} score={None if result is None else result['selected_score']}", flush=True)
    seal.verify(deps)
    old.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "plans.json"))
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS)
            or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json") or len(plans["rows"]) != 24):
        raise ValueError("manifest mismatch")
    old, summary = prior.prior, []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            matches = [row for row in plans["rows"] if row["root"] == root and row["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("missing or duplicate arm")
            row = matches[0]
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": "failed", "error": row["error"]})
                continue
            if row["task"] != task or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("task or trace mismatch")
            selected = np.asarray(row["result"]["selected"], np.float32)
            tokens, best, found = 0, float("inf"), False
            with np.load(OUT / row["trace"], allow_pickle=False) as data:
                banks.append(data["actions_0"].copy())
                for i in range(row["batches"]):
                    actions, positions, scores = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                    if (actions.dtype != np.float32 or actions.shape[1] % 5
                            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1):
                        raise ValueError("invalid action")
                    path, _ = old.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1), atol=1e-12, rtol=0)
                    tokens += len(actions) * actions.shape[1] // 5
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                            & (np.abs(scores-row["result"]["selected_score"]) < 1e-12)))
                if arm == "multistart_cem":
                    starts = row["result"]["starts"]
                    if len(starts) != 4 or sum(x["samples"] for x in starts) * 3 + len(banks[-1]) * 3 + 6 != tokens:
                        raise ValueError("multistart accounting")
            if tokens != row["counts"]["prediction_token_samples"] or tokens > CAP or not found or row["result"]["selected_score"] > best + 1e-12:
                raise ValueError("token or incumbent mismatch")
            summary.append({"root": root, "arm": arm, "status": "completed", "score": row["result"]["selected_score"],
                "success": row["result"]["selected_score"] < .05, "tokens": tokens,
                "svd_calls": row["result"].get("resource", {}).get("svd_calls", 0),
                "wall_seconds": row["wall_seconds"]})
        for bank in banks[1:]:
            np.testing.assert_array_equal(bank, banks[0])
    seal.verify(manifest["dependencies"])
    old.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"), "rows": summary,
                        "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
