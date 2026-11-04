"""Seal v19 E0 plans before the existing same-chain intervention."""
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as prior
from experiments import pc_v15_portfolio_e0_dev_20261104_codex as v15
from experiments import pc_v16_multinominal_e0_dev_20261104_codex as v16
from experiments import pc_v18_reachability_e0_dev_20261104_codex as v18_panel
from experiments import pc_v19_retention_fix_e0_dev_20261104_codex as v19_panel
from experiments import pc_response_retention_fixedbank_e0_dev_20261104_codex as fixed
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_global_reachability_v19 as compiler
from experiments.spectral_reachability import pc_recursive_reachability_v19 as core19
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_v19_e0_same_chain_mechanism_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V19_E0_SAME_CHAIN_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.21, .21), (-1.16, 1.16), (-.26, .26))]


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    panels = (v18_panel.tasks(), v19_panel.tasks(), v19_panel.previous.tasks(),
              v19_panel.previous.prior.tasks(), prior.tasks(), v15.tasks(), v16.tasks(),
              fixed.tasks(), *v15.source_panels())
    if len({str(task) for task in tasks()}) != len(tasks()) or any(task in panel for task in tasks() for panel in panels):
        raise ValueError("overlap with cited panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(compiler.__file__), Path(core19.__file__), Path(prior.__file__),
        Path(prior.compiler.__file__), Path(prior.compiler_child.__file__), Path(prior.intervention.__file__),
        Path(prior.intervention.local.__file__), Path(prior.intervention.core.__file__),
        Path(prior.exact.__file__), Path(prior.exact.wall.__file__), Path(prior.exact.wall.physics.__file__),
        Path(prior.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    prior.writer(OUT / "manifest.json", {"tasks": tasks(), "cap": 1920, "score_scale": 1.,
        "score_target": 0., "dependencies": deps, "scope": "development; provenance unknown"})
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in prior.previous.prior.core.NOMINALS)
    rows = []
    for root, task in enumerate(tasks()):
        trace, meter, goal = [], Meter(1920), np.asarray(task["goal"])

        def predict(actions):
            raw = np.asarray(actions, np.float32)
            def evaluate(batch):
                positions = prior.exact.endpoints(task, batch)
                return positions, np.linalg.norm(positions - goal, axis=1)
            position, score = meter.predict(raw, evaluate)
            trace.append((raw.copy(), position.copy(), score.copy()))
            return position, score

        try:
            config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominals, max_modes=1,
                token_cap=1920, seed=2029351000 + root)
            result = compiler.compile_program(predict, goal, config, score_scale=1., score_target=0.)
            if result["resource"]["prediction_tokens"] != meter.token_samples:
                raise ValueError("compiler meter mismatch")
            error = None
        except Exception:
            result, error = None, traceback.format_exc()
        trace_path = OUT / f"plan_root{root:02d}.npz"
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": value for i, triple in enumerate(trace)
                for name, value in zip(("actions", "positions", "scores"), triple, strict=True)})
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed",
            "error": error, "result": result, "counts": meter.as_dict(), "batches": len(trace),
            "trace": trace_path.name, "trace_sha256": seal.digest(trace_path)})
        print(f"plan root={root} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    prior.writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit_plans():
    output = OUT / "plans_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "plans.json"))
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or len(plans["rows"]) != len(tasks())):
        raise ValueError("manifest mismatch")
    for root, row in enumerate(plans["rows"]):
        if row["root"] != root or row["task"] != tasks()[root] or seal.digest(OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("task or trace mismatch")
        tokens, best, found = 0, float("inf"), False
        selected = None if row["result"] is None else np.asarray(row["result"]["selected"], np.float32)
        with np.load(OUT / row["trace"], allow_pickle=False) as saved:
            for i in range(row["batches"]):
                actions, positions, scores = (saved[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or np.max(np.abs(actions)) > 1 or actions.shape[1] % 5:
                    raise ValueError("action contract")
                path, _ = prior.previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                tokens += len(actions) * actions.shape[1] // 5
                if actions.shape[1] == 15:
                    best = min(best, float(scores.min()))
                    if selected is not None:
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores - row["result"]["selected_score"]) < 1e-12)))
        if tokens != row["counts"]["prediction_token_samples"] or tokens > 1920:
            raise ValueError("token mismatch")
        if selected is not None and (not found or row["result"]["selected_score"] > best + 1e-12):
            raise ValueError("incumbent mismatch")
    seal.verify(manifest["dependencies"])
    prior.writer(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
                          "scope": "same-environment geometry; not independent replication"})


def intervene():
    output = OUT / "interventions.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans, audit = (json.loads((OUT / name).read_text())
                              for name in ("manifest.json", "plans.json", "plans_audit.json"))
    seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(OUT / "plans.json"):
        raise ValueError("plans not sealed and audited")
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": "plan_failed", "error": row["error"]})
            continue
        try:
            data, arrays = prior.calculate(row)
            path = OUT / f"intervention_root{row['root']:02d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            rows.append({"root": row["root"], "status": "completed", "data": data,
                         "arrays": path.name, "arrays_sha256": seal.digest(path)})
        except Exception:
            rows.append({"root": row["root"], "status": "failed", "error": traceback.format_exc()})
        print(f"intervention root={row['root']} status={rows[-1]['status']}", flush=True)
    seal.verify(manifest["dependencies"])
    prior.writer(output, {"plans_sha256": seal.digest(OUT / "plans.json"),
                          "plans_audit_sha256": seal.digest(OUT / "plans_audit.json"), "rows": rows})


def audit_interventions():
    output = OUT / "interventions_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans, interventions = (json.loads((OUT / name).read_text())
                                     for name in ("manifest.json", "plans.json", "interventions.json"))
    seal.verify(manifest["dependencies"])
    if (interventions["plans_sha256"] != seal.digest(OUT / "plans.json")
            or interventions["plans_audit_sha256"] != seal.digest(OUT / "plans_audit.json")
            or len(interventions["rows"]) != len(tasks())):
        raise ValueError("intervention binding mismatch")
    for row, recorded in zip(plans["rows"], interventions["rows"], strict=True):
        if row["root"] != recorded["root"]:
            raise ValueError("root mismatch")
        if recorded["status"] != "completed":
            continue
        if seal.digest(OUT / recorded["arrays"]) != recorded["arrays_sha256"]:
            raise ValueError("intervention trace changed")
        rebuilt, arrays = prior.calculate(row)
        if rebuilt != recorded["data"]:
            raise ValueError("intervention reconstruction mismatch")
        with np.load(OUT / recorded["arrays"], allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        print(f"audited root={row['root']}", flush=True)
    seal.verify(manifest["dependencies"])
    prior.writer(output, {"status": "passed", "interventions_sha256": seal.digest(OUT / "interventions.json"),
        "attempted": len(interventions["rows"]),
        "completed": sum(row["status"] == "completed" for row in interventions["rows"]),
        "scope": "same-code deterministic reconstruction; not independent replication"})


if __name__ == "__main__":
    stages = {"plan": plan, "audit_plans": audit_plans, "intervene": intervene,
              "audit_interventions": audit_interventions}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit(f"choose one of {tuple(stages)}")
    stages[sys.argv[1]]()
