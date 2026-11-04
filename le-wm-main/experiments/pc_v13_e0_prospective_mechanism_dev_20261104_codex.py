"""Seal v13 plans, then evaluate the predeclared final-chain interventions."""
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_e0_v13_global_proposals_dev_20261104_codex as previous
from experiments import pc_e0_v14_carry_nominal_dev_20261104_codex as previous14
from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as old_panel
from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as intervention
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v13 as compiler
from experiments.spectral_reachability import pc_recursive_response_v12 as compiler_child
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = previous.ROOT
OUT = ROOT / "results/pc_v13_e0_prospective_mechanism_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V13_E0_PROSPECTIVE_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md"
exact = previous.prior.prior.previous.exact
writer = previous.prior.prior.previous.write


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.34, .06, .36), (-1.1, 1.4), (-.31, .27))]


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    panels = (previous.tasks(), previous14.tasks(), previous.remainder.tasks(),
              previous.small.tasks(), previous.prior.tasks(), previous.prior.previous_waypoint.tasks(),
              previous.prior.prior.tasks(), previous.prior.prior.previous.tasks(), old_panel.tasks())
    if any(task in panel for task in tasks() for panel in panels) or len({str(x) for x in tasks()}) != len(tasks()):
        raise ValueError("task overlap")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(previous.__file__), Path(previous14.__file__),
                         Path(compiler.__file__), Path(compiler_child.__file__),
                         Path(intervention.__file__), Path(intervention.local.__file__),
                         Path(intervention.core.__file__), Path(exact.__file__), Path(exact.wall.__file__),
                         Path(exact.wall.physics.__file__), Path(previous.independent.__code__.co_filename)])
    OUT.mkdir()
    writer(OUT / "manifest.json", {"tasks": tasks(), "seed": "2029131000+root", "cap": 1920,
                                    "dependencies": deps, "scope": "development; history independence unknown"})
    rows = []
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in previous.prior.core.NOMINALS)
    for root, task in enumerate(tasks()):
        trace, meter = [], Meter(1920)
        goal = np.asarray(task["goal"])

        def predict(actions):
            raw = np.asarray(actions, np.float32)

            def evaluate(batch):
                positions = exact.endpoints(task, batch)
                return positions, np.linalg.norm(positions - goal, axis=1)

            position, score = meter.predict(raw, evaluate)
            trace.append((raw.copy(), position.copy(), score.copy()))
            return position, score

        try:
            config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                                     model_packet_length=5, nominals=nominal, max_modes=1,
                                     token_cap=1920, seed=2029131000+root)
            result = compiler.compile_program(predict, goal, config)
            if result["resource"]["prediction_tokens"] != meter.token_samples:
                raise ValueError("meter mismatch")
            error = None
        except Exception:
            result, error = None, traceback.format_exc()
        path = OUT / f"plan_root{root:02d}.npz"
        arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                  for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        rows.append({"root": root, "task": task, "status": "failed" if error else "completed",
                     "error": error, "result": result, "counts": meter.as_dict(), "batches": len(trace),
                     "trace": path.name, "trace_sha256": seal.digest(path)})
        print(f"plan root={root} status={rows[-1]['status']}", flush=True)
    seal.verify(deps)
    writer(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit_plans():
    if (OUT / "plans_audit.json").exists():
        raise FileExistsError("plans audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    data = json.loads((OUT / "plans.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or data["manifest_sha256"] != seal.digest(OUT / "manifest.json"):
        raise ValueError("plan manifest mismatch")
    if len(data["rows"]) != len(tasks()):
        raise ValueError("missing root")
    for root, row in enumerate(data["rows"]):
        if row["root"] != root or row["task"] != tasks()[root]:
            raise ValueError("root mismatch")
        path = OUT / row["trace"]
        if seal.digest(path) != row["trace_sha256"]:
            raise ValueError("trace changed")
        tokens, full_scores, selected_found = 0, [], False
        selected = None if row["result"] is None else np.asarray(row["result"]["selected"], np.float32)
        with np.load(path, allow_pickle=False) as saved:
            for i in range(row["batches"]):
                actions, positions, scores = (saved[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or np.max(np.abs(actions)) > 1 or actions.shape[1] % 5:
                    raise ValueError("invalid actions")
                independent_path, _ = previous.independent(row["task"]["start"], actions)
                np.testing.assert_allclose(independent_path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                tokens += len(actions) * actions.shape[1] // 5
                if actions.shape[1] == 15:
                    full_scores.extend(scores.tolist())
                    if selected is not None:
                        score = row["result"]["selected_score"]
                        selected_found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                                       & (np.abs(scores-score) < 1e-12)))
        if tokens != row["counts"]["prediction_token_samples"] or tokens > 1920:
            raise ValueError("token mismatch")
        if selected is not None and (not selected_found or row["result"]["selected_score"] > min(full_scores)+1e-12):
            raise ValueError("selected full-chain incumbent mismatch")
    seal.verify(manifest["dependencies"])
    writer(OUT / "plans_audit.json", {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
                                      "roots": len(data["rows"]),
                                      "scope": "same-environment alternate geometry; not independent replication"})


def adapt_row(row):
    """Use the actual first-layer bank plus the zero global-bank prefix."""
    result = row["result"]
    zero = np.zeros((5, 2), np.float32)
    zero_score = float(np.linalg.norm(exact.endpoints(row["task"], zero[None])[0] - row["task"]["goal"]))
    layer = result["child"]["layers"][0]
    bank = [{"name": "zero", "prefix": zero.tolist()}]
    bank.extend({"name": f"response_{i}", "prefix": c["chain"]} for i, c in enumerate(layer["candidates"]))
    scores = [zero_score] + [c["score"] for c in layer["candidates"]]
    return {"root": row["root"], "task": row["task"], "result": {
        "selected": {"chain": result["selected"]}, "first_bank": bank, "prefix_scores": scores}}


def calculate(row):
    # The selected index is metadata; inserting random bridge shifts it.
    old_local = intervention.local

    class LocalAdapter:
        @staticmethod
        def build(value):
            output = old_local.build(value)
            output["controls"]["selected"]["candidate_index"] = -1
            return output

    try:
        intervention.local = LocalAdapter
        return intervention.calculate(adapt_row(row))
    finally:
        intervention.local = old_local


def intervene():
    audit = json.loads((OUT / "plans_audit.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(OUT / "plans.json"):
        raise ValueError("unsealed plans")
    if (OUT / "interventions.json").exists():
        raise FileExistsError("interventions already exist")
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": "plan_failed", "error": row["error"]})
            continue
        try:
            output, arrays = calculate(row)
            path = OUT / f"intervention_root{row['root']:02d}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            rows.append({"root": row["root"], "status": "completed", "data": output,
                         "arrays": path.name, "arrays_sha256": seal.digest(path)})
        except Exception:
            rows.append({"root": row["root"], "status": "failed", "error": traceback.format_exc()})
        print(f"intervention root={row['root']} status={rows[-1]['status']}", flush=True)
    seal.verify(manifest["dependencies"])
    writer(OUT / "interventions.json", {"plans_sha256": seal.digest(OUT / "plans.json"),
                                         "plans_audit_sha256": seal.digest(OUT / "plans_audit.json"),
                                         "rows": rows})


def audit_interventions():
    if (OUT / "interventions_audit.json").exists():
        raise FileExistsError("interventions audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    data = json.loads((OUT / "interventions.json").read_text())
    seal.verify(manifest["dependencies"])
    if data["plans_sha256"] != seal.digest(OUT / "plans.json") or data["plans_audit_sha256"] != seal.digest(OUT / "plans_audit.json"):
        raise ValueError("binding mismatch")
    if len(data["rows"]) != len(tasks()):
        raise ValueError("missing intervention records")
    for row, recorded in zip(plans["rows"], data["rows"], strict=True):
        if row["root"] != recorded["root"]:
            raise ValueError("root mismatch")
        if recorded["status"] != "completed":
            continue
        path = OUT / recorded["arrays"]
        if seal.digest(path) != recorded["arrays_sha256"]:
            raise ValueError("intervention arrays changed")
        rebuilt, arrays = calculate(row)
        if rebuilt != recorded["data"]:
            raise ValueError("intervention reconstruction mismatch")
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        print(f"audited root={row['root']}", flush=True)
    seal.verify(manifest["dependencies"])
    writer(OUT / "interventions_audit.json", {"status": "passed",
        "interventions_sha256": seal.digest(OUT / "interventions.json"),
        "completed": sum(x["status"] == "completed" for x in data["rows"]), "attempted": len(data["rows"]),
        "scope": "same-code deterministic reconstruction, not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit_plans", "intervene", "audit_interventions"):
        raise SystemExit("choose plan, audit_plans, intervene or audit_interventions")
    {"plan": plan, "audit_plans": audit_plans, "intervene": intervene,
     "audit_interventions": audit_interventions}[sys.argv[1]]()
