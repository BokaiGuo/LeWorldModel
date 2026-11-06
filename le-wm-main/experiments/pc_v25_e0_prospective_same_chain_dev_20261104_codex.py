"""Seal v25 plans, then run the unchanged prospective mechanism intervention."""
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as mechanism
from experiments import pc_v25_fixed_bank_suffix_e0_dev_20261104_codex as fixed
from experiments import pc_v25_successor_probe_cost_e0_dev_20261104_codex as cost
from experiments.spectral_reachability import pc_global_reachability_v25 as compiler

ROOT, OLD, seal = fixed.ROOT, fixed.OLD, fixed.seal
OUT = ROOT / "results/pc_v25_e0_prospective_same_chain_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_E0_PROSPECTIVE_SAME_CHAIN_DEV_PROTOCOL_20261104_CODEX.md"


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.17, .17), (-1.23, 1.23), (-.14, .14))]


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in old for task in tasks() for old in (*fixed.prior_panels(), fixed.tasks(), cost.tasks())):
        raise ValueError("task overlaps enumerated stopped panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(fixed.__file__), Path(cost.__file__),
        Path(compiler.__file__), Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(journal.__file__),
        Path(journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(mechanism.__file__), Path(mechanism.intervention.__file__),
        Path(mechanism.intervention.local.__file__), Path(mechanism.intervention.core.__file__),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "cap": 1920,
        "dependencies": deps, "scope": "development, broader provenance unknown"})
    print("Sealing v25 final plans before same-chain interventions.", flush=True)
    rows = []
    for root, task in enumerate(tasks()):
        recorder = journal.PredictionJournal(OUT / f"plan_root{root:02d}_calls", fixed.predictor(task),
                                              action_dim=2, packet_length=5, token_cap=1920)
        config = compiler.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
            nominals=nominals, token_cap=1920, seed=2030701000 + root, max_modes=2)
        try:
            result = compiler.compile_program(recorder, np.asarray(task["goal"]), config,
                                              score_scale=1., score_target=0.)
            if result["resource"]["prediction_tokens"] != recorder.tokens:
                raise ValueError("plan token mismatch")
            row = {"root": root, "task": task, "status": "completed", "result": result}
        except Exception:
            row = {"root": root, "task": task, "status": "failed", "error": traceback.format_exc()}
        row.update(journal_calls=recorder.calls, journal_tokens=recorder.tokens)
        journal.write_json(OUT / f"plan_root{root:02d}.json", row)
        rows.append(row)
        print(f"plan root={root} status={row['status']}", flush=True)
    seal.verify(deps)
    journal.write_json(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit_plans():
    output = OUT / "plans_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "plans.json"))
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or plans["manifest_sha256"] != seal.digest(OUT / "manifest.json")
            or len(plans["rows"]) != len(tasks())):
        raise ValueError("plan binding mismatch")
    for root, row in enumerate(plans["rows"]):
        if row["root"] != root or row["task"] != tasks()[root]:
            raise ValueError("task identity mismatch")
        if json.loads((OUT / f"plan_root{root:02d}.json").read_text()) != row:
            raise ValueError("per-root plan mismatch")
        directory = OUT / f"plan_root{root:02d}_calls"
        info = journal_audit.audit_journal(directory)
        if info["tokens"] != row["journal_tokens"] or len(info["calls"]) != row["journal_calls"]:
            raise ValueError("planning journal mismatch")
        if row["status"] != "completed":
            continue
        selected = np.asarray(row["result"]["selected"], np.float32)
        best, found = float("inf"), False
        for actions, features, scores in fixed.read_calls(directory):
            path, _ = OLD.previous.independent(row["task"]["start"], actions)
            np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(features-row["task"]["goal"], axis=1),
                                       atol=1e-12, rtol=0)
            if actions.shape[1] == 15:
                best = min(best, float(scores.min()))
                found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                     & (np.abs(scores-row["result"]["selected_score"]) < 1e-12)))
        if not found or row["result"]["selected_score"] > best + 1e-12:
            raise ValueError("selected plan is not recorded incumbent")
    seal.verify(manifest["dependencies"])
    journal.write_json(output, {"status": "passed", "plans_sha256": seal.digest(OUT / "plans.json"),
                               "scope": "alternate same-environment geometry; not independent replication"})


def calculate_with_journal(row, recorder):
    """Persist actual endpoint calls, including partial calls if analysis fails."""
    original = mechanism.exact.endpoints

    def wrapped(task, actions):
        if task != row["task"]:
            raise ValueError("intervention task mismatch")
        return recorder(actions)[0]

    mechanism.exact.endpoints = wrapped
    try:
        return mechanism.calculate(row)
    finally:
        mechanism.exact.endpoints = original


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
        root = row["root"]
        if row["status"] != "completed":
            rows.append({"root": root, "status": "plan_failed"})
            continue
        original = mechanism.exact.endpoints
        def predict(actions):
            positions = original(row["task"], actions)
            return positions, np.linalg.norm(positions - row["task"]["goal"], axis=1)
        recorder = journal.PredictionJournal(OUT / f"intervention_root{root:02d}_calls", predict,
                                            action_dim=2, packet_length=5, token_cap=1000000)
        try:
            data, arrays = calculate_with_journal(row, recorder)
            path = OUT / f"intervention_root{root:02d}.npz"
            journal.write_arrays(path, **arrays)
            entry = {"root": root, "status": "completed", "data": data, "arrays": path.name,
                     "arrays_sha256": seal.digest(path)}
        except Exception:
            entry = {"root": root, "status": "failed", "error": traceback.format_exc()}
        entry.update(journal_calls=recorder.calls, journal_tokens=recorder.tokens)
        journal.write_json(OUT / f"intervention_root{root:02d}.json", entry)
        rows.append(entry)
        print(f"intervention root={root} status={entry['status']}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(output, {"plans_sha256": seal.digest(OUT / "plans.json"),
        "plans_audit_sha256": seal.digest(OUT / "plans_audit.json"), "rows": rows})


def audit_interventions():
    output = OUT / "interventions_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest, plans, record = (json.loads((OUT / name).read_text())
        for name in ("manifest.json", "plans.json", "interventions.json"))
    seal.verify(manifest["dependencies"])
    if (record["plans_sha256"] != seal.digest(OUT / "plans.json")
            or record["plans_audit_sha256"] != seal.digest(OUT / "plans_audit.json")
            or len(record["rows"]) != len(tasks())):
        raise ValueError("intervention binding mismatch")
    for row, entry in zip(plans["rows"], record["rows"], strict=True):
        if row["root"] != entry["root"]:
            raise ValueError("root mismatch")
        if entry["status"] == "plan_failed":
            continue
        root = row["root"]
        if json.loads((OUT / f"intervention_root{root:02d}.json").read_text()) != entry:
            raise ValueError("per-root intervention mismatch")
        info = journal_audit.audit_journal(OUT / f"intervention_root{root:02d}_calls")
        if info["tokens"] != entry["journal_tokens"] or len(info["calls"]) != entry["journal_calls"]:
            raise ValueError("intervention journal mismatch")
        if entry["status"] != "completed":
            continue
        if seal.digest(OUT / entry["arrays"]) != entry["arrays_sha256"]:
            raise ValueError("intervention arrays changed")
        rebuilt, arrays = mechanism.calculate(row)
        if rebuilt != entry["data"]:
            raise ValueError("intervention reconstruction mismatch")
        with np.load(OUT / entry["arrays"], allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        print(f"audited root={root}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(output, {"status": "passed", "interventions_sha256": seal.digest(OUT / "interventions.json"),
        "completed": sum(x["status"] == "completed" for x in record["rows"]),
        "attempted": len(record["rows"]), "scope": "same-code reconstruction; not independent replication"})


if __name__ == "__main__":
    stages = {"plan": plan, "audit_plans": audit_plans, "intervene": intervene,
              "audit_interventions": audit_interventions}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit(f"choose one of {tuple(stages)}")
    stages[sys.argv[1]]()
