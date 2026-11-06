"""Sealed paired retention development and final-chain interventions."""
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_boundary_response_posthoc as boundary
from experiments import pc_final_chain_mechanism_measure_v1 as measure
from experiments import pc_final_chain_response_v1 as response
from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_v25_narrow_prospective_same_chain_dev_20261104_codex as old
from experiments.spectral_reachability import pc_global_reachability_lookahead_dev as compiler
from experiments.spectral_reachability import pc_response_temporal_search_v1 as cem

ROOT, seal = old.ROOT, old.seal
OUT = ROOT / "results/pc_lookahead_ridge_prospective_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_LOOKAHEAD_VS_RIDGE_PROSPECTIVE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("ridge", "probe_min")
exact = old.OLD.exact


def dependencies():
    return seal.capture([Path(__file__), PROTOCOL, Path(compiler.__file__),
        Path(compiler.core.__file__), Path(compiler.v13.__file__),
        Path(compiler.core.pc_shared_multistart_cem.__file__), Path(cem.__file__),
        Path(measure.__file__), Path(response.__file__), Path(boundary.__file__),
        Path(journal.__file__), Path(journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(exact.__file__), Path(exact.wall.__file__), Path(exact.wall.physics.__file__),
        Path(old.previous.fixed.__file__), Path(old.OLD.previous.independent.__code__.co_filename)])


def config(root, arm):
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in old.OLD.previous.prior.core.NOMINALS)
    return compiler.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
        nominals=nominal, token_cap=2880, seed=2032100000 + root, max_modes=2,
        beam_width=2, greedy_slots=1, successor_value=arm)


def plan():
    if OUT.exists():
        raise FileExistsError(OUT)
    deps = dependencies()
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"dependencies": deps, "tasks": old.tasks(),
        "arms": ARMS, "scope": "exposed tasks; paired prospective development, not confirmation"})
    rows = []
    for root, task in enumerate(old.tasks()):
        for arm in ARMS:
            recorder = journal.PredictionJournal(OUT / f"plan_{root:02d}_{arm}_calls",
                old.previous.fixed.predictor(task), action_dim=2, packet_length=5, token_cap=2880)
            row = {"root": root, "task": task, "arm": arm}
            try:
                result = compiler.compile_program(recorder, np.asarray(task["goal"]), config(root, arm),
                                                  score_scale=1., score_target=0.)
                if result["resource"]["prediction_tokens"] != recorder.tokens:
                    raise ValueError("plan budget mismatch")
                row.update(status="completed", result=result)
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row.update(tokens=recorder.tokens, calls=recorder.calls)
            journal.write_json(OUT / f"plan_{root:02d}_{arm}.json", row)
            rows.append(row)
            print(f"plan root={root} arm={arm} status={row['status']}", flush=True)
    seal.verify(deps)
    journal.write_json(OUT / "plans.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit_plans():
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    seal.verify(manifest["dependencies"])
    if plans["manifest_sha256"] != seal.digest(OUT / "manifest.json") or len(plans["rows"]) != 32:
        raise ValueError("incomplete or unbound plans")
    for row in plans["rows"]:
        root, arm, task = row["root"], row["arm"], row["task"]
        if task != old.tasks()[root] or arm not in ARMS:
            raise ValueError("task or arm mismatch")
        if json.loads((OUT / f"plan_{root:02d}_{arm}.json").read_text()) != row:
            raise ValueError("plan changed")
        directory = OUT / f"plan_{root:02d}_{arm}_calls"
        info = journal_audit.audit_journal(directory)
        if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
            raise ValueError("journal mismatch")
        found, best = False, float("inf")
        for actions, features, scores in old.previous.fixed.read_calls(directory):
            path, _ = old.OLD.previous.independent(task["start"], actions)
            np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1), atol=1e-12, rtol=0)
            if actions.shape[1] == 15 and row["status"] == "completed":
                selected = np.asarray(row["result"]["selected"], np.float32)
                best = min(best, float(scores.min()))
                found |= bool(np.any(np.all(actions == selected, axis=(1, 2)) &
                                     (np.abs(scores - row["result"]["selected_score"]) < 1e-12)))
        if row["status"] == "completed" and (not found or row["result"]["selected_score"] > best + 1e-12):
            raise ValueError("selected is not scored incumbent")
        print(f"audited plan root={root} arm={arm}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(OUT / "plans_audit.json", {"plans_sha256": seal.digest(OUT / "plans.json"),
                                                   "status": "passed"})


def intervene():
    manifest = json.loads((OUT / "manifest.json").read_text())
    plans = json.loads((OUT / "plans.json").read_text())
    audit = json.loads((OUT / "plans_audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(OUT / "plans.json"):
        raise ValueError("plans not sealed")
    entries = []
    for row in plans["rows"]:
        root, arm, task = row["root"], row["arm"], row["task"]
        entry = {"root": root, "arm": arm}
        if row["status"] != "completed":
            entries.append({**entry, "status": "plan_failed"})
            continue
        recorder = journal.PredictionJournal(OUT / f"intervention_{root:02d}_{arm}_calls",
            old.previous.fixed.predictor(task), action_dim=2, packet_length=5, token_cap=100000)
        event_dir = OUT / f"intervention_{root:02d}_{arm}_events"
        event_dir.mkdir()
        def emit(event):
            journal.write_json(event_dir / f"{len(list(event_dir.iterdir())):04d}.json", event)
        try:
            chain = np.asarray(row["result"]["selected"], np.float32)
            layer = row["result"]["child"]["layers"][0]
            greedy = int(np.argmin([x["score"] for x in layer["candidates"]]))
            rng = np.random.default_rng(2032200000 + root)
            random_prefix = np.clip(cem.interpolate(rng.normal(size=(1, 4, 2)), 5)[0], -1, 1).astype(np.float32)
            controls = {"selected": chain[:5], "zero": np.zeros((5, 2), np.float32),
                        "greedy": np.asarray(layer["candidates"][greedy]["chain"], np.float32),
                        "random": random_prefix}
            tails = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
            tails.append(chain[5:])
            suffix = {}
            for name, prefix in controls.items():
                common = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
                _, common_scores = recorder(common)
                runs = []
                for repeat in range(3):
                    zero = np.concatenate((prefix, np.zeros((10, 2), np.float32)))[None]
                    initial = float(recorder(zero)[1][0])
                    best_chain, score, _ = cem.cem(np.zeros((10, 2)), prefix, 480,
                        np.random.default_rng(2032300000 + root * 10 + repeat), recorder, initial, 4)
                    runs.append({"score": score, "chain": best_chain.tolist()})
                suffix[name] = {"prefix": prefix.tolist(), "common_distances": common_scores.tolist(),
                                "relative_zero": (common_scores - common_scores[0]).tolist(), "runs": runs}
            def branch(actions):
                trajectory, blocked = exact.wall.rollout(task["start"],
                    np.pad(actions, ((0, 0), (0, 1), (0, 0))) * exact.wall.HIGH)
                return trajectory[:, 15, :2], np.asarray(blocked)[:, :15]
            measurements = measure.measure(chain, controls, task["goal"], recorder, branch, emit)
            entry.update(status="completed", selected_chain=chain.tolist(), greedy_index=greedy,
                         suffix=suffix, measurements=measurements)
        except Exception:
            entry.update(status="failed", error=traceback.format_exc())
        entry.update(tokens=recorder.tokens, calls=recorder.calls)
        journal.write_json(OUT / f"intervention_{root:02d}_{arm}.json", entry)
        entries.append(entry)
        print(f"intervention root={root} arm={arm} status={entry['status']}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(OUT / "interventions.json", {"plans_sha256": seal.digest(OUT / "plans.json"),
        "plans_audit_sha256": seal.digest(OUT / "plans_audit.json"), "rows": entries})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit_plans", "intervene"):
        raise SystemExit("choose plan, audit_plans, or intervene")
    {"plan": plan, "audit_plans": audit_plans, "intervene": intervene}[sys.argv[1]]()
