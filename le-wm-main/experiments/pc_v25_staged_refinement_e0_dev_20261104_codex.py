"""End-to-end equal-token staged versus uniform refinement development panel."""
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_beam_resource_curve_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_v25_staged_refinement_dev as variant

ROOT, seal, journal, OLD = prior.ROOT, prior.seal, prior.journal, prior.OLD
OUT = ROOT / "results/pc_v25_staged_refinement_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_STAGED_REFINEMENT_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS, CAPS = ("response_uniform", "response_staged", "cem_uniform", "cem_staged"), prior.CAPS


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.175, .175), (-1.07, 1.07),
                                                   (-.13, -.05, .05, .13))]


def config(root, cap):
    return prior.config_for(root + 1000, "narrow", cap)


def plan(predict, target, cfg, arm):
    if arm == "response_uniform":
        return prior.compiler.compile_program(predict, target, cfg, score_scale=1., score_target=0.)
    if arm == "response_staged":
        return variant.compile_program(predict, target, cfg, score_scale=1., score_target=0.)
    if arm == "cem_uniform":
        return prior.baseline.plan(predict, cfg, seed=cfg.seed)
    if arm == "cem_staged":
        return variant.plan_cem(predict, cfg, seed=cfg.seed)
    raise ValueError("unknown arm")


def details(result, arm):
    if arm.startswith("cem_"):
        return result["prediction_tokens"], result["starts"], {}
    return result["resource"]["prediction_tokens"], result["child"]["refinement_starts"], result["child"]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    known = (*prior.source.prior.prior_panels(), prior.source.prior.tasks(),
             prior.source.prospective.tasks(), prior.source.prospective.cost.tasks(), prior.source.tasks(),
             prior.witness_panel.tasks(), prior.prefix_panel.tasks(), prior.random_panel.tasks(),
             prior.probe_panel.tasks(), prior.tasks())
    if any(task in panel for task in tasks() for panel in known):
        raise ValueError("overlap with enumerated prior panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(variant.__file__),
        Path(variant.staged.__file__), Path(variant.staged.shared.__file__),
        Path(prior.compiler.__file__), Path(prior.compiler.core.__file__), Path(prior.compiler.v13.__file__),
        Path(prior.baseline.__file__), Path(journal.__file__), Path(journal.write_json.__code__.co_filename),
        Path(prior.source.prospective.journal_audit.__file__),
        Path(prior.source.prior.predictor.__code__.co_filename),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "caps": CAPS,
        "dependencies": deps, "scope": "new development; source independence unknown"})
    print("Compare uniform and staged refinement for both v25 and shared-CEM at matched caps.", flush=True)
    for root, task in enumerate(tasks()):
        for cap in CAPS:
            for arm in ARMS:
                recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{cap}_{arm}_calls",
                    prior.source.prior.predictor(task), action_dim=2, packet_length=5, token_cap=cap)
                started = time.monotonic()
                try:
                    result = plan(recorder, np.asarray(task["goal"]), config(root, cap), arm)
                    if details(result, arm)[0] != recorder.tokens:
                        raise ValueError("token mismatch")
                    row = {"root": root, "task": task, "cap": cap, "arm": arm,
                           "status": "completed", "result": result}
                except Exception:
                    row = {"root": root, "task": task, "cap": cap, "arm": arm,
                           "status": "failed", "error": traceback.format_exc()}
                row.update(tokens=recorder.tokens, calls=recorder.calls,
                           elapsed_seconds_with_journal=time.monotonic() - started)
                journal.write_json(OUT / f"root{root:02d}_{cap}_{arm}.json", row)
                print(f"root={root} cap={cap} arm={arm} status={row['status']}", flush=True)
    seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS)
            or manifest["caps"] != list(CAPS)):
        raise ValueError("manifest mismatch")
    rows = []
    for root, task in enumerate(tasks()):
        for cap in CAPS:
            first = []
            for arm in ARMS:
                row = json.loads((OUT / f"root{root:02d}_{cap}_{arm}.json").read_text())
                if (row["root"], row["task"], row["cap"], row["arm"]) != (root, task, cap, arm):
                    raise ValueError("identity mismatch")
                directory = OUT / f"root{root:02d}_{cap}_{arm}_calls"
                info = prior.source.prospective.journal_audit.audit_journal(directory)
                if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                    raise ValueError("journal mismatch")
                if row["status"] != "completed":
                    rows.append({"root": root, "cap": cap, "arm": arm, "status": "failed",
                                 "error": row["error"], "tokens": info["tokens"]})
                    continue
                calls = list(prior.source.prior.read_calls(directory))
                first.append(calls[0][0])
                result = row["result"]
                selected, best, found = np.asarray(result["selected"], np.float32), float("inf"), False
                for actions, positions, scores in calls:
                    paths, _ = OLD.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(paths[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                               atol=1e-12, rtol=0)
                    if actions.shape[1] == 15:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                             & (np.abs(scores - result["selected_score"]) < 1e-12)))
                tokens, starts, child = details(result, arm)
                if not found or result["selected_score"] > best + 1e-12 or tokens != info["tokens"] or tokens > cap:
                    raise ValueError("selection or budget mismatch")
                replay = iter(calls)

                def predict(actions):
                    saved, features, scores = next(replay)
                    np.testing.assert_array_equal(actions, saved)
                    return features, scores

                rebuilt = plan(predict, np.asarray(task["goal"]), config(root, cap), arm)
                if next(replay, None) is not None:
                    raise ValueError("extra calls")
                for key in ("selected", "selected_score", "incumbent", "comparison", "paired_scores", "abstained"):
                    if rebuilt[key] != result[key]:
                        raise ValueError("planner replay mismatch")
                _, rebuilt_starts, _ = details(rebuilt, arm)
                if starts != rebuilt_starts:
                    raise ValueError("refinement allocation replay mismatch")
                rows.append({"root": root, "cap": cap, "arm": arm, "status": "completed",
                    "score": result["selected_score"], "success": result["selected_score"] < .05,
                    "tokens": tokens, "refinement_samples": sum(x["samples"] for x in starts),
                    "promoted": sum(bool(x.get("promoted")) for x in starts),
                    "staged_applied": any(x.get("allocation_policy") == "staged" for x in starts),
                    "successor_probes": sum(len(x["reachability"]) for x in child.get("layers", []))})
            if len(first) > 1:
                for bank in first[1:]:
                    np.testing.assert_array_equal(first[0], bank)
            print(f"audited root={root} cap={cap}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "same-environment alternate geometry and deterministic replay; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
