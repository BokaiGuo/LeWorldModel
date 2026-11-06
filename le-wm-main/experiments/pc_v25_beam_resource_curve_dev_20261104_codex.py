"""Paired E0 development curve: response beam widths and shared CEM."""
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_candidate_scale_e0_dev_20261104_codex as source
from experiments import pc_v25_new_witness_de_cem_dev_20261104_codex as witness_panel
from experiments import pc_v25_prefix_random_refinement_e0_dev_20261104_codex as prefix_panel
from experiments import pc_v25_random_refinement_e0_dev_20261104_codex as random_panel
from experiments import pc_v25_successor_probe_cost_e0_dev_20261104_codex as probe_panel
from experiments.spectral_reachability import pc_global_reachability_v25 as compiler
from experiments.spectral_reachability import pc_global_shared_cem_v23 as baseline

ROOT, seal, journal, OLD = source.ROOT, source.seal, source.journal, source.OLD
OUT = ROOT / "results/pc_v25_beam_resource_curve_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_BEAM_RESOURCE_CURVE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS, CAPS = ("wide", "narrow", "cem"), (1920, 2880)


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.165, .165), (-1.09, 1.09),
                                                   (-.14, -.06, .06, .14))]


def config_for(root, arm, cap):
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in OLD.previous.prior.core.NOMINALS)
    return compiler.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
        nominals=nominal, token_cap=cap, seed=2031301000 + root, max_modes=2,
        beam_width=2 if arm == "narrow" else 4, greedy_slots=1 if arm == "narrow" else 2)


def plan(predict, target, config, arm):
    if arm in ("wide", "narrow"):
        return compiler.compile_program(predict, target, config, score_scale=1., score_target=0.)
    if arm == "cem":
        return baseline.plan(predict, config, seed=config.seed)
    raise ValueError("unknown arm")


def token_count(result, arm):
    return result["prediction_tokens"] if arm == "cem" else result["resource"]["prediction_tokens"]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    panels = (*source.prior.prior_panels(), source.prior.tasks(), source.prospective.tasks(),
              source.prospective.cost.tasks(), source.tasks(), witness_panel.tasks(),
              prefix_panel.tasks(), random_panel.tasks(), probe_panel.tasks())
    if any(task in panel for task in tasks() for panel in panels):
        raise ValueError("overlap with enumerated prior panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(compiler.__file__),
        Path(compiler.core.__file__), Path(compiler.v13.__file__), Path(baseline.__file__),
        Path(baseline.cem.__file__), Path(journal.__file__),
        Path(source.prospective.journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(source.prior.predictor.__code__.co_filename),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS,
        "caps": CAPS, "dependencies": deps, "scope": "development; other provenance unknown"})
    print("Comparing response beam widths and shared CEM across two token caps.", flush=True)
    for root, task in enumerate(tasks()):
        for cap in CAPS:
            for arm in ARMS:
                recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{cap}_{arm}_calls",
                    source.prior.predictor(task), action_dim=2, packet_length=5, token_cap=cap)
                started = time.monotonic()
                try:
                    result = plan(recorder, np.asarray(task["goal"]), config_for(root, arm, cap), arm)
                    if token_count(result, arm) != recorder.tokens:
                        raise ValueError("planner and journal token mismatch")
                    row = {"root": root, "task": task, "arm": arm, "cap": cap,
                           "status": "completed", "result": result}
                except Exception:
                    row = {"root": root, "task": task, "arm": arm, "cap": cap,
                           "status": "failed", "error": traceback.format_exc()}
                row.update(tokens=recorder.tokens, calls=recorder.calls,
                           elapsed_seconds_with_journal=time.monotonic() - started)
                journal.write_json(OUT / f"root{root:02d}_{cap}_{arm}.json", row)
                print(f"root={root} cap={cap} arm={arm} status={row['status']} tokens={recorder.tokens}",
                      flush=True)
    seal.verify(deps)


def audit():
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS)
            or manifest["caps"] != list(CAPS)):
        raise ValueError("manifest mismatch")
    summary = []
    for root, task in enumerate(tasks()):
        for cap in CAPS:
            first = []
            for arm in ARMS:
                row = json.loads((OUT / f"root{root:02d}_{cap}_{arm}.json").read_text())
                if (row["root"], row["task"], row["arm"], row["cap"]) != (root, task, arm, cap):
                    raise ValueError("row identity mismatch")
                directory = OUT / f"root{root:02d}_{cap}_{arm}_calls"
                info = source.prospective.journal_audit.audit_journal(directory)
                if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                    raise ValueError("journal mismatch")
                if row["status"] != "completed":
                    summary.append({"root": root, "cap": cap, "arm": arm, "status": "failed",
                                    "tokens": info["tokens"], "error": row["error"]})
                    continue
                calls = list(source.prior.read_calls(directory))
                first.append(calls[0][0])
                result = row["result"]
                selected = np.asarray(result["selected"], np.float32)
                best, found, full_samples = float("inf"), False, 0
                for actions, positions, scores in calls:
                    path_geo, _ = OLD.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(path_geo[:, -1, :2], positions, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                               atol=1e-12, rtol=0)
                    if actions.shape[1] == 15:
                        full_samples += len(actions)
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                      & (np.abs(scores - result["selected_score"]) < 1e-12)))
                if not found or result["selected_score"] > best + 1e-12:
                    raise ValueError("selected chain not best evaluated complete chain")
                if token_count(result, arm) != info["tokens"] or info["tokens"] > cap:
                    raise ValueError("token budget mismatch")
                replay = iter(calls)

                def predict(actions):
                    saved, features, scores = next(replay)
                    np.testing.assert_array_equal(actions, saved)
                    return features, scores

                rebuilt = plan(predict, np.asarray(task["goal"]), config_for(root, arm, cap), arm)
                if next(replay, None) is not None:
                    raise ValueError("unconsumed prediction")
                for key in ("selected", "selected_score", "incumbent", "comparison",
                            "paired_scores", "abstained"):
                    if rebuilt[key] != result[key]:
                        raise ValueError("planner replay mismatch")
                child = result.get("child", {})
                starts = result["starts"] if arm == "cem" else child["refinement_starts"]
                summary.append({"root": root, "cap": cap, "arm": arm, "status": "completed",
                    "score": result["selected_score"], "success": result["selected_score"] < .05,
                    "tokens": info["tokens"], "full_horizon_samples": full_samples,
                    "refinement_samples": sum(x["samples"] for x in starts),
                    "successor_probes": sum(len(layer["reachability"]) for layer in child.get("layers", [])),
                    "svd_calls": result["resource"]["svd_calls"] if arm != "cem" else 0,
                    "elapsed_seconds_with_journal": row["elapsed_seconds_with_journal"]})
            if len(first) > 1:
                for bank in first[1:]:
                    np.testing.assert_array_equal(first[0], bank)
            print(f"audited root={root} cap={cap}", flush=True)
    seal.verify(manifest["dependencies"])
    journal.write_json(path, {"status": "passed", "rows": summary,
        "scope": "same-environment alternate geometry and deterministic replay; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
