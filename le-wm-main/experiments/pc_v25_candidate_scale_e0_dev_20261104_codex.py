"""Paired v25 candidate-amplitude development experiment with durable calls."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as prospective

prior = prospective.fixed
compiler, journal, seal, OLD = prospective.compiler, prospective.journal, prospective.seal, prospective.OLD
ROOT = prospective.ROOT
OUT = ROOT / "results/pc_v25_candidate_scale_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_CANDIDATE_SCALE_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = {"original": (0.25, 0.5), "smaller": (0.125, 0.25)}


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.155, .155), (-1.17, 1.17), (-.11, .11))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    existing = (*prior.prior_panels(), prior.tasks(), prospective.tasks(), prospective.cost.tasks())
    if any(task in old for task in tasks() for old in existing):
        raise ValueError("task overlaps enumerated stopped panels")
    files = [Path(__file__), PROTOCOL, Path(prospective.__file__), Path(prior.__file__),
             Path(compiler.__file__), Path(compiler.core.__file__), Path(compiler.v13.__file__),
             Path(compiler.core.pc_shared_multistart_cem.__file__), Path(journal.__file__),
             Path(prospective.journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
             Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
             Path(OLD.previous.independent.__code__.co_filename)]
    dependencies = seal.capture(files)
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": 1920,
        "dependencies": dependencies, "scope": "development; broader provenance unknown"})
    print("Comparing candidate scales with a common total prediction-token cap.", flush=True)
    for root, task in enumerate(tasks()):
        for arm, amplitudes in ARMS.items():
            recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls",
                prior.predictor(task), action_dim=2, packet_length=5, token_cap=1920)
            config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, token_cap=1920,
                seed=2030801000 + root, max_modes=2, amplitudes=amplitudes)
            started = time.monotonic()
            try:
                result = compiler.compile_program(recorder, np.asarray(task["goal"]), config,
                                                  score_scale=1., score_target=0.)
                if result["resource"]["prediction_tokens"] != recorder.tokens:
                    raise ValueError("prediction token mismatch")
                row = {"root": root, "task": task, "arm": arm, "status": "completed", "result": result}
            except Exception:
                row = {"root": root, "task": task, "arm": arm, "status": "failed",
                       "error": traceback.format_exc()}
            row.update(journal_calls=recorder.calls, journal_tokens=recorder.tokens,
                       elapsed_seconds_with_journal=time.monotonic() - started)
            journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={recorder.tokens}", flush=True)
    seal.verify(dependencies)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or manifest["arms"] != {k: list(v) for k, v in ARMS.items()}:
        raise ValueError("manifest mismatch")
    summaries = []
    for root, task in enumerate(tasks()):
        initial_banks = []
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("row identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["journal_tokens"] or len(info["calls"]) != row["journal_calls"]:
                raise ValueError("journal meter mismatch")
            if row["status"] != "completed":
                summaries.append({"root": root, "arm": arm, "status": "failed",
                                  "tokens": info["tokens"], "error": row["error"]})
                continue
            calls = list(prior.read_calls(directory))
            initial_banks.append(calls[0][0])
            result = row["result"]
            selected = np.asarray(result["selected"], np.float32)
            best, found, full_samples = float("inf"), False, 0
            for actions, features, scores in calls:
                path, _ = OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                if actions.shape[1] == 15:
                    full_samples += len(actions)
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                         & (np.abs(scores - result["selected_score"]) < 1e-12)))
            child = result["child"]
            if not found or result["selected_score"] > best + 1e-12:
                raise ValueError("selected chain not best recorded full chain")
            if result["resource"]["prediction_tokens"] != info["tokens"]:
                raise ValueError("compiler and journal token mismatch")
            summaries.append({"root": root, "arm": arm, "status": "completed",
                "score": result["selected_score"], "success": result["selected_score"] < .05,
                "tokens": info["tokens"], "full_horizon_samples": full_samples,
                "successor_probes": sum(len(layer["reachability"]) for layer in child["layers"]),
                "refinement_samples": sum(start["samples"] for start in child["refinement_starts"]),
                "svd_calls": result["resource"]["svd_calls"],
                "elapsed_seconds_with_journal": row["elapsed_seconds_with_journal"]})
        if len(initial_banks) == 2:
            np.testing.assert_array_equal(*initial_banks)
    seal.verify(manifest["dependencies"])
    journal.write_json(output, {"status": "passed", "rows": summaries,
        "scope": "alternate same-environment geometry; not independent replication"})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
