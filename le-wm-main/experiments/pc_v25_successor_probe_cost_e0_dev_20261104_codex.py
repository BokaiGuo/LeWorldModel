"""Equal-token paired v25 successor-probe ablation on new exact-Wall roots."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_prediction_journal_audit_v1 as journal_audit
from experiments import pc_prediction_journal_v1 as journal
from experiments import pc_v25_fixed_bank_suffix_e0_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_global_reachability_v25 as frozen
from experiments.spectral_reachability import pc_global_reachability_v25_probe_ablation as ablation

ROOT, OLD, seal = prior.ROOT, prior.OLD, prior.seal
OUT = ROOT / "results/pc_v25_successor_probe_cost_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_SUCCESSOR_PROBE_COST_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("response", "without_successor_probes")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.19, .19), (-1.21, 1.21), (-.18, .18))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if any(task in old for task in tasks() for old in (*prior.prior_panels(), prior.tasks())):
        raise ValueError("overlap with enumerated panels")
    regression = ROOT / "tests/test_pc_v25_probe_cost_ablation.py"
    dependencies = seal.capture([Path(__file__), PROTOCOL, regression, Path(prior.__file__),
        Path(frozen.__file__), Path(frozen.core.__file__), Path(frozen.v13.__file__),
        Path(ablation.__file__), Path(ablation.core.__file__),
        Path(ablation.core.pc_shared_multistart_cem.__file__), Path(journal.__file__),
        Path(journal_audit.__file__), Path(journal.write_json.__code__.co_filename),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    OLD.writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": 1920,
        "dependencies": dependencies, "scope": "development, wider provenance unknown"})
    print("Compare successor probes versus no-probe fallback at the same total token cap.", flush=True)
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        def predict(actions):
            positions = OLD.exact.endpoints(task, actions)
            return positions, np.linalg.norm(positions - goal, axis=1)
        for arm in ARMS:
            recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls", predict,
                                                  action_dim=2, packet_length=5, token_cap=1920)
            module = frozen if arm == "response" else ablation
            extra = {} if arm == "response" else {"successor_probe_enabled": False}
            config = module.Config(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                nominals=nominal, token_cap=1920, seed=2030601000 + root, max_modes=2, **extra)
            started = time.monotonic()
            try:
                result = module.compile_program(recorder, goal, config, score_scale=1., score_target=0.)
                if result["resource"]["prediction_tokens"] != recorder.tokens:
                    raise ValueError("token meter mismatch")
                row = {"root": root, "task": task, "arm": arm, "status": "completed", "result": result}
            except Exception:
                row = {"root": root, "task": task, "arm": arm, "status": "failed",
                       "error": traceback.format_exc()}
            row.update(journal_calls=recorder.calls, journal_tokens=recorder.tokens,
                       elapsed_seconds_with_journal=time.monotonic() - started)
            OLD.writer(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']}", flush=True)
    seal.verify(dependencies)


def audit():
    out = OUT / "audit.json"
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or manifest["arms"] != list(ARMS):
        raise ValueError("manifest mismatch")
    summary = []
    for root, task in enumerate(tasks()):
        banks = []
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if row["root"] != root or row["task"] != task or row["arm"] != arm:
                raise ValueError("row identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = journal_audit.audit_journal(directory)
            if info["tokens"] != row["journal_tokens"] or len(info["calls"]) != row["journal_calls"]:
                raise ValueError("journal meter mismatch")
            if row["status"] != "completed":
                summary.append({"root": root, "arm": arm, "status": "failed", "error": row["error"],
                                "tokens": info["tokens"]})
                continue
            calls = list(prior.read_calls(directory))
            banks.append(calls[0][0])
            selected = np.asarray(row["result"]["selected"], np.float32)
            selected_score = row["result"]["selected_score"]
            best, found, full_samples = float("inf"), False, 0
            for actions, features, scores in calls:
                path, _ = OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], features, rtol=0, atol=1e-12)
                np.testing.assert_allclose(scores, np.linalg.norm(features - goal_for(task), axis=1),
                                           rtol=0, atol=1e-12)
                if actions.shape[1] == 15:
                    full_samples += len(actions)
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                         & (np.abs(scores - selected_score) < 1e-12)))
            result = row["result"]
            layers = result["child"]["layers"]
            probes = sum(len(layer["reachability"]) for layer in layers)
            if (not found or selected_score > best + 1e-12 or result["resource"]["prediction_tokens"] != info["tokens"]
                    or (arm == "without_successor_probes" and probes != 0)):
                raise ValueError("selection, probe or budget invariant failed")
            summary.append({"root": root, "arm": arm, "status": "completed", "score": selected_score,
                "success": selected_score < .05, "tokens": info["tokens"], "successor_probes": probes,
                "full_horizon_samples": full_samples,
                "refinement_samples": sum(x["samples"] for x in result["child"]["refinement_starts"]),
                "svd_calls": result["resource"]["svd_calls"],
                "elapsed_seconds_with_journal": row["elapsed_seconds_with_journal"]})
        if len(banks) == 2:
            np.testing.assert_array_equal(banks[0], banks[1])
    seal.verify(manifest["dependencies"])
    OLD.writer(out, {"status": "passed", "rows": summary,
                    "scope": "alternate same-environment geometry, not independent replication"})
    print(json.dumps(summary, indent=2))


def goal_for(task):
    return np.asarray(task["goal"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
