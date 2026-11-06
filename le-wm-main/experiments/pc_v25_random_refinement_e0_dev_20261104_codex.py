"""Equal-token v25 refinement seed-policy development panel."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_v25_candidate_scale_e0_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_v25_random_refinement_dev as candidate

seal, journal, OLD = prior.seal, prior.journal, prior.OLD
ROOT = prior.ROOT
OUT = ROOT / "results/pc_v25_random_refinement_e0_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V25_RANDOM_REFINEMENT_E0_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v25", "random_refinement")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.145, .145), (-1.15, 1.15), (-.09, .09))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    known = (*prior.prior.prior_panels(), prior.prior.tasks(), prior.prospective.tasks(),
             prior.prospective.cost.tasks(), prior.tasks())
    if any(task in panel for task in tasks() for panel in known):
        raise ValueError("overlap with enumerated stopped panels")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(candidate.__file__),
        Path(candidate.frozen.__file__), Path(candidate.frozen.core.__file__),
        Path(candidate.frozen.core.pc_shared_multistart_cem.__file__), Path(candidate.frozen.v13.__file__),
        Path(journal.__file__), Path(prior.prospective.journal_audit.__file__),
        Path(journal.write_json.__code__.co_filename),
        Path(OLD.exact.__file__), Path(OLD.exact.wall.__file__), Path(OLD.exact.wall.physics.__file__),
        Path(OLD.previous.independent.__code__.co_filename)])
    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                    for item in OLD.previous.prior.core.NOMINALS)
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": 1920,
        "dependencies": deps, "scope": "new development tasks, wider provenance unknown"})
    print("Comparing v25 with replaced full-chain refinement seeds at equal token cap.", flush=True)
    for root, task in enumerate(tasks()):
        for arm in ARMS:
            recorder = journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls",
                prior.prior.predictor(task), action_dim=2, packet_length=5, token_cap=1920)
            config = candidate.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, token_cap=1920, seed=2031101000 + root, max_modes=2)
            started = time.monotonic()
            try:
                result = candidate.compile_program(recorder, np.asarray(task["goal"]), config,
                    score_scale=1., score_target=0., random_refinement=arm == "random_refinement")
                if result["resource"]["prediction_tokens"] != recorder.tokens:
                    raise ValueError("prediction token mismatch")
                row = {"root": root, "task": task, "arm": arm, "status": "completed", "result": result}
            except Exception:
                row = {"root": root, "task": task, "arm": arm, "status": "failed", "error": traceback.format_exc()}
            row.update(tokens=recorder.tokens, calls=recorder.calls,
                       elapsed_seconds_with_journal=time.monotonic() - started)
            journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={recorder.tokens}", flush=True)
    seal.verify(deps)


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
        first = []
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("row identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = prior.prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
                raise ValueError("journal mismatch")
            if row["status"] != "completed":
                summary.append(row)
                continue
            calls = list(prior.prior.read_calls(directory))
            first.append(calls[0][0])
            result = row["result"]
            selected, best, found = np.asarray(result["selected"], np.float32), float("inf"), False
            for actions, positions, scores in calls:
                path, _ = OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1), atol=1e-12, rtol=0)
                if actions.shape[1] == 15:
                    best = min(best, float(scores.min()))
                    found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                                         & (np.abs(scores - result["selected_score"]) < 1e-12)))
            if not found or result["selected_score"] > best + 1e-12:
                raise ValueError("selected chain not best recorded complete chain")
            if result["resource"]["prediction_tokens"] != info["tokens"]:
                raise ValueError("compiler token mismatch")
            replay = iter(calls)
            def predict(actions):
                saved, features, scores = next(replay)
                np.testing.assert_array_equal(actions, saved)
                return features, scores
            rebuilt = candidate.compile_program(predict, np.asarray(task["goal"]),
                candidate.Config(**result["config"]), score_scale=1., score_target=0.,
                random_refinement=arm == "random_refinement")
            if next(replay, None) is not None:
                raise ValueError("unconsumed prediction")
            for key in ("selected", "selected_score", "incumbent", "comparison", "paired_scores", "abstained"):
                if rebuilt[key] != result[key]:
                    raise ValueError("deterministic compiler replay mismatch")
            if arm == "random_refinement" and rebuilt["refinement_seed_policy"] != result["refinement_seed_policy"]:
                raise ValueError("random seed policy mismatch")
            child = result["child"]
            summary.append({"root": root, "arm": arm, "status": "completed",
                "score": result["selected_score"], "success": result["selected_score"] < .05,
                "tokens": info["tokens"], "refinement_samples": sum(x["samples"] for x in child["refinement_starts"]),
                "refinement_seed_sources": [x["source"] for x in child["refinement_starts"]],
                "successor_probes": sum(len(layer["reachability"]) for layer in child["layers"]),
                "svd_calls": result["resource"]["svd_calls"], "elapsed_seconds": row["elapsed_seconds_with_journal"]})
        if len(first) == 2:
            np.testing.assert_array_equal(first[0], first[1])
    seal.verify(manifest["dependencies"])
    journal.write_json(out, {"status": "passed", "rows": summary,
        "scope": "alternate same-environment geometry and compiler replay, not independent replication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
