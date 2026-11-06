"""Posthoc calibration of the actual mechanism suffix optimizer."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_fixed_opportunity_witness_calibration_posthoc_20261104_codex as base
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core

panel = base.panel
OUT = panel.ROOT / "results/pc_four_knot_witness_calibration_posthoc_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_FOUR_KNOT_WITNESS_CALIBRATION_POSTHOC_PROTOCOL_20261104_CODEX.md"


def optimize(predict, prefixes, root):
    rows = []
    for index, prefix in enumerate(prefixes):
        runs = []
        for restart in range(4):
            observed = []

            def scored(actions):
                features, scores = predict(np.asarray(actions, np.float32))
                observed.extend(np.asarray(scores).tolist())
                return features, scores

            initial = np.concatenate((prefix, np.zeros((10, 2), np.float32)))
            initial_score = float(scored(initial[None])[1][0])
            seed = 2031900000 + root * 10 + restart
            chain, score, generations = core.cem(
                np.zeros((10, 2)), prefix, 480, np.random.default_rng(seed),
                scored, initial_score, 4)
            if len(observed) != 481 or score != min(observed):
                raise ValueError("sample count or incumbent mismatch")
            runs.append({"seed": seed, "score": score, "chain": chain.tolist(),
                         "generations": generations, "evaluations": len(observed)})
        rows.append({"index": index, "prefix": prefix.tolist(), "runs": runs,
                     "curve": {str(n * 481): min(x["score"] for x in runs[:n]) for n in (1, 2, 4)}})
    return rows


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    witness = json.loads(base.WITNESS.read_text())
    panel.seal.verify(witness["dependencies"])
    deps = panel.seal.capture([Path(__file__), PROTOCOL, Path(base.__file__),
                               Path(core.__file__), base.WITNESS,
                               Path(panel.source.prior.predictor.__code__.co_filename),
                               Path(panel.OLD.exact.__file__), Path(panel.OLD.exact.wall.__file__),
                               Path(panel.OLD.exact.wall.physics.__file__)])
    OUT.mkdir()
    panel.journal.write_json(OUT / "manifest.json", {"dependencies": deps,
        "scope": "posthoc exposed-task privileged-prefix calibration", "threshold": .05})
    print("Calibrating four-knot mechanism solver, not running a planner.", flush=True)
    for root, task in enumerate(panel.tasks()):
        row = {"root": root, "task": task}
        prefixes = base.prefixes_for(witness, root, task)
        if len(prefixes) < 2:
            row.update(status="unknown_insufficient_witnesses", tokens=0)
        else:
            recorder = panel.journal.PredictionJournal(OUT / f"root{root:02d}_calls",
                panel.source.prior.predictor(task), action_dim=2, packet_length=5,
                token_cap=2 * 4 * 481 * 3)
            try:
                row.update(status="completed", prefixes=optimize(recorder, prefixes, root))
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row["tokens"] = recorder.tokens
        panel.journal.write_json(OUT / f"root{root:02d}.json", row)
        print(f"root={root} status={row['status']}", flush=True)
    panel.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    witness = json.loads(base.WITNESS.read_text())
    results = []
    for root, task in enumerate(panel.tasks()):
        row = json.loads((OUT / f"root{root:02d}.json").read_text())
        if row["root"] != root or row["task"] != task:
            raise ValueError("identity mismatch")
        if row["status"] != "completed":
            results.append({"root": root, "status": row["status"]})
            continue
        directory = OUT / f"root{root:02d}_calls"
        info = panel.source.prospective.journal_audit.audit_journal(directory)
        if info["tokens"] != row["tokens"] or info["tokens"] != 2 * 4 * 481 * 3:
            raise ValueError("budget mismatch")
        calls = list(panel.source.prior.read_calls(directory))
        for actions, features, scores in calls:
            trajectory, _ = panel.OLD.previous.independent(task["start"], actions)
            np.testing.assert_allclose(trajectory[:, -1, :2], features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1),
                                       atol=1e-12, rtol=0)
        replay = iter(calls)

        def predict(actions):
            recorded, features, scores = next(replay)
            np.testing.assert_array_equal(actions, recorded)
            return features, scores

        rebuilt = optimize(predict, base.prefixes_for(witness, root, task), root)
        if next(replay, None) is not None or rebuilt != row["prefixes"]:
            raise ValueError("replay mismatch")
        results.append({"root": root, "status": "completed",
                        "curves": [item["curve"] for item in rebuilt]})
        print(f"audited root={root}", flush=True)
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"status": "passed", "rows": results,
                                     "scope": "same-environment geometry and exact action replay"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
