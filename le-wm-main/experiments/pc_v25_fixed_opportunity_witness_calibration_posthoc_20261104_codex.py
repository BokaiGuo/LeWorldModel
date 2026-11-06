"""Calibrate fixed-opportunity suffix search against privileged feasible prefixes."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v25_response_random_fixed_opportunity_posthoc_20261104_codex as fixed
from experiments.spectral_reachability import pc_shared_multistart_cem as cem

panel = fixed.panel
OUT = panel.ROOT / "results/pc_v25_fixed_opportunity_witness_calibration_posthoc_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_V25_FIXED_OPPORTUNITY_WITNESS_CALIBRATION_POSTHOC_PROTOCOL_20261104_CODEX.md"
WITNESS = panel.OUT / "feasibility_posthoc.json"
RESTARTS, SAMPLES, FIRST = 4, 121, 49


def prefixes_for(witness, root, task):
    item = witness["rows"][root]
    if item["root"] != root or item["task"] != task:
        raise ValueError("witness task mismatch")
    success = [record for record in item.get("witnesses", []) if record.get("success")]
    return np.asarray([record["actions"][:5] for record in success[:2]], np.float32)


def start(root, restart):
    if restart == 0:
        return fixed.initial_tail(root)
    rng = np.random.default_rng(np.random.SeedSequence([2031501000, root, restart]))
    return rng.uniform(-1, 1, (1, 10, 2)).astype(np.float32)


def optimize(predict, prefixes, root):
    output = []
    for index, prefix in enumerate(np.asarray(prefixes, np.float32)):
        runs = []
        for restart in range(RESTARTS):
            observed = []

            def score(actions):
                _, values = predict(actions)
                observed.extend(float(value) for value in values)
                return values

            result = cem.optimize(score, prefix, start(root, restart), SAMPLES,
                seed=2031404000 + root if restart == 0 else 2031502000 + root * 10 + restart,
                batch_size=12, elites=4)
            if len(observed) != SAMPLES or result["score"] != min(observed):
                raise ValueError("optimizer incumbent or sample mismatch")
            runs.append({"score": result["score"], "first49": min(observed[:FIRST]),
                         "actions": result["actions"].tolist(), "samples": result["samples"]})
        output.append({"index": index, "runs": runs, "curve": {
            "1x49": runs[0]["first49"], "1x121": runs[0]["score"],
            "2x121": min(r["score"] for r in runs[:2]),
            "4x121": min(r["score"] for r in runs)}})
    return output


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path, audit_path = panel.OUT / "manifest.json", panel.OUT / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    panel.seal.verify(manifest["dependencies"])
    witness = json.loads(WITNESS.read_text())
    panel.seal.verify(witness["dependencies"])
    if json.loads(audit_path.read_text())["status"] != "passed":
        raise ValueError("source panel not audited")
    deps = panel.seal.capture([Path(__file__), PROTOCOL, Path(fixed.__file__), Path(cem.__file__),
                               manifest_path, audit_path, WITNESS])
    OUT.mkdir()
    panel.journal.write_json(OUT / "manifest.json", {"dependencies": deps,
        "restarts": RESTARTS, "samples": SAMPLES, "scope": "privileged exposed-task calibration"})
    print("Calibrating fixed-opportunity suffix search on witnessed prefixes, not running a planner.", flush=True)
    for root, task in enumerate(panel.tasks()):
        row = {"root": root, "task": task}
        prefixes = prefixes_for(witness, root, task)
        if len(prefixes) < 2:
            row.update(status="unknown_insufficient_witnesses", tokens=0)
        else:
            recorder = panel.journal.PredictionJournal(
                OUT / f"root{root:02d}_calls", panel.source.prior.predictor(task),
                action_dim=2, packet_length=5, token_cap=2 * RESTARTS * SAMPLES * 3)
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
    witness = json.loads(WITNESS.read_text())
    rows = []
    for root, task in enumerate(panel.tasks()):
        row = json.loads((OUT / f"root{root:02d}.json").read_text())
        if (row["root"], row["task"]) != (root, task):
            raise ValueError("identity mismatch")
        if row["status"] != "completed":
            rows.append(row)
            continue
        prefixes = prefixes_for(witness, root, task)
        directory = OUT / f"root{root:02d}_calls"
        info = panel.source.prospective.journal_audit.audit_journal(directory)
        if info["tokens"] != row["tokens"] or info["tokens"] != 2 * RESTARTS * SAMPLES * 3:
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

        rebuilt = optimize(predict, prefixes, root)
        if next(replay, None) is not None or rebuilt != row["prefixes"]:
            raise ValueError("optimizer or checkpoint replay mismatch")
        rows.append({"root": root, "status": "completed", "tokens": info["tokens"],
                     "curves": [item["curve"] for item in rebuilt]})
        print(f"audited root={root}", flush=True)
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "privileged calibration, deterministic replay and alternate same-environment geometry"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
