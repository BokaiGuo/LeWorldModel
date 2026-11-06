"""Budget-matched restart curves on sealed privileged prefixes."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_witness_suffix_expression_dev_20261104_codex as previous

source, common, witness, cem = previous.source, previous.common, previous.witness, previous.cem
OUT = source.ROOT / "results/pc_v25_witness_suffix_budget_curve_dev_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_WITNESS_SUFFIX_BUDGET_CURVE_DEV_PROTOCOL_20261104_CODEX.md"
ARMS, CHECKPOINTS = ("step_zero", "step_random"), (2, 4, 8)


def optimize(predict, prefixes, root, arm):
    if arm not in ARMS:
        raise ValueError("invalid arm")
    records = []
    for index, prefix in enumerate(np.asarray(prefixes, np.float32)):
        runs = []
        for restart in range(CHECKPOINTS[-1]):
            if arm == "step_zero":
                initial = np.zeros((1, 10, 2), np.float32)
            else:
                initial = np.random.default_rng(np.random.SeedSequence(
                    [2030910000, root, restart, index])).uniform(-1, 1, (1, 10, 2)).astype(np.float32)
            result = cem.optimize(lambda chains: predict(chains)[1], prefix, initial, common.SAMPLES,
                seed=2030902000 + root + restart, batch_size=12, elites=4)
            runs.append({"restart": restart, "score": result["score"], "actions": result["actions"].tolist(),
                         "samples": result["samples"], "starts": result["starts"]})
        records.append({"index": index, "runs": runs, "curve": {
            str(k): min(run["score"] for run in runs[:k]) for k in CHECKPOINTS}})
    return records


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old_manifest = json.loads((previous.OUT / "manifest.json").read_text())
    old_audit = json.loads((previous.OUT / "audit.json").read_text())
    source.seal.verify(old_manifest["dependencies"])
    if old_audit["status"] != "passed" or len(old_audit["rows"]) != 24:
        raise ValueError("source experiment not audited")
    witnesses = json.loads(witness.OUT.read_text())
    source.seal.verify(witnesses["dependencies"])
    deps = source.seal.capture([Path(__file__), PROTOCOL, Path(previous.__file__), Path(cem.__file__),
        witness.OUT, previous.OUT / "manifest.json", previous.OUT / "audit.json"])
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {"dependencies": deps, "arms": ARMS,
        "checkpoints": CHECKPOINTS, "samples_per_restart": common.SAMPLES,
        "scope": "post-hoc privileged prefix restart-budget development diagnostic"})
    print("Comparing predeclared 2/4/8-restart curves on identical witness prefixes.", flush=True)
    for root, task in enumerate(source.tasks()):
        prefixes = previous.prior.prefixes_for(root, task, witnesses)
        for arm in ARMS:
            recorder = source.journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls",
                source.prior.predictor(task), action_dim=2, packet_length=5,
                token_cap=2 * CHECKPOINTS[-1] * common.SAMPLES * 3)
            try:
                result = optimize(recorder, prefixes, root, arm)
                row = {"root": root, "arm": arm, "task": task, "status": "completed", "prefixes": result}
            except Exception:
                row = {"root": root, "arm": arm, "task": task, "status": "failed",
                       "error": traceback.format_exc()}
            row["tokens"] = recorder.tokens
            source.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']}", flush=True)
    source.seal.verify(deps)


def audit():
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    witnesses = json.loads(witness.OUT.read_text())
    old = json.loads((previous.OUT / "audit.json").read_text())
    rows = []
    for root, task in enumerate(source.tasks()):
        prefixes = previous.prior.prefixes_for(root, task, witnesses)
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = source.prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"]:
                raise ValueError("token mismatch")
            if row["status"] != "completed":
                rows.append(row)
                continue
            calls = list(source.prior.read_calls(directory))
            for actions, positions, scores in calls:
                trajectory, _ = source.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(trajectory[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            replay = iter(calls)
            def predict(actions):
                recorded, positions, scores = next(replay)
                np.testing.assert_array_equal(np.asarray(actions, np.float32), recorded)
                return positions, scores
            result = optimize(predict, prefixes, root, arm)
            if next(replay, None) is not None or result != row["prefixes"]:
                raise ValueError("optimizer replay mismatch")
            if info["tokens"] != 2 * CHECKPOINTS[-1] * common.SAMPLES * 3:
                raise ValueError("budget mismatch")
            earlier = next(item for item in old["rows"] if item["root"] == root and item["arm"] == arm)
            if [item["curve"]["2"] for item in result] != earlier["prefix_scores"]:
                raise ValueError("two-restart reference mismatch")
            rows.append({"root": root, "arm": arm, "status": "completed",
                         "curves": [item["curve"] for item in result], "tokens": info["tokens"]})
            print(f"audited root={root} arm={arm}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(path, {"status": "passed", "rows": rows,
        "scope": "alternate same-environment geometry and optimizer replay, not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
