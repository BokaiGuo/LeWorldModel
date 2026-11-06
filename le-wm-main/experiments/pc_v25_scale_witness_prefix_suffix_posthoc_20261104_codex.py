"""Common CEM on privileged fixed prefixes; never feeds a planner."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_scale_fixed_bank_suffix_posthoc_20261104_codex as common
from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as witness

source, suffix = common.source, common.suffix
OUT = source.ROOT / "results/pc_v25_scale_witness_prefix_suffix_posthoc_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_SCALE_WITNESS_PREFIX_SUFFIX_POSTHOC_PROTOCOL_20261104_CODEX.md"


def prefixes_for(root, task, witnesses):
    row = witnesses["rows"][root]
    if row["root"] != root or row["task"] != task or len(row["witnesses"]) != 2:
        raise ValueError("witness identity mismatch")
    if any(not record.get("success") for record in row["witnesses"]):
        raise ValueError("both prefixes must come from verified successful witnesses")
    return np.asarray([record["actions"][:5] for record in row["witnesses"]], np.float32)


def optimize(predict, prefixes, root):
    return suffix.evaluate_bank(predict, prefixes, {"both": [0, 1]}, tail_steps=10,
        samples=common.SAMPLES, restarts=common.RESTARTS, seed=2030902000 + root,
        batch_size=12, elites=4)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    witnesses = json.loads(witness.OUT.read_text())
    source.seal.verify(witnesses["dependencies"])
    deps = source.seal.capture([Path(__file__), PROTOCOL, witness.OUT, Path(common.__file__),
        Path(suffix.__file__), Path(suffix.pc_shared_multistart_cem.__file__)])
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {"dependencies": deps,
        "scope": "post-hoc privileged prefixes, never planner input"})
    print("Evaluating common CEM on the two verified witness prefixes per task.", flush=True)
    for root, task in enumerate(source.tasks()):
        recorder = source.journal.PredictionJournal(OUT / f"root{root:02d}_calls",
            source.prior.predictor(task), action_dim=2, packet_length=5,
            token_cap=2 * common.SAMPLES * common.RESTARTS * 3)
        try:
            prefixes = prefixes_for(root, task, witnesses)
            result = optimize(recorder, prefixes, root)
            row = {"root": root, "task": task, "status": "completed", "diagnostic": result}
        except Exception:
            row = {"root": root, "task": task, "status": "failed", "error": traceback.format_exc()}
        row["tokens"] = recorder.tokens
        source.journal.write_json(OUT / f"root{root:02d}_result.json", row)
        print(f"root={root} status={row['status']}", flush=True)
    source.seal.verify(deps)


def audit():
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    witnesses = json.loads(witness.OUT.read_text())
    rows = []
    for root, task in enumerate(source.tasks()):
        row = json.loads((OUT / f"root{root:02d}_result.json").read_text())
        if row["root"] != root or row["task"] != task:
            raise ValueError("task mismatch")
        directory = OUT / f"root{root:02d}_calls"
        info = source.prospective.journal_audit.audit_journal(directory)
        if info["tokens"] != row["tokens"]:
            raise ValueError("journal meter mismatch")
        if row["status"] != "completed":
            rows.append(row)
            continue
        calls = list(source.prior.read_calls(directory))
        for actions, positions, scores in calls:
            trajectory, _ = source.OLD.previous.independent(task["start"], actions)
            np.testing.assert_allclose(trajectory[:, -1, :2], positions, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                       rtol=0, atol=1e-12)
        replay = iter(calls)
        def predict(actions):
            recorded, features, scores = next(replay)
            np.testing.assert_array_equal(actions, recorded)
            return features, scores
        result = optimize(predict, prefixes_for(root, task, witnesses), root)
        if next(replay, None) is not None or result != row["diagnostic"]:
            raise ValueError("optimizer replay mismatch")
        rows.append({"root": root, "status": "completed", "tokens": info["tokens"],
                     "prefix_scores": [candidate["best_score"] for candidate in result["candidates"]]})
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(path, {"status": "passed", "rows": rows,
        "scope": "same-environment alternate geometry and deterministic replay; not independent replication"})
    print(f"Audited {len(rows)} privileged-prefix diagnostics.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
