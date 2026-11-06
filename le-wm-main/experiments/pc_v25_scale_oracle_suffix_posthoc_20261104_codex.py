"""Fixed privileged suffix-family diagnostic on sealed generated prefixes."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as feasibility

source = feasibility.source
OUT = source.ROOT / "results/pc_v25_scale_oracle_suffix_posthoc_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_SCALE_ORACLE_SUFFIX_POSTHOC_PROTOCOL_20261104_CODEX.md"


def summarize(prefixes, scores, selections):
    prefixes, scores = np.asarray(prefixes), np.asarray(scores, dtype=float)
    if prefixes.ndim != 3 or scores.shape != (len(prefixes), 2) or not np.isfinite(scores).all():
        raise ValueError("invalid candidate scores")
    best = scores.min(axis=1)
    output = {}
    for name, indices in selections.items():
        if (not indices or len(set(indices)) != len(indices)
                or any(not isinstance(i, int) or i < 0 or i >= len(prefixes) for i in indices)):
            raise ValueError("invalid selection")
        winner = min(indices, key=lambda i: (best[i], i))
        output[name] = {"indices": indices, "winner": winner, "distance": float(best[winner]),
                        "success": bool(best[winner] < .05)}
    winner = int(np.argmin(best))
    return {"rules": output, "bank_winner": winner, "bank_distance": float(best[winner]),
            "bank_success": bool(best[winner] < .05)}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit_path = source.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    witnesses = json.loads(feasibility.OUT.read_text())
    source.seal.verify(manifest["dependencies"])
    source.seal.verify(witnesses["dependencies"])
    if (audit["status"] != "passed" or len(audit["rows"]) != 16
            or any(row["status"] != "completed" for row in audit["rows"])
            or len(witnesses["rows"]) != len(source.tasks())):
        raise ValueError("source panel or witnesses incomplete")
    files = [Path(__file__), PROTOCOL, feasibility.OUT, audit_path, source.OUT / "manifest.json"]
    files += [source.OUT / f"root{root:02d}_{arm}.json"
              for root in range(len(source.tasks())) for arm in source.ARMS]
    dependencies = source.seal.capture(files)
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {
        "dependencies": dependencies, "scope": "post-hoc privileged suffix family"})
    print("Evaluating two fixed oracle suffixes on every frozen generated first-layer prefix.", flush=True)
    for root, task in enumerate(source.tasks()):
        witness = witnesses["rows"][root]
        if witness["task"] != task or any(not w.get("success") for w in witness["witnesses"]):
            raise ValueError("two successful witnesses required by frozen protocol")
        tails = np.asarray([w["actions"][5:] for w in witness["witnesses"]], np.float32)
        for arm in source.ARMS:
            path = source.OUT / f"root{root:02d}_{arm}.json"
            plan = json.loads(path.read_text())
            recorder = source.journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls",
                source.prior.predictor(task), action_dim=2, packet_length=5, token_cap=20000)
            try:
                layer = plan["result"]["child"]["layers"][0]
                prefixes = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                actions = np.concatenate((np.repeat(prefixes, 2, axis=0),
                                          np.tile(tails, (len(prefixes), 1, 1))), axis=1)
                _, values = recorder(actions)
                values = values.reshape(len(prefixes), 2)
                summary = summarize(prefixes, values, layer["retention_alternatives"])
                row = {"root": root, "arm": arm, "task": task, "status": "completed",
                       "scores": values.tolist(), "summary": summary, "tokens": recorder.tokens,
                       "response_added": layer["response_added"], "response_removed": layer["response_removed"]}
            except Exception:
                row = {"root": root, "arm": arm, "task": task, "status": "failed",
                       "error": traceback.format_exc(), "tokens": recorder.tokens}
            source.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']}", flush=True)
    source.seal.verify(dependencies)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    witnesses = json.loads(feasibility.OUT.read_text())
    summaries = []
    for root, task in enumerate(source.tasks()):
        tails = np.asarray([w["actions"][5:] for w in witnesses["rows"][root]["witnesses"]], np.float32)
        for arm in source.ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["arm"], row["task"]) != (root, arm, task):
                raise ValueError("identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = source.prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"]:
                raise ValueError("journal cost mismatch")
            if row["status"] != "completed":
                summaries.append(row)
                continue
            plan = json.loads((source.OUT / f"root{root:02d}_{arm}.json").read_text())
            layer = plan["result"]["child"]["layers"][0]
            prefixes = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
            calls = list(source.prior.read_calls(directory))
            if len(calls) != 1:
                raise ValueError("expected one candidate bank call")
            actions, positions, scores = calls[0]
            np.testing.assert_array_equal(actions[:, :5], np.repeat(prefixes, 2, axis=0))
            np.testing.assert_array_equal(actions[:, 5:], np.tile(tails, (len(prefixes), 1, 1)))
            path, _ = source.OLD.previous.independent(task["start"], actions)
            np.testing.assert_allclose(path[:, -1, :2], positions, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                       atol=1e-12, rtol=0)
            values = scores.reshape(len(prefixes), 2)
            if values.tolist() != row["scores"] or summarize(prefixes, values, layer["retention_alternatives"]) != row["summary"]:
                raise ValueError("summary mismatch")
            summaries.append({"root": root, "arm": arm, "status": "completed", **row["summary"]})
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(output, {"status": "passed", "rows": summaries,
        "scope": "alternate same-environment geometry; not independent replication"})
    print(f"Audited {len(summaries)} diagnostic rows.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
