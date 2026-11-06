"""Common suffix optimization on sealed v25 candidate-scale banks."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_scale_oracle_suffix_posthoc_20261104_codex as previous
from experiments.spectral_reachability import pc_fixed_bank_suffix_diagnostic_v1 as suffix

source = previous.source
OUT = source.ROOT / "results/pc_v25_scale_fixed_bank_suffix_posthoc_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_SCALE_FIXED_BANK_SUFFIX_POSTHOC_PROTOCOL_20261104_CODEX.md"
SAMPLES, RESTARTS, LOOKAHEAD = 121, 2, 24


def tail_bank(root):
    tails = np.random.default_rng(2030901000 + root).uniform(-1, 1, (LOOKAHEAD, 10, 2)).astype(np.float32)
    return np.concatenate((np.zeros((1, 10, 2), np.float32), tails))


def selection(layer, lookahead):
    return source.prior.make_selections(layer, lookahead)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit_path = source.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    source.seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or len(audit["rows"]) != 16
            or any(row["status"] != "completed" for row in audit["rows"])):
        raise ValueError("source panel not completed and audited")
    files = [Path(__file__), PROTOCOL, Path(suffix.__file__),
             Path(suffix.pc_shared_multistart_cem.__file__), audit_path, source.OUT / "manifest.json"]
    files += [source.OUT / f"root{root:02d}_{arm}.json"
              for root in range(len(source.tasks())) for arm in source.ARMS]
    dependencies = source.seal.capture(files)
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {
        "dependencies": dependencies, "samples": SAMPLES, "restarts": RESTARTS,
        "lookahead_random_tails": LOOKAHEAD, "scope": "post-hoc fixed-bank development diagnostic"})
    print("Optimizing each fixed generated prefix with equal common CEM opportunities.", flush=True)
    for root, task in enumerate(source.tasks()):
        for arm in source.ARMS:
            row = {"root": root, "arm": arm, "task": task}
            try:
                plan = json.loads((source.OUT / f"root{root:02d}_{arm}.json").read_text())
                layer = plan["result"]["child"]["layers"][0]
                prefixes = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                scout = source.journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_lookahead",
                    source.prior.predictor(task), action_dim=2, packet_length=5,
                    token_cap=len(prefixes) * (LOOKAHEAD + 1) * 3)
                lookahead = []
                for prefix in prefixes:
                    actions = np.concatenate((np.broadcast_to(prefix, (LOOKAHEAD + 1, 5, 2)),
                                              tail_bank(root)), axis=1)
                    _, scores = scout(actions)
                    lookahead.append(float(scores.min()))
                selections = selection(layer, lookahead)
                select_path = OUT / f"root{root:02d}_{arm}_selection.json"
                source.journal.write_json(select_path, {"root": root, "arm": arm, "task": task,
                    "lookahead": lookahead, "selections": selections, "lookahead_tokens": scout.tokens})
                selected_digest = source.seal.digest(select_path)
                recorder = source.journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_suffix",
                    source.prior.predictor(task), action_dim=2, packet_length=5,
                    token_cap=len(prefixes) * SAMPLES * RESTARTS * 3)
                result = suffix.evaluate_bank(recorder, prefixes, selections, tail_steps=10,
                    samples=SAMPLES, restarts=RESTARTS, seed=2030902000 + root,
                    batch_size=12, elites=4)
                if source.seal.digest(select_path) != selected_digest:
                    raise ValueError("selection changed after sealing")
                row.update(status="completed", selection_sha256=selected_digest,
                           suffix_tokens=recorder.tokens, diagnostic=result)
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            source.journal.write_json(OUT / f"root{root:02d}_{arm}_result.json", row)
            print(f"root={root} arm={arm} status={row['status']}", flush=True)
    source.seal.verify(dependencies)


def audit():
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(source.tasks()):
        for arm in source.ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}_result.json").read_text())
            if (row["root"], row["arm"], row["task"]) != (root, arm, task):
                raise ValueError("row identity mismatch")
            if row["status"] != "completed":
                rows.append(row)
                continue
            select_path = OUT / f"root{root:02d}_{arm}_selection.json"
            if source.seal.digest(select_path) != row["selection_sha256"]:
                raise ValueError("selection hash mismatch")
            selected = json.loads(select_path.read_text())
            plan = json.loads((source.OUT / f"root{root:02d}_{arm}.json").read_text())
            layer = plan["result"]["child"]["layers"][0]
            prefixes = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
            dirs = [OUT / f"root{root:02d}_{arm}_{name}" for name in ("lookahead", "suffix")]
            counts = [source.prospective.journal_audit.audit_journal(d) for d in dirs]
            if [count["tokens"] for count in counts] != [selected["lookahead_tokens"], row["suffix_tokens"]]:
                raise ValueError("journal token mismatch")
            calls = [list(source.prior.read_calls(d)) for d in dirs]
            for batches in calls:
                for actions, positions, scores in batches:
                    trajectory, _ = source.OLD.previous.independent(task["start"], actions)
                    np.testing.assert_allclose(trajectory[:, -1, :2], positions, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                               rtol=0, atol=1e-12)
            if len(calls[0]) != len(prefixes):
                raise ValueError("lookahead call count mismatch")
            for i, (actions, _, scores) in enumerate(calls[0]):
                np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(prefixes[i], (LOOKAHEAD + 1, 5, 2)))
                np.testing.assert_array_equal(actions[:, 5:], tail_bank(root))
                if float(scores.min()) != selected["lookahead"][i]:
                    raise ValueError("lookahead score mismatch")
            if selection(layer, selected["lookahead"]) != selected["selections"]:
                raise ValueError("retention selection mismatch")
            replay = iter(calls[1])
            def predict(actions):
                saved, positions, scores = next(replay)
                np.testing.assert_array_equal(actions, saved)
                return positions, scores
            result = suffix.evaluate_bank(predict, prefixes, selected["selections"], tail_steps=10,
                samples=SAMPLES, restarts=RESTARTS, seed=2030902000 + root, batch_size=12, elites=4)
            if next(replay, None) is not None or result != row["diagnostic"]:
                raise ValueError("suffix replay mismatch")
            if row["suffix_tokens"] != len(prefixes) * SAMPLES * RESTARTS * 3:
                raise ValueError("suffix sample count mismatch")
            rows.append({"root": root, "arm": arm, "status": "completed", "candidate_count": len(prefixes),
                         "all_bank_best": result["all_bank_best"], "rules": result["rules"],
                         "lookahead_tokens": counts[0]["tokens"], "suffix_tokens": counts[1]["tokens"]})
            print(f"audited root={root} arm={arm}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(path, {"status": "passed", "rows": rows,
        "scope": "alternate same-environment geometry and deterministic optimizer replay; not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
