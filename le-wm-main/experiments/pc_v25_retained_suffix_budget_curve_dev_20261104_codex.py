"""Equal-budget optimizer curves on previously retained automatic prefixes."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_witness_suffix_budget_curve_dev_20261104_codex as previous

source, common = previous.source, previous.common
OUT = source.ROOT / "results/pc_v25_retained_suffix_budget_curve_dev_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_RETAINED_SUFFIX_BUDGET_CURVE_DEV_PROTOCOL_20261104_CODEX.md"


def retained(root, arm):
    plan = json.loads((source.OUT / f"root{root:02d}_{arm}.json").read_text())
    layer = plan["result"]["child"]["layers"][0]
    indices = layer["retention_alternatives"]["response"]
    if indices != layer["retained"] or len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("response selection does not match actual retained beam")
    prefixes = np.asarray([layer["candidates"][i]["chain"] for i in indices], np.float32)
    if prefixes.shape != (4, 5, 2):
        raise ValueError("invalid retained prefixes")
    return indices, prefixes


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest = json.loads((source.OUT / "manifest.json").read_text())
    source_audit = json.loads((source.OUT / "audit.json").read_text())
    prev_manifest = json.loads((previous.OUT / "manifest.json").read_text())
    prev_audit = json.loads((previous.OUT / "audit.json").read_text())
    source.seal.verify(source_manifest["dependencies"])
    source.seal.verify(prev_manifest["dependencies"])
    if source_audit["status"] != "passed" or prev_audit["status"] != "passed":
        raise ValueError("source audits not passed")
    files = [Path(__file__), PROTOCOL, Path(previous.__file__), Path(previous.cem.__file__),
             source.OUT / "manifest.json", source.OUT / "audit.json",
             previous.OUT / "manifest.json", previous.OUT / "audit.json"]
    files += [source.OUT / f"root{root:02d}_{arm}.json"
              for root in range(len(source.tasks())) for arm in source.ARMS]
    deps = source.seal.capture(files)
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {"dependencies": deps,
        "checkpoints": previous.CHECKPOINTS, "samples_per_restart": common.SAMPLES,
        "scope": "post-hoc generated retained prefix development diagnostic"})
    print("Evaluating equal-budget suffix restart curves on frozen retained automatic prefixes.", flush=True)
    for root, task in enumerate(source.tasks()):
        for plan_arm in source.ARMS:
            indices, prefixes = retained(root, plan_arm)
            for init in previous.ARMS:
                recorder = source.journal.PredictionJournal(OUT / f"root{root:02d}_{plan_arm}_{init}_calls",
                    source.prior.predictor(task), action_dim=2, packet_length=5,
                    token_cap=4 * previous.CHECKPOINTS[-1] * common.SAMPLES * 3)
                try:
                    result = previous.optimize(recorder, prefixes, root, init)
                    row = {"root": root, "task": task, "plan_arm": plan_arm, "init": init,
                           "status": "completed", "indices": indices, "prefixes": result}
                except Exception:
                    row = {"root": root, "task": task, "plan_arm": plan_arm, "init": init,
                           "status": "failed", "error": traceback.format_exc()}
                row["tokens"] = recorder.tokens
                source.journal.write_json(OUT / f"root{root:02d}_{plan_arm}_{init}.json", row)
                print(f"root={root} plan={plan_arm} init={init} status={row['status']}", flush=True)
    source.seal.verify(deps)


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(source.tasks()):
        for plan_arm in source.ARMS:
            indices, prefixes = retained(root, plan_arm)
            for init in previous.ARMS:
                row = json.loads((OUT / f"root{root:02d}_{plan_arm}_{init}.json").read_text())
                if (row["root"], row["task"], row["plan_arm"], row["init"]) != (root, task, plan_arm, init):
                    raise ValueError("row identity mismatch")
                directory = OUT / f"root{root:02d}_{plan_arm}_{init}_calls"
                info = source.prospective.journal_audit.audit_journal(directory)
                if row["tokens"] != info["tokens"]:
                    raise ValueError("token mismatch")
                if row["status"] != "completed":
                    rows.append(row)
                    continue
                if row["indices"] != indices:
                    raise ValueError("retention order mismatch")
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
                result = previous.optimize(predict, prefixes, root, init)
                if next(replay, None) is not None or result != row["prefixes"]:
                    raise ValueError("optimizer replay mismatch")
                if info["tokens"] != 4 * previous.CHECKPOINTS[-1] * common.SAMPLES * 3:
                    raise ValueError("budget mismatch")
                rows.append({"root": root, "plan_arm": plan_arm, "init": init, "status": "completed",
                             "indices": indices, "curves": [item["curve"] for item in result],
                             "tokens": info["tokens"]})
                print(f"audited root={root} plan={plan_arm} init={init}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(output, {"status": "passed", "rows": rows,
        "scope": "same-environment geometry and optimizer replay, not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
