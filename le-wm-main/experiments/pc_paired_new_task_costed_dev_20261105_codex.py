"""New-parameter E0 comparison with shared global parent generation charged."""
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_paired_segment_repair_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_paired_new_task_costed_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PAIRED_NEW_TASK_COSTED_DEV_PROTOCOL_20261105_CODEX.md"


def tasks():
    return [{"start": [-.5, y, angle], "goal": [.5, gy]}
            for y, angle, gy in itertools.product((-.14, .14), (-.8, .8), (-.13, -.07, .07, .13))]


def parent_bank(seed):
    a = np.random.default_rng(seed).uniform(-1, 1, (64, 15, 2)).astype(np.float32)
    a[0] = 0
    return a


def cem_arm(predict, parent, seed):
    result = fixed.optimizer.optimize(lambda a: predict(a)[1], np.empty((0, 2), np.float32),
                                      np.asarray(parent, np.float32)[None], 157,
                                      seed=seed, batch_size=12, elites=4)
    return {"final_actions": result["actions"].tolist(), "final_score": float(result["score"])}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest = json.loads((prior.OUT / "manifest.json").read_text())
    source_audit = json.loads((prior.OUT / "audit.json").read_text())
    if source_audit["status"] != "passed" or source_audit["summary_sha256"] != prior.old.prior.digest(prior.OUT / "summary.json"):
        raise ValueError("prior audit mismatch")
    if any(task in source_manifest["tasks"] for task in tasks()):
        raise ValueError("overlap with exposed ranked tasks")
    seal, journal = fixed.prior.prior.prior.seal, fixed.prior.prior.prior.journal
    seal.verify(source_manifest["dependencies"])
    seal.verify(source_manifest["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(fixed.optimizer.__file__),
                         Path(fixed.prior.prior.prior.journal.__file__)])
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"tasks": tasks(), "dependencies": deps,
                        "source_dependencies": source_manifest["source_dependencies"]})
    rows = []
    for root, task in enumerate(tasks()):
        for rep in range(2):
            prefix = f"root{root:02d}_rep{rep}"
            parent_seed = 2051700000 + 2*root + rep
            gen_seed = 2051800000 + 2*root + rep
            opt_seed = 2051900000 + 2*root + rep
            predict = fixed.prior.prior.prior.old.previous.fixed.predictor(task)
            recorder = journal.PredictionJournal(OUT / f"{prefix}_parent_calls", predict,
                                                action_dim=2, packet_length=5, token_cap=192)
            bank = parent_bank(parent_seed)
            start = time.monotonic()
            _, scores = recorder(bank)
            elapsed = time.monotonic()-start
            index = int(np.argmin(scores))
            parent = bank[index].copy()
            journal.write_json(OUT / f"{prefix}_parent.json", {"seed": parent_seed, "selected_index": index,
                               "selected_score": float(scores[index]), "tokens": recorder.tokens,
                               "calls": recorder.calls, "elapsed_seconds": elapsed})
            for arm in ("pair", "single", "cem"):
                name = f"{prefix}_{arm}"
                rec = journal.PredictionJournal(OUT / f"{name}_calls", predict,
                                                action_dim=2, packet_length=5, token_cap=471)
                row = {"root": root, "repeat": rep, "arm": arm, "proposal_seed": gen_seed, "optimizer_seed": opt_seed}
                start = time.monotonic()
                try:
                    result = (cem_arm(rec, parent, opt_seed) if arm == "cem" else
                              prior.run_arm(rec, parent, arm, gen_seed, opt_seed))
                    if rec.tokens != 471 or result["final_score"] > float(scores[index]) + 1e-12:
                        raise ValueError("budget or incumbent mismatch")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=rec.tokens, shared_parent_tokens=recorder.tokens,
                           total_allocated_tokens=rec.tokens + recorder.tokens,
                           calls=rec.calls, elapsed_seconds=time.monotonic()-start)
                rows.append(row)
                journal.write_json(OUT / f"{name}.json", row)
                print(name, row["status"], row["total_allocated_tokens"], flush=True)
    seal.verify(deps)
    seal.verify(source_manifest["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Testing paired edits on a predeclared new-parameter E0 panel, including parent cost.", flush=True)
    main()
