"""Remove direction edits while retaining six identical response probes."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_cheap_direction_edit_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_cheap_probe_ablation_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_CHEAP_DIRECTION_PROBE_ABLATION_DEV_PROTOCOL_20261105_CODEX.md"


def probe_arm(predict, parent, seed, total):
    parent = np.asarray(parent, np.float32)
    _, score = predict(parent[None])
    chosen, best = parent.copy(), float(score[0])
    rng = np.random.default_rng(seed + 200000)
    probes = []
    for segment in range(3):
        signed = rng.choice((-1., 1.), size=(5, 2))
        pair = np.repeat(parent[None], 2, axis=0)
        sl = slice(5*segment, 5*(segment+1))
        pair[0, sl] = np.clip(parent[sl].astype(np.float64) + .05*signed, -1, 1)
        pair[1, sl] = np.clip(parent[sl].astype(np.float64) - .05*signed, -1, 1)
        probes.extend(pair)
    probes = np.stack(probes)
    _, scores = predict(probes)
    index = int(np.argmin(scores))
    if scores[index] < best:
        chosen, best = probes[index].copy(), float(scores[index])
    samples = 150 if total else 84
    result = fixed.optimizer.optimize(lambda a: predict(a)[1], np.empty((0, 2), np.float32),
                                      chosen[None], samples, seed=seed, batch_size=12, elites=4)
    if float(result["score"]) > best + 1e-12:
        raise ValueError("incumbent lost")
    return {"start_actions": chosen.tolist(), "start_score": best, "refine_samples": samples,
            "final_actions": result["actions"].tolist(), "final_score": float(result["score"])}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads((prior.OUT / "manifest.json").read_text())
    audit = json.loads((prior.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["summary_sha256"] != prior.prior.digest(prior.OUT / "summary.json"):
        raise ValueError("prior audit mismatch")
    seal, journal = fixed.prior.prior.prior.seal, fixed.prior.prior.prior.journal
    seal.verify(previous["dependencies"])
    seal.verify(previous["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), prior.OUT / "manifest.json",
                         prior.OUT / "summary.json", prior.OUT / "audit.json"])
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"dependencies": deps, "source_dependencies":
                                               previous["source_dependencies"], "tasks": previous["tasks"]})
    rows = []
    for root, task in enumerate(previous["tasks"]):
        source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(source["result"])[0][0]
        for repeat in range(2):
            seed = 2051100000 + root*2 + repeat
            for arm in ("probe_fixed", "probe_total"):
                name = f"root{root:02d}_rep{repeat}_{arm}"
                budget = 157 if arm.endswith("total") else 91
                rec = journal.PredictionJournal(
                    OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=budget*3)
                row, started = {"root": root, "repeat": repeat, "arm": arm, "seed": seed}, time.monotonic()
                try:
                    result = probe_arm(rec, parent, seed, arm.endswith("total"))
                    if rec.tokens != budget*3:
                        raise ValueError("incorrect budget")
                    row.update(status="completed", result=result)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                row.update(tokens=rec.tokens, calls=rec.calls, elapsed_seconds=time.monotonic()-started)
                rows.append(row)
                journal.write_json(OUT / f"{name}.json", row)
                print(name, row["status"], flush=True)
    seal.verify(deps)
    seal.verify(previous["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Testing whether direction edits add value beyond their six shared probes.", flush=True)
    main()
