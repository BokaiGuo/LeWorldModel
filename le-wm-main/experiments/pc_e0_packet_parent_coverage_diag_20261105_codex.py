"""Packet-correlated parent bank against prior step-independent parent bank."""
import json
import time
import traceback
from pathlib import Path

import numpy as np
from experiments import pc_paired_new_task_costed_dev_20261105_codex as prior

ROOT, fixed = prior.ROOT, prior.fixed
OUT = ROOT / "results/pc_e0_packet_parent_coverage_diag_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_PACKET_PARENT_COVERAGE_DIAG_PROTOCOL_20261105_CODEX.md"


def packet_bank(seed):
    rng = np.random.default_rng(seed)
    packet = rng.uniform(-1, 1, (64, 3, 2)).astype(np.float32)
    packet[0] = 0
    return np.repeat(packet, 5, axis=1)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = json.loads((prior.OUT / "manifest.json").read_text())
    audit = json.loads((prior.OUT / "audit.json").read_text())
    digest = prior.prior.old.prior.digest
    if audit["status"] != "passed" or audit["summary_sha256"] != digest(prior.OUT / "summary.json"):
        raise ValueError("source audit mismatch")
    seal, journal = fixed.prior.prior.prior.seal, fixed.prior.prior.prior.journal
    seal.verify(old["dependencies"])
    seal.verify(old["source_dependencies"])
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__), Path(fixed.optimizer.__file__),
                         prior.OUT / "manifest.json", prior.OUT / "summary.json", prior.OUT / "audit.json"])
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"dependencies": deps, "source_dependencies":
                                               old["source_dependencies"], "tasks": old["tasks"]})
    rows = []
    for root, task in enumerate(old["tasks"]):
        for rep in range(2):
            name = f"root{root:02d}_rep{rep}"
            seed, opt_seed = 2052000000 + root*2 + rep, 2051900000 + root*2 + rep
            rec = journal.PredictionJournal(OUT / f"{name}_calls",
                    fixed.prior.prior.prior.old.previous.fixed.predictor(task),
                    action_dim=2, packet_length=5, token_cap=663)
            row, start = {"root": root, "repeat": rep, "seed": seed, "optimizer_seed": opt_seed}, time.monotonic()
            try:
                bank = packet_bank(seed)
                _, scores = rec(bank)
                index = int(np.argmin(scores))
                result = prior.cem_arm(rec, bank[index], opt_seed)
                if rec.tokens != 663 or result["final_score"] > float(scores[index]) + 1e-12:
                    raise ValueError("budget or incumbent mismatch")
                row.update(status="completed", parent_index=index, parent_score=float(scores[index]), result=result)
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
            row.update(tokens=rec.tokens, calls=rec.calls, elapsed_seconds=time.monotonic()-start)
            rows.append(row)
            journal.write_json(OUT / f"{name}.json", row)
            print(name, row["status"], row["tokens"], flush=True)
    seal.verify(deps)
    seal.verify(old["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Testing packet-correlated versus step-independent E0 global parent libraries.", flush=True)
    main()
