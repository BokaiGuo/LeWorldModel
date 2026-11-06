"""Independently inspect saved batches and returned incumbent per restart."""
import json

import numpy as np

from experiments import pc_v25_new_witness_de_cem_dev_20261104_codex as experiment

OUT = experiment.OUT / "incumbent_audit.json"


def audit_restart(calls, prefix, result):
    count = 0
    best = float("inf")
    selected = np.asarray(result["actions"], np.float32)
    found = False
    while count < experiment.SAMPLES:
        actions, _, scores = next(calls)
        count += len(actions)
        if count > experiment.SAMPLES:
            raise ValueError("batch crossed restart boundary")
        np.testing.assert_array_equal(actions[:, :5], np.broadcast_to(prefix, actions[:, :5].shape))
        best = min(best, float(np.min(scores)))
        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))
                             & np.isclose(scores, result["score"], atol=1e-12, rtol=0)))
    if not found or not np.isclose(best, result["score"], atol=1e-12, rtol=0):
        raise ValueError("returned action is not a scored best candidate")
    return best


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((experiment.OUT / "manifest.json").read_text())
    experiment.source.seal.verify(manifest["dependencies"])
    rows = []
    for root, task in enumerate(experiment.tasks()):
        witness = json.loads((experiment.OUT / f"root{root:02d}_witness.json").read_text())
        valid = [w for w in witness["witnesses"] if w.get("success")]
        if len(valid) < 2:
            rows.append({"root": root, "status": "unknown_insufficient_witnesses"})
            continue
        prefixes = np.asarray([w["actions"][:5] for w in valid[:2]], np.float32)
        for arm in experiment.ARMS:
            row = json.loads((experiment.OUT / f"root{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed":
                rows.append({"root": root, "arm": arm, "status": row["status"]})
                continue
            calls = iter(experiment.source.prior.read_calls(
                experiment.OUT / f"root{root:02d}_{arm}_calls"))
            bests = []
            for index, prefix in enumerate(prefixes):
                bests.append([audit_restart(calls, prefix, run)
                              for run in row["prefixes"][index]["runs"]])
                if min(bests[-1]) != row["prefixes"][index]["best_score"]:
                    raise ValueError("incorrect restart aggregation")
            if next(calls, None) is not None:
                raise ValueError("unexpected trailing calls")
            rows.append({"root": root, "arm": arm, "status": "passed", "restart_scores": bests})
    experiment.source.seal.verify(manifest["dependencies"])
    experiment.source.journal.write_json(OUT, {"status": "passed", "rows": rows,
        "scope": "saved-batch incumbent check; not an independent scientific replication"})


if __name__ == "__main__":
    main()
