"""Retrospective input-order sensitivity, using the unchanged legacy audit."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import legacy_p1_failure_attribution_20261102_exclusive as source
from . import run_cross_environment_transfer_diagnostic as cross
from . import run_matched_learned_transfer_benchmark as benchmark

OUT = source.RESULTS / "legacy_p1_order_sensitivity_20261103_codex_RUN01"
N = 32
SEED = 20261103


def key(row):
    return (row["environment"], row["training_stage"], row["latent_dimension"], row["action_distribution"])


def snapshot(result):
    rows = {}
    for row in result["strata"]:
        selectors = row["selectors"]
        rows[str(key(row))] = {
            "eligible": sorted(row["eligible_checkpoints"]),
            "identifiable": row["identifiable"],
            "reason": row["reason_if_gated"],
            "mse": selectors.get("mse_only", {}).get("checkpoint"),
            "full": selectors.get("full_mse_action_risk_geometry", {}).get("checkpoint"),
            "delta": row.get("delta_risk_mse_minus_full"),
        }
    return {"rows": rows, "primary": result["summary"]["primary_delta_risk_mse_minus_full"],
            "claim_ready": result["summary"]["claim_ready_for_positive_delta"]}


def changes(reference, variant):
    if reference["rows"].keys() != variant["rows"].keys():
        raise ValueError("strata changed")
    fields = ("eligible", "identifiable", "reason", "mse", "full", "delta")
    return {k: [f for f in fields if reference["rows"][k][f] != v[f]]
            for k, v in variant["rows"].items()
            if any(reference["rows"][k][f] != v[f] for f in fields)}


def panel(paths, historical, label, bootstrap_seed, rng):
    before = [source.file_record(p) for p in paths]
    records, failures = cross._load_all(paths, require_external=True)
    if failures or len(records) != len(paths):
        raise ValueError(f"incomplete panel: {failures}")
    baseline = snapshot(benchmark.audit(records, require_external=True, bootstrap_seed=bootstrap_seed))
    expected = snapshot(historical)
    if changes(expected, baseline) or baseline["primary"] != expected["primary"]:
        raise ValueError("baseline does not reproduce saved original")
    variants = []
    for i in range(N):
        order = rng.permutation(len(records))
        run = snapshot(benchmark.audit([records[int(j)] for j in order], require_external=True,
                                       bootstrap_seed=bootstrap_seed))
        variants.append({"index": i, "permutation_indices": order.tolist(),
                         "changed_strata": changes(baseline, run), **run})
    after = [source.file_record(p) for p in paths]
    if before != after:
        raise ValueError("input changed during audit")
    return {"label": label, "input": before, "baseline": baseline, "permutations": variants}


def main():
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(OUT)
    inputs = [source.OLD, source.EXTERNAL]
    hashes = [source.file_record(p) for p in inputs]
    old = json.loads(inputs[0].read_bytes())
    external = json.loads(inputs[1].read_bytes())
    old_paths = sorted(p for root in source.OLD_ROOTS for p in root.rglob("checkpoint_record.json"))
    external_paths = [source.ROOT / p for p in external["input"]["paths"]]
    rng = np.random.default_rng(SEED)
    panels = [panel(old_paths, old, "provisional", 20261011, rng),
              panel(external_paths, external["global_audit"], "external", 20261013, rng)]
    if hashes != [source.file_record(p) for p in inputs]:
        raise ValueError("historical report changed")
    report = {"seed": SEED, "permutations_per_panel": N, "historical_reports": hashes,
              "interpretation": "Retrospective finite input-order sensitivity; not a new gate or independent evidence.",
              "panels": panels}
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    os.mkdir(OUT, 0o700)
    fd = os.open(OUT / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    print(json.dumps({"output": str(OUT), "sha256": hashlib.sha256(data).hexdigest(),
                      "changed_runs": {p["label"]: sum(bool(v["changed_strata"]) for v in p["permutations"])
                                       for p in panels}}))


if __name__ == "__main__":
    main()
