"""Independent developmental selector and fixed-pool retrospective sensitivity."""
from collections import defaultdict
import hashlib
import json
import math
import os

import numpy as np

from . import legacy_p1_failure_attribution_20261102_exclusive as source
from . import run_cross_environment_transfer_diagnostic as cross
from . import run_matched_learned_transfer_benchmark as old

OUT = source.RESULTS / "legacy_p1_stable_rank_dev_20261103_codex_RUN01"
METRICS = ("prediction_mse", "action_inverse_risk", *old.METRICS)


def select(records):
    """Select from a fixed pool; no held-out values participate."""
    if not records:
        raise ValueError("empty candidate pool")
    ids = [r.get("checkpoint") for r in records]
    if any(not isinstance(x, str) or not x or x != x.strip() for x in ids):
        raise ValueError("invalid checkpoint identity")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate checkpoint identity")
    try:
        values = np.asarray([[float(r["diagnostics"][m]) for m in METRICS]
                             for r in records], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid metric") from exc
    if values.shape != (len(records), len(METRICS)) or not np.isfinite(values).all():
        raise ValueError("invalid or non-finite metrics")
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        start = 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and values[order[end], column] == values[order[start], column]:
                end += 1
            ranks[order[start:end], column] = (start + end - 1) / 2
            start = end
    scores = ranks.mean(axis=1)
    full = min(range(len(ids)), key=lambda i: (scores[i], ids[i]))
    mse = min(range(len(ids)), key=lambda i: (values[i, 0], ids[i]))
    return {"full": ids[full], "mse": ids[mse],
            "scores": {ids[i]: float(scores[i]) for i in range(len(ids))},
            "ranks": {ids[i]: ranks[i].tolist() for i in range(len(ids))}}


def panel(paths, historical, label, seed):
    before = [source.file_record(p) for p in paths]
    records, failures = cross._load_all(paths, require_external=True)
    if failures or len(records) != len(paths):
        raise ValueError(f"incomplete input: {failures}")
    grouped = defaultdict(dict)
    for record in records:
        key = old._stratum_key(record)
        identity = record.get("checkpoint")
        if not isinstance(identity, str) or not identity or identity != identity.strip() or identity in grouped[key]:
            raise ValueError(f"invalid or duplicate identity in {key}")
        grouped[key][identity] = record
    rows, deltas, saved_deltas = [], [], []
    expected_keys = set()
    for row in historical["strata"]:
        key = (row["environment"], row["training_stage"], row["latent_dimension"], row["action_distribution"])
        if key in expected_keys:
            raise ValueError("duplicate saved stratum")
        expected_keys.add(key)
        eligible = row["eligible_checkpoints"]
        if len(set(eligible)) != len(eligible) or not set(eligible) <= grouped[key].keys():
            raise ValueError("saved pool cannot be reconstructed")
        if not row["identifiable"]:
            continue
        candidates = [grouped[key][identity] for identity in eligible]
        picked = select(candidates)
        reversed_pick = select(list(reversed(candidates)))
        if picked != reversed_pick:
            raise ValueError("selector depends on input order")
        scale = float(row["planning_risk_scale_train_only"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid saved risk scale")
        by_id = grouped[key]
        risk = lambda name: float(by_id[name]["heldout"]["planning_risk"]) / scale
        delta = risk(picked["mse"]) - risk(picked["full"])
        if not math.isfinite(delta):
            raise ValueError("invalid risk")
        saved_delta = float(row["delta_risk_mse_minus_full"])
        saved_mse = row["selectors"]["mse_only"]["checkpoint"]
        saved_full = row["selectors"]["full_mse_action_risk_geometry"]["checkpoint"]
        if saved_mse not in eligible or saved_full not in eligible or not math.isclose(
                risk(saved_mse) - risk(saved_full), saved_delta, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("saved selector or delta does not reproduce")
        rows.append({"stratum": list(key), "saved_mse": saved_mse, "saved_full": saved_full,
                     "dev_mse": picked["mse"], "dev_full": picked["full"],
                     "saved_delta": saved_delta, "dev_delta": delta, "dev_scores": picked["scores"]})
        saved_deltas.append(saved_delta)
        deltas.append(delta)
    if expected_keys != set(grouped):
        raise ValueError("stratum mismatch")
    if before != [source.file_record(p) for p in paths]:
        raise ValueError("input changed")
    def stats(values):
        result = old._positive_delta_test(values, seed + 500)
        result.pop("positive_gate")
        return result
    saved_stats = stats(saved_deltas)
    expected_stats = dict(historical["summary"]["primary_delta_risk_mse_minus_full"])
    expected_stats.pop("positive_gate")
    if saved_stats != expected_stats:
        raise ValueError("saved primary statistics do not reproduce")
    return {"label": label, "inputs": before, "rows": rows,
            "saved_statistics": saved_stats, "development_statistics": stats(deltas)}


def main():
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(OUT)
    reports = [source.OLD, source.EXTERNAL]
    hashes = [source.file_record(p) for p in reports]
    historic, external = [json.loads(p.read_bytes()) for p in reports]
    paths_a = sorted(p for root in source.OLD_ROOTS for p in root.rglob("checkpoint_record.json"))
    paths_b = [source.ROOT / p for p in external["input"]["paths"]]
    panels = [panel(paths_a, historic, "provisional", 20261011),
              panel(paths_b, external["global_audit"], "external", 20261013)]
    if hashes != [source.file_record(p) for p in reports]:
        raise ValueError("historical report changed")
    data = (json.dumps({"scope": "retrospective development only; no gate or independent confirmation",
                        "reports": hashes, "panels": panels}, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    os.mkdir(OUT, 0o700)
    fd = os.open(OUT / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    print(json.dumps({"output": str(OUT), "sha256": hashlib.sha256(data).hexdigest(),
                      "statistics": {p["label"]: p["development_statistics"] for p in panels}}))


if __name__ == "__main__":
    main()
