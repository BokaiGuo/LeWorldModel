"""Summarize paired reduced-budget JEPA runner traces."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def load(seed, arm):
    stem = f"pc_jepa_multi_unroll_{arm}_seed{seed}_r3_20261106_codex"
    raw = (ROOT / f"{stem}.trace.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    paths = list((ROOT / f"{stem}.run").rglob("eval.csv"))
    if len(paths) != 1:
        raise ValueError(f"{stem}: expected exactly one eval.csv")
    with paths[0].open() as stream:
        metrics = list(csv.DictReader(stream))
    if len(metrics) != 1:
        raise ValueError(f"{stem}: expected aggregate metrics")
    return raw, rows, metrics[0]


def main():
    output = ROOT / "pc_jepa_multi_unroll_summary_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for seed in (1, 2):
        local_raw, local, local_metric = load(seed, "local")
        pinned_raw, pinned, pinned_metric = load(seed, "pinned")
        counter = Counter(row["kind"] for row in local)
        if counter != Counter(row["kind"] for row in pinned):
            raise ValueError("kind counts differ")
        cases.append({
            "seed": seed, "trace_byte_identical": local_raw == pinned_raw,
            "sha256_local": hashlib.sha256(local_raw).hexdigest(),
            "sha256_pinned": hashlib.sha256(pinned_raw).hexdigest(),
            "record_counts": dict(counter),
            "context_shapes": sorted({tuple(row["context"]["shape"]) for row in local
                                      if row["kind"] == "unroll"}),
            "local_success_rate": float(local_metric["episode_success"]),
            "pinned_success_rate": float(pinned_metric["episode_success"]),
        })
    with output.open("x") as stream:
        json.dump({"cases": cases,
                   "scope": "two 5-episode reduced-budget runner comparisons",
                   "limitations": [
                       "Trace includes expert-reference and policy steps.",
                       "Only unroll differs; other local changes and DINO shim remain.",
                       "Single-frame context, one open-loop plan; no multi-replan parity.",
                       "CEM 8x2 is not paper budget and tasks are development data.",
                   ]}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
