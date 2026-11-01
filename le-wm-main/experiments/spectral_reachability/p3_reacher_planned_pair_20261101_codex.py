"""Two fresh-process, source-bound real Reacher planned-chunk probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import codex_p3_reacher_planned_exclusive_20261101 as planned

ROOT = Path(__file__).resolve().parents[2]
NAME = "p3_reacher_planned_pair_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P3_REACHER_PLANNED_PAIR_20261101_CODEX.md"
POLICY = Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
STATS = ROOT / "results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json"


def digest(path):
    return planned.file_hash(path)


def compare(fixed, dynamic, action_fixed, action_dynamic):
    complete = fixed["failure"] is None and dynamic["failure"] is None
    roots = complete and fixed["root_identity"] == dynamic["root_identity"] and fixed["root_identity"] is not None
    raw_a, raw_b = np.asarray(action_fixed["physical_actions"], np.float32), np.asarray(action_dynamic["physical_actions"], np.float32)
    tokens_a, tokens_b = np.asarray(action_fixed["normalized_tokens"], np.float32), np.asarray(action_dynamic["normalized_tokens"], np.float32)
    candidate = (raw_a.shape == raw_b.shape == (15, 2) and tokens_a.shape == tokens_b.shape
                 and np.array_equal(raw_a, raw_b) and np.array_equal(tokens_a, tokens_b))
    return dict(complete=complete, same_recorded_root=bool(roots), identical_candidate=bool(candidate),
                matched_descriptive_pair=bool(roots and candidate),
                limitation="root state not independently authenticated; no matched total-compute or success claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists():
        raise ValueError("exclusive absolute output")
    files = [Path(__file__).resolve(), DOC, STATS, Path(planned.__file__).resolve(),
             *sorted(p for p in POLICY.rglob("*") if p.is_file())]
    sources = {str(p): digest(p) for p in files}
    out.mkdir()
    protocol = dict(schema=NAME + ".protocol.v1", seed=20261101,
                    thresholds=dict(min_improvement=.01, max_disagreement=1.0),
                    cem=dict(samples=8, steps=2, elite=2), arms=["fixed", "dynamic"],
                    sources=sources, scope="exploratory single chunk, not P3/P1 admission")
    planned.write_once(out / "protocol.json", protocol)
    env = os.environ.copy()
    env.update(MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    outcomes = {}
    for arm in ("fixed", "dynamic"):
        print(json.dumps(dict(arm=arm, status="started")), flush=True)
        command = [sys.executable, "-m", "experiments.spectral_reachability.codex_p3_reacher_planned_exclusive_20261101",
                   "--policy", str(POLICY), "--adapter-artifact", str(STATS),
                   "--output", str(out / arm), "--seed", "20261101", "--arm", arm,
                   "--min-improvement", "0.01", "--max-disagreement", "1.0",
                   "--cem-samples", "8", "--cem-steps", "2", "--cem-elite", "2"]
        try:
            proc = subprocess.run(command, cwd=ROOT, env=env, timeout=1200, check=False)
            outcomes[arm] = dict(returncode=proc.returncode)
        except subprocess.TimeoutExpired:
            outcomes[arm] = dict(returncode=None, error="timeout")
        print(json.dumps(dict(arm=arm, **outcomes[arm])), flush=True)
    receipts = {}
    for arm in outcomes:
        path = out / arm / "receipt.json"
        if path.is_file():
            receipts[arm] = json.loads(path.read_bytes())
            outcomes[arm]["receipt_sha256"] = digest(path)
    result = None
    if (len(receipts) == 2 and all((out / arm / "action_binding.json").is_file() for arm in outcomes)):
        actions = {arm: json.loads((out / arm / "action_binding.json").read_bytes()) for arm in outcomes}
        result = compare(receipts["fixed"], receipts["dynamic"], actions["fixed"], actions["dynamic"])
    unchanged = all(digest(path) == value for path, value in sources.items())
    summary = dict(schema=NAME + ".report.v1", outcomes=outcomes, pair=result, sources_unchanged=unchanged,
                   P1_admission=False, P3_admission=False, equal_compute=False)
    planned.write_once(out / "report.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not unchanged or any(item["returncode"] != 0 for item in outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
