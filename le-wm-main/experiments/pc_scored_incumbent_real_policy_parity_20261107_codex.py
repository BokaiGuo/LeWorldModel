"""Real PushT closed-loop parity of historical and fail-closed solver outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/pc_scored_incumbent_real_policy_parity_20261107_codex"
DATA = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
PROTOCOL = ROOT.parent / "docs/PC_SCORED_INCUMBENT_REAL_POLICY_PARITY_PROTOCOL_20261107_CODEX.md"
OLD = "experiments.pc_explicit_prefix_scored_incumbent_20261107_codex.ExplicitPrefixScoredIncumbent"
NEW = "experiments.pc_explicit_prefix_output_contract_20261107_codex.ExplicitPrefixSelectedActionOnly"


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def differences(a, b, path, result):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key == "elapsed_seconds":
                continue
            if key not in a or key not in b:
                result.append(f"{path}.{key}: missing")
            else:
                differences(a[key], b[key], f"{path}.{key}", result)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            result.append(f"{path}: different lengths {len(a)} {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                differences(x, y, f"{path}[{i}]", result)
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        tol = 1e-5 if any(f".{part}[" in path for part in ("before", "after", "observed_state")) else 0
        if not np.isclose(a, b, atol=tol, rtol=0):
            result.append(f"{path}: {a} != {b}")
    elif a != b:
        result.append(f"{path}: {a!r} != {b!r}")


def main():
    if BASE.exists():
        raise FileExistsError(BASE)
    BASE.mkdir()
    with h5py.File(DATA, "r") as data:
        actions = data["action"][:]
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid action scaler")
    mean = "[" + ",".join(repr(float(v)) for v in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(v)) for v in scaler.scale_) + "]"
    print("Run paired four-task real PushT policy parity; this is not a performance claim.", flush=True)
    for name, solver in (("historical", OLD), ("action_only", NEW)):
        trace, audit, output = BASE / f"{name}.trace.jsonl", BASE / f"{name}.audit.jsonl", BASE / name
        env = os.environ.copy()
        env.update(PC_PHYSICS_RESET_SEED="871001", PC_PHYSICS_TRACE_PATH=str(trace))
        subprocess.run(
            [sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
             "policy=pusht/lewm", "seed=85", "solver.seed=42", "eval.num_eval=4",
             f"solver._target_={solver}", "+solver.prefix_mode=lookahead",
             "solver.num_samples=100", "solver.n_steps=3", "solver.topk=10",
             f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
             f"+solver.audit_path={audit}", f"output.dir={output}"],
            cwd=ROOT, env=env, check=True)
    results = []
    audit_old, audit_new = (records(BASE / f"{name}.audit.jsonl") for name in ("historical", "action_only"))
    trace_old, trace_new = (records(BASE / f"{name}.trace.jsonl") for name in ("historical", "action_only"))
    trace_old.sort(key=lambda x: (x["reset_seed"], x["step"]))
    trace_new.sort(key=lambda x: (x["reset_seed"], x["step"]))
    differences(audit_old, audit_new, "audit", results)
    differences(trace_old, trace_new, "trace", results)
    metrics_old = json.loads((BASE / "historical/eval_metrics.json").read_text())
    metrics_new = json.loads((BASE / "action_only/eval_metrics.json").read_text())
    differences(metrics_old, metrics_new, "metrics", results)
    report = {"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "historical_audit_sha256": file_hash(BASE / "historical.audit.jsonl"),
              "action_only_audit_sha256": file_hash(BASE / "action_only.audit.jsonl"),
              "tasks": 4, "steps_historical": len(trace_old), "steps_action_only": len(trace_new),
              "decisions": sum(len(x["environments"]) for x in audit_old),
              "difference_count": len(results), "differences": results}
    with (BASE / "comparison.json").open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
    print(f"Comparison found {len(results)} differences; see {BASE / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
