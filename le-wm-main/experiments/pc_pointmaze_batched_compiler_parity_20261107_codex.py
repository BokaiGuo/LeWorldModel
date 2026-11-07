"""State-assisted serial/batched shared-compiler comparison on exposed inputs."""
import json
import time

import numpy as np
import torch
import yaml

from experiments import pc_fixed_prefix_state_adapter_20261107_codex as serial
from experiments import pc_fixed_prefix_state_batched_20261107_codex as batched
from experiments import pc_pointmaze_multimodal_objective_20261107_codex as inputs
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = inputs.ROOT.parent / "docs/PC_POINTMAZE_BATCHED_COMPILER_PARITY_PROTOCOL_20261107_CODEX.md"
OUT = inputs.ROOT / "results/pc_pointmaze_batched_compiler_parity_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model, dataset, audit = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    cfg = yaml.safe_load(inputs.CONFIG.read_text())
    if cfg["planner"]["planning_objective"]["alpha"] != 0.1:
        raise ValueError("unexpected objective")
    fingerprint = model_fingerprint(model)
    rows = []
    with np.load(inputs.PUBLIC, allow_pickle=False) as panel, torch.inference_mode():
        for i in range(3):
            rng = np.random.default_rng(27132000 + i)
            parents = rng.uniform(-.1, .1, (3, 25, 2)).astype(np.float32)
            probes = rng.uniform(-.1, .1, (2, 15, 2)).astype(np.float32)
            tails = rng.uniform(-.1, .1, (2, 15, 2)).astype(np.float32)
            current = {"visual": rgb_tensor(panel["current"][i, 0]),
                       "proprio": torch.as_tensor(panel["proprio"][i, 0], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(panel["goal"][i, 0]),
                    "proprio": torch.as_tensor(panel["goal_proprio"][i, 0], dtype=torch.float32, device="cuda")[None, None]}
            params = dict(mean=dataset.action_mean, std=dataset.action_std,
                          objective_factory=lambda target: ReprTargetDistMPCObjective(cfg, target, alpha=.1),
                          prefix_steps=10, retain_count=1, strategy="lookahead", budget=11, seed=42)

            def execute(kind):
                boundary = StateAssistedPointMaze(model, audit)
                torch.cuda.synchronize()
                started = time.perf_counter()
                if kind == "serial":
                    result = serial.compile_state(boundary, current, goal, parents, probes, tails, **params)
                else:
                    result = batched.compile_state_batched(boundary, current, goal, parents, probes, tails,
                                                           batch_size=4, **params)
                torch.cuda.synchronize()
                return result, time.perf_counter() - started, boundary.receipts

            execute("serial")
            execute("batched")
            trials = []
            for repeat in range(3):
                # Reverse order every other repeat to reduce systematic ordering bias.
                kinds = ("serial", "batched") if repeat % 2 == 0 else ("batched", "serial")
                observations = {kind: execute(kind) for kind in kinds}
                a, at, ar = observations["serial"]
                b, bt, br = observations["batched"]
                errors = {key: float(np.max(np.abs(a[key] - b[key])))
                          for key in ("parent_cost", "probe_cost", "tail_cost")}
                trials.append({"serial_seconds": at, "batched_seconds": bt, "max_score_errors": errors,
                               "same_retention": a["retained_prefixes"] == b["retained_prefixes"],
                               "same_choice": (a["choice"].source, a["choice"].index) == (b["choice"].source, b["choice"].index),
                               "same_actions": bool(np.array_equal(a["actions"], b["actions"])),
                               "serial_calls": a["predict_calls"], "batched_calls": b["predict_calls"],
                               "logical_queries": b["model_queries"], "batch_lengths": b["batch_lengths"],
                               "serial_receipts": ar, "batched_receipts": br})
            rows.append({"root": 1884 + i, "camera": 0, "trials": trials})
            print(f"root={1884+i} done", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    trials = [trial for row in rows for trial in row["trials"]]
    summary = {"max_score_error": max(max(t["max_score_errors"].values()) for t in trials),
               "same_retention": sum(t["same_retention"] for t in trials),
               "same_choice": sum(t["same_choice"] for t in trials),
               "same_actions": sum(t["same_actions"] for t in trials),
               "trials": len(trials),
               "serial_median_seconds": float(np.median([t["serial_seconds"] for t in trials])),
               "batched_median_seconds": float(np.median([t["batched_seconds"] for t in trials]))}
    with OUT.open("x") as f:
        json.dump({"track": STATE, "scope": "state-assisted, exposed inputs, engineering parity",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "serial_sha256": file_hash(serial.__file__), "batched_sha256": file_hash(batched.__file__),
                   "input_sha256": file_hash(inputs.PUBLIC), "model_sha256": fingerprint,
                   "audit_hash": audit.fingerprint, "rows": rows, "summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
