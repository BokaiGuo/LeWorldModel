"""Compare serial and batched retention decisions with two retained prefixes."""
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

PROTOCOL = inputs.ROOT.parent / "docs/PC_POINTMAZE_BATCHED_VECTOR_PARITY_PROTOCOL_20261107_CODEX.md"
OUT = inputs.ROOT / "results/pc_pointmaze_batched_vector_parity_20261107_codex.json"


def snapshot(result, seconds, receipts):
    return {"parent_cost": result["parent_cost"].tolist(),
            "probe_cost": result["probe_cost"].tolist(), "tail_cost": result["tail_cost"].tolist(),
            "retained": result["retained_prefixes"], "actions": result["actions"].tolist(),
            "choice": {"source": result["choice"].source, "index": result["choice"].index,
                       "cost": result["choice"].cost},
            "logical_queries": result["model_queries"], "predict_calls": result["predict_calls"],
            "batch_lengths": result.get("batch_lengths"), "seconds": seconds, "receipts": receipts}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model, dataset, audit = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(inputs.CONFIG.read_text())
    if cfg["planner"]["planning_objective"]["alpha"] != .1:
        raise ValueError("unexpected objective")
    fingerprint = model_fingerprint(model)
    rows = []
    with np.load(inputs.PUBLIC, allow_pickle=False) as panel, torch.inference_mode():
        for i in range(3):
            rng = np.random.default_rng(27143000 + i)
            parents = rng.uniform(-.1, .1, (3, 25, 2)).astype(np.float32)
            probes = rng.uniform(-.1, .1, (4, 15, 2)).astype(np.float32)
            tails = rng.uniform(-.1, .1, (2, 15, 2)).astype(np.float32)
            current = {"visual": rgb_tensor(panel["current"][i, 0]),
                       "proprio": torch.as_tensor(panel["proprio"][i, 0], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(panel["goal"][i, 0]),
                    "proprio": torch.as_tensor(panel["goal_proprio"][i, 0], dtype=torch.float32, device="cuda")[None, None]}
            methods = {}
            for strategy in ("parent", "lookahead", "vector", "shuffled"):
                common = dict(mean=dataset.action_mean, std=dataset.action_std,
                              objective_factory=lambda target: ReprTargetDistMPCObjective(cfg, target, alpha=.1),
                              prefix_steps=10, retain_count=2, strategy=strategy, budget=19, seed=27144000 + i)
                results = {}
                for name in ("serial", "batched"):
                    boundary = StateAssistedPointMaze(model, audit)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if name == "serial":
                        value = serial.compile_state(boundary, current, goal, parents, probes, tails, **common)
                    else:
                        value = batched.compile_state_batched(boundary, current, goal, parents, probes, tails,
                                                             batch_size=4, **common)
                    torch.cuda.synchronize()
                    results[name] = snapshot(value, time.perf_counter() - start, boundary.receipts)
                first, second = results["serial"], results["batched"]
                same_retention = first["retained"] == second["retained"]
                diffs = {key: float(np.max(np.abs(np.asarray(first[key]) - np.asarray(second[key]))))
                         for key in ("parent_cost", "probe_cost")}
                diffs["tail_cost"] = (float(np.max(np.abs(np.asarray(first["tail_cost"]) - np.asarray(second["tail_cost"]))))
                                      if same_retention else None)
                methods[strategy] = {"serial": first, "batched": second, "score_errors": diffs,
                                     "same_retention": same_retention,
                                     "same_choice": (first["choice"]["source"], first["choice"]["index"])
                                                    == (second["choice"]["source"], second["choice"]["index"]),
                                     "same_actions": first["actions"] == second["actions"]}
            rows.append({"root": 1884 + i, "methods": methods})
            print(f"root={1884+i} finished", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    cases = [m for row in rows for m in row["methods"].values()]
    summary = {"cases": len(cases),
               "max_score_error": max(v for x in cases for v in x["score_errors"].values() if v is not None),
               "same_retention": sum(x["same_retention"] for x in cases),
               "same_choice": sum(x["same_choice"] for x in cases),
               "same_actions": sum(x["same_actions"] for x in cases)}
    summary["panel_decision_parity"] = (summary["max_score_error"] <= 1e-5
                                        and summary["same_retention"] == len(cases)
                                        and summary["same_choice"] == len(cases)
                                        and summary["same_actions"] == len(cases))
    with OUT.open("x") as f:
        json.dump({"track": STATE, "scope": "exposed state-assisted engineering parity",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "serial_sha256": file_hash(serial.__file__), "batched_sha256": file_hash(batched.__file__),
                   "input_sha256": file_hash(inputs.PUBLIC), "model_sha256": fingerprint,
                   "rows": rows, "summary": summary}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
