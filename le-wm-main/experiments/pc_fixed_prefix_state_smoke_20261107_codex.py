"""Run the shared fixed-prefix core with frozen state-assisted PointMaze JEPA."""
import json

import numpy as np
import torch
import yaml

from experiments import pc_fixed_prefix_shared_20261107_codex as core
from experiments import pc_fixed_prefix_state_adapter_20261107_codex as adapter
from experiments import pc_pointmaze_multimodal_objective_20261107_codex as inputs
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

OUT = inputs.ROOT / "results/pc_fixed_prefix_state_smoke_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model, dataset, audit = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    config = yaml.safe_load(inputs.CONFIG.read_text())
    if config["planner"]["planning_objective"]["alpha"] != 0.1:
        raise ValueError("unexpected official objective configuration")
    before = model_fingerprint(model)
    boundary = StateAssistedPointMaze(model, audit)
    rng = np.random.default_rng(271101)
    parents = rng.uniform(-0.1, 0.1, (3, 25, 2)).astype(np.float32)
    probes = rng.uniform(-0.1, 0.1, (2, 15, 2)).astype(np.float32)
    tails = rng.uniform(-0.1, 0.1, (2, 15, 2)).astype(np.float32)
    with np.load(inputs.PUBLIC, allow_pickle=False) as panel, torch.inference_mode():
        current = {"visual": rgb_tensor(panel["current"][0, 0]),
                   "proprio": torch.as_tensor(panel["proprio"][0, 0], dtype=torch.float32, device="cuda")[None, None]}
        goal = {"visual": rgb_tensor(panel["goal"][0, 0]),
                "proprio": torch.as_tensor(panel["goal_proprio"][0, 0], dtype=torch.float32, device="cuda")[None, None]}
        result = adapter.compile_state(
            boundary, current, goal, parents, probes, tails, mean=dataset.action_mean, std=dataset.action_std,
            objective_factory=lambda target: ReprTargetDistMPCObjective(config, target, alpha=0.1),
            prefix_steps=10, retain_count=1, strategy="lookahead", budget=11, seed=42)
    if model_fingerprint(model) != before:
        raise ValueError("model weights changed")
    record = {"track": STATE, "root": 1884, "camera": 0,
              "scope": "state-assisted engineering smoke; full goal and current proprio; not RGB evidence",
              "input_sha256": file_hash(inputs.PUBLIC), "config_sha256": file_hash(inputs.CONFIG),
              "shared_core_sha256": file_hash(core.__file__), "adapter_sha256": file_hash(adapter.__file__),
              "model_sha256": before, "audit_hash": audit.fingerprint,
              "actions": result["actions"].tolist(),
              "choice": {"source": result["choice"].source, "index": result["choice"].index,
                         "cost": result["choice"].cost},
              "retained_prefixes": result["retained_prefixes"], "logical_queries": result["model_queries"],
              "predict_calls": result["predict_calls"], "goal_encoding_calls": result["goal_encoding_calls"],
              "receipts": boundary.receipts}
    with OUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({k: record[k] for k in ("track", "choice", "logical_queries", "predict_calls",
                                             "goal_encoding_calls")}, indent=2))


if __name__ == "__main__":
    main()
