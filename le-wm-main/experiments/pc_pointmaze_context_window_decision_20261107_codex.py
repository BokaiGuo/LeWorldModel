"""Compare local context-1 and config-bound context-2 action decisions."""
import json

import numpy as np
import torch
import yaml

from experiments import pc_fixed_prefix_state_batched_20261107_codex as adapter
from experiments import pc_pointmaze_multimodal_objective_20261107_codex as inputs
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = inputs.ROOT.parent / "docs/PC_POINTMAZE_CONTEXT_WINDOW_DECISION_PROTOCOL_20261107_CODEX.md"
OUT = inputs.ROOT / "results/pc_pointmaze_context_window_decision_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model, dataset, audit1 = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(inputs.CONFIG.read_text())
    if cfg["task_specification"]["num_frames"] != 1 or cfg["planner"]["planning_objective"]["alpha"] != .1:
        raise ValueError("unexpected evaluation configuration")
    fingerprint = model_fingerprint(model)
    if model.ctxt_window != 1:
        raise ValueError("old loader no longer context 1")
    audit2 = configure_context(model, audit1)
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
            means = dataset.action_mean.to("cuda")
            stds = dataset.action_std.to("cuda")
            packets = ((torch.as_tensor(parents[0], device="cuda") - means) / stds).reshape(5, 1, 10)
            predictions = {}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                output = boundary.predict(current, packets)
                predictions[window] = {key: output[key].cpu().numpy() for key in ("visual", "proprio")}
            per_step = {key: [float(np.max(np.abs(predictions[1][key][step] - predictions[2][key][step])))
                              for step in range(6)] for key in ("visual", "proprio")}
            if any(per_step[key][1] > 1e-5 for key in per_step):
                raise ValueError(f"first packet mismatch: {per_step}")
            methods = {}
            for strategy in ("parent", "lookahead", "vector", "shuffled"):
                results = {}
                for window, audit in ((1, audit1), (2, audit2)):
                    model.ctxt_window = window
                    boundary = StateAssistedPointMaze(model, audit)
                    result = adapter.compile_state_batched(
                        boundary, current, goal, parents, probes, tails,
                        mean=dataset.action_mean, std=dataset.action_std,
                        objective_factory=lambda target: ReprTargetDistMPCObjective(cfg, target, alpha=.1),
                        prefix_steps=10, retain_count=2, strategy=strategy,
                        budget=19, seed=27144000 + i, batch_size=4)
                    results[str(window)] = {
                        "parent_cost": result["parent_cost"].tolist(), "probe_cost": result["probe_cost"].tolist(),
                        "tail_cost": result["tail_cost"].tolist(), "retained": result["retained_prefixes"],
                        "actions": result["actions"].tolist(),
                        "choice": {"source": result["choice"].source, "index": result["choice"].index,
                                   "cost": result["choice"].cost},
                        "queries": result["model_queries"], "predict_calls": result["predict_calls"],
                        "audit_fingerprint": audit.fingerprint, "receipts": boundary.receipts}
                a, b = results["1"], results["2"]
                methods[strategy] = {"windows": results, "retention_changed": a["retained"] != b["retained"],
                                     "actions_changed": a["actions"] != b["actions"],
                                     "max_shared_score_diff": max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
                                                                  for k in ("parent_cost", "probe_cost"))}
            rows.append({"root": 1884 + i, "per_step_feature_max_abs_diff": per_step, "methods": methods})
            print(f"root={1884+i} completed", flush=True)
    model.ctxt_window = 2
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    cases = [x for row in rows for x in row["methods"].values()]
    summary = {"conditions": len(cases), "retention_changed": sum(x["retention_changed"] for x in cases),
               "actions_changed": sum(x["actions_changed"] for x in cases),
               "max_shared_score_diff": max(x["max_shared_score_diff"] for x in cases),
               "max_first_packet_feature_diff": max(row["per_step_feature_max_abs_diff"][key][1]
                                                    for row in rows for key in ("visual", "proprio"))}
    with OUT.open("x") as f:
        json.dump({"track": STATE, "scope": "exposed state-assisted context ablation, not official full evaluation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "config_sha256": file_hash(inputs.CONFIG), "input_sha256": file_hash(inputs.PUBLIC),
                   "model_sha256": fingerprint, "rows": rows, "summary": summary}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
