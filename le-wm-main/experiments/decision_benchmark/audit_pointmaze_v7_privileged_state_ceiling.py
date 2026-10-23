"""Diagnostic-only privileged-state ceiling for a sealed V7 panel.

It answers one mechanism question after outcomes have been opened: can the
declared 20-step candidate bank be ranked better when the controller receives
the true current PointMaze state?  This arm is never eligible for deployment
or model selection.  A positive state ceiling localizes a V7 failure to visual
state observability; a negative one localizes it to bank/horizon/task geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v7_task_contrast as v7
from experiments.gpu_preflight import require_cuda_recurrent


SCHEMA = "pointmaze_v7_privileged_state_ceiling_audit.v1"
HIDDEN, EPOCHS, SEED = 64, 300, 2026102901


class StateTaskRanker(nn.Module):
    """Upper-bound task ranker with the same pulse encoder as V7."""

    def __init__(self, action_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(4, hidden), nn.SiLU())
        self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, state: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        context = self.state_encoder(state)
        scale, shift = self.film(context).chunk(2, dim=-1)
        action = self.candidate_encoder(candidate)
        return self.head((1.0 + scale) * action + shift).squeeze(-1)


def _fit(source: list[dict[str, Any]], arrays: Any, *, device: str) -> tuple[StateTaskRanker, dict[str, np.ndarray]]:
    _history, candidate, state, _goal = v6._tensors(source)
    cost = v7._source_task_costs(source, arrays)
    v6._seed(SEED)
    sm, ss = state.mean(axis=0), np.maximum(state.std(axis=0), 1.0e-3)
    cm, cs = candidate.mean(axis=(0, 1)), np.maximum(candidate.std(axis=(0, 1)), 1.0e-3)
    ym, ys = cost.mean(), max(float(cost.std()), 1.0e-3)
    model = StateTaskRanker(candidate.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    state_tensor = torch.tensor((state - sm) / ss, device=device)
    candidate_tensor = torch.tensor((candidate - cm) / cs, device=device)
    target = torch.tensor((cost - ym) / ys, device=device)
    groups = v7._root_index_groups(source)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(state_tensor, candidate_tensor)
        loss = (prediction - target).square().mean() + v7._rank_loss(prediction, target, groups)
        loss.backward(); optimizer.step()
    return model.eval(), {"state_mean": sm, "state_scale": ss, "candidate_mean": cm, "candidate_scale": cs, "target_mean": np.asarray(ym, dtype=np.float32), "target_scale": np.asarray(ys, dtype=np.float32)}


def _predict(model: StateTaskRanker, fit: dict[str, np.ndarray], records: list[dict[str, Any]]) -> np.ndarray:
    _history, candidate, state, _goal = v6._tensors(records)
    device = next(model.parameters()).device
    with torch.inference_mode():
        predicted = model(
            torch.tensor((state - fit["state_mean"]) / fit["state_scale"], device=device),
            torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device),
        ).cpu().numpy()
    return np.asarray(predicted * fit["target_scale"] + fit["target_mean"], dtype=np.float64)


def audit(source: Path, source_arrays: Path, context: Path, frozen: Path, outcome: Path, output: Path, *, device: str = "cuda") -> dict[str, Any]:
    if not outcome.is_file():
        raise FileNotFoundError("privileged-state ceiling requires an already sealed V7 outcome artifact")
    runtime = require_cuda_recurrent(device)
    _source_metadata, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    context_metadata, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, context_metadata, strict_counts=True)
    payload = json.loads(outcome.read_text(encoding="utf-8"))
    if payload.get("schema") != "pointmaze_temporal_ood_v7_generated_outcomes.v1" or not isinstance(payload.get("labels"), dict):
        raise ValueError("expected a sealed V7 generated-outcome map")
    recurrent, action, fit = v7._load_model(json.loads(frozen.read_text(encoding="utf-8")), frozen, device=device)
    arrays = np.load(source_arrays, allow_pickle=False)
    state_model, state_fit = _fit(source_rows, arrays, device=device)
    outcome_rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    actual = np.asarray([payload["labels"][v6._key(row)] for row in outcome_rows], dtype=np.float64)
    visual_cost, action_cost = v7._predict(recurrent, action, fit, outcome_rows)
    state_cost = _predict(state_model, state_fit, outcome_rows)
    visual_result = v6._selection(outcome_rows, visual_cost, actual)
    action_result = v6._selection(outcome_rows, action_cost, actual)
    state_result = v6._selection(outcome_rows, state_cost, actual)
    result = {
        "schema": SCHEMA, "diagnostic_only": True, "deployment_eligible": False,
        "source_sha256": v6._sha256(source), "context_sha256": v6._sha256(context), "outcome_sha256": v6._sha256(outcome),
        "state_ranker": state_result, "visual_ranker": visual_result, "action_only": action_result,
        "paired_gain_state_over_action_only": v6._paired(state_result, action_result),
        "paired_gain_state_over_visual": v6._paired(state_result, visual_result),
        "paired_gain_visual_over_action_only": v6._paired(visual_result, action_result),
        "device": str(device), "cuda_preflight": runtime,
        "claim_boundary": "Post-outcome privileged-state ceiling. It diagnoses information bottlenecks and is not an online visual controller or an authorization result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite a privileged-state ceiling audit")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True); parser.add_argument("--source-arrays", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True); parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
