"""Task-aligned V7 PointMaze ranking experiment.

V6 learns successor *effects* with MSE and turns them into a task order only
afterwards.  This runner tests the narrower, falsifiable repair that matters
for planning: learn the within-root ordering of candidate task costs directly
from source-only realised successors.  It deliberately retains an action-only
ranker with the same candidate encoder and parameter budget.

The command is sealed by default.  Calibration uses only calibration labels,
and ``score`` refuses an output directory that already contains outcomes.  A
debug replay over an already-opened panel requires ``--debug-opened-outcomes``
and is written as non-confirmatory evidence.
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
from experiments.gpu_preflight import require_cuda_recurrent


FROZEN_SCHEMA = "pointmaze_temporal_ood_v7_task_contrast_frozen.v1"
MODEL_SCHEMA = "pointmaze_temporal_ood_v7_task_contrast_model.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v7_task_contrast_score.v1"
TRAIN_SEED = 2026102701
HIDDEN_WIDTHS = (32, 64)
MARGIN_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)


class HistoryTaskContrastRanker(nn.Module):
    """History-conditioned scalar task-cost ranker with FiLM action effects."""

    def __init__(self, history_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.history_encoder = nn.Sequential(nn.Linear(history_dim, hidden), nn.SiLU())
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, history: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        _, final = self.history_gru(self.history_encoder(history))
        memory = final[-1]
        scale, shift = self.film(memory).chunk(2, dim=-1)
        action = self.candidate_encoder(candidate)
        return self.head((1.0 + scale) * action + shift).squeeze(-1)


class ActionOnlyTaskContrastRanker(nn.Module):
    """Matched task ranker that cannot use visual or action history."""

    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.head(self.candidate_encoder(candidate)).squeeze(-1)


def _normalizer(value: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return value.mean(axis=axes), np.maximum(value.std(axis=axes), 1.0e-3)


def _seed(seed: int) -> None:
    v6._seed(seed)


def _source_task_costs(source: list[dict[str, Any]], arrays: Any) -> np.ndarray:
    terminal = np.asarray([arrays[v6._array_key(row["realized_terminal_state_ref"])] for row in source], dtype=np.float32)
    goals = np.asarray([row["goal_raw"][:2] for row in source], dtype=np.float32)
    return np.linalg.norm(terminal[:, :2] - goals, axis=1).astype(np.float32)


def _root_index_groups(records: list[dict[str, Any]]) -> list[np.ndarray]:
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        positions.setdefault(str(row["instance_id"]), []).append(index)
    result = []
    for root, indices in sorted(positions.items()):
        ordered = sorted(indices, key=lambda index: int(records[index]["candidate_id"]))
        candidate_ids = [int(records[index]["candidate_id"]) for index in ordered]
        if candidate_ids != list(range(len(candidate_ids))):
            raise ValueError(f"task-contrast source root {root!r} has non-contiguous candidates")
        result.append(np.asarray(ordered, dtype=np.int64))
    if not result or len({len(value) for value in result}) != 1:
        raise ValueError("task-contrast source needs equal non-empty candidate banks")
    return result


def _rank_loss(prediction: torch.Tensor, target: torch.Tensor, groups: list[np.ndarray]) -> torch.Tensor:
    """Pairwise logistic loss: lower realised cost must obtain lower score."""
    losses = []
    for group in groups:
        index = torch.as_tensor(group, dtype=torch.long, device=prediction.device)
        predicted, actual = prediction[index], target[index]
        left, right = torch.triu_indices(len(group), len(group), offset=1, device=prediction.device)
        signed_margin = torch.sign(actual[left] - actual[right]) * (predicted[left] - predicted[right])
        # Ties contain no ordering information and must not push arbitrary ranks.
        informative = torch.abs(actual[left] - actual[right]) > 1.0e-6
        if bool(informative.any()):
            losses.append(nn.functional.softplus(-signed_margin[informative]).mean())
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _fit(
    source: list[dict[str, Any]], arrays: Any, *, hidden: int, epochs: int, device: str,
) -> tuple[HistoryTaskContrastRanker, ActionOnlyTaskContrastRanker, dict[str, np.ndarray]]:
    history, candidate, _state, _goal = v6._tensors(source)
    cost = _source_task_costs(source, arrays)
    _seed(TRAIN_SEED)
    hm, hs = _normalizer(history, (0, 1))
    cm, cs = _normalizer(candidate, (0, 1))
    ym, ys = _normalizer(cost, (0,))
    recurrent = HistoryTaskContrastRanker(history.shape[-1], candidate.shape[-1], hidden).to(device)
    action = ActionOnlyTaskContrastRanker(candidate.shape[-1], hidden).to(device)
    recurrent_optimizer = torch.optim.AdamW(recurrent.parameters(), lr=2e-3, weight_decay=1e-4)
    action_optimizer = torch.optim.AdamW(action.parameters(), lr=2e-3, weight_decay=1e-4)
    history_tensor = torch.tensor((history - hm) / hs, device=device)
    candidate_tensor = torch.tensor((candidate - cm) / cs, device=device)
    target = torch.tensor((cost - ym) / ys, device=device)
    groups = _root_index_groups(source)
    for _ in range(int(epochs)):
        recurrent_optimizer.zero_grad(set_to_none=True)
        recurrent_prediction = recurrent(history_tensor, candidate_tensor)
        recurrent_loss = (recurrent_prediction - target).square().mean() + _rank_loss(recurrent_prediction, target, groups)
        recurrent_loss.backward(); recurrent_optimizer.step()
        action_optimizer.zero_grad(set_to_none=True)
        action_prediction = action(candidate_tensor)
        action_loss = (action_prediction - target).square().mean() + _rank_loss(action_prediction, target, groups)
        action_loss.backward(); action_optimizer.step()
    return recurrent.eval(), action.eval(), {
        "history_mean": hm, "history_scale": hs, "candidate_mean": cm, "candidate_scale": cs,
        "target_mean": np.asarray(ym, dtype=np.float32), "target_scale": np.asarray(ys, dtype=np.float32),
    }


def _predict(
    recurrent: HistoryTaskContrastRanker, action: ActionOnlyTaskContrastRanker,
    fit: dict[str, np.ndarray], records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    history, candidate, _state, _goal = v6._tensors(records)
    device = next(recurrent.parameters()).device
    with torch.inference_mode():
        history_tensor = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        candidate_tensor = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        recurrent_cost = recurrent(history_tensor, candidate_tensor).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
        action_cost = action(candidate_tensor).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
    return np.asarray(recurrent_cost, dtype=np.float64), np.asarray(action_cost, dtype=np.float64)


def _margin_gated_cost(
    records: list[dict[str, Any]], recurrent_cost: np.ndarray, action_cost: np.ndarray, *, threshold: float,
) -> np.ndarray:
    """Use the history ranker only for roots with a frozen, confident ranking.

    The gate is root-level: all candidates of a root are routed together.  It
    therefore cannot become a hidden per-candidate selector.  A threshold of
    zero is the unconditional recurrent ranker; a very large threshold is the
    action-only carrier.
    """
    recurrent_cost, action_cost = np.asarray(recurrent_cost), np.asarray(action_cost)
    if recurrent_cost.shape != action_cost.shape or recurrent_cost.shape != (len(records),):
        raise ValueError("task-contrast gate costs must align with records")
    routed = action_cost.copy()
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        positions.setdefault(str(row["instance_id"]), []).append(index)
    for root, indices in positions.items():
        ordered = sorted(indices, key=lambda index: int(records[index]["candidate_id"]))
        values = np.sort(recurrent_cost[ordered])
        if len(values) < 2:
            raise ValueError(f"task-contrast root {root!r} needs at least two candidates")
        margin = float(values[1] - values[0])
        if margin >= float(threshold):
            routed[ordered] = recurrent_cost[ordered]
    return routed


def _policy_cost(
    records: list[dict[str, Any]], recurrent_cost: np.ndarray, action_cost: np.ndarray, policy: dict[str, Any],
) -> np.ndarray:
    kind = str(policy.get("kind", ""))
    if kind == "action_only":
        return np.asarray(action_cost, dtype=np.float64)
    if kind == "recurrent":
        return np.asarray(recurrent_cost, dtype=np.float64)
    if kind == "margin_gate":
        return _margin_gated_cost(records, recurrent_cost, action_cost, threshold=float(policy["threshold"]))
    raise ValueError(f"unsupported task-contrast policy {kind!r}")


def _model_path(frozen_path: Path) -> Path:
    return frozen_path.with_name(f"{frozen_path.stem}_model.pt")


def _write_model(
    path: Path, recurrent: HistoryTaskContrastRanker, action: ActionOnlyTaskContrastRanker,
    fit: dict[str, np.ndarray], *, hidden: int,
) -> str:
    payload = {
        "schema": MODEL_SCHEMA, "hidden": int(hidden),
        "history_dim": int(fit["history_mean"].shape[-1]), "action_dim": int(fit["candidate_mean"].shape[-1]),
        "recurrent_state_dict": {key: value.detach().cpu() for key, value in recurrent.state_dict().items()},
        "action_state_dict": {key: value.detach().cpu() for key, value in action.state_dict().items()},
        "normalizer": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in fit.items()},
        "runner_source_sha256": v6._sha256(Path(__file__)),
    }
    torch.save(payload, path)
    return v6._sha256(path)


def _load_model(frozen: dict[str, Any], frozen_path: Path, *, device: str) -> tuple[HistoryTaskContrastRanker, ActionOnlyTaskContrastRanker, dict[str, np.ndarray]]:
    path = Path(str(frozen.get("model_checkpoint_path", "")))
    if not path.is_file():
        path = _model_path(frozen_path)
    expected = str(frozen.get("model_checkpoint_sha256", ""))
    if not expected or not path.is_file() or v6._sha256(path) != expected:
        raise ValueError("task-contrast frozen checkpoint is missing or hash-mismatched")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA or payload.get("runner_source_sha256") != v6._sha256(Path(__file__)):
        raise ValueError("task-contrast model has an incompatible schema or runner")
    recurrent = HistoryTaskContrastRanker(int(payload["history_dim"]), int(payload["action_dim"]), int(payload["hidden"])).to(device)
    action = ActionOnlyTaskContrastRanker(int(payload["action_dim"]), int(payload["hidden"])).to(device)
    recurrent.load_state_dict(payload["recurrent_state_dict"], strict=True)
    action.load_state_dict(payload["action_state_dict"], strict=True)
    normalizer = payload.get("normalizer")
    required = {"history_mean", "history_scale", "candidate_mean", "candidate_scale", "target_mean", "target_scale"}
    if not isinstance(normalizer, dict) or set(normalizer) != required:
        raise ValueError("task-contrast model lacks a complete normalizer")
    fit = {key: value.detach().cpu().numpy().astype(np.float32, copy=False) for key, value in normalizer.items()}
    return recurrent.eval(), action.eval(), fit


def _calibration_costs(path: Path, records: list[dict[str, Any]]) -> np.ndarray:
    return v6._calibration_costs(path, records, require_identity=True)


def _ensure_fresh_calibration_output(context: Path, output: Path) -> None:
    """Forbid refitting after either a frozen fit or a held outcome exists."""
    outcome = context.parent / "temporal_ood_v6_outcome_labels.json"
    model = _model_path(output)
    if output.exists() or model.exists() or outcome.exists():
        raise FileExistsError(
            "task-contrast calibration is sealed: an output/model or held outcome already exists"
        )


def calibrate(
    source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *,
    epochs: int = 300, device: str = "cuda", manifest: Path | None = None,
) -> dict[str, Any]:
    _ensure_fresh_calibration_output(context, output)
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    cuda = require_cuda_recurrent(device)
    _source_metadata, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    context_metadata, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, context_metadata, strict_counts=True)
    calibration_rows = [row for row in context_rows if str(row["split"]) == "calibration"]
    labels = _calibration_costs(calibration_labels, calibration_rows)
    cost_by_key = {v6._key(row): value for row, value in zip(calibration_rows, labels)}
    partition = v6.deterministic_calibration_split(context_rows)
    selection_roots, authorization_roots = set(partition["selection"]), set(partition["authorization"])
    select = [row for row in calibration_rows if str(row["instance_id"]) in selection_roots]
    authorize = [row for row in calibration_rows if str(row["instance_id"]) in authorization_roots]
    arrays = np.load(source_arrays, allow_pickle=False)
    choices: list[dict[str, Any]] = []
    for hidden in HIDDEN_WIDTHS:
        recurrent, action, fit = _fit(source_rows, arrays, hidden=hidden, epochs=epochs, device=device)
        recurrent_cost, action_cost = _predict(recurrent, action, fit, select)
        actual = np.asarray([cost_by_key[v6._key(row)] for row in select])
        for policy in ([{"kind": "action_only"}, {"kind": "recurrent"}] + [
            {"kind": "margin_gate", "threshold": float(threshold)} for threshold in MARGIN_THRESHOLDS if threshold > 0.0
        ]):
            selected = v6._selection(select, _policy_cost(select, recurrent_cost, action_cost, policy), actual)
            choices.append({"hidden": hidden, "policy": policy, "selected": selected})
    chosen = min(
        choices,
        key=lambda value: (value["selected"]["mean_regret"], value["hidden"], 0 if value["policy"]["kind"] == "action_only" else 1),
    )
    recurrent, action, fit = _fit(source_rows, arrays, hidden=int(chosen["hidden"]), epochs=epochs, device=device)
    recurrent_cost, action_cost = _predict(recurrent, action, fit, authorize)
    actual = np.asarray([cost_by_key[v6._key(row)] for row in authorize])
    action_result = v6._selection(authorize, action_cost, actual)
    selected_result = v6._selection(authorize, _policy_cost(authorize, recurrent_cost, action_cost, chosen["policy"]), actual)
    gain = v6._paired(selected_result, action_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(output)
    model_sha256 = _write_model(model_path, recurrent, action, fit, hidden=int(chosen["hidden"]))
    report = {
        "schema": FROZEN_SCHEMA, "objective": "source_successor_task_cost_pairwise_ranking",
        "source_sha256": v6._sha256(source), "source_arrays_sha256": v6._sha256(source_arrays),
        "context_sha256": v6._sha256(context), "manifest_sha256": v6._sha256(manifest),
        "calibration_labels_sha256": v6._sha256(calibration_labels), "calibration_labels_path": str(calibration_labels.resolve()),
        "calibration_partition": partition, "choice_grid": choices, "selected_hidden": int(chosen["hidden"]), "selected_policy": chosen["policy"],
        "model_checkpoint_path": str(model_path.resolve()), "model_checkpoint_sha256": model_sha256,
        "authorization": {"selected_policy_over_action_only": gain, "permitted": bool(chosen["policy"]["kind"] != "action_only" and gain["lower"] > 0.0), "criterion": "disjoint authorization paired bootstrap lower CI > 0"},
        "epochs": int(epochs), "device": str(device), "cuda_preflight": cuda,
        "outcome_labels_opened": False,
        "claim_boundary": "Task-aligned source-successor ranking with disjoint calibration authorization; confirmation requires sealed fresh outcome roots.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score(
    source: Path, source_arrays: Path, context: Path, frozen_selection: Path, output: Path, *,
    device: str = "cuda", manifest: Path | None = None, debug_opened_outcomes: bool = False,
) -> dict[str, Any]:
    cuda = require_cuda_recurrent(device)
    frozen = json.loads(frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("schema") != FROZEN_SCHEMA:
        raise ValueError("task-contrast score requires a current frozen selection")
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_path)
    if frozen.get("manifest_sha256") != v6._sha256(manifest):
        raise ValueError("task-contrast frozen selection does not bind the current manifest")
    if (frozen.get("source_sha256"), frozen.get("source_arrays_sha256"), frozen.get("context_sha256")) != (v6._sha256(source), v6._sha256(source_arrays), v6._sha256(context)):
        raise ValueError("task-contrast frozen selection does not bind source/context")
    _source_metadata, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    context_metadata, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, context_metadata, strict_counts=True)
    recurrent, action, fit = _load_model(frozen, frozen_selection, device=device)
    output.mkdir(parents=True, exist_ok=True)
    label_path = output / "temporal_ood_v7_task_contrast_outcomes.json"
    preexisting_v6_outcome = context.parent / "temporal_ood_v6_outcome_labels.json"
    if (label_path.exists() or preexisting_v6_outcome.exists()) and not debug_opened_outcomes:
        raise FileExistsError("outcomes are already materialized; pass --debug-opened-outcomes only for a non-confirmatory replay")
    outcome_rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = {}
    for row in outcome_rows:
        terminal = v6._simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64))
        labels[v6._key(row)] = float(np.linalg.norm(np.asarray(terminal[:2]) - np.asarray(row["goal_raw"][:2])))
    if not debug_opened_outcomes:
        label_path.write_text(json.dumps({"schema": "pointmaze_temporal_ood_v7_generated_outcomes.v1", "generated_after_frozen_selection_sha256": v6._sha256(frozen_selection), "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recurrent_cost, action_cost = _predict(recurrent, action, fit, outcome_rows)
    actual = np.asarray([labels[v6._key(row)] for row in outcome_rows])
    recurrent_result, action_result = v6._selection(outcome_rows, recurrent_cost, actual), v6._selection(outcome_rows, action_cost, actual)
    selected_policy = dict(frozen.get("selected_policy", {"kind": "recurrent"}))
    selected_result = v6._selection(outcome_rows, _policy_cost(outcome_rows, recurrent_cost, action_cost, selected_policy), actual)
    result = {
        "schema": SCORE_SCHEMA, "debug_opened_outcomes": bool(debug_opened_outcomes),
        "outcome_labels_generated_after_frozen_selection": not bool(debug_opened_outcomes),
        "recurrent": recurrent_result, "action_only": action_result, "selected_policy": selected_policy, "selected": selected_result,
        "paired_recurrent_over_action_only": v6._paired(recurrent_result, action_result),
        "paired_selected_over_action_only": v6._paired(selected_result, action_result),
        "active_policy": selected_policy["kind"] if frozen["authorization"]["permitted"] else "action_only_fallback",
        "active": selected_result if frozen["authorization"]["permitted"] else action_result,
        "device": str(device), "cuda_preflight": cuda,
        "claim_boundary": "Debug scores are not confirmation. Non-debug scores are renderer-backed offline decision-transfer evidence, not live visual planning.",
    }
    (output / "task_contrast_score_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "score"):
        item = sub.add_parser(name)
        item.add_argument("--source", type=Path, required=True)
        item.add_argument("--source-arrays", type=Path, required=True)
        item.add_argument("--context", type=Path, required=True)
        item.add_argument("--frozen-selection", type=Path, required=name == "score")
        item.add_argument("--manifest", type=Path, default=None)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--device", choices=("cuda",), default="cuda")
    calibration = sub.choices["calibrate"]
    calibration.add_argument("--calibration-labels", type=Path, required=True)
    calibration.add_argument("--epochs", type=int, default=300)
    sub.choices["score"].add_argument("--debug-opened-outcomes", action="store_true")
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "calibrate":
        args.pop("frozen_selection", None)
        result = calibrate(**args)
    else:
        result = score(**args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
