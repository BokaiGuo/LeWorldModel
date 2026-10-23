"""Sealed PointMaze visual-belief intervention for V8.

V7 established that the declared candidate bank is rankable from current
PointMaze state but not from a direct visual-history score.  V8 tests the
specific, deployable repair implied by that localization: learn a causal
visual-history -> control-state belief using source-only state labels, then
condition the same pulse-aware task ranker on that *predicted* belief.

The evaluator may read source state only for source supervision and post-hoc
readout metrics.  No scoring path passes source state into the deployed
belief ranker.  A cyclic-history intervention, direct visual ranker, and
matched action-only ranker are carried as controls.
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


FROZEN_SCHEMA = "pointmaze_temporal_ood_v8_belief_state_frozen.v1"
MODEL_SCHEMA = "pointmaze_temporal_ood_v8_belief_state_model.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v8_belief_state_score.v1"
TRAIN_SEED = 2026103001
HIDDEN_WIDTHS = (32, 64)
DEFAULT_EPOCHS = 500


class VisualBeliefStateTaskRanker(nn.Module):
    """Causal visual belief with a task-cost readout conditioned on belief."""

    def __init__(self, history_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.history_encoder = nn.Sequential(nn.Linear(history_dim, hidden), nn.SiLU())
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.state_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden)
        self.film = nn.Linear(4, hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # The initial cost path is candidate-only; state effects must be learnt
        # from source data rather than injected by an initialization artifact.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, history: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, final = self.history_gru(self.history_encoder(history))
        belief = self.state_head(final[-1])
        scale, shift = self.film(belief).chunk(2, dim=-1)
        action = self.candidate_encoder(candidate)
        return self.head((1.0 + scale) * action + shift).squeeze(-1), belief


def _normalizer(value: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return value.mean(axis=axes).astype(np.float32), np.maximum(value.std(axis=axes), 1.0e-3).astype(np.float32)


def _source_task_costs(source: list[dict[str, Any]], arrays: Any) -> np.ndarray:
    return v7._source_task_costs(source, arrays)


def _fit(
    source: list[dict[str, Any]], arrays: Any, *, hidden: int, epochs: int, device: str,
) -> tuple[VisualBeliefStateTaskRanker, v7.HistoryTaskContrastRanker, v7.ActionOnlyTaskContrastRanker, dict[str, np.ndarray]]:
    history, candidate, state, _goal = v6._tensors(source)
    cost = _source_task_costs(source, arrays)
    v6._seed(TRAIN_SEED)
    hm, hs = _normalizer(history, (0, 1))
    cm, cs = _normalizer(candidate, (0, 1))
    sm, ss = _normalizer(state, (0,))
    ym, ys = _normalizer(cost, (0,))
    belief = VisualBeliefStateTaskRanker(history.shape[-1], candidate.shape[-1], hidden).to(device)
    direct = v7.HistoryTaskContrastRanker(history.shape[-1], candidate.shape[-1], hidden).to(device)
    action = v7.ActionOnlyTaskContrastRanker(candidate.shape[-1], hidden).to(device)
    bopt = torch.optim.AdamW(belief.parameters(), lr=2e-3, weight_decay=1e-4)
    dopt = torch.optim.AdamW(direct.parameters(), lr=2e-3, weight_decay=1e-4)
    aopt = torch.optim.AdamW(action.parameters(), lr=2e-3, weight_decay=1e-4)
    th = torch.tensor((history - hm) / hs, device=device)
    tc = torch.tensor((candidate - cm) / cs, device=device)
    ts = torch.tensor((state - sm) / ss, device=device)
    ty = torch.tensor((cost - ym) / ys, device=device)
    groups = v7._root_index_groups(source)
    for _ in range(int(epochs)):
        bopt.zero_grad(set_to_none=True)
        bp, bs = belief(th, tc)
        # State supervision targets the entire sufficient control state, while
        # task/ranking losses force retained state coordinates to be useful.
        b_loss = (bp - ty).square().mean() + v7._rank_loss(bp, ty, groups) + (bs - ts).square().mean()
        b_loss.backward(); bopt.step()
        dopt.zero_grad(set_to_none=True)
        dp = direct(th, tc)
        d_loss = (dp - ty).square().mean() + v7._rank_loss(dp, ty, groups)
        d_loss.backward(); dopt.step()
        aopt.zero_grad(set_to_none=True)
        ap = action(tc)
        a_loss = (ap - ty).square().mean() + v7._rank_loss(ap, ty, groups)
        a_loss.backward(); aopt.step()
    return belief.eval(), direct.eval(), action.eval(), {
        "history_mean": hm, "history_scale": hs, "candidate_mean": cm, "candidate_scale": cs,
        "state_mean": sm, "state_scale": ss, "target_mean": np.asarray(ym, dtype=np.float32),
        "target_scale": np.asarray(ys, dtype=np.float32),
    }


def _predict(
    belief: VisualBeliefStateTaskRanker, direct: v7.HistoryTaskContrastRanker,
    action: v7.ActionOnlyTaskContrastRanker, fit: dict[str, np.ndarray], records: list[dict[str, Any]], *,
    cyclic_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history, candidate, state, _goal = v6._tensors(records)
    if cyclic_history:
        history = v6._cyclic_root_history(records, history)
    device = next(belief.parameters()).device
    with torch.inference_mode():
        th = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        tc = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        bc, bstate = belief(th, tc)
        dc = direct(th, tc)
        ac = action(tc)
    scale, offset = fit["target_scale"], fit["target_mean"]
    return (
        np.asarray(bc.cpu().numpy() * scale + offset, dtype=np.float64),
        np.asarray(dc.cpu().numpy() * scale + offset, dtype=np.float64),
        np.asarray(ac.cpu().numpy() * scale + offset, dtype=np.float64),
        np.asarray(bstate.cpu().numpy() * fit["state_scale"] + fit["state_mean"], dtype=np.float64),
        np.asarray(state, dtype=np.float64),
    )


def _policy_cost(kind: str, belief: np.ndarray, direct: np.ndarray, action: np.ndarray) -> np.ndarray:
    choices = {"belief_state": belief, "direct_visual": direct, "action_only": action}
    if kind not in choices:
        raise ValueError(f"unsupported V8 policy {kind!r}")
    return choices[kind]


def _state_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("belief-state metrics require aligned [rows, 4] arrays")
    error = predicted - actual
    nonzero = np.abs(actual[:, 2:]) > 1.0e-4
    velocity_sign = np.sign(predicted[:, 2:]) == np.sign(actual[:, 2:])
    return {
        "position_rmse": float(np.sqrt(np.mean(np.square(error[:, :2])))),
        "velocity_rmse": float(np.sqrt(np.mean(np.square(error[:, 2:])))),
        "state_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "velocity_sign_accuracy": float(velocity_sign[nonzero].mean()) if bool(nonzero.any()) else float("nan"),
    }


def _model_path(frozen_path: Path) -> Path:
    return frozen_path.with_name(f"{frozen_path.stem}_model.pt")


def _write_model(
    path: Path, belief: VisualBeliefStateTaskRanker, direct: v7.HistoryTaskContrastRanker,
    action: v7.ActionOnlyTaskContrastRanker, fit: dict[str, np.ndarray], *, hidden: int,
) -> str:
    payload = {
        "schema": MODEL_SCHEMA, "hidden": int(hidden), "history_dim": int(fit["history_mean"].shape[-1]),
        "action_dim": int(fit["candidate_mean"].shape[-1]),
        "belief_state_dict": {key: value.detach().cpu() for key, value in belief.state_dict().items()},
        "direct_state_dict": {key: value.detach().cpu() for key, value in direct.state_dict().items()},
        "action_state_dict": {key: value.detach().cpu() for key, value in action.state_dict().items()},
        "normalizer": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in fit.items()},
        "runner_source_sha256": v6._sha256(Path(__file__)),
    }
    torch.save(payload, path)
    return v6._sha256(path)


def _load_model(
    frozen: dict[str, Any], frozen_path: Path, *, device: str,
) -> tuple[VisualBeliefStateTaskRanker, v7.HistoryTaskContrastRanker, v7.ActionOnlyTaskContrastRanker, dict[str, np.ndarray]]:
    path = Path(str(frozen.get("model_checkpoint_path", "")))
    if not path.is_file():
        path = _model_path(frozen_path)
    expected = str(frozen.get("model_checkpoint_sha256", ""))
    if not expected or not path.is_file() or v6._sha256(path) != expected:
        raise ValueError("V8 frozen checkpoint is missing or hash-mismatched")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA or payload.get("runner_source_sha256") != v6._sha256(Path(__file__)):
        raise ValueError("V8 model has an incompatible schema or runner")
    belief = VisualBeliefStateTaskRanker(int(payload["history_dim"]), int(payload["action_dim"]), int(payload["hidden"])).to(device)
    direct = v7.HistoryTaskContrastRanker(int(payload["history_dim"]), int(payload["action_dim"]), int(payload["hidden"])).to(device)
    action = v7.ActionOnlyTaskContrastRanker(int(payload["action_dim"]), int(payload["hidden"])).to(device)
    belief.load_state_dict(payload["belief_state_dict"], strict=True)
    direct.load_state_dict(payload["direct_state_dict"], strict=True)
    action.load_state_dict(payload["action_state_dict"], strict=True)
    required = {"history_mean", "history_scale", "candidate_mean", "candidate_scale", "state_mean", "state_scale", "target_mean", "target_scale"}
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != required:
        raise ValueError("V8 model lacks a complete normalizer")
    fit = {key: value.detach().cpu().numpy().astype(np.float32, copy=False) for key, value in normalizer.items()}
    return belief.eval(), direct.eval(), action.eval(), fit


def _ensure_fresh_calibration_output(context: Path, output: Path) -> None:
    outcome = context.parent / "temporal_ood_v8_belief_state_outcomes.json"
    model = _model_path(output)
    if output.exists() or model.exists() or outcome.exists():
        raise FileExistsError("V8 calibration is sealed: an output/model or held outcome already exists")


def calibrate(
    source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *,
    epochs: int = DEFAULT_EPOCHS, device: str = "cuda", manifest: Path | None = None,
) -> dict[str, Any]:
    _ensure_fresh_calibration_output(context, output)
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    runtime = require_cuda_recurrent(device)
    _sm, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    cm, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, cm, strict_counts=True)
    calibration = [row for row in context_rows if str(row["split"]) == "calibration"]
    labels = v6._calibration_costs(calibration_labels, calibration, require_identity=True)
    cost_by_key = {v6._key(row): value for row, value in zip(calibration, labels)}
    partition = v6.deterministic_calibration_split(context_rows)
    selected_roots, authorization_roots = set(partition["selection"]), set(partition["authorization"])
    select = [row for row in calibration if str(row["instance_id"]) in selected_roots]
    authorize = [row for row in calibration if str(row["instance_id"]) in authorization_roots]
    arrays = np.load(source_arrays, allow_pickle=False)
    choices: list[dict[str, Any]] = []
    for hidden in HIDDEN_WIDTHS:
        belief, direct, action, fit = _fit(source_rows, arrays, hidden=hidden, epochs=epochs, device=device)
        bc, dc, ac, _bp, _actual_state = _predict(belief, direct, action, fit, select)
        actual = np.asarray([cost_by_key[v6._key(row)] for row in select])
        for kind in ("belief_state", "direct_visual", "action_only"):
            choices.append({"hidden": hidden, "policy": {"kind": kind}, "selected": v6._selection(select, _policy_cost(kind, bc, dc, ac), actual)})
    chosen = min(choices, key=lambda row: (row["selected"]["mean_regret"], row["hidden"], row["policy"]["kind"]))
    belief, direct, action, fit = _fit(source_rows, arrays, hidden=int(chosen["hidden"]), epochs=epochs, device=device)
    bc, dc, ac, _bp, _actual_state = _predict(belief, direct, action, fit, authorize)
    actual = np.asarray([cost_by_key[v6._key(row)] for row in authorize])
    authorization_arms = {
        "belief_state": v6._selection(authorize, bc, actual),
        "direct_visual": v6._selection(authorize, dc, actual),
        "action_only": v6._selection(authorize, ac, actual),
    }
    baseline = authorization_arms["action_only"]
    selected = authorization_arms[str(chosen["policy"]["kind"])]
    gain = v6._paired(selected, baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = _model_path(output)
    model_hash = _write_model(model, belief, direct, action, fit, hidden=int(chosen["hidden"]))
    report = {
        "schema": FROZEN_SCHEMA, "objective": "causal_visual_history_to_control_belief_to_task_ranking",
        "source_sha256": v6._sha256(source), "source_arrays_sha256": v6._sha256(source_arrays),
        "context_sha256": v6._sha256(context), "manifest_sha256": v6._sha256(manifest),
        "calibration_labels_sha256": v6._sha256(calibration_labels), "calibration_labels_path": str(calibration_labels.resolve()),
        "calibration_partition": partition, "choice_grid": choices, "selected_hidden": int(chosen["hidden"]), "selected_policy": chosen["policy"],
        "model_checkpoint_path": str(model.resolve()), "model_checkpoint_sha256": model_hash,
        "authorization": {"arms": authorization_arms, "selected_policy_over_action_only": gain, "permitted": bool(chosen["policy"]["kind"] != "action_only" and gain["lower"] > 0.0), "criterion": "disjoint authorization paired bootstrap lower CI > 0"},
        "epochs": int(epochs), "device": str(device), "cuda_preflight": runtime, "outcome_labels_opened": False,
        "runtime_contract": {"uses_visual_history_and_prior_actions": True, "uses_privileged_state_at_runtime": False, "source_state_role": "source-supervised belief target and post-hoc readout only"},
        "claim_boundary": "Source-supervised visual belief with disjoint calibration authorization; confirmation requires fresh sealed outcomes.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score(
    source: Path, source_arrays: Path, context: Path, frozen_selection: Path, output: Path, *,
    device: str = "cuda", manifest: Path | None = None, debug_opened_outcomes: bool = False,
) -> dict[str, Any]:
    runtime = require_cuda_recurrent(device)
    frozen = json.loads(frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("schema") != FROZEN_SCHEMA:
        raise ValueError("V8 score requires a current frozen selection")
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    calibration_path = Path(str(frozen.get("calibration_labels_path", "")))
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_path)
    if frozen.get("manifest_sha256") != v6._sha256(manifest) or (frozen.get("source_sha256"), frozen.get("source_arrays_sha256"), frozen.get("context_sha256")) != (v6._sha256(source), v6._sha256(source_arrays), v6._sha256(context)):
        raise ValueError("V8 frozen selection does not bind this panel")
    _sm, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA)
    cm, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, cm, strict_counts=True)
    belief, direct, action, fit = _load_model(frozen, frozen_selection, device=device)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "temporal_ood_v8_belief_state_outcomes.json"
    if labels_path.exists() and not debug_opened_outcomes:
        raise FileExistsError("V8 outcomes are already materialized; debug replay is non-confirmatory")
    outcome = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = {v6._key(row): float(np.linalg.norm(v6._simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64))[:2] - np.asarray(row["goal_raw"][:2], dtype=np.float64))) for row in outcome}
    if not debug_opened_outcomes:
        labels_path.write_text(json.dumps({"schema": "pointmaze_temporal_ood_v8_generated_outcomes.v1", "generated_after_frozen_selection_sha256": v6._sha256(frozen_selection), "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bc, dc, ac, bstate, actual_state = _predict(belief, direct, action, fit, outcome)
    cyclic_bc, cyclic_dc, _cyclic_ac, cyclic_state, _cyclic_actual = _predict(belief, direct, action, fit, outcome, cyclic_history=True)
    actual = np.asarray([labels[v6._key(row)] for row in outcome])
    results = {"belief_state": v6._selection(outcome, bc, actual), "direct_visual": v6._selection(outcome, dc, actual), "action_only": v6._selection(outcome, ac, actual), "cyclic_belief_state": v6._selection(outcome, cyclic_bc, actual), "cyclic_direct_visual": v6._selection(outcome, cyclic_dc, actual)}
    selected_kind = str(frozen["selected_policy"]["kind"])
    selected = results[selected_kind]
    report = {
        "schema": SCORE_SCHEMA, "debug_opened_outcomes": bool(debug_opened_outcomes), "outcome_labels_generated_after_frozen_selection": not bool(debug_opened_outcomes),
        **results, "selected_policy": frozen["selected_policy"], "selected": selected,
        "paired_belief_over_action_only": v6._paired(results["belief_state"], results["action_only"]),
        "paired_direct_over_action_only": v6._paired(results["direct_visual"], results["action_only"]),
        "paired_belief_over_direct": v6._paired(results["belief_state"], results["direct_visual"]),
        "paired_belief_over_cyclic": v6._paired(results["belief_state"], results["cyclic_belief_state"]),
        "belief_readout": {"matched_history": _state_metrics(bstate, actual_state), "cyclic_history": _state_metrics(cyclic_state, actual_state)},
        "active_policy": selected_kind if frozen["authorization"]["permitted"] else "action_only_fallback",
        "active": selected if frozen["authorization"]["permitted"] else results["action_only"],
        "device": str(device), "cuda_preflight": runtime,
        "runtime_contract": {"uses_privileged_state_at_runtime": False, "held_state_used_only_for_posthoc_readout": True},
        "claim_boundary": "Sealed renderer-backed offline visual decision-transfer evidence, not live visual planning. Belief gains require a positive authorization gate.",
    }
    (output / "belief_state_score_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "score"):
        item = sub.add_parser(name)
        item.add_argument("--source", type=Path, required=True); item.add_argument("--source-arrays", type=Path, required=True)
        item.add_argument("--context", type=Path, required=True); item.add_argument("--manifest", type=Path, default=None)
        item.add_argument("--output", type=Path, required=True); item.add_argument("--device", choices=("cuda",), default="cuda")
    calibration = sub.choices["calibrate"]
    calibration.add_argument("--calibration-labels", type=Path, required=True); calibration.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    scoring = sub.choices["score"]
    scoring.add_argument("--frozen-selection", type=Path, required=True); scoring.add_argument("--debug-opened-outcomes", action="store_true")
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "calibrate":
        result = calibrate(**args)
    else:
        result = score(**args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
