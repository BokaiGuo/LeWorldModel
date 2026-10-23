"""V9 preregistered confirmation of a fixed direct visual task ranker.

V8 is an opened development panel.  Its selection split happened to prefer an
action-only carrier despite both visual arms improving the held panel.  V9
therefore makes one narrow, predeclared confirmation test: a 32-wide,
500-epoch causal direct visual task ranker versus its matched action-only
arm.  There is no model-family or policy selection at calibration time.

Calibration is used only as a disjoint authorization gate.  Thus V9 cannot
convert a favourable held result into a controller by post-hoc selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v7_task_contrast as v7
from experiments.gpu_preflight import require_cuda_recurrent


FROZEN_SCHEMA = "pointmaze_temporal_ood_v9_fixed_direct_frozen.v1"
MODEL_SCHEMA = "pointmaze_temporal_ood_v9_fixed_direct_model.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v9_fixed_direct_score.v1"
HIDDEN, EPOCHS = 32, 500


def _model_path(frozen: Path) -> Path:
    return frozen.with_name(f"{frozen.stem}_model.pt")


def _write_model(path: Path, direct: v7.HistoryTaskContrastRanker, action: v7.ActionOnlyTaskContrastRanker, fit: dict[str, np.ndarray]) -> str:
    torch.save({
        "schema": MODEL_SCHEMA, "hidden": HIDDEN, "history_dim": int(fit["history_mean"].shape[-1]), "action_dim": int(fit["candidate_mean"].shape[-1]),
        "direct_state_dict": {key: value.detach().cpu() for key, value in direct.state_dict().items()},
        "action_state_dict": {key: value.detach().cpu() for key, value in action.state_dict().items()},
        "normalizer": {key: torch.as_tensor(value, dtype=torch.float32) for key, value in fit.items()},
        "runner_source_sha256": v6._sha256(Path(__file__)),
    }, path)
    return v6._sha256(path)


def _load_model(frozen: dict[str, Any], frozen_path: Path, *, device: str) -> tuple[v7.HistoryTaskContrastRanker, v7.ActionOnlyTaskContrastRanker, dict[str, np.ndarray]]:
    path, expected = _model_path(frozen_path), str(frozen.get("model_checkpoint_sha256", ""))
    if not expected or not path.is_file() or v6._sha256(path) != expected:
        raise ValueError("V9 fixed-direct model is missing or hash-mismatched")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != MODEL_SCHEMA or payload.get("runner_source_sha256") != v6._sha256(Path(__file__)):
        raise ValueError("V9 fixed-direct model has an incompatible runner")
    direct = v7.HistoryTaskContrastRanker(int(payload["history_dim"]), int(payload["action_dim"]), HIDDEN).to(device)
    action = v7.ActionOnlyTaskContrastRanker(int(payload["action_dim"]), HIDDEN).to(device)
    direct.load_state_dict(payload["direct_state_dict"], strict=True); action.load_state_dict(payload["action_state_dict"], strict=True)
    required = {"history_mean", "history_scale", "candidate_mean", "candidate_scale", "target_mean", "target_scale"}
    normalizer = payload.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != required:
        raise ValueError("V9 fixed-direct model lacks a complete normalizer")
    return direct.eval(), action.eval(), {key: value.detach().cpu().numpy().astype(np.float32, copy=False) for key, value in normalizer.items()}


def _predict(direct: v7.HistoryTaskContrastRanker, action: v7.ActionOnlyTaskContrastRanker, fit: dict[str, np.ndarray], rows: list[dict[str, Any]], *, cyclic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    history, candidate, _state, _goal = v6._tensors(rows)
    if cyclic:
        history = v6._cyclic_root_history(rows, history)
    device = next(direct.parameters()).device
    with torch.inference_mode():
        h = torch.tensor((history - fit["history_mean"]) / fit["history_scale"], device=device)
        c = torch.tensor((candidate - fit["candidate_mean"]) / fit["candidate_scale"], device=device)
        direct_cost = direct(h, c).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
        action_cost = action(c).cpu().numpy() * fit["target_scale"] + fit["target_mean"]
    return np.asarray(direct_cost, dtype=np.float64), np.asarray(action_cost, dtype=np.float64)


def _ensure_fresh(context: Path, output: Path) -> None:
    if output.exists() or _model_path(output).exists() or (context.parent / "temporal_ood_v9_fixed_direct_outcomes.json").exists():
        raise FileExistsError("V9 calibration is sealed: output/model/outcome already exists")


def calibrate(source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *, device: str = "cuda", manifest: Path | None = None) -> dict[str, Any]:
    _ensure_fresh(context, output)
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    runtime = require_cuda_recurrent(device)
    _meta, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA); cm, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    # V10-style enlarged calibration panels retain the exact candidate and
    # causality contract, but deliberately exceed the historical 64/48/32
    # root budget to reduce authorization variance.
    v6._validate_builder_contract(source_rows, context_rows, cm, strict_counts=False)
    calibration_rows = [row for row in context_rows if str(row["split"]) == "calibration"]
    costs = v6._calibration_costs(calibration_labels, calibration_rows, require_identity=True)
    cost_by_key = {v6._key(row): value for row, value in zip(calibration_rows, costs)}
    partition = v6.deterministic_calibration_split(context_rows)
    authorization_roots = set(partition["authorization"])
    authorization_rows = [row for row in calibration_rows if str(row["instance_id"]) in authorization_roots]
    arrays = np.load(source_arrays, allow_pickle=False)
    direct, action, fit = v7._fit(source_rows, arrays, hidden=HIDDEN, epochs=EPOCHS, device=device)
    direct_cost, action_cost = _predict(direct, action, fit, authorization_rows)
    actual = np.asarray([cost_by_key[v6._key(row)] for row in authorization_rows])
    direct_result, action_result = v6._selection(authorization_rows, direct_cost, actual), v6._selection(authorization_rows, action_cost, actual)
    gain = v6._paired(direct_result, action_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = _write_model(_model_path(output), direct, action, fit)
    report = {
        "schema": FROZEN_SCHEMA, "objective": "preregistered_fixed_direct_visual_task_ranking", "fixed_policy": "direct_visual", "fixed_hidden": HIDDEN, "fixed_epochs": EPOCHS,
        "source_sha256": v6._sha256(source), "source_arrays_sha256": v6._sha256(source_arrays), "context_sha256": v6._sha256(context), "manifest_sha256": v6._sha256(manifest),
        "calibration_labels_sha256": v6._sha256(calibration_labels), "calibration_labels_path": str(calibration_labels.resolve()), "calibration_partition": partition,
        "model_checkpoint_sha256": model_hash,
        "authorization": {"direct_visual": direct_result, "action_only": action_result, "direct_over_action_only": gain, "permitted": bool(gain["lower"] > 0.0), "criterion": "fixed direct policy; disjoint authorization paired bootstrap lower CI > 0"},
        "outcome_labels_opened": False, "device": str(device), "cuda_preflight": runtime,
        "claim_boundary": "Preregistered no-selector V9 confirmation; source-only training and disjoint authorization, with fresh outcomes still required.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score(source: Path, source_arrays: Path, context: Path, frozen_selection: Path, output: Path, *, device: str = "cuda", manifest: Path | None = None) -> dict[str, Any]:
    runtime = require_cuda_recurrent(device)
    frozen = json.loads(frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("schema") != FROZEN_SCHEMA or frozen.get("fixed_policy") != "direct_visual":
        raise ValueError("V9 score requires its fixed direct frozen protocol")
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"; calibration = Path(str(frozen.get("calibration_labels_path", "")))
    v6._validate_manifest(manifest, source, source_arrays, context, calibration)
    if frozen.get("manifest_sha256") != v6._sha256(manifest) or (frozen.get("source_sha256"), frozen.get("source_arrays_sha256"), frozen.get("context_sha256")) != (v6._sha256(source), v6._sha256(source_arrays), v6._sha256(context)):
        raise ValueError("V9 frozen protocol does not bind this panel")
    _meta, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA); cm, context_rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, context_rows, cm, strict_counts=False)
    direct, action, fit = _load_model(frozen, frozen_selection, device=device)
    output.mkdir(parents=True, exist_ok=True); labels_path = output / "temporal_ood_v9_fixed_direct_outcomes.json"
    if labels_path.exists():
        raise FileExistsError("V9 outcomes are already materialized")
    rows = [row for row in context_rows if str(row["split"]) == "outcome"]
    labels = {v6._key(row): float(np.linalg.norm(v6._simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64))[:2] - np.asarray(row["goal_raw"][:2], dtype=np.float64))) for row in rows}
    labels_path.write_text(json.dumps({"schema": "pointmaze_temporal_ood_v9_generated_outcomes.v1", "generated_after_frozen_selection_sha256": v6._sha256(frozen_selection), "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    direct_cost, action_cost = _predict(direct, action, fit, rows); cyclic_cost, _ = _predict(direct, action, fit, rows, cyclic=True)
    actual = np.asarray([labels[v6._key(row)] for row in rows])
    direct_result, action_result, cyclic_result = v6._selection(rows, direct_cost, actual), v6._selection(rows, action_cost, actual), v6._selection(rows, cyclic_cost, actual)
    report = {
        "schema": SCORE_SCHEMA, "outcome_labels_generated_after_frozen_selection": True, "direct_visual": direct_result, "action_only": action_result, "cyclic_direct_visual": cyclic_result,
        "paired_direct_over_action_only": v6._paired(direct_result, action_result), "paired_direct_over_cyclic": v6._paired(direct_result, cyclic_result),
        "active_policy": "direct_visual" if frozen["authorization"]["permitted"] else "action_only_fallback", "active": direct_result if frozen["authorization"]["permitted"] else action_result,
        "device": str(device), "cuda_preflight": runtime, "runtime_contract": {"uses_privileged_state_at_runtime": False},
        "claim_boundary": "Sealed root-disjoint renderer-backed decision-transfer confirmation, not a live visual planning result.",
    }
    (output / "fixed_direct_score_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "score"):
        item = sub.add_parser(name)
        item.add_argument("--source", type=Path, required=True); item.add_argument("--source-arrays", type=Path, required=True); item.add_argument("--context", type=Path, required=True)
        item.add_argument("--manifest", type=Path, default=None); item.add_argument("--output", type=Path, required=True); item.add_argument("--device", choices=("cuda",), default="cuda")
    sub.choices["calibrate"].add_argument("--calibration-labels", type=Path, required=True)
    sub.choices["score"].add_argument("--frozen-selection", type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    print(json.dumps(calibrate(**args) if command == "calibrate" else score(**args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
