"""V11 fixed visual-belief confirmation after V10 refuted direct ranking.

The direct visual task-ranker was fixed before V9/V10 and failed its enlarged
confirmation.  V11 is a separate, preregistered mechanism test: retain V8's
source-only control-state supervision and route a candidate ranker through
the *predicted* visual belief.  The policy, width and epoch count are fixed;
calibration is an authorization gate only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v8_belief_state as v8
from experiments.gpu_preflight import require_cuda_recurrent

FROZEN_SCHEMA = "pointmaze_temporal_ood_v11_fixed_belief_frozen.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v11_fixed_belief_score.v1"
HIDDEN, EPOCHS = 32, 500


def _model_path(frozen: Path) -> Path:
    return frozen.with_name(f"{frozen.stem}_model.pt")


def _ensure_fresh(context: Path, output: Path) -> None:
    if output.exists() or _model_path(output).exists() or (context.parent / "temporal_ood_v11_fixed_belief_outcomes.json").exists():
        raise FileExistsError("V11 calibration is sealed: output/model/outcome already exists")


def calibrate(source: Path, source_arrays: Path, context: Path, calibration_labels: Path, output: Path, *, device: str = "cuda", manifest: Path | None = None) -> dict[str, Any]:
    _ensure_fresh(context, output)
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"
    v6._validate_manifest(manifest, source, source_arrays, context, calibration_labels)
    runtime = require_cuda_recurrent(device)
    _meta, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA); cm, rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, rows, cm, strict_counts=False)
    calibration = [row for row in rows if str(row["split"]) == "calibration"]
    labels = v6._calibration_costs(calibration_labels, calibration, require_identity=True)
    cost = {v6._key(row): value for row, value in zip(calibration, labels)}
    partition = v6.deterministic_calibration_split(rows)
    authorization = [row for row in calibration if str(row["instance_id"]) in set(partition["authorization"])]
    arrays = np.load(source_arrays, allow_pickle=False)
    belief, direct, action, fit = v8._fit(source_rows, arrays, hidden=HIDDEN, epochs=EPOCHS, device=device)
    belief_cost, _direct_cost, action_cost, _state, _actual_state = v8._predict(belief, direct, action, fit, authorization)
    actual = np.asarray([cost[v6._key(row)] for row in authorization])
    belief_result, action_result = v6._selection(authorization, belief_cost, actual), v6._selection(authorization, action_cost, actual)
    gain = v6._paired(belief_result, action_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = v8._write_model(_model_path(output), belief, direct, action, fit, hidden=HIDDEN)
    report = {
        "schema": FROZEN_SCHEMA, "objective": "preregistered_fixed_visual_belief_task_ranking", "fixed_policy": "belief_state", "fixed_hidden": HIDDEN, "fixed_epochs": EPOCHS,
        "source_sha256": v6._sha256(source), "source_arrays_sha256": v6._sha256(source_arrays), "context_sha256": v6._sha256(context), "manifest_sha256": v6._sha256(manifest),
        "calibration_labels_sha256": v6._sha256(calibration_labels), "calibration_labels_path": str(calibration_labels.resolve()), "calibration_partition": partition,
        "model_checkpoint_path": str(_model_path(output).resolve()), "model_checkpoint_sha256": model_hash,
        "authorization": {"belief_state": belief_result, "action_only": action_result, "belief_over_action_only": gain, "permitted": bool(gain["lower"] > 0.0), "criterion": "fixed belief policy; disjoint authorization paired bootstrap lower CI > 0"},
        "outcome_labels_opened": False, "device": str(device), "cuda_preflight": runtime,
        "runtime_contract": {"uses_privileged_state_at_runtime": False, "source_state_role": "source-supervised belief target and post-hoc readout only"},
        "claim_boundary": "Preregistered fixed visual-belief confirmation; fresh sealed outcomes are required.",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score(source: Path, source_arrays: Path, context: Path, frozen_selection: Path, output: Path, *, device: str = "cuda", manifest: Path | None = None) -> dict[str, Any]:
    runtime = require_cuda_recurrent(device)
    frozen = json.loads(frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("schema") != FROZEN_SCHEMA or frozen.get("fixed_policy") != "belief_state":
        raise ValueError("V11 score requires its fixed-belief frozen protocol")
    manifest = manifest or context.parent / "temporal_ood_v6_manifest.json"; calibration = Path(str(frozen.get("calibration_labels_path", "")))
    v6._validate_manifest(manifest, source, source_arrays, context, calibration)
    if frozen.get("manifest_sha256") != v6._sha256(manifest) or (frozen.get("source_sha256"), frozen.get("source_arrays_sha256"), frozen.get("context_sha256")) != (v6._sha256(source), v6._sha256(source_arrays), v6._sha256(context)):
        raise ValueError("V11 frozen protocol does not bind this panel")
    _meta, source_rows = v6._read_panel(source, v6.SOURCE_SCHEMA); cm, rows = v6._read_panel(context, v6.CONTEXT_SCHEMA)
    v6._validate_builder_contract(source_rows, rows, cm, strict_counts=False)
    belief, direct, action, fit = v8._load_model(frozen, frozen_selection, device=device)
    output.mkdir(parents=True, exist_ok=True); labels_path = output / "temporal_ood_v11_fixed_belief_outcomes.json"
    if labels_path.exists(): raise FileExistsError("V11 outcomes are already materialized")
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    labels = {v6._key(row): float(np.linalg.norm(v6._simulate_sequence(np.asarray(row["source_state_raw"], dtype=np.float64), np.asarray(row["action_sequence"], dtype=np.float64))[:2] - np.asarray(row["goal_raw"][:2], dtype=np.float64))) for row in outcome}
    labels_path.write_text(json.dumps({"schema": "pointmaze_temporal_ood_v11_generated_outcomes.v1", "generated_after_frozen_selection_sha256": v6._sha256(frozen_selection), "labels": labels}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bc, _dc, ac, state, actual_state = v8._predict(belief, direct, action, fit, outcome)
    cyclic_bc, _cyclic_dc, _cyclic_ac, cyclic_state, _ = v8._predict(belief, direct, action, fit, outcome, cyclic_history=True)
    actual = np.asarray([labels[v6._key(row)] for row in outcome])
    belief_result, action_result, cyclic_result = v6._selection(outcome, bc, actual), v6._selection(outcome, ac, actual), v6._selection(outcome, cyclic_bc, actual)
    report = {
        "schema": SCORE_SCHEMA, "outcome_labels_generated_after_frozen_selection": True, "belief_state": belief_result, "action_only": action_result, "cyclic_belief_state": cyclic_result,
        "paired_belief_over_action_only": v6._paired(belief_result, action_result), "paired_belief_over_cyclic": v6._paired(belief_result, cyclic_result),
        "belief_readout": {"matched_history": v8._state_metrics(state, actual_state), "cyclic_history": v8._state_metrics(cyclic_state, actual_state)},
        "active_policy": "belief_state" if frozen["authorization"]["permitted"] else "action_only_fallback", "active": belief_result if frozen["authorization"]["permitted"] else action_result,
        "device": str(device), "cuda_preflight": runtime, "runtime_contract": {"uses_privileged_state_at_runtime": False, "held_state_used_only_for_posthoc_readout": True},
        "claim_boundary": "Sealed root-disjoint renderer-backed decision-transfer confirmation, not live visual planning.",
    }
    (output / "fixed_belief_score_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "score"):
        item = sub.add_parser(name); item.add_argument("--source", type=Path, required=True); item.add_argument("--source-arrays", type=Path, required=True); item.add_argument("--context", type=Path, required=True); item.add_argument("--manifest", type=Path, default=None); item.add_argument("--output", type=Path, required=True); item.add_argument("--device", choices=("cuda",), default="cuda")
    sub.choices["calibrate"].add_argument("--calibration-labels", type=Path, required=True); sub.choices["score"].add_argument("--frozen-selection", type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    print(json.dumps(calibrate(**args) if command == "calibrate" else score(**args), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
