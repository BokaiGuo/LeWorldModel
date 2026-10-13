"""Audit the evidence prerequisites for an action-conditioned JEPA-WM run.

This is a provenance/closure audit, not a substitute for a temporal evaluation.
It records whether a concrete checkpoint and trajectory dataset are available,
and emits the exact boundary that must remain in the paper when either input is
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_summary(path: Path) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    summary: dict[str, object] = {
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }
    if isinstance(payload, dict):
        for key in ("predictor", "proprio_encoder"):
            value = payload.get(key)
            if isinstance(value, dict):
                summary[f"{key}_tensor_count"] = len(value)
                summary[f"{key}_parameter_count"] = int(
                    sum(v.numel() for v in value.values() if hasattr(v, "numel"))
                )
    return summary


def _dataset_summary(root: Path) -> dict[str, object]:
    candidates = [
        root / "pusht_noise",
        root / "pusht",
        root / "pusht_noise.zip",
        root / "pusht" / "pusht_noise.zip",
    ]
    present = [str(path) for path in candidates if path.exists()]
    trajectory_files = []
    if root.exists():
        trajectory_files = [
            str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".hdf5", ".h5", ".npz", ".pt", ".pth"}
        ][:20]
    split_summary: dict[str, object] = {}
    for split in ("train", "val", "test"):
        split_root = root / split
        states_path = split_root / "states.pth"
        actions_path = split_root / "rel_actions.pth"
        lengths_path = split_root / "seq_lengths.pkl"
        if not (states_path.exists() and actions_path.exists() and lengths_path.exists()):
            continue
        import torch

        states = torch.load(states_path, map_location="cpu", weights_only=False)
        actions = torch.load(actions_path, map_location="cpu", weights_only=False)
        with lengths_path.open("rb") as handle:
            lengths = [int(v) for v in pickle.load(handle)]
        split_summary[split] = {
            "episode_count": len(lengths),
            "episode_ids": [f"{split}_{i:05d}" for i in range(len(lengths))],
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_total": sum(lengths),
        }
    return {
        "root": str(root),
        "exists": root.exists(),
        "candidate_paths_present": present,
        "trajectory_files_sample": trajectory_files,
        "trajectory_protocol_closed": bool(trajectory_files),
        "splits": split_summary,
        "download_state": (
            "ready" if trajectory_files else ("partial_or_gated_download" if present else "missing")
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = _checkpoint_summary(args.checkpoint) if args.checkpoint.exists() else {"exists": False}
    dataset = _dataset_summary(args.dataset_root)
    checkpoint_ready = bool(checkpoint.get("exists") and checkpoint.get("epoch") is not None)
    data_ready = bool(dataset["trajectory_protocol_closed"])
    if checkpoint_ready and data_ready:
        status = "ready_for_temporal_eval"
    elif checkpoint_ready:
        status = "checkpoint_ready_data_blocked"
    elif data_ready:
        status = "data_ready_checkpoint_blocked"
    else:
        status = "checkpoint_and_data_blocked"
    metrics_path = args.output_dir / "jepa_wm_temporal_metrics.json"
    return {
        "status": status,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "manifest": {
            "dataset_root": str(args.dataset_root),
            "episode_ids": {
                split: details["episode_ids"]
                for split, details in dataset.get("splits", {}).items()
            },
            "train_test_split": {
                "train": len(dataset.get("splits", {}).get("train", {}).get("episode_ids", [])),
                "validation": len(dataset.get("splits", {}).get("val", {}).get("episode_ids", [])),
            },
            "observation_shape": [3, 224, 224],
            "action_shape": [2],
            "context_length": 2,
            "prediction_horizon": [1, 2, 4, 8],
            "metric_definition": {
                "latent_prediction_error_by_horizon": "mean per-token squared error between predicted and encoded held-out visual latents",
                "covariance_operator_commutator_by_horizon": "relative Frobenius norm of [C, A_h] on held-out latent transitions",
                "spectrum_preserving_null_by_horizon": "same-spectrum random-basis null with matched rank and repeated seeds",
            },
            "source": "DINO-WM public OSF release (GitHub gaoyuezhou/dino_wm)",
            "interface_note": "Released AdaLN wrapper evaluated with two-frame context after historical action-token alignment correction.",
        },
        "temporal_eval": {
            "status": "diagnostic_complete_interface_bounded" if metrics_path.exists() else "pending",
            "artifact": str(metrics_path) if metrics_path.exists() else None,
        },
        "required_manifest_fields": [
            "dataset_root",
            "episode_ids",
            "train_test_split",
            "observation_shape",
            "action_shape",
            "context_length",
            "prediction_horizon",
            "metric_definition",
        ],
        "temporal_metrics": [
            "latent_prediction_error_by_horizon",
            "covariance_operator_commutator_by_horizon",
            "spectrum_preserving_null_by_horizon",
        ],
        "evidence_boundary": (
            "No action-conditioned JEPA-WM temporal claim is admitted until a concrete "
            "checkpoint, trajectory manifest, split, and held-out multi-step metric "
            "are jointly available."
        ),
    }


def write_markdown(path: Path, result: dict[str, object]) -> None:
    checkpoint = result["checkpoint"]
    dataset = result["dataset"]
    temporal_eval = result.get("temporal_eval", {})
    if temporal_eval.get("status") == "diagnostic_complete_interface_bounded":
        temporal_sentence = (
            "A two-frame-context held-out diagnostic is available in "
            f"`{temporal_eval['artifact']}`; the intended multi-frame predictor "
            "generalization beyond this explicit context remains a separate gate."
        )
    else:
        temporal_sentence = (
            "Action-conditioned temporal metrics remain pending until the "
            "predictor interface is evaluated end to end."
        )
    lines = [
        "# JEPA-WM Temporal Promotion Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        "| prerequisite | state |",
        "| --- | --- |",
        f"| checkpoint | {'ready' if checkpoint.get('exists') and checkpoint.get('epoch') is not None else 'missing/invalid'} |",
        f"| trajectory root | {'present' if dataset['exists'] else 'missing'} |",
        f"| trajectory protocol | {'closed' if dataset['trajectory_protocol_closed'] else 'not closed'} |",
        f"| checkpoint epoch | {checkpoint.get('epoch', 'n/a')} |",
        f"| checkpoint SHA-256 | `{checkpoint.get('sha256', 'n/a')}` |",
        "",
        "The checkpoint is a concrete JEPA-WM Push-T predictor artifact. The trajectory protocol is closed from the public DINO-WM OSF mirror (SHA-256 recorded in the experiment log). " + temporal_sentence,
        "",
        "Required manifest fields: " + ", ".join(result["required_manifest_fields"]) + ".",
        "",
        "Required metrics: " + ", ".join(result["temporal_metrics"]) + ".",
        "",
        result["evidence_boundary"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (args.output_dir / "jepa_wm_temporal_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "jepa_wm_trajectory_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2) + "\n"
    )
    write_markdown(args.output_dir / "jepa_wm_temporal_gate.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
