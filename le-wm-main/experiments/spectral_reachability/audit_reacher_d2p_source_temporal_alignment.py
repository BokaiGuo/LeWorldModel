"""CUDA-only audit of Reacher action-packet/frame temporal alignment.

The D2P runtime predicts the successor of an observed RGB root from the
three preceding five-control packets.  A one-packet error in the HDF5/action
index convention would leave gradients and deterministic replay intact while
making the physical successor ranking appear to fail.  This audit therefore
compares the released checkpoint's one-step latent error for the declared
packet alignment and its immediate neighbours on source-held-out clips.

It is diagnostic only: it neither updates a checkpoint nor opens a planning
outcome panel.  CPU execution is deliberately rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import _load_action_stats, install_missing_field_compat


SCHEMA = "reacher_d2p_source_temporal_alignment_audit.v1"
FRAME_SKIP = 5
HISTORY = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packetize(actions: np.ndarray, *, start: int, packet_shift: int) -> np.ndarray:
    """Return three consecutive time-major five-control action packets.

    For source RGB frames x[t], x[t+5], x[t+10], x[t+15], the declared
    alignment is shift=0: packets beginning at t, t+5, and t+10 predict the
    observed successor x[t+15].  Neighbouring shifts are a falsification
    probe, never a deployment choice.
    """

    values = np.asarray(actions, dtype=np.float32)
    begin = int(start) + int(packet_shift) * FRAME_SKIP
    end = begin + HISTORY * FRAME_SKIP
    if values.ndim != 2 or values.shape[1] != 2 or begin < 0 or end > len(values):
        raise ValueError("source action packet window is out of bounds")
    packet = values[begin:end].reshape(HISTORY, FRAME_SKIP * values.shape[1])
    if not np.isfinite(packet).all():
        raise ValueError("source action packet window contains non-finite values")
    return np.ascontiguousarray(packet)


def _valid_clip_starts(lengths: np.ndarray, *, packet_shifts: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return (episode, local start) pairs valid for RGB and every shift."""

    if not packet_shifts:
        raise ValueError("at least one packet shift is required")
    lower = max(0, -min(packet_shifts) * FRAME_SKIP)
    # Four frames require x[t + 15]; the largest action probe needs three
    # packets ending at t + (shift + 3) * 5.
    upper_padding = max(HISTORY * FRAME_SKIP, (max(packet_shifts) + HISTORY) * FRAME_SKIP)
    clips: list[tuple[int, int]] = []
    for episode, length in enumerate(np.asarray(lengths, dtype=np.int64).tolist()):
        last_start = int(length) - max(HISTORY * FRAME_SKIP + 1, upper_padding)
        clips.extend((episode, start) for start in range(lower, last_start + 1))
    if not clips:
        raise ValueError("source HDF5 contains no clips valid for every temporal-alignment probe")
    return clips


def _one_step_error(model: torch.nn.Module, pixels: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return per-example squared latent error for x[t+15]."""

    if pixels.ndim != 5 or pixels.shape[1] != HISTORY + 1:
        raise ValueError("source pixels must contain exactly four RGB observations")
    if tuple(actions.shape[:2]) != (pixels.shape[0], HISTORY):
        raise ValueError("source actions must contain exactly three action packets")
    with torch.no_grad():
        embeddings = model.encode({"pixels": pixels})["emb"]
        prediction = model.predict(
            embeddings[:, :HISTORY], model.action_encoder(actions),
        )[:, -1]
    return (prediction - embeddings[:, HISTORY]).square().mean(dim=-1)


def _load_batch(
    handle: h5py.File, samples: list[tuple[int, int]], *, packet_shift: int, root_anchored: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    pixels: list[np.ndarray] = []
    packets: list[np.ndarray] = []
    for episode, start in samples:
        offset = int(offsets[episode])
        # The deployed P0/P1 adapter predicts after its observed planning
        # root: [x_{t+5}, x_{t+10}, x_{t+15}] plus a future action should
        # predict x_{t+20}. The historical source check remains available as
        # the default; this branch tests that root-anchored convention.
        frame_offset = FRAME_SKIP if root_anchored else 0
        frame_rows = offset + int(start) + frame_offset + np.arange(HISTORY + 1, dtype=np.int64) * FRAME_SKIP
        frames = np.asarray(handle["pixels"][frame_rows], dtype=np.uint8)
        if frames.shape != (HISTORY + 1, 224, 224, 3):
            raise ValueError(f"unexpected official Reacher RGB layout: {frames.shape}")
        episode_actions = np.asarray(handle["action"][offset : offset + int(handle["ep_len"][episode])], dtype=np.float32)
        pixels.append(frames)
        packet_start = int(start) + frame_offset
        packets.append(_packetize(episode_actions, start=packet_start, packet_shift=int(packet_shift)))
    return np.stack(pixels), np.stack(packets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026102501)
    parser.add_argument("--root-anchored", action="store_true", help="Audit the deployed observed-root [t+5,t+10,t+15] convention.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher temporal-alignment audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.samples < 8 or args.output.exists():
        raise ValueError("samples must be at least eight and output must be new")
    policy, adapter, dataset, output = (args.policy.resolve(), args.adapter_artifact.resolve(), args.dataset.resolve(), args.output.resolve())
    if not dataset.is_file():
        raise FileNotFoundError(f"official Reacher source is unavailable: {dataset}")
    install_missing_field_compat()
    device = torch.device("cuda")
    torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    shifts = (-1, 0, 1)
    with h5py.File(dataset, "r", swmr=True) as handle:
        if not all(name in handle for name in ("pixels", "action", "ep_len", "ep_offset")):
            raise ValueError("official Reacher source has an incompatible HDF5 schema")
        lengths = np.asarray(handle["ep_len"])
        if args.root_anchored:
            # The added observed-root offset consumes one preceding packet.
            lengths = np.maximum(lengths - FRAME_SKIP, 0)
        valid = _valid_clip_starts(lengths, packet_shifts=shifts)
        rng = np.random.default_rng(int(args.seed))
        picked = [valid[int(index)] for index in rng.choice(len(valid), size=args.samples, replace=False)]
        frame_values, _ = _load_batch(handle, picked, packet_shift=0, root_anchored=args.root_anchored)
        errors: dict[str, list[float]] = {}
        for shift in shifts:
            _, packet_values = _load_batch(handle, picked, packet_shift=shift, root_anchored=args.root_anchored)
            prepared = preprocessor._prepare_info({"pixels": frame_values, "action": packet_values})
            values = _one_step_error(model, prepared["pixels"].to(device), prepared["action"].to(device))
            errors[str(shift)] = [float(value) for value in values.detach().cpu()]
    means = {key: float(np.mean(value)) for key, value in errors.items()}
    declared = means["0"]
    best_shift = min(means, key=means.get)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": "source-held-out CUDA temporal-alignment diagnostic; no checkpoint or planning outcome was changed",
        "inputs": {
            "policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"),
            "adapter_artifact": str(adapter), "adapter_sha256": _sha256(adapter),
            "dataset": str(dataset),
        },
        "protocol": {
            "device": str(device), "samples": int(args.samples), "seed": int(args.seed),
            "frame_rows": "[t+5,t+10,t+15,t+20]" if args.root_anchored else "[t,t+5,t+10,t+15]", "declared_packet_shift": 0,
            "packet_layout": "three time-major consecutive [5,2] packets",
            "root_anchored": bool(args.root_anchored),
        },
        "mean_latent_mse_by_packet_shift": means,
        "per_clip_latent_mse_by_packet_shift": errors,
        "best_packet_shift": int(best_shift),
        "declared_shift_is_strict_best": bool(best_shift == "0" and all(declared < means[key] for key in means if key != "0")),
        "finding": (
            "alignment defect indicated" if best_shift != "0" else
            "no one-packet source temporal-alignment defect indicated"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("best_packet_shift", "declared_shift_is_strict_best", "mean_latent_mse_by_packet_shift", "finding")}, sort_keys=True))


if __name__ == "__main__":
    main()
