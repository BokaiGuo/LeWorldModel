"""I/O helpers shared by LeWM experiment scripts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_empty_chunk_dir(path: str | Path, pattern: str = "chunk_*.npz") -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob(pattern))
    if existing:
        sample = ", ".join(p.name for p in existing[:3])
        suffix = "..." if len(existing) > 3 else ""
        raise FileExistsError(
            f"refusing to overwrite existing chunk files in {path}: {sample}{suffix}. "
            "Move the chunk directory, choose a fresh output path, or remove stale chunks "
            "before restarting the audit."
        )
    return path


def _chunk_index(path: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)\.npz", path.name)
    if not match:
        raise ValueError(f"unexpected chunk filename: {path.name}")
    return int(match.group(1))


def list_chunk_paths(path: str | Path, pattern: str = "chunk_*.npz") -> list[Path]:
    return sorted(Path(path).glob(pattern), key=_chunk_index)


def validate_contiguous_chunks(paths: list[Path]) -> None:
    indices = [_chunk_index(path) for path in paths]
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            f"chunk files must be contiguous from chunk_000000.npz; "
            f"got indices {indices[:8]}{'...' if len(indices) > 8 else ''}"
        )


def prepare_resumable_chunk_dir(
    path: str | Path,
    *,
    resume: bool = False,
    pattern: str = "chunk_*.npz",
) -> tuple[Path, list[Path], int]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = list_chunk_paths(path, pattern=pattern)
    if existing and not resume:
        sample = ", ".join(p.name for p in existing[:3])
        suffix = "..." if len(existing) > 3 else ""
        raise FileExistsError(
            f"refusing to overwrite existing chunk files in {path}: {sample}{suffix}. "
            "Pass --resume to append after existing chunks, move the chunk directory, "
            "choose a fresh output path, or remove stale chunks before restarting."
        )
    validate_contiguous_chunks(existing)
    return path, existing, len(existing)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def write_run_status(output: str | Path, stage: str, status: str, **fields: Any) -> dict[str, Any]:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **fields,
    }
    output = Path(output)
    save_json(output / "run_status.json", record)
    append_jsonl(output / "run_events.jsonl", record)
    return record


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = ensure_parent(path)
    np.savez_compressed(path, **arrays)


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _remap_legacy_vit_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map older Transformers ViT checkpoint keys onto current ViTModel keys."""

    replacements = (
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
        (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
        (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_before\.", r"encoder.layers.\1.layernorm_before."),
        (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_after\.", r"encoder.layers.\1.layernorm_after."),
    )
    remapped: dict[str, torch.Tensor] = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key
        for pattern, repl in replacements:
            new_key = re.sub(pattern, repl, new_key)
        changed = changed or new_key != key
        remapped[new_key] = value
    return remapped if changed else state_dict


def _load_pretrained_compat(policy: str):
    import stable_worldmodel as swm
    from hydra.utils import instantiate

    cache_dir = swm.data.get_cache_dir(sub_folder="checkpoints")
    checkpoint_path, config = swm.wm.utils._resolve(policy, cache_dir)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = instantiate(config)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(_remap_legacy_vit_state_dict(state_dict))
    return model


def load_policy(policy: str, device: str = "cuda"):
    """Load a LeWM policy without poisoning CUDA on legacy checkpoints.

    The installed ``stable_worldmodel`` loader cannot read the legacy ViT key
    layout used by the checked-in LeWM checkpoint.  Calling it first raises
    during state-dict loading and, on the current CUDA/cu13 runtime, can leave
    the CUDA initialization path in a bad state before the compatibility
    loader is reached.  Resolve and remap the checkpoint directly for named
    LeWM policies; use the generic loader only for explicit checkpoint paths.
    """
    path = Path(policy)
    if str(device).startswith("cuda"):
        # Initialize CUDA before importing stable_worldmodel.  Its import path
        # initializes JAX/registries and can otherwise expose the host's NVML
        # mismatch as a spurious "No CUDA GPUs" error on the first tensor
        # allocation.
        torch.empty(0, device=device)
    if path.exists():
        model = torch.load(path, map_location="cpu", weights_only=False)
    else:
        model = _load_pretrained_compat(policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model

